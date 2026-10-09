"""Quant market-system layer: regime HMM, Kalman fair value, dependency graph, GARCH + Monte Carlo,
PCMCI+ causal graph, entropy pooling (skfolio). See docs/wayfinder/map.md T2-T3."""
import numpy as np
import pandas as pd


def regime(daily: pd.DataFrame) -> dict:
    """4-state Gaussian HMM on daily returns of [XAU, SPX, VIX, USDX]. States labelled by rule, not by index."""
    from hmmlearn.hmm import GaussianHMM

    r = np.log(daily).diff().dropna()
    m = GaussianHMM(4, covariance_type="full", n_iter=200, random_state=7).fit(r.values)
    p = m.predict_proba(r.values)[-1]
    names = []
    for mu in m.means_:
        xau, spx = mu[0], mu[1]
        names.append("Hedge failure" if xau < 0 and spx < 0 else "Risk-off (hedge attivo)" if xau > 0 > spx
                     else "Risk-on" if spx > 0 else "Transizione")
    probs: dict[str, float] = {}
    for n, q in zip(names, p):
        probs[n] = probs.get(n, 0.0) + float(q)
    return {"regime": max(probs, key=lambda k: probs[k]), "probs": probs}


def kalman_fair_value(y: pd.Series, X: pd.DataFrame, delta: float = 1e-4) -> dict:
    """Random-walk-beta Kalman regression y_t = x_t·b_t + e. Fair value = prior prediction; z = innovation / sd."""
    Xv = np.column_stack([np.ones(len(X)), X.values])
    k = Xv.shape[1]
    b, P = np.zeros(k), np.eye(k)
    Q, R = delta / (1 - delta) * np.eye(k), float(np.var(y.values)) * 0.01
    for t in range(len(y)):
        x = Xv[t]
        P = P + Q
        yhat, S = x @ b, x @ P @ x + R
        K = P @ x / S
        e = y.values[t] - yhat
        b, P = b + K * e, P - np.outer(K, x) @ P
    z = float(e / np.sqrt(S))
    return {"fair_value": float(np.exp(yhat)), "z": round(z, 2), "beta": dict(zip(["const", *X.columns], b.round(4))),
            "bias": float(np.clip(-z / 2, -1, 1))}  # gold rich vs drivers -> bearish lean


def dependency_graph(rets: pd.DataFrame, thr: float = 0.05) -> dict:
    """Sparse conditional-dependence graph (graphical lasso): edges only where a direct link survives."""
    from sklearn.covariance import GraphicalLassoCV

    z = (rets - rets.mean()) / rets.std()
    prec = GraphicalLassoCV().fit(z.dropna().values).precision_
    d = np.sqrt(np.diag(prec))
    pcor = -prec / np.outer(d, d)
    cols, edges = list(rets.columns), []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(pcor[i, j]) > thr:
                edges.append({"a": cols[i], "b": cols[j], "w": round(float(pcor[i, j]), 3)})
    return {"nodes": cols, "edges": edges}


def garch_mc(close: pd.Series, price: float, target: float | None, stop: float | None,
             drift: float = 0.0, horizon: int = 32, n: int = 4000) -> dict:
    """GARCH(1,1)-t sigma + bootstrapped standardized residuals -> P(target before stop) within horizon bars."""
    from arch import arch_model

    r = 1e4 * np.log(close).diff().dropna().tail(2000)  # basis points: arch wants scale 1..1000
    res = arch_model(r, p=1, q=1, dist="t").fit(disp="off")
    sigma = float(np.sqrt(res.forecast(horizon=1).variance.values[-1, 0])) / 1e4
    out = {"sigma_bar": sigma}
    if target is None or stop is None:
        return out
    std = (res.resid / res.conditional_volatility).dropna().values
    shocks = np.random.default_rng(0).choice(std, size=(n, horizon)) * sigma + drift
    paths = price * np.exp(np.cumsum(shocks, axis=1))
    up = target > price
    hit_t = (paths >= target) if up else (paths <= target)
    hit_s = (paths <= stop) if up else (paths >= stop)
    first_t = np.where(hit_t.any(1), hit_t.argmax(1), horizon + 1)
    first_s = np.where(hit_s.any(1), hit_s.argmax(1), horizon + 1)
    out.update(p_target_first=float((first_t < first_s).mean()), p_stop_first=float((first_s < first_t).mean()))
    return out


def causal_graph(daily_chg: pd.DataFrame, tau_max: int = 3) -> list[dict]:
    """PCMCI+ (tigramite) on daily changes of the drivers. Lagged links tell who leads gold."""
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    df = daily_chg.dropna()
    res = PCMCI(dataframe=pp.DataFrame(df.values, var_names=list(df.columns)),
                cond_ind_test=ParCorr()).run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=0.01)
    names, links = list(df.columns), []
    for i, j, t in zip(*np.where(res["p_matrix"] < 0.01)):
        if res["graph"][i, j, t] in ("-->", "o-o") and (t > 0 or i < j):
            links.append({"src": names[i], "dst": names[j], "lag": int(t), "val": round(float(res["val_matrix"][i, j, t]), 3)})
    return links


def entropy_pool(rets: pd.DataFrame, views: list[str]) -> dict:
    """skfolio EntropyPooling: tilt the empirical return distribution to honour the views; read P(XAUUSD up)."""
    from skfolio.prior import EntropyPooling

    ep = EntropyPooling(mean_views=views).fit(rets.values)
    w = ep.return_distribution_.sample_weight
    x = rets["XAUUSD"].values
    return {"p_up": float(w[x > 0].sum()), "exp_ret": float(w @ x)}


def _demo():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"ry": np.cumsum(rng.normal(0, .02, 300))})
    y = pd.Series(7 - 0.5 * X["ry"] + rng.normal(0, .001, 300))
    fv = kalman_fair_value(y, X)
    assert abs(fv["beta"]["ry"] + 0.5) < 0.1, fv
    print("system ok", fv["beta"])


if __name__ == "__main__":
    _demo()
