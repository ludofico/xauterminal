"""Offline calibration of confluence weights + NO TRADE threshold. Run by hand, never in the live loop:
    ./.env/bin/python calibrate.py            # writes calibration.json, picked up by signals.py at startup

Method: replay the price-derived components on history (M15, ~60 days), label each sample with the
directional forward return over HORIZON bars, grid-search weight multipliers + min score on the first 70%
(train), report the untouched last 30% (test) after a purge gap of HORIZON bars so labels never overlap.
Components that need data with no history here (news, forecast, positioning, fair value) are held at 0:
their weights stay at the defaults."""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import prices
import signals

HORIZON = 16      # M15 bars = 4h forward outcome
STEP = 4          # sample every hour
REPLAYABLE = ["htf", "strength", "liquidity", "smt", "dwell", "fvg", "session"]
GRID = [0.0, 0.5, 1.0, 1.5]
OUT = Path(__file__).with_name("calibration.json")


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


def samples() -> pd.DataFrame:
    frames, fx = prices.xau_crosses("M15", 6000)
    usd, usdx = frames["XAUUSD"], prices.usd_index(fx)
    rows = []
    for t in range(400, len(usd) - HORIZON, STEP):
        ts = usd.index[t]
        now = ts.to_pydatetime()
        cut = {k: v.iloc[: t + 1] for k, v in frames.items()}
        u = cut["XAUUSD"]
        lv = signals.levels(u, now)
        ctx = {"price": float(u["close"].iloc[-1]), "levels": lv, "sweeps": signals.sweeps(u, lv, now),
               "fvgs": signals.fvgs(u), "cross": signals.cross_strength(cut, usdx.iloc[: t + 1]),
               "smt": signals.smt(cut, now), "session": signals.session_now(now),
               "mtf": signals.mtf({"H4": _resample(u, "4h"), "H1": _resample(u, "1h"), "M15": u})}
        conf = signals.confluence(ctx)
        d = conf["direction"]
        if not d:
            continue
        comp = {c["name"]: c["value"] for c in conf["components"]}
        sess = ctx["session"]
        row = {k: max(0.0, comp.get(k, 0.0) * d) for k in REPLAYABLE if k != "session"}
        row["session"] = 1.0 if sess["silver_bullet"] else 0.7 if sess["killzone"] else 0.0
        fwd = float(usd["close"].iloc[t + HORIZON] / usd["close"].iloc[t] - 1)
        rows.append({"ts": ts, "dir": d, "ret": d * fwd, **row})
    return pd.DataFrame(rows).set_index("ts")


def evaluate(df: pd.DataFrame, w: dict, min_score: float) -> dict:
    fixed = sum(v for k, v in signals.BASE_W.items() if k not in REPLAYABLE)  # non-replayable weights, value 0 here
    total = sum(w.values()) + fixed
    score = 100 * sum(df[k] * w[k] for k in REPLAYABLE) / total
    taken = df[score >= min_score]
    n = len(taken)
    if n < 20:  # too few trades to judge: never pick a config that barely trades
        return {"n": n, "sharpe": -np.inf, "hit": None, "avg_bp": None}
    r = taken["ret"]
    return {"n": n, "sharpe": float(r.mean() / r.std() * np.sqrt(n)) if r.std() > 0 else 0.0,
            "hit": round(float((r > 0).mean()), 3), "avg_bp": round(float(1e4 * r.mean()), 2)}


def calibrate(df: pd.DataFrame) -> dict:
    cut = int(len(df) * 0.7)
    train, test = df.iloc[:cut], df.iloc[cut + HORIZON // STEP:]  # purge: no overlapping forward windows
    best = None
    for mult in itertools.product(GRID, repeat=len(REPLAYABLE)):
        if not any(mult):
            continue
        w = {k: signals.BASE_W[k] * m for k, m in zip(REPLAYABLE, mult)}
        for min_score in range(20, 75, 5):
            ev = evaluate(train, w, min_score)
            if best is None or ev["sharpe"] > best[0]["sharpe"]:
                best = (ev, w, min_score)
    ev_train, w, min_score = best
    base = {k: signals.BASE_W[k] for k in REPLAYABLE}
    return {"weights": {**signals.BASE_W, **{k: round(v, 2) for k, v in w.items()}}, "min_score": min_score,
            "train": ev_train, "test": evaluate(test, w, min_score),
            "baseline_test": evaluate(test, base, signals.MIN_SCORE),
            "samples": len(df), "span": [str(df.index[0]), str(df.index[-1])], "horizon_bars": HORIZON}


def _demo():
    """Self-check: on synthetic samples where only 'liquidity' predicts returns, calibration must keep it."""
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({k: rng.choice([0.0, 1.0], n) for k in REPLAYABLE},
                      index=pd.date_range("2026-01-01", periods=n, freq="h"))
    df["ret"] = 0.002 * (df["liquidity"] - 0.5) + rng.normal(0, 0.001, n)
    res = calibrate(df)
    assert res["weights"]["liquidity"] > 0 and res["test"]["avg_bp"] > 0, res
    print("calibrate ok", res["test"])


if __name__ == "__main__":
    import sys

    if "--demo" in sys.argv:
        _demo()
    else:
        data = samples()
        print(f"{len(data)} samples replayed")
        result = calibrate(data)
        OUT.write_text(json.dumps(result, indent=2, default=str))
        print(json.dumps({k: result[k] for k in ("min_score", "train", "test", "baseline_test")}, indent=2))
        print(f"written {OUT}")
