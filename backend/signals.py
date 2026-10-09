"""TA + SMC/ICT liquidity + cross-pair strength + confluence. Pure pandas/numpy, no I/O.

Conventions: frames are UTC-indexed OHLCV; every directional component is in [-1, 1], +1 = bullish gold."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NY = ZoneInfo("America/New_York")


# ---------- indicators ----------
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up, dn = d.clip(lower=0).ewm(alpha=1 / n).mean(), (-d.clip(upper=0)).ewm(alpha=1 / n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n).mean()


def ta(df: pd.DataFrame) -> dict:
    c = df["close"]
    return {"ema20": float(ema(c, 20).iloc[-1]), "ema50": float(ema(c, 50).iloc[-1]),
            "ema200": float(ema(c, 200).iloc[-1]), "rsi": float(rsi(c).iloc[-1]), "atr": float(atr(df).iloc[-1])}


# ---------- sessions (ICT, New York time) ----------
WINDOWS = {"Asia": (20, 24), "London KZ": (2, 5), "NY KZ": (7, 10),
           "Silver Bullet LDN": (3, 4), "Silver Bullet AM": (10, 11), "Silver Bullet PM": (14, 15)}


def session_now(now: datetime) -> dict:
    t = now.astimezone(NY)
    h = t.hour + t.minute / 60
    active = [k for k, (a, b) in WINDOWS.items() if a <= h < b]
    return {"ny_time": t.strftime("%H:%M"), "active": active,
            "killzone": any("KZ" in a for a in active), "silver_bullet": any("Silver" in a for a in active)}


def _ny_slice(df: pd.DataFrame, day: datetime, h0: float, h1: float) -> pd.DataFrame:
    start = datetime(day.year, day.month, day.day, tzinfo=NY) + timedelta(hours=h0)
    end = datetime(day.year, day.month, day.day, tzinfo=NY) + timedelta(hours=h1)
    return df[(df.index >= start) & (df.index < end)]


def levels(df: pd.DataFrame, now: datetime) -> dict:
    """Liquidity pools: Asia range (20:00-24:00 NY, the session that just closed), London range, previous day."""
    today = now.astimezone(NY).date()
    yday = today - timedelta(days=1 if today.weekday() != 0 else 3)
    asia = _ny_slice(df, yday, 20, 24)
    ldn = _ny_slice(df, today, 2, 5)
    pday = _ny_slice(df, yday, 0, 24)
    out = {}
    for name, part in (("Asia", asia), ("London", ldn), ("PD", pday)):
        if len(part):
            out[f"{name} high"], out[f"{name} low"] = float(part["high"].max()), float(part["low"].min())
    return out


# ---------- liquidity sweeps ----------
def sweep(df: pd.DataFrame, level: float, side: str, since: pd.Timestamp) -> dict | None:
    """side='buy' (stops above a high) or 'sell' (stops below a low). A sweep = trade through, then close back inside.
    dwell = bars spent beyond the level before reclaim (fast reclaim = strength, long dwell = zone being consumed)."""
    d = df[df.index >= since]
    beyond = d["high"] > level if side == "buy" else d["low"] < level
    if not beyond.any():
        return None
    i0 = int(np.argmax(beyond.values))
    after = d.iloc[i0:]
    inside = after["close"] < level if side == "buy" else after["close"] > level
    reclaimed = bool(inside.any())
    dwell = int(np.argmax(inside.values)) + 1 if reclaimed else len(after)
    extreme = float(after["high"].max() if side == "buy" else after["low"].min())
    return {"level": level, "side": side, "time": d.index[i0].isoformat(), "reclaimed": reclaimed,
            "dwell": dwell, "extreme": extreme, "direction": (-1 if side == "buy" else 1) if reclaimed else 0}


def sweeps(df: pd.DataFrame, lv: dict, now: datetime) -> list[dict]:
    since = df.index[-1] - pd.Timedelta(hours=10)
    out = []
    for name, price in lv.items():
        s = sweep(df, price, "buy" if name.endswith("high") else "sell", since)
        if s:
            out.append({"name": name, **s})
    return sorted(out, key=lambda s: s["time"])


def fvgs(df: pd.DataFrame, lookback: int = 40) -> list[dict]:
    """3-candle fair value gaps in the last `lookback` bars, flagged filled if price traded back through them."""
    d = df.tail(lookback)
    h, l, out = d["high"].values, d["low"].values, []
    for i in range(2, len(d)):
        for direction, top, bot in ((1, l[i], h[i - 2]), (-1, l[i - 2], h[i])):
            if top > bot:
                later = d.iloc[i + 1:]
                filled = bool((later["low"] <= bot).any() if direction == 1 else (later["high"] >= top).any())
                out.append({"dir": direction, "top": float(top), "bottom": float(bot),
                            "time": d.index[i].isoformat(), "filled": filled})
    return out


# ---------- cross-pair ----------
def cross_strength(frames: dict[str, pd.DataFrame], usdx: pd.Series, n: int = 12) -> dict:
    """Real gold strength vs USD weakness. n bars lookback."""
    rets = {k: float(v["close"].iloc[-1] / v["close"].iloc[-n - 1] - 1) for k, v in frames.items()}
    vols = {k: float(v["close"].pct_change().rolling(200).std().iloc[-1] * np.sqrt(n)) for k, v in frames.items()}
    z = {k: rets[k] / vols[k] if vols[k] else 0.0 for k in rets}
    sign = np.sign(rets["XAUUSD"]) or 1.0
    breadth = sum(np.sign(r) == sign for r in rets.values()) / len(rets)
    others = [r for k, r in rets.items() if k != "XAUUSD"]
    usdx_ret = float(usdx.iloc[-1] / usdx.iloc[-n - 1] - 1)
    usd_driven = bool(sign * np.median(others) <= 0 and sign * usdx_ret < 0)
    comp = float(np.clip(np.mean(list(z.values())) / 2, -1, 1)) * (breadth if not usd_driven else 0.25)
    return {"returns": rets, "z": z, "breadth": breadth, "usdx_ret": usdx_ret, "usd_driven": usd_driven,
            "component": comp}


def smt(frames: dict[str, pd.DataFrame], now: datetime) -> dict:
    """SMT divergence: same liquidity pool swept on some XAU crosses but not on others."""
    swept: dict[str, list[str]] = {}
    for pair, df in frames.items():
        for s in sweeps(df, levels(df, now), now):
            if s["reclaimed"]:
                swept.setdefault(s["name"], []).append(pair)
    div = {k: v for k, v in swept.items() if 0 < len(v) < len(frames)}
    return {"swept": swept, "divergent": div}


# ---------- multi-timeframe (top-down: HTF bias -> LTF trigger) ----------
HTF = {"D1": 0.45, "H4": 0.35, "H1": 0.20}   # bias weights
LTF = {"M15": 0.5, "M5": 0.3, "M1": 0.2}     # trigger weights


def tf_read(df: pd.DataFrame, k: int = 2) -> dict:
    """One timeframe: EMA trend, swing structure (HH/HL vs LH/LL), break of structure, premium/discount."""
    c = df["close"]
    e20, e50 = ema(c, 20), ema(c, 50)
    trend = int(np.sign(e20.iloc[-1] - e50.iloc[-1]))
    h, l = df["high"].values, df["low"].values
    sh = [i for i in range(k, len(df) - k) if h[i] == h[i - k:i + k + 1].max()]
    sl = [i for i in range(k, len(df) - k) if l[i] == l[i - k:i + k + 1].min()]
    structure = 0
    if len(sh) >= 2 and len(sl) >= 2:
        hh, hl = h[sh[-1]] > h[sh[-2]], l[sl[-1]] > l[sl[-2]]
        structure = 1 if hh and hl else -1 if not hh and not hl else 0
    bos = 0
    if sh and c.iloc[-1] > h[sh[-1]]:
        bos = 1
    elif sl and c.iloc[-1] < l[sl[-1]]:
        bos = -1
    rng = df.tail(50)
    lo, hi = float(rng["low"].min()), float(rng["high"].max())
    pd_pct = (float(c.iloc[-1]) - lo) / (hi - lo) if hi > lo else 0.5
    bias = float(np.clip(0.4 * trend + 0.4 * structure + 0.2 * bos, -1, 1))
    return {"trend": trend, "structure": structure, "bos": bos, "pd_pct": round(pd_pct, 3),
            "zone": "premium" if pd_pct > 0.5 else "discount", "rsi": round(float(rsi(c).iloc[-1]), 1),
            "bias": round(bias, 3)}


def mtf(frames: dict[str, pd.DataFrame]) -> dict:
    """frames: {"D1","H4","H1","M15","M5","M1"} -> per-TF read + weighted HTF bias / LTF trigger + alignment."""
    reads = {tf: tf_read(df) for tf, df in frames.items() if df is not None and len(df) > 60}
    htf = sum(w * reads[t]["bias"] for t, w in HTF.items() if t in reads)
    ltf = sum(w * reads[t]["bias"] for t, w in LTF.items() if t in reads)
    return {"tf": reads, "htf_bias": round(htf, 3), "ltf_bias": round(ltf, 3),
            "aligned": bool(np.sign(htf) == np.sign(ltf) and abs(htf) > 0.2 and abs(ltf) > 0.2),
            "h1_zone": reads.get("H1", {}).get("zone")}


# ---------- confluence ----------
BASE_W = {"htf": 15, "strength": 20, "liquidity": 20, "smt": 10, "dwell": 10, "session": 10, "fvg": 10,
          "macro": 10, "forecast": 10, "positioning": 10, "fair_value": 10}
# Regime-dependent multipliers (see wayfinder T2): in hedge failure gold is sold for liquidity, macro-bull news means less.
REGIME_MULT = {"Hedge failure": {"macro": 0.5, "strength": 1.3},
               "Risk-off (hedge attivo)": {"macro": 1.3},
               "Risk-on": {"forecast": 0.7, "macro": 0.8}}
MIN_SCORE = 60

# calibrate.py output overrides the defaults (weights + NO TRADE threshold) when present.
_CAL = __import__("pathlib").Path(__file__).with_name("calibration.json")
if _CAL.exists():
    _c = __import__("json").loads(_CAL.read_text())
    BASE_W.update({k: v for k, v in _c.get("weights", {}).items() if k in BASE_W})
    MIN_SCORE = _c.get("min_score", MIN_SCORE)


def confluence(c: dict) -> dict:
    """c: dict of raw inputs (see main.compute). Returns explainable score + direction + NO TRADE reasons."""
    sw = [s for s in c["sweeps"] if s["reclaimed"]]
    last = sw[-1] if sw else None
    direction = last["direction"] if last else int(np.sign(c["cross"]["component"]))
    comps: dict[str, tuple[float, str]] = {}
    comps["liquidity"] = (float(direction) if last else 0.0,
                          f"sweep {last['name']} con rientro" if last else "nessuna presa di liquidità")
    comps["strength"] = (c["cross"]["component"],
                         f"breadth {c['cross']['breadth']:.0%}" + (" · solo debolezza USD" if c["cross"]["usd_driven"] else ""))
    div_pairs = c["smt"]["divergent"].get(last["name"], []) if last else []
    comps["smt"] = (float(direction) if div_pairs else 0.0,
                    f"SMT: sweep solo su {', '.join(div_pairs)}" if div_pairs else "nessuna SMT")
    if last:
        dwell_score = 1.0 if last["dwell"] <= 2 else 0.5 if last["dwell"] <= 5 else -0.5
        comps["dwell"] = (direction * dwell_score, f"permanenza {last['dwell']} barre oltre il livello")
    else:
        comps["dwell"] = (0.0, "—")
    fresh = [f for f in c["fvgs"] if not f["filled"] and last and f["time"] > last["time"]]
    comps["fvg"] = (float(np.sign(sum(f["dir"] for f in fresh))) if fresh else 0.0,
                    f"{len(fresh)} FVG dopo lo sweep" if fresh else "nessun FVG di displacement")
    m = c.get("mtf") or {}
    if m:
        zone_ok = (m.get("h1_zone") == "discount") if direction > 0 else (m.get("h1_zone") == "premium")
        comps["htf"] = (float(m["htf_bias"]), f"bias D1/H4/H1 {m['htf_bias']:+.2f}"
                        + (" · zona H1 ok" if zone_ok else " · zona H1 sfavorevole") + (" · allineato LTF" if m["aligned"] else ""))
    else:
        comps["htf"] = (0.0, "multi-timeframe n/d")
    comps["macro"] = (c.get("macro_bias", 0.0), "sentiment news pesato (calibrato)" if c.get("calibrated") else "sentiment news (non calibrato)")
    comps["forecast"] = (c.get("forecast_bias", 0.0), c.get("forecast_note", "forecast n/d"))
    comps["positioning"] = (c.get("positioning_bias", 0.0), c.get("positioning_note", "posizionamento n/d"))
    comps["fair_value"] = (c.get("fair_value_bias", 0.0), c.get("fair_value_note", "fair value n/d"))

    mult = REGIME_MULT.get(c.get("regime", ""), {})
    w = {k: BASE_W[k] * mult.get(k, 1.0) for k in BASE_W}
    total = sum(w.values())
    d = direction or 1
    pts = sum(w[k] * max(0.0, v * d) for k, (v, _) in comps.items())
    sess = c["session"]
    pts += w["session"] * (1.0 if sess["silver_bullet"] else 0.7 if sess["killzone"] else 0.0)
    score = round(100 * pts / total)

    reasons_no = []
    if not direction:
        reasons_no.append("nessuna direzione")
    if score < MIN_SCORE:
        reasons_no.append(f"score {score} < {MIN_SCORE}")
    for k in ("liquidity", "strength", "htf"):
        if comps[k][0] * d < -0.5:
            reasons_no.append(f"conflitto: {k} contro la direzione")
    if c.get("event_block"):
        reasons_no.append(c["event_block"])
    gate = c.get("gate")
    if gate and (gate["p_take"] < 0.75 or gate["direction"] != d):
        reasons_no.append(f"gate System One: P={gate['p_take']:.2f}, dir {gate['direction']:+d}")

    target, stop = None, None
    if last and direction:
        pools = [v for k, v in c["levels"].items() if (v > c["price"] if direction > 0 else v < c["price"])]
        target = (min(pools) if direction > 0 else max(pools)) if pools else None
        stop = last["extreme"]
    return {"score": score, "direction": direction, "no_trade": bool(reasons_no), "no_trade_reasons": reasons_no,
            "components": [{"name": k, "value": round(v, 3), "weight": round(w[k], 1), "note": n}
                           for k, (v, n) in comps.items()],
            "trapped": (f"breakout trader {'short' if direction > 0 else 'long'} oltre {last['name']}" if last else None),
            "invalidation": stop, "target_liquidity": target}


def _demo():
    """Self-check on synthetic data: a sell-side sweep of the Asia low with fast reclaim must read bullish."""
    idx = pd.date_range("2026-10-05 00:00", periods=300, freq="5min", tz=NY).tz_convert("UTC")
    base = np.full(300, 2650.0)
    df = pd.DataFrame({"open": base, "high": base + 1, "low": base - 1, "close": base, "volume": 1.0}, index=idx)
    now = idx[-1].to_pydatetime()
    lv = {"Asia low": 2640.0}
    df.iloc[-5, df.columns.get_loc("low")] = 2638.0  # wick through Asia low, close back above
    s = sweeps(df, lv, now)
    assert s and s[0]["reclaimed"] and s[0]["direction"] == 1 and s[0]["dwell"] == 1, s
    assert sweep(df, 2660.0, "buy", df.index[0]) is None
    x = np.arange(200.0)
    mid = x + 6 * np.sin(x / 3)  # uptrend with pullbacks -> higher highs and higher lows
    up = pd.DataFrame({"open": mid - .2, "close": mid + .2, "high": mid + 1, "low": mid - 1, "volume": 1.0},
                      index=pd.date_range("2026-01-01", periods=200, freq="h", tz="UTC"))
    r = mtf({"D1": up, "H4": up, "H1": up, "M15": up, "M5": up, "M1": up})
    assert r["htf_bias"] > 0.5 and r["tf"]["H1"]["trend"] == 1 and r["aligned"], r
    print("signals ok")


if __name__ == "__main__":
    _demo()
