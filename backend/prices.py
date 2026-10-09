"""Price feed, realtime first: MT5 bridge -> OANDA v20 (realtime candles incl. the forming one, free practice
token) -> yfinance (DELAYED, last resort). Every frame is UTC-indexed OHLCV."""
import os

import httpx
import pandas as pd

MT5_URL = os.getenv("MT5_BRIDGE_URL", "http://127.0.0.1:8080").rstrip("/")
MT5_KEY = os.getenv("MT5_API_KEY", "")
MT5_SUFFIX = os.getenv("MT5_SYMBOL_SUFFIX", "")
# MT5 bar times are broker server time. Calibration knob: most EU/US brokers run UTC+2 (winter) / UTC+3 (summer).
MT5_TZ_OFFSET_H = float(os.getenv("MT5_TZ_OFFSET_H", "3"))

FX = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD"]
# cross -> (fx pair, op): XAUEUR = XAUUSD / EURUSD, XAUJPY = XAUUSD * USDJPY
CROSSES = {"XAUEUR": ("EURUSD", "div"), "XAUGBP": ("GBPUSD", "div"), "XAUJPY": ("USDJPY", "mul"),
           "XAUCHF": ("USDCHF", "mul"), "XAUAUD": ("AUDUSD", "div")}
PAIRS = ["XAUUSD", *CROSSES]

YF = {"XAUUSD": "GC=F", **{p: f"{p}=X" for p in FX}, "GVZ": "^GVZ", "VIX": "^VIX", "SPX": "^GSPC", "USDCNH": "CNH=X"}
YF_INTERVAL = {"M1": ("1m", "5d"), "M5": ("5m", "5d"), "M15": ("15m", "30d"), "H1": ("60m", "60d"), "D1": ("1d", "2y")}
OHLC = ["open", "high", "low", "close"]

OANDA_TOKEN = os.getenv("OANDA_TOKEN", "")
OANDA_URL = os.getenv("OANDA_URL", "https://api-fxpractice.oanda.com")
OANDA_GRAN = {"M1": "M1", "M5": "M5", "M15": "M15", "H1": "H1", "D1": "D"}
_oanda = httpx.Client(timeout=3, headers={"Authorization": f"Bearer {OANDA_TOKEN}"})  # keep-alive: ~50 ms/poll

source = {"name": "init"}


def _oanda_candles(sym: str, tf: str, n: int) -> pd.DataFrame:
    """OANDA quotes XAU_EUR, XAU_JPY... natively, so crosses are real quotes, not synthetic."""
    if not OANDA_TOKEN:
        raise RuntimeError("OANDA_TOKEN not set")
    inst = f"{sym[:3]}_{sym[3:]}"
    r = _oanda.get(f"{OANDA_URL}/v3/instruments/{inst}/candles",
                   params={"granularity": OANDA_GRAN[tf], "count": min(n, 5000), "price": "M"})
    r.raise_for_status()
    rows = [{"t": c["time"], "open": float(c["mid"]["o"]), "high": float(c["mid"]["h"]), "low": float(c["mid"]["l"]),
             "close": float(c["mid"]["c"]), "volume": float(c["volume"])} for c in r.json()["candles"]]
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df.pop("t"), utc=True)
    return df


def _mt5(sym: str, tf: str, n: int) -> pd.DataFrame:
    r = httpx.get(f"{MT5_URL}/rates/{sym}{MT5_SUFFIX}/{tf}", params={"count": n},
                  headers={"x-api-key": MT5_KEY} if MT5_KEY else {}, timeout=5)
    j = r.json()
    if not j.get("success"):
        raise RuntimeError(j.get("message", "mt5 error"))
    df = pd.DataFrame(j["bars"], columns=j["fields"])
    df.index = pd.to_datetime(df["time"] - MT5_TZ_OFFSET_H * 3600, unit="s", utc=True)
    return df[[*OHLC, "tick_volume"]].rename(columns={"tick_volume": "volume"}).astype(float)


def _yf(sym: str, tf: str, n: int) -> pd.DataFrame:
    import yfinance as yf

    interval, period = YF_INTERVAL[tf]
    df = None
    for _ in range(2):  # Yahoo's crumb endpoint times out intermittently; one retry clears most of them
        df = yf.download(YF[sym], interval=interval, period=period, progress=False, auto_adjust=False)
        if df is not None and not df.empty:
            break
    if df is None or df.empty:
        raise RuntimeError(f"yfinance empty for {sym}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[[*OHLC, "volume"]].astype(float).dropna(subset=OHLC).tail(n)
    df.index = df.index.tz_convert("UTC") if df.index.tz else df.index.tz_localize("UTC")
    return df


def raw(sym: str, tf: str, n: int = 500) -> pd.DataFrame:
    import stream

    df = stream.frame(sym, tf, n)  # XauStream EA: tick-built bars incl. the forming one, zero delay
    if df is not None:
        source["name"] = "MT5 stream"
        return df
    if sym in YF and sym not in PAIRS and sym not in FX:  # indices: yfinance only
        return _yf(sym, tf, n)
    for name, fn in (("MT5", _mt5), ("OANDA", _oanda_candles)):
        try:
            df = fn(sym, tf, n)
            source["name"] = name
            return df
        except Exception:
            continue
    source["name"] = "yfinance (DELAYED)"
    return _yf(sym, tf, n)


def xau_crosses(tf: str, n: int = 500) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """XAU against 6 currencies. Broker quote if MT5 has it, else synthetic from XAUUSD and the FX leg."""
    usd = raw("XAUUSD", tf, n)
    fx = {p: raw(p, tf, n).reindex(usd.index, method="ffill") for p in FX}
    out = {"XAUUSD": usd}
    for name, (leg, op) in CROSSES.items():
        import stream

        live = stream.frame(name, tf, n)
        if live is not None:  # broker quotes the cross natively through the stream
            out[name] = live
            continue
        direct = {"MT5": _mt5, "OANDA": _oanda_candles}.get(source["name"])
        if direct:  # realtime feeds quote the XAU crosses natively
            try:
                out[name] = direct(name, tf, n)
                continue
            except Exception:
                pass
        # ponytail: one FX close per bar for all of O/H/L/C; intrabar FX moves ignored, fine at M5+.
        f = fx[leg]["close"]
        synth = usd[OHLC].div(f, axis=0) if op == "div" else usd[OHLC].mul(f, axis=0)
        out[name] = synth.assign(volume=usd["volume"]).dropna()
    return out, fx


def usd_index(fx: dict[str, pd.DataFrame]) -> pd.Series:
    """Equal-weight USD strength proxy (DXY-like): geometric mean of USD vs 5 majors, rebased to 100."""
    import numpy as np

    legs = [1 / fx["EURUSD"]["close"], 1 / fx["GBPUSD"]["close"], fx["USDJPY"]["close"],
            fx["USDCHF"]["close"], 1 / fx["AUDUSD"]["close"]]
    logm = sum(np.log(s) for s in legs) / len(legs)
    return (100 * np.exp(logm - logm.dropna().iloc[0])).rename("USDX")
