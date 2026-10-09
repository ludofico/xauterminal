"""In-memory realtime store fed by mt5/XauStream.mq5 (POST /api/ingest). Read-only: no trading path exists.
History arrives as broker-time bars, ticks as broker-time ms; the EA sends the exact server-UTC offset."""
import threading
import time

import pandas as pd

TF_RULE = {"M1": "1min", "M5": "5min", "M15": "15min", "H1": "1h", "D1": "1D"}
STALE_S = 30  # no tick for this long -> feed considered down, prices.py falls back

_lock = threading.Lock()
_bars: dict[tuple[str, str], pd.DataFrame] = {}
_last: dict[str, dict] = {}
_offset = {"s": 0}
stats = {"ticks": 0, "last_ingest": 0.0, "series": 0}


def ingest(p: dict) -> None:
    off = int(p.get("offset", _offset["s"]))
    with _lock:
        _offset["s"] = off
        for h in p.get("history", []):
            df = pd.DataFrame(h["bars"], columns=["t", "open", "high", "low", "close", "volume"])
            df.index = pd.to_datetime(df.pop("t") - off, unit="s", utc=True)
            if h["tf"] == "D1":  # keep the broker's daily boundary (labels stay on broker midnight)
                df.index = df.index + pd.Timedelta(seconds=off)
            df = df.astype(float)
            old = _bars.get((h["sym"], h["tf"]))
            if old is not None:  # merge: short resync snapshots must not erase the long history
                df = pd.concat([old[old.index < df.index[0]], df, old[old.index > df.index[-1]]])
            _bars[(h["sym"], h["tf"])] = df
        for t in p.get("ticks", []):
            ts = pd.Timestamp(t["t"] / 1000 - off, unit="s", tz="UTC")
            px = float(t["bid"])  # MT5 charts are bid-based; keep bars consistent with history
            _last[t["sym"]] = {"bid": px, "ask": float(t["ask"]), "t": ts, "rx": time.time()}
            stats["ticks"] += 1
            for tf, rule in TF_RULE.items():
                df = _bars.get((t["sym"], tf))
                if df is None:
                    continue
                bucket = ts.floor(rule)
                last = df.index[-1]
                if bucket == last:
                    row = df.loc[last]
                    df.loc[last, ["high", "low", "close", "volume"]] = [max(row["high"], px), min(row["low"], px),
                                                                       px, row["volume"] + 1]
                elif bucket > last:
                    df.loc[bucket] = [px, px, px, px, 1.0]
                    if len(df) > 6000:
                        _bars[(t["sym"], tf)] = df.iloc[-5000:]
        stats["last_ingest"] = time.time()
        stats["series"] = len(_bars)


def live() -> bool:
    return bool(_last) and time.time() - max(v["rx"] for v in _last.values()) < STALE_S


def frame(sym: str, tf: str, n: int) -> pd.DataFrame | None:
    with _lock:
        df = _bars.get((sym, tf))
        return df.tail(n).copy() if df is not None and live() else None


def quotes() -> dict:
    with _lock:
        return {k: {"bid": v["bid"], "ask": v["ask"], "spread": round(v["ask"] - v["bid"], 5), "t": v["t"].isoformat()}
                for k, v in _last.items()}


def _demo():
    now = int(time.time()) // 300 * 300
    ingest({"offset": 10800, "history": [{"sym": "XAUUSD", "tf": "M5",
            "bars": [[now + 10800 - 600, 1, 2, 0.5, 1.5, 10], [now + 10800 - 300, 1.5, 2, 1, 1.8, 10]]}]})
    ingest({"offset": 10800, "ticks": [{"sym": "XAUUSD", "t": (now + 10800 + 5) * 1000, "bid": 1.9, "ask": 1.95}]})
    df = frame("XAUUSD", "M5", 10)
    assert df is not None and len(df) == 3 and df["close"].iloc[-1] == 1.9, df
    assert df.index[-1] == pd.Timestamp(now, unit="s", tz="UTC"), df.index  # offset removed -> true UTC
    ingest({"offset": 10800, "ticks": [{"sym": "XAUUSD", "t": (now + 10800 + 30) * 1000, "bid": 2.5, "ask": 2.55}]})
    assert frame("XAUUSD", "M5", 10)["high"].iloc[-1] == 2.5
    print("stream ok")


if __name__ == "__main__":
    _demo()
