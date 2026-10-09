"""xau-terminal API: background loops + REST + SSE. Run: .env/bin/uvicorn main:app --port 8000"""
import asyncio
import threading
import json
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import forecast
import macro
import news
import nlp
import prices
import signals
import stream as feed  # not "stream": the SSE route function below is named stream()
import system

S: dict = {"news": deque(maxlen=500), "errors": {}, "frames": {}, "fx": {}}
seen: list[str] = []


def _safe(name, fn, *a, **k):
    try:
        out = fn(*a, **k)
        S["errors"].pop(name, None)
        return out
    except Exception as e:  # every source is optional; record and keep going
        S["errors"][name] = f"{type(e).__name__}: {e}"[:200]
        return None


def compute() -> None:
    now = datetime.now(timezone.utc)
    frames, fx = prices.xau_crosses("M1", 1500)  # minute stream: price, sweeps, FVG, cross-pair
    S["frames"], S["fx"] = frames, fx
    usd = frames["XAUUSD"]
    usdx = prices.usd_index(fx)
    lv = signals.levels(usd, now)
    sw = signals.sweeps(usd, lv, now)
    items = classified()[-60:]
    w = [it["nlp"]["impact"] or 1 for it in items if it.get("nlp")]
    bias = [it["nlp"]["bias"].get("bull", 0) - it["nlp"]["bias"].get("bear", 0) for it in items if it.get("nlp")]
    sys_ = S.get("system", {})
    ctx = {"price": float(usd["close"].iloc[-1]), "levels": lv, "sweeps": sw, "fvgs": signals.fvgs(usd),
           "cross": signals.cross_strength(frames, usdx), "smt": signals.smt(frames, now),
           "session": signals.session_now(now), "mtf": S.get("mtf"), "regime": sys_.get("regime", {}).get("regime", ""),
           "macro_bias": float(np.average(bias, weights=w)) if bias else 0.0,
           "calibrated": any(it["nlp"]["calibrated"] for it in items if it.get("nlp")),
           "event_block": macro.event_block(S.get("calendar") or [], now)}
    if fc := S.get("forecast"):
        ctx["forecast_bias"], ctx["forecast_note"] = fc["bias"], f"TimesFM {fc['move_pct']:+.2f}% (hit-rate {fc['hit_rate']})"
    if fv := sys_.get("fair_value"):
        ctx["fair_value_bias"], ctx["fair_value_note"] = fv["bias"], f"Kalman z={fv['z']}"
    pos = [b for b in (S.get("cot", {}).get("bias"), (S.get("oanda") or {}).get("bias")) if b is not None]
    if pos:
        ctx["positioning_bias"], ctx["positioning_note"] = float(np.mean(pos)), "COT + OANDA retail (contrarian)"
    conf = signals.confluence(ctx)
    # System One gate: one calibrated decision over the whole picture.
    state = (f"Gold {ctx['price']:.2f}. Regime {ctx['regime'] or 'unknown'}. Session {ctx['session']['active']}. "
             f"HTF bias {(S.get('mtf') or {}).get('htf_bias')}. Sweeps {[ (s['name'], s['reclaimed'], s['dwell']) for s in sw]}. Cross breadth {ctx['cross']['breadth']:.0%}, "
             f"usd-driven {ctx['cross']['usd_driven']}. Score {conf['score']} dir {conf['direction']:+d}. "
             f"News: " + " | ".join(it["title"] for it in items[-8:]))
    g = nlp.lev_ready() and nlp.system_one(state, {"take_trade": {"type": "noul", "instructions": "Is this a high-quality gold setup to take now?"},
                               "direction": {"type": "choice", "instructions": "Direction", "criteria": {"long": "buy gold", "short": "sell gold"}}})
    if g:
        d = g["answers"]["direction"].get("probabilities") or {}
        ctx["gate"] = {"p_take": float(g["answers"]["take_trade"].get("noul") or 0),
                       "direction": 1 if d.get("long", 0) >= d.get("short", 0) else -1, "engine": g["engine"]}
        conf = signals.confluence(ctx)
    fc_now = S.get("forecast")
    if fc_now and fc_now.get("source") != prices.source["name"] and not S.get("fc_busy"):
        # feed changed (e.g. yfinance futures -> MT5 spot): recompute now instead of waiting 15 min
        S["fc_busy"] = True
        threading.Thread(target=lambda: (_safe("fc_refresh", fc_refresh), S.update(fc_busy=False)), daemon=True).start()
    mc = _safe("mc", system.garch_mc, usd["close"], ctx["price"], conf["target_liquidity"], conf["invalidation"])
    S["snapshot"] = {"ts": now.isoformat(), "source": prices.source["name"], "price": ctx["price"],
                     "ticker": {k: float(v["close"].iloc[-1]) for k, v in frames.items()} | {"USDX": float(usdx.iloc[-1])},
                     "ta": signals.ta(usd), "levels": lv, "sweeps": sw, "fvgs": ctx["fvgs"][-10:], "cross": ctx["cross"],
                     "smt": ctx["smt"], "session": ctx["session"], "mtf": S.get("mtf"), "confluence": conf, "gate": ctx.get("gate"), "mc": mc}


def slow_refresh() -> None:
    """Daily/hourly sources and models."""
    S["cot"] = _safe("cot", macro.cot) or S.get("cot", {})
    S["calendar"] = _safe("calendar", macro.calendar) or S.get("calendar", [])
    daily = {k: _safe(k, prices.raw, k, "D1", 500) for k in ("XAUUSD", "SPX", "VIX", "GVZ")}
    fxd = {p: _safe(p, prices.raw, p, "D1", 500) for p in prices.FX}
    if all(v is not None for v in (*daily.values(), *fxd.values())):
        usdx = prices.usd_index(fxd)
        D = pd.concat({k: v["close"] for k, v in daily.items()} | {"USDX": usdx}, axis=1).ffill().dropna()
        D.index = D.index.normalize()
        sysd = {"regime": _safe("regime", system.regime, D[["XAUUSD", "SPX", "VIX", "USDX"]])}
        ry = _safe("fred", macro.fred, "DFII10")
        if ry is not None:
            ry.index = ry.index.normalize()
            J = pd.concat([np.log(D["XAUUSD"]), ry, np.log(D["USDX"])], axis=1).dropna()
            J.columns = ["y", "real_yield", "usdx"]
            sysd["fair_value"] = _safe("kalman", system.kalman_fair_value, J["y"], J[["real_yield", "usdx"]])
            chg = pd.concat([D.pct_change(), ry.diff().rename("DFII10")], axis=1).dropna()
            sysd["causal"] = _safe("pcmci", system.causal_graph, chg)
        S["system"] = {**S.get("system", {}), **{k: v for k, v in sysd.items() if v is not None}}
    if S.get("frames"):
        rets = pd.DataFrame({k: v["close"] for k, v in S["frames"].items()}).pct_change().dropna().tail(500)
        S.setdefault("system", {})["graph"] = _safe("graph", system.dependency_graph, rets)
        price = float(S["frames"]["XAUUSD"]["close"].iloc[-1])
        S["oanda"] = _safe("oanda", macro.oanda_books, price) if macro.OANDA_TOKEN else None  # optional source
        cnh = _safe("USDCNH", prices.raw, "USDCNH", "H1", 5)
        if cnh is not None:
            S["sge"] = _safe("sge", macro.sge_premium, price, float(cnh["close"].iloc[-1]))


def mtf_refresh() -> None:
    """Top-down read of XAUUSD on D1 / H4 / H1 / M15 / M5. H4 resampled from H1 (yfinance has no 4h)."""
    h1 = prices.raw("XAUUSD", "H1", 1500)
    h4 = h1.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    S["mtf"] = signals.mtf({"D1": prices.raw("XAUUSD", "D1", 300), "H4": h4, "H1": h1,
                            "M15": prices.raw("XAUUSD", "M15", 400), "M5": prices.raw("XAUUSD", "M5", 600),
                            "M1": S["frames"].get("XAUUSD")})


def news_refresh() -> None:
    """Fetch only: headlines show immediately, classification happens in classify_pending."""
    fresh = news.dedup(news.fetch(), seen, nlp.similarity)
    for it in sorted(fresh, key=lambda i: i["ts"]):
        it["nlp"], it["entities"] = None, []
        S["news"].append(it)


def classify_pending() -> None:
    """Newest first, one parallel Jev batch per cycle (keyword extractor when Jev is unavailable)."""
    todo = [it for it in reversed(S["news"]) if it["nlp"] is None][:32]
    for it, res in zip(todo, nlp.classify_many([it["title"] for it in todo])):
        it["nlp"], it["entities"] = res, nlp.entities(it["title"])


def classified() -> list[dict]:
    return [it for it in S["news"] if it.get("nlp")]


def fc_refresh() -> None:
    m15 = _safe("m15", prices.raw, "XAUUSD", "M15", 600)
    if m15 is None:
        return
    closes = m15["close"]
    forecast.settle(lambda ts, k: (lambda i: float(closes.iloc[i + k]) if i + k < len(closes) else None)(
        closes.index.get_indexer([pd.Timestamp(ts)], method="nearest")[0]))
    fc = _safe("timesfm", forecast.forecast, closes.values, closes.index[-1].isoformat())
    S["forecast"] = {**fc, "source": prices.source["name"]} if fc else S.get("forecast")


async def loop(fn, every: int):
    while True:
        await asyncio.to_thread(_safe, fn.__name__, fn)
        await asyncio.sleep(every)


@asynccontextmanager
async def lifespan(_):
    tasks = [asyncio.create_task(loop(f, s)) for f, s in
             ((compute, 1), (mtf_refresh, 60), (news_refresh, 60), (classify_pending, 3), (fc_refresh, 900), (slow_refresh, 3600))]
    asyncio.create_task(asyncio.to_thread(nlp.warmup))
    yield
    for t in tasks:
        t.cancel()


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _clean(x):
    """NaN/inf are not valid JSON: the browser's JSON.parse rejects them. Map to null, recursively."""
    if isinstance(x, float):
        return x if np.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.generic):
        return _clean(x.item())
    return x


def _view() -> dict:
    return _clean({**S.get("snapshot", {}), "system": S.get("system", {}), "forecast": S.get("forecast"),
            "cot": S.get("cot"), "oanda": S.get("oanda"), "sge": S.get("sge"), "calendar": (S.get("calendar") or [])[:30],
            "news_status": {**news.status, **nlp.status}, "quotes": feed.quotes(), "stream": feed.stats, "errors": S["errors"]})


STREAM_KEY = __import__("os").getenv("STREAM_KEY", "")


@app.post("/api/ingest")
async def ingest(body: dict, x_stream_key: str | None = Header(default=None)):
    """Feed from mt5/XauStream.mq5 (read-only). Local only; set STREAM_KEY to require the EA's key."""
    if STREAM_KEY and x_stream_key != STREAM_KEY:
        raise HTTPException(401, "bad stream key")
    feed.ingest(body)
    return {"ok": True}


@app.get("/api/snapshot")
def snapshot():
    return _view()


@app.get("/api/candles")
def candles(sym: str = "XAUUSD", n: int = 300):
    df = S["frames"].get(sym)
    if df is None:
        return []
    d = df.tail(n)
    return [{"time": int(t.timestamp()), "open": r.open, "high": r.high, "low": r.low, "close": r.close}
            for t, r in d.iterrows()]


@app.get("/api/news")
def get_news(q: str = "", limit: int = 100):
    items = [i for i in S["news"] if q.lower() in i["title"].lower()] if q else list(S["news"])
    return items[-limit:][::-1]


@app.post("/api/ask")
async def ask(body: dict):
    return await asyncio.to_thread(nlp.ask, body.get("q", ""), list(S["news"]))


@app.get("/api/stream")
async def stream():
    async def gen():
        last_news = 0
        while True:
            payload = _view()
            items = list(S["news"])
            payload["news_delta"] = _clean(items[last_news:] if last_news <= len(items) else items)
            last_news = len(items)
            yield f"data: {json.dumps(payload, default=str)}\n\n"
            await asyncio.sleep(1)
    return StreamingResponse(gen(), media_type="text/event-stream")
