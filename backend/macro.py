"""Structured macro / positioning sources (no scraping). Each fetcher returns plain dicts and raises on failure;
main.py isolates failures per source. See docs/wayfinder/map.md T1."""
import io
import os
from datetime import datetime, timedelta, timezone

import httpx
import pandas as pd

OANDA_TOKEN = os.getenv("OANDA_TOKEN", "")  # free practice-account token
OANDA_URL = os.getenv("OANDA_URL", "https://api-fxpractice.oanda.com")
TROY_OZ_G = 31.1034768


def cot() -> dict:
    """CFTC COT gold (legacy futures-only) via futuresbench mirror. Contrarian bias only at 52w extremes."""
    df = pd.read_csv("https://futuresbench.com/data/cot/gold.csv", parse_dates=["report_date"]).tail(156)
    net = df["net_noncommercial"]
    last52 = net.tail(52)
    idx = float((net.iloc[-1] - last52.min()) / (last52.max() - last52.min()))
    bias = -1.0 if idx > 0.9 else 1.0 if idx < 0.1 else 0.0  # crowded long -> downside risk, and vice versa
    return {"date": df["report_date"].iloc[-1].date().isoformat(), "net": int(net.iloc[-1]),
            "wow": int(net.iloc[-1] - net.iloc[-2]), "cot_index_52w": round(idx, 3), "bias": bias,
            "history": [{"t": d.date().isoformat(), "net": int(n)} for d, n in zip(df["report_date"].tail(52), last52)]}


def calendar() -> list[dict]:
    """ForexFactory weekly calendar (JSON), high/medium impact only."""
    rows = httpx.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=10).json()
    return [{"title": r["title"], "country": r["country"], "impact": r["impact"], "forecast": r.get("forecast"),
             "previous": r.get("previous"), "ts": datetime.fromisoformat(r["date"]).astimezone(timezone.utc).isoformat()}
            for r in rows if r.get("impact") in ("High", "Medium")]


def event_block(events: list[dict], now: datetime, minutes: int = 15) -> str | None:
    for e in events:
        if e["impact"] == "High" and e["country"] in ("USD", "All"):
            dt = datetime.fromisoformat(e["ts"])
            if abs(dt - now) <= timedelta(minutes=minutes):
                return f"evento macro: {e['title']} ({dt:%H:%M}Z)"
    return None


def fred(series: str) -> pd.Series:
    """FRED public CSV, no key needed (e.g. DFII10 real yield, DTWEXBGS broad USD, T10YIE breakeven)."""
    txt = httpx.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}", timeout=15).text
    df = pd.read_csv(io.StringIO(txt), na_values=".")
    s = df.set_index(pd.to_datetime(df.iloc[:, 0], utc=True)).iloc[:, 1].dropna().astype(float)
    return s.rename(series)


def oanda_books(price: float) -> dict:
    """OANDA order book (where resting stops/limits sit) + position book (retail net). Needs OANDA_TOKEN."""
    if not OANDA_TOKEN:
        raise RuntimeError("OANDA_TOKEN not set")
    h = {"Authorization": f"Bearer {OANDA_TOKEN}"}
    ob = httpx.get(f"{OANDA_URL}/v3/instruments/XAU_USD/orderBook", headers=h, timeout=10).json()["orderBook"]
    pb = httpx.get(f"{OANDA_URL}/v3/instruments/XAU_USD/positionBook", headers=h, timeout=10).json()["positionBook"]
    b = pd.DataFrame(ob["buckets"]).astype(float)
    near = b[(b["price"] - price).abs() < price * 0.02]
    # Above price: buy orders = buy stops (short stops / breakout entries) -> buy-side liquidity. Below: sell stops.
    above = near[near["price"] > price].nlargest(3, "longCountPercent")
    below = near[near["price"] < price].nlargest(3, "shortCountPercent")
    p = pd.DataFrame(pb["buckets"]).astype(float)
    long_pct = float(p["longCountPercent"].sum() / (p["longCountPercent"].sum() + p["shortCountPercent"].sum()))
    bias = -1.0 if long_pct > 0.65 else 1.0 if long_pct < 0.35 else 0.0  # retail crowd is the fuel, fade extremes
    return {"time": ob["time"], "buy_side_pools": above["price"].tolist(), "sell_side_pools": below["price"].tolist(),
            "retail_long_pct": round(long_pct, 3), "bias": bias}


def sge_premium(xauusd: float, usdcnh: float) -> dict:
    """Shanghai Gold Exchange Au99.99 vs international price in CNY/g. Positive premium = strong Chinese demand."""
    import akshare as ak

    q = ak.spot_quotations_sge(symbol="Au99.99")
    sge = float(q["现价"].dropna().iloc[-1])  # last row is often an empty tick
    intl = xauusd * usdcnh / TROY_OZ_G
    return {"sge_cny_g": sge, "intl_cny_g": round(intl, 2), "premium_pct": round(100 * (sge / intl - 1), 3)}
