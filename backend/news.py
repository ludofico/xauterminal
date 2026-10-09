"""News aggregation: RSS + Google News + GDELT + SMM China metals (via AKShare). Returns normalized, de-duplicated items."""
import calendar
import hashlib
import re
from datetime import datetime, timedelta, timezone

import feedparser
import httpx

FEEDS = {  # verified 2026-10-05 (ForexLive, Mining.com return 403)
    "FXStreet": "https://www.fxstreet.com/rss/news",
    "FXEmpire": "https://www.fxempire.com/api/v1/en/articles/rss/news",
    "Investing": "https://www.investing.com/rss/news_11.rss",
    "Investing Metals": "https://www.investing.com/rss/commodities_Metals.rss",
    "Fed": "https://www.federalreserve.gov/feeds/press_all.xml",
    "ECB": "https://www.ecb.europa.eu/rss/press.html",
    "CNBC Economy": "https://www.cnbc.com/id/20910258/device/rss/rss.html",
    "MarketWatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "Google News": "https://news.google.com/rss/search?q=gold+OR+XAUUSD+OR+fed+OR+inflation+when:1d&hl=en-US&gl=US&ceid=US:en",
}
GDELT = ("https://api.gdeltproject.org/api/v2/doc/doc?query=(gold%20OR%20bullion%20OR%20%22federal%20reserve%22)"
         "%20sourcelang:english&mode=artlist&format=json&maxrecords=50&timespan=2h&sort=datedesc")
UA = {"User-Agent": "Mozilla/5.0 xau-terminal"}
CN_TZ = timezone(timedelta(hours=8))  # Chinese feeds use Beijing time

status: dict[str, str] = {}  # per-source health, shown in UI


def _item(source: str, title: str, url: str, ts: datetime, lang: str = "en") -> dict:
    title = re.sub(r"<[^>]+>", "", title).strip()
    return {"id": hashlib.sha1((title.lower() + source).encode()).hexdigest()[:12], "source": source,
            "title": title, "url": url, "ts": ts.astimezone(timezone.utc).isoformat(), "lang": lang}


def _rss(client: httpx.Client, name: str, url: str) -> list[dict]:
    feed = feedparser.parse(client.get(url).content)
    out = []
    for e in feed.entries[:40]:
        t = e.get("published_parsed") or e.get("updated_parsed")
        ts = datetime.fromtimestamp(calendar.timegm(t), timezone.utc) if t else datetime.now(timezone.utc)
        out.append(_item(name, e.get("title", ""), e.get("link", ""), ts))
    return out


def _gdelt(client: httpx.Client) -> list[dict]:
    arts = client.get(GDELT, timeout=20).json().get("articles", [])
    return [_item(f"GDELT·{a.get('domain', '')}", a["title"], a["url"],
                  datetime.strptime(a["seendate"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)) for a in arts]


def _shmet() -> list[dict]:
    """Chinese metals newsflash (Shanghai Metals Market) via AKShare. Jin10's js_news was removed from AKShare."""
    import akshare as ak  # optional dependency

    df = ak.futures_news_shmet(symbol="全部")
    out = []
    for _, r in df.tail(40).iterrows():
        ts = datetime.fromisoformat(str(r.iloc[0])).replace(tzinfo=CN_TZ)
        out.append(_item("SMM", str(r.iloc[1]), "https://news.metal.com/", ts, "zh"))
    return out


def fetch() -> list[dict]:
    items: list[dict] = []
    with httpx.Client(headers=UA, timeout=10, follow_redirects=True) as c:
        jobs = {name: (lambda n=name, u=url: _rss(c, n, u)) for name, url in FEEDS.items()}
        jobs["GDELT"] = lambda: _gdelt(c)
        jobs["SMM"] = _shmet
        for name, job in jobs.items():
            try:
                got = job()
                items += got
                status[name] = f"ok ({len(got)})"
            except Exception as e:  # one dead feed must never kill the loop
                status[name] = f"error: {type(e).__name__}"
    return items


def dedup(items: list[dict], seen_titles: list[str], sim) -> list[dict]:
    """Drop items whose title is near-identical to one already seen. `sim(a, b) -> float` in [0, 1]."""
    fresh = []
    for it in items:
        if any(sim(it["title"], s) > 0.9 for s in seen_titles[-300:]):
            continue
        seen_titles.append(it["title"])
        fresh.append(it)
    return fresh
