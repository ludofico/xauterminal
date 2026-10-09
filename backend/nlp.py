"""News NLP without an LLM. Primary: Laya (open, Jev-protocol System One, ModernBERT, in-process on CPU) ->
calibrated typed decisions, one forward pass per headline with every question. Optional: Jev (hosted).
The predefined keyword set lives in the question criteria (it steers Jev) and doubles as a deterministic
offline extractor with the same output shape when Jev is unavailable. No torch on the default path."""
import difflib
import hashlib
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import httpx

JEV_URL = os.getenv("JEV_URL", "https://api.typesafe.ai/v1/systemone")
JEV_KEY = os.getenv("TYPESAFE_API_KEY", "")
JEV_MODEL = os.getenv("JEV_MODEL", "jev-latest")
JEV_TIMEOUT = float(os.getenv("JEV_TIMEOUT", "2.0"))   # Jev answers in 70-500 ms; past 2 s fall back, never block
JEV_PARALLEL = int(os.getenv("JEV_PARALLEL", "8"))
LEV_LOCAL = False  # Lev 4B removed from the path: Laya covers the same protocol at a fraction of the weight
LAYA_LOCAL = os.getenv("LAYA_LOCAL", "1") == "1"
LAYA_MODEL = os.getenv("LAYA_MODEL", "convaiinnovations/laya")
LAYA_SUBFOLDER = os.getenv("LAYA_SUBFOLDER", "typed-decisions")  # fine-tuned checkpoint; base ones need training

# ---------- predefined keyword set (single source of truth: steers Jev AND drives the offline extractor) ----------
# (regex, weight). Matched case-insensitively on the headline.
BIAS_KW = {
    "bull": [(r"\b(gold|bullion|xau/?usd) (rises|rallies|jumps|climbs|gains|surges|soars|breaks out|hits record|record high)", 2.0),
             (r"\brate cuts?\b|\bcuts? rates?\b|\bdovish\b|\beasing\b", 1.5),
             (r"\bsafe[- ]haven\b|\bflight to safety\b", 1.5),
             (r"\b(war|attack|missile|strike|invasion|escalat\w*|sanctions?|tensions?)\b", 1.0),
             (r"\bcentral banks? (buy|buying|purchas\w*|add\w*)\b|\bgold reserves?\b", 1.5),
             (r"(?<!xau/)(?<!xau)\b(dollar|greenback|dxy|usd) (falls|slides|drops|weakens|slumps|tumbles|retreats|eases)\b", 1.5),
             (r"\byields? (fall|drop|slide|decline|ease|retreat|cool)\w*\b", 1.0),
             (r"\brecession\b|\bslowdown\b|\bweak (jobs|payrolls|data)\b", 1.0),
             (r"\binflows?\b", 0.5)],
    "bear": [(r"\b(gold|bullion|xau/?usd) (falls|slides|drops|slips|tumbles|declines|retreats|plunges|eases)\b", 2.0),
             (r"\brate hikes?\b|\bhikes? rates?\b|\bhawkish\b|\btightening\b|\bhigher for longer\b", 1.5),
             (r"(?<!xau/)(?<!xau)\b(dollar|greenback|dxy|usd) (rises|jumps|surges|gains|strengthens|rallies|climbs|firms)\b", 1.5),
             (r"\byields? (rise|jump|surge|climb|spike)\b", 1.0),
             (r"\bceasefire\b|\bpeace (deal|talks)\b|\bde-?escalat\w*\b", 1.0),
             (r"\brisk[- ]on\b|\bstocks? (rally|surge|record)\b", 0.5),
             (r"\bprofit[- ]taking\b|\boutflows?\b", 0.7),
             (r"\b(strong|hot|robust) (jobs|payrolls|data|cpi)\b|\binflation (cools|eases)\b", 1.0)],
}
THEME_KW = {
    "fed": r"\bfed\b|\bfomc\b|\bpowell\b|\bfederal reserve\b|\brate (cut|hike|decision)s?\b",
    "inflation": r"\bcpi\b|\bpce\b|\bppi\b|\binflation\b",
    "usd": r"\bdollar\b|\bgreenback\b|\bdxy\b|\busd\b|\btreasur(y|ies)\b|\byields?\b",
    "geo": r"\bwar\b|\battack\b|\bmissile\b|\biran\b|\bisrael\b|\brussia\b|\bukraine\b|\bsanction|\btariff|\bgeopolit",
    "cb_buying": r"\bcentral banks?\b.*\bgold\b|\bpboc\b|\bgold reserves?\b|\bworld gold council\b",
    "risk": r"\bstocks?\b|\bequit(y|ies)\b|\bs&p\b|\bnasdaq\b|\bvix\b|\brisk[- ](on|off)\b",
    "china": r"\bchina\b|\bchinese\b|\bshanghai\b|\bsge\b|\byuan\b|黄金|央行",
}
SURPRISE_KW = (r"\bunexpected(ly)?\b|\bsurpris\w*\b|\bbeats?\b|\bmiss(es|ed)?\b|\bshock\w*\b"
               r"|\b(above|below) (expectations|forecast|consensus)\b")
IMPACT_KW = [(r"\bfomc\b|\bnfp\b|\bnonfarm\b|\bpayrolls\b|\bcpi\b|\bpowell\b|\brate decision\b|\bwar\b", 3),
             (r"\bfed\b|\becb\b|\binflation\b|\byields?\b|\bdollar\b|\brecord\b", 2),
             (r"\bgold\b|\bbullion\b|\bxau\b", 1)]
USD_KW = r"\bdollar\b|\bgreenback\b|\bdxy\b|\busd\b|\btreasur(y|ies)\b|\byields?\b|\bfed\b"
ENTITY_KW = {"Fed": r"\bfed\b|\bfomc\b|\bfederal reserve\b", "Powell": r"\bpowell\b", "ECB": r"\becb\b|\blagarde\b",
             "PBoC": r"\bpboc\b|\bpeople'?s bank of china\b", "BoJ": r"\bboj\b|\bbank of japan\b",
             "Trump": r"\btrump\b", "China": r"\bchina\b|\bchinese\b", "Iran": r"\biran\b", "Israel": r"\bisrael\b",
             "Russia": r"\brussia\b", "Ukraine": r"\bukraine\b", "OPEC": r"\bopec\b", "WGC": r"\bworld gold council\b"}

THEMES = {"fed": "Fed / US rates: FOMC, Powell, rate cut/hike",
          "inflation": "inflation data: CPI, PCE, PPI",
          "usd": "US dollar and Treasury yields: DXY, greenback",
          "geo": "geopolitics: war, attack, sanctions, tariffs, Iran, Russia",
          "cb_buying": "central bank gold buying: PBoC, reserves, World Gold Council",
          "risk": "equity risk-on/risk-off: stocks, S&P, Nasdaq, VIX",
          "china": "China gold demand: Shanghai, SGE, yuan",
          "other": "none of the above"}

# Keywords are written into the criteria so Jev decides against the same vocabulary the offline extractor uses.
NEWS_QUESTIONS = {
    "gold_bias": {"type": "choice", "instructions": "Likely effect of this headline on the gold price in the next hours",
                  "criteria": {"bull": "pushes gold up: rate cuts, dovish Fed, safe-haven demand, war/escalation, "
                                       "central bank buying, weaker dollar, falling yields, recession fears",
                               "bear": "pushes gold down: rate hikes, hawkish Fed, stronger dollar, rising yields, "
                                       "ceasefire/de-escalation, risk-on, profit-taking, strong data",
                               "none": "irrelevant for gold"}},
    "theme": {"type": "choice", "instructions": "Main macro theme of this headline", "criteria": THEMES},
    "surprise": {"type": "noul", "instructions": "Is this a surprise versus expectations (beat, miss, unexpected, shock)?"},
    "impact": {"type": "score", "instructions": "Expected price impact on gold (FOMC, CPI, NFP, war = high)",
               "criteria": ["none", "low", "medium", "high"]},
    "usd_driven": {"type": "noul", "instructions": "Does the effect on gold go mainly through the US dollar or yields?"},
}


# ---------- offline keyword extractor (same output shape as Jev) ----------
def _hits(text: str, rules) -> float:
    return sum(w for rx, w in rules if re.search(rx, text, re.I))


def _softmax(scores: dict, temp: float = 1.0) -> dict:
    m = max(scores.values())
    e = {k: math.exp((v - m) / temp) for k, v in scores.items()}
    z = sum(e.values())
    return {k: round(v / z, 3) for k, v in e.items()}


def keyword_classify(title: str) -> dict:
    """Deterministic typed extraction. Probabilities = softmax over keyword evidence: monotone, NOT calibrated."""
    bull, bear = _hits(title, BIAS_KW["bull"]), _hits(title, BIAS_KW["bear"])
    bias = _softmax({"bull": bull, "bear": bear, "none": 1.0 if bull == bear == 0 else 0.0}, temp=0.7)
    theme_scores = {k: float(len(re.findall(rx, title, re.I))) for k, rx in THEME_KW.items()}
    theme = max(theme_scores, key=lambda k: theme_scores[k]) if any(theme_scores.values()) else "other"
    impact = max((lvl for rx, lvl in IMPACT_KW if re.search(rx, title, re.I)), default=0)
    return {"bias": bias, "theme": theme, "theme_p": 1.0 if theme != "other" else 0.0,
            "surprise": 1.0 if re.search(SURPRISE_KW, title, re.I) else 0.0, "impact": float(impact),
            "usd_driven": 1.0 if re.search(USD_KW, title, re.I) else 0.0, "confidence": None,
            "engine": "keywords", "calibrated": False}


def entities(title: str) -> list[str]:
    return [name for name, rx in ENTITY_KW.items() if re.search(rx, title, re.I)]


# ---------- Jev (TypeSafe System One) ----------
_client = httpx.Client(timeout=JEV_TIMEOUT, limits=httpx.Limits(max_keepalive_connections=JEV_PARALLEL),
                       headers={"authorization": f"Bearer {JEV_KEY}", "content-type": "application/json"})
status = {"laya": "loading" if LAYA_LOCAL else "off", "jev": "no key" if not JEV_KEY else "idle"}


@lru_cache(maxsize=1)
def _laya():
    import laya

    return laya.load(LAYA_MODEL, subfolder=LAYA_SUBFOLDER or None, device="cpu")  # cpu: AMD-GPU Macs crash on MPS


def laya_ready() -> bool:
    return LAYA_LOCAL and _laya.cache_info().currsize > 0


def system_one(state, questions: dict) -> dict | None:
    """One Jev call: {"answers": {name: {type, choice|noul|score, probabilities, confidence}}, "engine"} or None."""
    if laya_ready():
        try:
            res = _laya().system_one(state, questions)
            ans = res.get("answers", res) if isinstance(res, dict) else res.answers
            return {"answers": {k: _as_dict(v) for k, v in ans.items()}, "engine": "laya"}
        except Exception as e:
            status["laya"] = f"error: {type(e).__name__}"
    if not JEV_KEY:
        return None
    try:
        r = _client.post(JEV_URL, json={"model": JEV_MODEL, "state": state, "questions": questions})
        r.raise_for_status()
        status["jev"] = f"ok {r.elapsed.total_seconds() * 1000:.0f} ms"
        return {"answers": r.json()["answers"], "engine": "jev"}
    except Exception as e:
        status["jev"] = f"fallback: {type(e).__name__}"  # 401/429/529/timeout -> keywords, never block the loop
        return None


def _from_jev(a: dict, engine: str) -> dict | None:
    bias, theme = a.get("gold_bias", {}).get("probabilities"), a.get("theme", {}).get("probabilities")
    if not bias or not theme:
        return None
    impact = a.get("impact", {}).get("score")
    return {"bias": bias, "theme": a["theme"].get("choice") or max(theme, key=lambda k: theme[k]),
            "theme_p": max(theme.values()), "surprise": a.get("surprise", {}).get("noul"),
            "impact": impact, "usd_driven": a.get("usd_driven", {}).get("noul"),
            "confidence": a["gold_bias"].get("confidence"), "engine": engine, "calibrated": True}


@lru_cache(maxsize=4096)
def _classify_cached(key: str, title: str) -> tuple:
    res = system_one(title, NEWS_QUESTIONS)
    out = _from_jev(res["answers"], res["engine"]) if res else None
    return tuple(sorted((out or keyword_classify(title)).items()))


def classify_news(title: str) -> dict:
    return dict(_classify_cached(hashlib.sha1(title.encode()).hexdigest(), title))


_pool = ThreadPoolExecutor(max_workers=JEV_PARALLEL)


def classify_many(titles: list[str]) -> list[dict]:
    """Parallel batch: N headlines in ~one Jev round-trip instead of N."""
    return list(_pool.map(classify_news, titles))


def warmup() -> None:
    """Load Laya once in the background (first run downloads ~1.7 GB). Keywords answer until it is ready."""
    global LAYA_LOCAL
    if LAYA_LOCAL:
        try:
            _laya()
            status["laya"] = "ready"
        except Exception as e:
            status["laya"] = f"disabled: {type(e).__name__}"
            LAYA_LOCAL = False


def lev_ready() -> bool:  # kept for main.py's gate check; Laya is the in-process engine now
    return laya_ready()


def _as_dict(ans) -> dict:
    if isinstance(ans, dict):
        return ans
    return {k: getattr(ans, k, None) for k in ("probabilities", "choice", "noul", "score", "confidence")}


# ---------- similarity / "ask" without an LLM ----------
def similarity(a: str, b: str) -> float:
    # ponytail: stdlib ratio for dedup (titles are short, near-dupes are near-identical strings).
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}|[一-鿿]+", s.lower()))


def ask(question: str, items: list[dict], k: int = 12) -> dict:
    """Keyword retrieval + aggregation of the typed answers into a short Italian summary. Nothing generated."""
    if not items:
        return {"answer": "Nessuna news in memoria.", "sources": []}
    q, qtheme = _tokens(question), keyword_classify(question)["theme"]
    scored = [(len(q & _tokens(i["title"])) + (2 if (i.get("nlp") or {}).get("theme") == qtheme != "other" else 0), n)
              for n, i in enumerate(items)]
    top = [items[n] for s, n in sorted(scored, key=lambda x: (-x[0], -x[1])) if s > 0][:k] or items[-k:]
    cls = [t.get("nlp") or keyword_classify(t["title"]) for t in top]
    bull = sum(c["bias"].get("bull", 0) for c in cls)
    bear = sum(c["bias"].get("bear", 0) for c in cls)
    themes: dict[str, int] = {}
    for c in cls:
        themes[c["theme"]] = themes.get(c["theme"], 0) + 1
    lean = "rialzista" if bull > bear * 1.2 else "ribassista" if bear > bull * 1.2 else "neutra"
    main = ", ".join(f"{t} ({n})" for t, n in sorted(themes.items(), key=lambda x: -x[1])[:3])
    usd = sum(1 for c in cls if (c.get("usd_driven") or 0) > 0.5) / len(cls)
    answer = (f"{len(top)} news rilevanti. Lettura sull'oro: {lean} (bull {bull:.1f} vs bear {bear:.1f}). "
              f"Temi: {main}. {usd:.0%} passano da dollaro/rendimenti"
              + (" → più debolezza USD che forza dell'oro." if usd > 0.5 else ".")
              + "\n" + "\n".join(f"• [{t['source']} {t['ts'][11:16]}Z] {t['title']}" for t in top[:6]))
    return {"answer": answer, "sources": top}


def _demo():
    a = keyword_classify("Gold rallies to record high as Fed signals faster rate cuts")
    assert a["bias"]["bull"] > a["bias"]["bear"] and a["theme"] == "fed" and a["impact"] >= 2, a
    b = keyword_classify("Dollar surges after hot CPI, gold slides")
    assert b["bias"]["bear"] > b["bias"]["bull"] and b["usd_driven"] == 1.0, b
    c = keyword_classify("Company X opens new office")
    assert c["bias"]["none"] > 0.5 and c["theme"] == "other", c
    assert entities("Powell says Fed will watch Iran") == ["Fed", "Powell", "Iran"]
    d = keyword_classify("Gold Price Forecast: XAU/USD rallies to $4,200 as US yields retreat")
    assert d["bias"]["bull"] > 0.9, d  # "XAU/USD rallies" is gold up, not "USD rallies"
    j = _from_jev({"gold_bias": {"choice": "bull", "probabilities": {"bull": .8, "bear": .1, "none": .1}, "confidence": .7},
                   "theme": {"choice": "fed", "probabilities": {"fed": .9, "other": .1}},
                   "surprise": {"noul": .2}, "impact": {"score": 2.4}, "usd_driven": {"noul": .6}}, "jev")
    assert j["calibrated"] and j["theme"] == "fed" and j["confidence"] == .7, j
    print("nlp ok")


if __name__ == "__main__":
    _demo()
