# Plan Plus — xau-terminal

> Terminale leggero per XAU: news realtime + sentiment macro/micro, TA + liquidità SMC/ICT,
> forecast TimesFM/Kronos, algoritmo di incrocio cross-pair. Tutti i modelli on-device.
> Ricerca: [`docs/00-research/models-and-sources.md`](../../00-research/models-and-sources.md)

## Executive Summary
| Prospettiva | Contenuto |
|---|---|
| **Problema** | Il setup sull'oro richiede di incrociare a mano news, macro, sessioni, liquidità e 6+ coppie XAU: lento, soggettivo, e si salta il filtro "forza vera vs debolezza USD". |
| **Soluzione** | FastAPI con pipeline ML on-device (sentiment, zero-shot, NER, embedding, RAG, TimesFM 3.0, Kronos) + algoritmo di confluenza cross-pair; frontend Next.js bento in streaming SSE. |
| **Funzione / UX** | Una schermata: ticker, grafico con sweep/FVG/banda forecast, heatmap e grafo cross-pair, gauge di confluenza con motivazioni, feed news cercabile con sentiment, box "chiedi alle news". |
| **Valore core** | Score di confluenza 0–100 spiegabile e stato "NO TRADE" esplicito: lo strumento dice anche quando stare fuori. |

## 1. User Intent Discovery
- **Problema core**: aggregare e incrociare in tempo reale tutti i fattori della strategia (SMC/ICT, sessioni, macro, cross-pair) in un solo punto decisionale.
- **Utente**: trader discrezionale singolo, uso locale su Mac + server .60.
- **Criteri di successo**:
  1. Latenza prezzo→UI < 2 s con MT5 attivo.
  2. News nuove classificate e visibili < 90 s dalla pubblicazione RSS.
  3. Score di confluenza ricalcolato a ogni chiusura M5, ogni componente visibile.
  4. Tutta l'inferenza ML locale, RAM del backend < 3 GB.
  5. Hit-rate del forecast tracciato rolling (onestà del modello).

## 2. Alternatives Explored
| | A: Next.js + FastAPI (**scelta utente**) | B: FastAPI + 1 HTML vanilla | C: Ricostruire FinceptTerminal |
|---|---|---|---|
| Pro | Bento componibile, TS, ecosistema chart | zero build, minimo | feature enormi |
| Contro | build step, 2 processi | UI meno componibile | C++/Qt, mesi di lavoro |
| Esito | ✅ | scartata dall'utente | fuori scala |

## 3. YAGNI Review
**Incluso v1** (utente: "tutto"):
- News realtime (RSS + GDELT) + ricerca + dedup/cluster semantico
- Sentiment micro (per titolo) e macro (temi: Fed/tassi, inflazione, USD, geopolitica, banche centrali, risk-on/off, ETF/flussi)
- NER (banche centrali, paesi, persone) per i filtri
- RAG "chiedi alle news" + riassunto di sessione (LLM piccolo locale)
- TA + liquidità SMC/ICT
- Cross-pair confluence
- Forecast TimesFM 3.0 + Kronos-mini (a confronto)

**Rinviato / rimosso**:
| Voce | Motivo |
|---|---|
| RL | nessuna reward affidabile; spinge verso l'auto-trading → rimosso |
| Video-Text, Visual Document Retrieval, Table QA | nessun input di quel tipo nel flusso → rimossi |
| Traduzione | titoli in EN, il riassunto LLM esce già in italiano → coperto |
| Esecuzione ordini dalla UI | l'MCP MT5 fa già trading; UI **sola lettura** per sicurezza |
| Auth, DB persistente | locale, mono-utente; stato in memoria (deque). SQLite quando serve lo storico |
| Fine-tuning modelli | prima misurare lo zero-shot |

## 4. Design

### 4.1 Architettura
```
MT5 bridge :8080 ──┐                       ┌── Next.js :3000 (bento)
yfinance fallback ─┤                       │      ▲ SSE /api/stream
RSS / GDELT ───────┼──► FastAPI :8000 ─────┘      │ REST /api/*
                   │   ├ prices   (poll 2s)
                   │   ├ news     (poll 60s) → nlp (sentiment, zero-shot, NER, embed)
                   │   ├ signals  (TA + SMC/ICT + cross-pair + confluenza)
                   │   ├ forecast (TimesFM 3.0 MLX/torch, Kronos-mini) a ogni M15
                   │   └ ask      (RAG: embed → top-k → LLM locale via Ollama)
Ollama .60 / locale ┘
```

### 4.2 Componenti
**Backend `backend/`**
| File | Responsabilità |
|---|---|
| `main.py` | FastAPI, loop in background, SSE, route |
| `prices.py` | MT5 `/rates/{sym}/{tf}` → fallback yfinance (`GC=F`, `EURUSD=X`…); cross XAU sintetici |
| `news.py` | poll RSS + GDELT, normalizzazione, dedup per similarità coseno |
| `nlp.py` | lazy-load dei modelli: sentiment, zero-shot macro, NER, embedding, RAG |
| `signals.py` | indicatori, swing, sweep, FVG, sessioni/killzone/Silver Bullet, dwell-time, SMT, confluenza |
| `forecast.py` | TimesFM 3.0 (multivariato: XAUUSD + EURUSD come covariata) + Kronos-mini |

**Frontend `web/`** (Next.js App Router, una pagina)
Ticker · Chart (lightweight-charts: candele, sweep, FVG, banda quantili) · Gauge di confluenza + ragioni · Heatmap cross-pair · Grafo correlazioni · Orologio sessioni · Feed news (ricerca, chip sentiment/tema) · Barre macro · Box Ask.

### 4.3 Algoritmo di incrocio (il cuore)
Coppie: XAUUSD, XAUEUR, XAUGBP, XAUJPY, XAUCHF, XAUAUD (+ proxy DXY dai cambi).
Sintetici se il broker non li quota: `XAUEUR = XAUUSD / EURUSD`, `XAUJPY = XAUUSD × USDJPY`.

| Componente | Calcolo | Peso |
|---|---|---|
| **Forza oro reale** (breadth) | quota di coppie XAU con rendimento a N barre nello stesso verso + z-score medio | 20 |
| **Filtro debolezza USD** | XAUUSD ↑ ma XAUEUR piatto/↓ e DXY ↓ → penalità | (−) |
| **Liquidità presa** | sweep del range Asia / high-low precedente con chiusura di rientro | 20 |
| **SMT divergence** | sweep su una coppia XAU ma non sull'altra → indica lo strumento "pulito" | 10 |
| **Tempo di permanenza** | barre oltre il livello sweepato: rientro rapido = forza, permanenza = zona che si consuma | 10 |
| **Sessione** | dentro killzone Londra/NY o finestra Silver Bullet (03–04, 10–11, 14–15 ora di NY) | 10 |
| **FVG / displacement** | FVG creato nella direzione dopo lo sweep | 10 |
| **Macro sentiment** | media pesata per recency × rilevanza dei temi (bull/bear oro) | 10 |
| **Forecast** | segno della mediana TimesFM + Kronos, scalato per la larghezza della banda | 10 |

Output: `score 0–100`, `direzione`, `strumento suggerito`, `ragioni[]`; `NO TRADE` se score < 60
o se i componenti principali sono in conflitto. "Chi è intrappolato": i breakout trader al livello
sweepato → target = pozza di liquidità opposta successiva (liquidità annidata).

### 4.4 Modelli (default, sostituibili via env)
| Compito | Modello | Dim. |
|---|---|---|
| Sentiment micro | `mikeysharma/finance-sentiment-analysis` (TinyBERT) | ~55 MB |
| Macro zero-shot | `MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33` | ~90 MB |
| NER | `dslim/distilbert-NER` | ~260 MB |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` | ~90 MB |
| LLM (riassunto/RAG) | Ollama `qwen3:0.6b` (o modello su .60) | esterno |
| Forecast | `google/timesfm-3.0-pytorch`, `NeoQuasar/Kronos-mini` | ~1.3 GB / 16 MB |

### 4.5 Flusso dati
1. Loop prezzi ogni 2 s → cache OHLCV M5/M15/H1 per coppia → `signals.compute()` alla chiusura della candela.
2. Loop news ogni 60 s → nuovi item → embed → dedup (cos > 0.9) → sentiment + tema + NER → deque(500).
3. Loop forecast alla chiusura M15.
4. SSE `/api/stream` invia `{ticker, confluence, news_delta, forecast}`; il frontend fa merge dello stato.

## 5. Rischi
| Rischio | Mitigazione |
|---|---|
| MT5 offline | fallback yfinance con badge "DELAYED" |
| TimesFM pesante su CPU | backend MLX su Apple silicon, cadenza M15, lazy load, flag per disattivarlo |
| Forecast scambiato per segnale | peso 10/100, hit-rate mostrato accanto |
| Feed RSS che cambiano | ogni feed in try/except, stato per fonte visibile in UI |

## 6. Brainstorming Log
- Prezzi: MT5 + fallback yfinance (utente).
- Sentiment/ML: "tutti light e on device cpu o gpu", bento con grafi (utente).
- Stack: Next.js + FastAPI (utente). Scope v1: tutto (utente).
- `/wayfinder` richiesto ma non installato → saltato.

## 7. Next
`/pdca design xau-terminal` → `/pdca do xau-terminal`

## 8. Revisione 2 (2026-10-05) — dalla mappa wayfinder
Decisioni in [`docs/wayfinder/map.md`](../../wayfinder/map.md) (T1–T4). Delta rispetto alle sezioni sopra:
- **Classificazione news e gate finale** → System One model (Lev su .60, Jev API come riserva, TinyBERT/DeBERTa solo come ultimo fallback "UNCALIBRATED"). Traduzione fuori scope (Lev legge il cinese).
- **Nuove fonti**: COT (futuresbench), calendario ForexFactory, OANDA order/position book, FRED, GVZ, AKShare (SGE, Jin10). File `backend/macro.py`.
- **Layer quantitativo** `backend/system.py`: HMM di regime, Kalman fair value, grafo di dipendenza (GraphicalLassoCV di skfolio), PCMCI+ (tigramite) giornaliero, Monte Carlo dai quantili.
- **Confluenza** pesata per regime + componenti "Posizionamento" e "Fair value" + blocco evento macro; decisione = score spiegabile **e** P calibrata da Lev > 0.75.
- **Hardware reale**: Mac Intel i7 (niente MLX/MPS) → TimesFM su CPU a cadenza M15; Lev e LLM sul server .60.
- Rinviati: Kronos (richiede il repo, non è su pip → attivabile con `KRONOS_PATH`), `calibrate.py` (serve prima lo storico esportato), MarS.
