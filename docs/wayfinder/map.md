# Map — xau-terminal  `wayfinder:map`
> Tracker: markdown locale (nessun issue tracker configurato). Ticket = sezioni sotto.

## Destination
Plan + design di xau-terminal bloccati: fonti dati, modelli e algoritmo di confluenza decisi, pronti per `/pdca do`.

## Notes
Dominio: trading XAU discrezionale (SMC/ICT, sessioni, cross-pair). Skill: research, grilling.
Preferenza: tutto on-device, fonti gratuite/API ufficiali prima dello scraping. UI in sola lettura.

## Decisions so far
- [Fonti dati oltre lo scraping](#t1-fonti-dati-oltre-lo-scraping--research--closed): API pubbliche gratuite coprono posizionamento, liquidità retail, macro, odds Fed, Cina; scraping solo come ultimo fallback.
- Stack, modelli, scope v1 → vedi [plan](../01-plan/features/xau-terminal.plan.md).

- [Modelli statistici del mercato](#t2-modelli-statistici-del-mercato--research--closed): HMM di regime + Kalman (beta dinamica e fair value) + grafo causale PCMCI+ + GARCH; MarS rinviato (serve dato order-level che l'oro spot OTC non ha).

- [skfolio](#t3-skfolio--research--closed): sì, per 5 pezzi (grafo di dipendenza, copule di stress, entropy pooling, walk-forward CV, rischio combinato); no ottimizzazione di portafoglio azionario.

- [System One models (Jev / Lev)](#t4-system-one-models-jev--lev--research--closed): Lev (open, 4B, Apache-2.0) diventa il **motore decisionale unico** per le news e il gate finale del setup; Jev via API come opzione hosted.

## Not yet specified
- Dove gira Lev: MPS sul Mac (non dichiarato dagli autori) o GPU del server .60 → misurare la latenza.
- Pesi della confluenza: calibrazione via `CombinatorialPurgedCV` di skfolio (vedi T3) su storico MT5/Dukascopy — serve prima l'export dei dati.
- Validazione dell'order book OANDA come proxy degli stop "veri" (quanto anticipa gli sweep?).

## Out of scope
- RL, esecuzione ordini dalla UI (vedi plan §3).

---

## T1. Fonti dati oltre lo scraping — `research` · CLOSED

### Question
Quali fonti strutturate (API/file ufficiali, gratuite) alimentano il terminale senza scraping stealth?

### Resolution
| Fattore della strategia | Fonte | Accesso | Frequenza | Uso nel terminale |
|---|---|---|---|---|
| **Liquidità retail / stop** | OANDA v20 `GET /v3/instruments/XAU_USD/orderBook` e `/positionBook` ([doc](https://oanda-api-v20.readthedocs.io/en/stable/endpoints/instruments/instrumentorderbook.html)) | token di un conto demo gratuito | snapshot ~20 min | heatmap ordini/posizioni per livello → dove stanno gli stop, "chi è intrappolato" |
| **Posizionamento istituzionale** | CFTC COT, Socrata dataset `72hh-3qpy` (disaggregated, futures only), senza token ([CFTC](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm)); mirror CSV `futuresbench.com/data/cot/gold.csv` ([doc](https://futuresbench.com/api/)) | nessuna chiave | settimanale (ven. sera, dati del martedì) | net managed money, COT index 52w → bias macro |
| **Driver macro** | FRED: `DFII10` (rendimenti reali 10y), `DTWEXBGS` (USD broad), `T10YIE` (breakeven) | chiave gratuita | giornaliera | correlazione rolling oro vs rendimenti reali e dollaro |
| **Volatilità implicita oro** | Cboe GVZ (`^GVZ` via yfinance) ([Cboe](https://www.cboe.com/us/indices/dashboard/gvz/)) | nessuna chiave | intraday ritardato | regime di volatilità → dimensione stop/target |
| **Flussi ETF** | SPDR GLD holdings; `ak.macro_cons_gold()` (AKShare) | nessuna chiave | giornaliera | domanda fisica occidentale |
| **Domanda Cina** | AKShare: `spot_quotations_sge("Au99.99")` (realtime SGE), `spot_golden_benchmark_sge()`, `macro_china_fx_gold()` (riserve PBoC), `futures_comex_inventory("黄金")` ([AKShare](https://akshare.akfamily.xyz/data/spot/spot.html)) | nessuna chiave | realtime / mensile | **premio SGE vs XAUUSD×USDCNH** = domanda asiatica; riserve PBoC |
| **News Cina realtime** | AKShare `js_news` → flash di Jin10 金十数据 (ultime 4h) | nessuna chiave | ~minuti | feed in cinese → traduzione + sentiment |
| **Probabilità Fed** | Polymarket Gamma API `gamma-api.polymarket.com/markets` · Kalshi `trade-api/v2/markets` (lettura pubblica) | nessuna chiave per la lettura | live | probabilità di taglio/rialzo → driver dei rendimenti reali |
| **Calendario macro** | ForexFactory JSON `nfs.faireconomy.media/ff_calendar_thisweek.json` | nessuna chiave | settimanale | NO TRADE ±15 min dagli eventi high-impact |
| **Storico tick** | Dukascopy (download storico gratuito) | nessuna chiave | storico | backtest dei pesi di confluenza |

Verificate in questa sessione: OANDA, CFTC/futuresbench, Cboe GVZ, AKShare (SGE, COMEX, Jin10, SPDR).
Test `curl` del 2026-10-05: ForexFactory JSON **200 ✓**, futuresbench `gold.csv` **200 ✓**,
Polymarket Gamma e Kalshi **nessuna connessione (000)** da questa rete → probabilmente geo-block o
blocco di rete; da riprovare via VPN/.60, altrimenti fallback su CME FedWatch. Dukascopy non testato.

### Impatto sul plan
1. Nuovo componente di confluenza **"Posizionamento"** (peso 10): cluster di stop dall'order book OANDA + COT estremo.
2. Nuovo filtro **"Evento macro"**: NO TRADE automatico intorno agli eventi high-impact.
3. Nuovo pannello bento **"Cina"**: premio SGE, riserve PBoC, flash Jin10.
4. La traduzione torna in scope (feed Jin10 in cinese) → `Helsinki-NLP/opus-mt-zh-en` (~300 MB) oppure l'LLM locale.

---

## T2. Modelli statistici del mercato — `research` · CLOSED
> Nota: nato da un'interpretazione errata di "system models" (vedi T4), ma resta valido come layer quantitativo.

### Question
Quali modelli del *sistema* mercato (regimi, stato latente, causalità, simulazione) servono al terminale, leggeri e on-device?

### Resolution
| Livello | Modello | Libreria | Costo | Cosa dà al trader |
|---|---|---|---|---|
| **Regime** | Gaussian HMM 4 stati su feature XAU + DXY + VIX + rendimenti reali, refit rolling (niente look-ahead) | `hmmlearn` | ms, CPU | regime corrente: *Risk-off/hedge attivo*, *Hedge failure* (oro venduto per liquidità), *Risk-on*, *Transizione*. Uno studio del 2026 su XAUUSD H1 da MT5 identifica proprio questi 4 regimi ([dataset](https://exa.ai/library/publication/3wgntnk5y0w)) |
| **Stato latente** | Filtro di Kalman: beta dinamica di XAU vs `DFII10` e DXY → fair value + residuo z-score | numpy (~20 righe) | µs | "l'oro è caro/economico rispetto ai suoi driver adesso"; il residuo estremo segnala l'anomalia |
| **Causalità** (incrocio dati vero) | PCMCI+ su {XAU, DXY, rendimenti reali, GVZ, COT net, premio SGE, sentiment macro}; RPCMCI per i grafi dipendenti dal regime | `tigramite` ([repo](https://github.com/jakobrunge/tigramite)) | secondi, refit giornaliero | **grafo causale con lag** nel pannello "grafo": chi guida l'oro *oggi* (dollaro? tassi? Cina?) invece della sola correlazione |
| **Volatilità** | GARCH(1,1) condizionato al regime | `arch` | ms | stop/target in ATR "di regime" invece che fissi |
| **Simulazione / world model** | MarS (Microsoft Research Asia) — LMM order-level ([repo](https://github.com/microsoft/MarS)) | Ray + modello pesante | GPU, serve l'order book completo | **rinviato**: XAU spot è OTC, nessun dato order-level completo. Sostituto: Monte Carlo dai quantili TimesFM/Kronos (N percorsi → probabilità di toccare la prossima pozza di liquidità prima dello stop) |

Riferimento accademico (Cina): un modello MS-GARCH-MIDAS mostra che i driver macro USA e Cina dell'oro cambiano segno a seconda del regime ([Cheng et al. 2025](https://ideas.repec.org/a/spr/empeco/v69y2025i3d10.1007_s00181-025-02772-7.html)) → giustifica lo score di confluenza **pesato per regime**.

### Impatto sul plan
1. Nuovo file `backend/system.py`: HMM, Kalman, PCMCI+, GARCH, Monte Carlo.
2. Pesi di confluenza **diversi per regime** (es. in *Hedge failure* il sentiment macro bull pesa meno).
3. Nuovo componente di confluenza **"Fair value"**: residuo Kalman nella direzione del setup.
4. Pannello bento "Sistema": regime + probabilità, grafo causale, gauge fair value, P(target prima dello stop).

---

## T3. skfolio — `research` · CLOSED

### Question
skfolio ([repo](https://github.com/skfolio/skfolio), BSD-3, API scikit-learn, 2.3k★) serve a un terminale XAU mono-asset?

### Resolution
Il suo cuore (ottimizzazione di portafoglio, factor model azionario a 46 descrittori) **non serve**. Servono 5 moduli:

| Modulo skfolio | Uso nel terminale | Sostituisce |
|---|---|---|
| `GraphicalLassoCV` + `DistanceCorrelation` / `VariationOfInformation` | grafo di **dipendenza condizionata** tra le 6 coppie XAU e i driver: archi solo dove c'è legame diretto, anche non lineare | heatmap Pearson (ingannevole: tutte le coppie XAU sono correlate tramite XAU) |
| `VineCopula` con campionamento condizionato + copule Clayton/Gumbel | **stress test**: "se DXY +1% / VIX +30%, distribuzione di XAUUSD e XAUEUR"; dipendenza di coda oro↔azioni = misura diretta dell'*hedge failure* | scenari a mano |
| `EntropyPooling` / `OpinionPooling` | **motore di incrocio con base teorica**: ogni fonte è una "view" con confidenza (sentiment macro, COT, forecast, fair value Kalman, regime) → distribuzione a posteriori dei rendimenti XAU → P(rialzo), rendimento atteso, coda | somma pesata a mano dello score (resta come vista spiegabile) |
| `WalkForward`, `CombinatorialPurgedCV` | **calibrazione dei pesi** di confluenza e del filtro NO TRADE senza leakage | backtest ingenuo |
| `RiskBudgeting` + `RegimeAdjustedEWCovariance` | se apri XAUUSD e XAUEUR insieme: rischio combinato reale e sizing per budget di rischio, covarianza che si adatta al regime | lotti decisi a occhio |

### Impatto sul plan
1. `backend/system.py` usa skfolio per grafo, stress, entropy pooling.
2. Lo score di confluenza diventa doppio: **score spiegabile 0–100** (somma pesata, per leggere le ragioni) + **P(rialzo) da entropy pooling** (per decidere). Se divergono → NO TRADE.
3. Nuovo script offline `backend/calibrate.py`: walk-forward sui pesi, eseguito a mano, non nel loop live.
4. Pannello "Rischio": stress scenario + rischio combinato delle posizioni aperte (lette in sola lettura dall'MCP MT5).

---

## T4. System One models (Jev / Lev) — `research` · CLOSED

### Question
I "System One models" di ultimissima generazione (Jev di TypeSafe, Lev open) possono sostituire i classificatori e diventare il motore decisionale del terminale?

### Resolution
**Cosa sono.** Modelli che non generano testo: ricevono uno *state* + domande tipizzate (`noul` sì/no, `choice`, `score` 2–10 livelli) e restituiscono **probabilità calibrate** per ogni opzione in un solo forward pass. Etichette fuori lista impossibili, zero token di output. Addestrati con RLCD (Reinforcement Learning for Calibrated Decisions): se dicono 0.8, hanno ragione l'80% delle volte.

| | **Jev 1.13** (TypeSafe, 15 set 2026) | **Lev** (Interfaze, open) |
|---|---|---|
| Pesi | chiusi | **Apache-2.0**, Qwen3.5-4B + LoRA |
| S1Bench (13 subset) | 76.1% | 68.9% (miglior open ≤4B) |
| Calibrazione ECE | 0.091 | 0.115 |
| Latenza | 70–500 ms (API) | ~69 ms su GPU; **secondi su CPU** |
| Costo | $0.042 / 1M token input, output gratis | locale; ~8 GB base + 200 MB adapter |
| API | `/systemone` | `lev serve` → `/v1/systemone` (stesso protocollo) |

Fonti: [lancio Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) · [TechCrunch](https://techcrunch.com/2026/09/18/a-new-kind-of-ai-model-from-a-chatgpt-inventor-is-thrilling-developers/) · [Lev](https://interfaze.ai/blog/jev-now-open-source-lev) · [Jev non è open](https://www.orcarouter.ai/blog/jev-open-source).
Non verificato in modo indipendente: S1Bench è dichiarato dagli autori di Lev; nessuna valutazione terza su testi finanziari.

### Perché è il pezzo giusto qui
Lo score di confluenza e l'entropy pooling (T3) hanno bisogno di **probabilità oneste**, non di etichette. I classificatori di sentiment danno softmax non calibrati; un LLM dà testo. Lev/Jev danno esattamente l'input che serve.

### Uso nel terminale
**1. News: una sola chiamata per titolo** (sostituisce sentiment + zero-shot + parte del NER):
```python
questions = {
  "gold_bias":  {"type": "choice", "instructions": "Effetto probabile sull'oro nelle prossime ore",
                 "criteria": {"bull": "rialzista", "bear": "ribassista", "none": "irrilevante"}},
  "theme":      {"type": "choice", "instructions": "Tema macro principale",
                 "criteria": {"fed": "Fed/tassi USA", "inflation": "inflazione", "usd": "dollaro",
                              "geo": "geopolitica/guerra", "cb_buying": "acquisti banche centrali",
                              "risk": "risk-on/off azionario", "china": "domanda Cina", "other": "altro"}},
  "surprise":   {"type": "noul", "instructions": "Il dato/evento è una sorpresa rispetto alle attese?"},
  "impact":     {"type": "score", "instructions": "Impatto atteso sul prezzo",
                 "criteria": ["nullo", "basso", "medio", "alto"]},
  "usd_driven": {"type": "noul", "instructions": "L'effetto passa principalmente dal dollaro?"},
}
```
`usd_driven` alimenta direttamente il filtro "forza vera vs debolezza USD".

**2. Gate finale del setup.** Lo *state* è il riassunto testuale dei segnali (regime, sweep, SMT, sessione, fair value, COT, news delle ultime 2 h) e le domande sono `take_trade` (noul), `direction` (choice), `instrument` (choice tra le 6 coppie XAU). Policy: **si opera solo se P > 0.75** e la direzione concorda con lo score e con l'entropy pooling; altrimenti NO TRADE.

**3. Feed cinese (Jin10).** Qwen3.5 è multilingue: lo state in cinese funziona senza traduzione → il task Translation esce di nuovo dallo scope.

### Deployment
- Primario: `lev serve` sul server .60 (GPU), chiamato dal backend; batch dei titoli.
- Fallback: Jev via API (costo trascurabile ai volumi di news) con `system-one-adapter-python`, così l'interfaccia resta portabile.
- Ultimo fallback: classificatori piccoli del plan §4.4 (senza calibrazione, badge "UNCALIBRATED").

### Impatto sul plan
1. `nlp.py`: `system_one(state, questions)` come unica interfaccia di classificazione; TinyBERT e DeBERTa zero-shot restano solo come fallback.
2. Nuovo gate decisionale calibrato in `signals.py`.
3. UI: ogni chip mostra la probabilità (es. "bull 0.82"), non solo l'etichetta.
4. Nuovo task di validazione: confrontare la calibrazione di Lev sulle news dell'oro con l'esito di prezzo a 1 h/4 h (reliability diagram nel pannello "Sistema").
