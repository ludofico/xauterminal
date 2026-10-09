# XAU Terminal — Ricerca modelli on-device e fonti dati
*Generato: 2026-10-05 | Fonti: 14 | Confidenza: Media-Alta*

## Executive Summary
Tutto lo stack ML può girare on-device (CPU/MPS/MLX) sotto ~2 GB RAM totali. Per il forecasting
TimesFM 3.0 (Google, set 2026) supporta multivariato + covariate e ha backend MLX nativo per Apple
silicon (~11 ms batch 1 su M4 Max). Kronos (sviluppatori cinesi, Tsinghua) è l'unico foundation
model nativo per candele OHLCV (mini = 4.1M param). Per il sentiment, i classificatori distillati
(TinyBERT/ALBERT/DistilBERT, 12–67M) eguagliano FinBERT con 1/10 della dimensione; esiste un
adapter LoRA specifico per l'oro. Avvertenza: la letteratura 2026 mostra che i TSFM battono il
random walk sui rendimenti finanziari solo di poco e raramente in modo significativo → il
forecast è un **input di confluenza**, non un segnale.

## 1. Forecasting time series
| Modello | Origine | Param | Note |
|---|---|---|---|
| `google/timesfm-3.0-pytorch` | Google (US) | 330M | multivariato + covariate passate/future, 9 quantili, backend MLX ([repo](https://github.com/google-research/timesfm)) |
| `NeoQuasar/Kronos-mini` / `-small` | Tsinghua (CN) | 4.1M / 24.7M | foundation model K-line, OHLCV in → OHLCV out, 45 borse ([repo](https://github.com/shiyu-coder/Kronos)) |
| Chronos-Bolt small / Chronos-2 | Amazon (US) | 48M / 120M | Chronos-2 ben calibrato (95% coverage al 90% nominale) su CPU ([arXiv 2602.10848](https://arxiv.org/pdf/2602.10848.pdf)) |
| Moirai-2 small | Salesforce (US) | 11M | velocissimo (25 ms CPU) ma overconfident (~70% coverage) |

- Benchmark rendimenti azionari: TimesFM-2.5 e Moirai-2 i migliori rank medi, ma il test di
  Diebold-Mariano rifiuta il random walk solo in 2 casi su 10 ([arXiv 2606.27100](https://arxiv.org/html/2606.27100)).
- Chronos-Bolt: energia e latenza più basse tra 10 TSFM ([PMLR v309](https://proceedings.mlr.press/v309/guibert26a.html)).

## 2. Sentiment finanziario (micro)
| Modello | Param | Accuratezza | Note |
|---|---|---|---|
| `hadangvu/pkd-albert-student` | 11.7M | 97.4% FPB-allagree | distillato da FinBERT, 44 MB ([HF](https://huggingface.co/hadangvu/pkd-albert-student)) |
| `AnkitAI/FinSense-ModernBERT-…` | 149M | 0.8675 FPB-50agree held-out | il più accurato su split pulito ([HF](https://huggingface.co/AnkitAI/FinSense-ModernBERT-Financial-News-Sentiment)) |
| `misraanay/finbert-tone-gold-lora-final` | 110M + 1.2 MB LoRA | 87% | **specifico per notizie sull'oro**, 4 classi ([HF](https://huggingface.co/misraanay/finbert-tone-gold-lora-final)) |
| `mikeysharma/finance-sentiment-analysis` | TinyBERT 14M | 89% | 28 ms CPU |

Macro (tema hawkish/dovish, risk-on/off, geopolitica): zero-shot NLI (`bart-large-mnli` o
DeBERTa-v3 xsmall) — pattern usato anche in [Finpipe](https://github.com/MwkosP/Finpipe).

## 3. Embedding / RAG / generazione
- Qwen3-Embedding-0.6B (Alibaba, CN): miglior retrieval F1 sotto 1B, 32K context; ma ~1.2 GB ([bench ONNX](https://iam-tsr.github.io/blog/cutting-edge-onnx-embedding-models/)).
- EmbeddingGemma-300M (Google, US): #1 MTEB sotto 500M ([arXiv 2509.20354](https://arxiv.org/abs/2509.20354)).
- all-MiniLM-L6/L12: >100 QPS, ok per dedup/clustering titoli (non per RAG fine).
- Generazione/QA locale: Qwen3-0.6B / Gemma-4 E2B via Ollama o LiteRT ([edgerag](https://github.com/Danselem/edgerag)).

## 4. Fonti news gratuite
| Fonte | Limite | Uso |
|---|---|---|
| RSS via `feedparser` (Reuters, FXStreet, Kitco, ForexLive, Fed, ECB) | illimitato | feed grezzo principale |
| GDELT DOC 2.0 | nessuna chiave, refresh 15 min | tono globale, geopolitica |
| Finnhub | chiave gratuita, 60/min | news forex + sentiment pre-calcolato |
| Forex News API | chiave, piani a pagamento | calendario economico |

Pattern dashboard di riferimento: [situation-monitor](https://github.com/ramenwang/situation-monitor) (RSS+GDELT+Finnhub).
Riferimento architetturale: [FinceptTerminal](https://github.com/Fincept-Corporation/FinceptTerminal) (C++/Qt + Python, 100+ connettori) — troppo pesante da replicare, ne prendiamo solo l'idea dei "desk".

## Key Takeaways
- Forecast = un voto nella confluenza, con banda quantili visibile; mai trade sul solo forecast.
- Cross-pair via TimesFM 3.0 multivariato: XAUUSD target, EURUSD/DXY come covariate.
- Sentiment a due livelli: classificatore piccolo per titolo (micro) + zero-shot per tema (macro).

## Gap
- Nessun benchmark trovato di TSFM su XAU intraday → da validare in-app (hit-rate rolling).
- Accuratezza del LoRA-oro non verificata indipendentemente.

## Methodology
4 query Exa + 3 fetch diretti (README TimesFM/Kronos/FinceptTerminal) + API HF per gli ID modello.
Sottodomande: TSFM leggeri; sentiment finanziario piccolo; embedding/LLM on-device; fonti news gratuite.
