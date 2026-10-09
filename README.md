# XAU Terminal

Terminale XAU tutto locale (Mac Intel, CPU): news realtime + sentiment calibrato (Lev System One), TA + liquidità SMC/ICT,
cross-pair XAU×6 valute, regime HMM, fair value Kalman, grafi di dipendenza/causali, TimesFM 3.0, LLM leggero per il RAG.
Docs: `docs/01-plan/features/xau-terminal.plan.md`, decisioni in `docs/wayfinder/map.md`.

## Avvio
```bash
# backend (Python 3.12 + torch da conda-forge: su Mac Intel pip si ferma a torch 2.2)
cd backend
mkdir -p .tools && curl -Ls https://micro.mamba.pm/api/micromamba/osx-64/latest | tar -xj -C .tools bin/micromamba
MAMBA_ROOT_PREFIX=$PWD/.tools/root .tools/bin/micromamba create -y -p ./.env -c conda-forge python=3.12 "pytorch>=2.6"
uv pip install -p ./.env/bin/python -r requirements.txt -r requirements-ml.txt
./.env/bin/uvicorn main:app --port 8000

# LLM leggero (RAG "Chiedi alle news")
ollama pull qwen3.5:0.8b

# frontend
cd ../web && pnpm install && pnpm dev   # http://localhost:3000
```

## Variabili (tutte opzionali)
| Var | Default | Uso |
|---|---|---|
| `MT5_BRIDGE_URL` / `MT5_API_KEY` | `http://127.0.0.1:8080` | prezzi realtime dal bridge MT5; se giù → yfinance (DELAYED) |
| `MT5_TZ_OFFSET_H` | `3` | ora server broker − UTC (**da calibrare**: sbagliato = sessioni/killzone sfasate) |
| `MT5_SYMBOL_SUFFIX` | `""` | es. `.m`, `-ECN` |
| `LEV_LOCAL` | `1` | Lev in-process (~8 GB, secondi/titolo su CPU). `0` → fallback piccoli non calibrati |
| `LEV_URL` / `JEV_URL` + `TYPESAFE_API_KEY` | — | System One remoto opzionale |
| `LLM_URL` / `LLM_MODEL` | Ollama `/v1` · `qwen3.5:0.8b` | qualsiasi endpoint OpenAI-compatible |
| `OANDA_TOKEN` | — | order/position book XAU_USD (conto demo gratuito) |

## Check
`./.env/bin/python signals.py` · `./.env/bin/python system.py`

UI in sola lettura: nessun ordine parte da qui.
