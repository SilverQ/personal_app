# Configuration

## Files and Environment
- `config.ini` (root)
  - `[GEMINI_API_KEY]` — multiple keys supported (one per entry)
  - `[KIWOOM_API]`
    - `appkey`, `secretkey`, `mode` (`real|mock`)
    - Base URL inferred: real → `https://api.kiwoom.com`, mock → `https://mockapi.kiwoom.com`
- Environment variables (override)
  - `GEMINI_API_KEY` or `GOOGLE_API_KEY` — comma‑separated keys

## Session Initialization
- `ConfigManager` loads config/env
- `GeminiAPIHandler` uses v1 client with key rotation
- `KiwoomAPIHandler` acquires token and caches expiry in `st.session_state`

## Dependencies (key)
- Python: streamlit, pandas, numpy, plotly, pillow, kaleido, requests, google‑genai client
- Image export requires `kaleido` (and `pillow` for composition)

## Data Locations
- Store root: `@data/` under repo root
- Cache: `cache/` (e.g., `stock_market_caps.csv`)
- Reports: `reports/charts/`

## Operational Notes
- Prefer local store; fall back to API when coverage is insufficient
- Monthly series validated for cadence; rebuilt from daily/weekly if sparse
- Gemini calls display model used and maintain a simple call counter

