# Stock Analysis v3

Entrypoint: `apps/stock_analysis_app_v3.py:1775` (`def main()`)

## Responsibilities
- Configure API handlers and session state
- Acquire + persist market data (daily/weekly/monthly candles, investor flows)
- Plot daily/weekly/monthly charts; export combined square JPG
- Run Gemini analysis, parse sections to state
- Build/search similarity indices (returns/price vectors, and feature‑based)
- Batch jobs: metadata collection and chart generation

## Key Classes
- `ConfigManager` (`apps/stock_analysis_app_v3.py:460`)
  - Reads `config.ini` and env (`GEMINI_API_KEY`, `GOOGLE_API_KEY`)
  - Provides Kiwoom config (appkey/secret/mode/url)

- `GeminiAPIHandler` (`apps/stock_analysis_app_v3.py:495`)
  - Key rotation; primary `gemini-2.0-flash` with fallback `gemini-2.5-flash`
  - `generate_content(prompt, system_instruction)` with v1 client

- `KiwoomAPIHandler` (`apps/stock_analysis_app_v3.py:617`)
  - Token acquisition and caching
  - `get_stock_info`, `fetch_all_chart_data` (D/W/M), `fetch_investor_data`
  - Fallback daily (close‑line) fetch, investor holdings helpers

- `ValuationCalculator` (`apps/stock_analysis_app_v3.py:956`)
  - Justified PBR, target price, investment opinion

## Local Store Helpers (JSON)
- Upsert loaders/savers
  - `upsert_daily_candles_store` (`apps/stock_analysis_app_v3.py:159`)
  - `upsert_investor_flows_store` (`apps/stock_analysis_app_v3.py:199`)
  - `load_daily_candles_from_store` (`apps/stock_analysis_app_v3.py:236`)
  - `load_investor_flows_from_store` (`apps/stock_analysis_app_v3.py:262`)
- Derivation and validation
  - `derive_weekly_monthly_from_daily` (resample) (`apps/stock_analysis_app_v3.py:300`)
  - `ensure_monthly_frequency` (rebuild if quarterly‑sparse) (`apps/stock_analysis_app_v3.py:318`)

## Charting & Export
- Daily panel (3 rows): candles, volume, cumulative investor net buy/sell
- Weekly (1Y) / Monthly (3Y) panels
- Export
  - `save_plotly_fig_as_jpg` (single) and `save_combined_charts_as_jpg` (square mosaic)
  - Combined export path: `reports/charts/<company>_<timestamp>.jpg`
  - Headless helper: `generate_and_save_combined_chart_headless` (`apps/stock_analysis_app_v3.py:1100`)

## Similarity (Returns/Price vectors)
- Build index from local store: `build_similarity_index` (`apps/stock_analysis_app_v3.py:1322`)
- Query: `find_similar` (`apps/stock_analysis_app_v3.py:1386`)
- Lightweight overlay figure for pairs

## Similarity (Feature‑based)
- Extract features from local JSON store: `_compute_features_from_store`
- Build feature stats: `build_data_similarity_index` (`apps/stock_analysis_app_v3.py:1666`)
- Query by Euclidean distance on standardized vectors: `find_similar_by_data` (`apps/stock_analysis_app_v3.py:1754`)

## Planned Enhancements (v2 Multiview Clustering)
- New vectorization per view:
  - Daily log‑return z‑scores (60d), Weekly (52w), Monthly (36m) + last‑price position percentile
  - Investor features: 20/60‑day net buys per type; recent foreign net‑buy signal
- Per‑view distances fused with weights and bonus/penalty for foreigner signal agreement
- Build multiview index + distance matrix; cluster via agglomerative (avg linkage) with K selection; fallback DBSCAN
- Persist `@data/similarity/multiview-index-...` and `@data/clusters/charts-<date>.json`; mirror images to `reports/charts_clusters/<date>/<cluster_id>/`
- Streamlit batch UI: parameters (windows, weights, market filter, rank slice), progress, and links to outputs

### Cluster Visualization (New Panel)
- Add a “Cluster Viewer” inside the Multiview section:
  - Select manifest/date and cluster id
  - Adjust: sort by code or distance‑to‑medoid, filter top‑K, grid columns (2–5), image size
  - Render combined chart images from `reports/charts/` or clustered folder
  - Optional regenerate‑missing toggle using `generate_and_save_combined_chart_headless`

## UI Flow (Main)
1) Initialize handlers and state (Gemini/Kiwoom)
2) Sidebar: select stock (from `stock_list.csv`), date range, toggles
3) Fetch/persist data; render charts
4) Optional AI analysis (Gemini), parse sections into:
   - Main business summary, Investment summary
   - Master analysis Q&A list
5) Export combined chart (button and/or auto when all three figures exist)
6) Similarity search panels
7) Batch tools: metadata scrape and chart generation by market caps/ranges

## Session State (selected)
- `gemini_analysis`, `main_business`, `investment_summary`
- `kiwoom_data`, `df_forecast`, `master_analysis_results`
- `full_gemini_prompt`, `gemini_api_calls`, `analyst_model`
- Chart caches: `_last_fig_daily`, `_last_fig_weekly`, `_last_fig_monthly`
