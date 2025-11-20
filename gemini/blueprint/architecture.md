# Architecture

## Context (C4 L1)
- User interacts with a Streamlit UI.
- App integrates with:
  - Kiwoom REST API (market data, investor flows, listings)
  - Gemini API (analyst analysis, text generation)
  - Local JSON store (`@data`) for time‑series and indices
  - Local cache (`cache/`) and report exports (`reports/charts/`)

## Containers (C4 L2)
- Streamlit App (`main_app.py`) — hosts navigation and orchestrates sub‑apps
- Sub‑app: Stock Analysis v3 (`apps/stock_analysis_app_v3.py:1775`) — end‑to‑end stock workflow
- Data Store — JSON files under `@data/` (candles, investor flows, manifests, instruments, similarity)
- External Services — Kiwoom API, Gemini API

## Component Overview
- Main App
  - Sidebar app switch; calls `stock_analysis_app_v3.main()` from `main_app.py:28`
- Stock Analysis v3
  - Config & session initialization (Gemini/Kiwoom handlers)
  - Data acquisition from Kiwoom (with local store reuse)
  - Charting (Plotly) and combined export (JPG)
  - Gemini analyst analysis parsing
  - Similarity search (returns/price vectors, and feature‑based)
  - v2 clustering (proposed): multiview similarity and chart image clustering
  - Cluster visualization (proposed): side‑by‑side image preview per cluster
  - Batch runners (charts and indices)

## Data Flow (Happy Path)
1) User selects stock → sub‑app loads listing (`stock_list.csv`)
2) Try local store for last 3M daily → else fetch from Kiwoom → persist
3) Derive weekly/monthly from wider daily window (or fetch) → ensure monthly cadence
4) Fetch/persist investor flows → render daily panel with cumulative investor lines
5) Optional: call Gemini with prompt; parse sections → show analysis and master Q&A
6) Export combined chart (square) to `reports/charts/`
7) Build/search similarity indices from local store

## Outputs
- Reports: `reports/charts/*.jpg`
- Clustered reports (proposed): `reports/charts_clusters/<date>/<cluster_id>/`
- Data store: `@data/timeseries/<code>/candles_daily-YYYY.json`, `@data/timeseries/<code>/investor_flows-YYYY.json`
- Manifests: `@data/manifests/<code>.json`
- Instruments: `@data/instruments/<code>.json`
- Similarity: `@data/similarity/index-<window>-<feature>.json`
- Multiview similarity: `@data/similarity/multiview-index-d<ND>-w<NW>-m<NM>.json`
- Clusters manifest: `@data/clusters/charts-<date>.json`
- Data similarity index: feature stats map

## Risks & Mitigations
- API quotas/latency → local store reuse; key rotation for Gemini
- Sparse monthly series → derive from daily and validate cadence
- Encoding issues in UI text → English fallbacks in log/export paths
