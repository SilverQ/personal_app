# Data Schemas (Local JSON Store)

Root: `@data/` (under repo root)

## Timeseries: Daily Candles
- Path: `@data/timeseries/<code>/candles_daily-YYYY.json`
- Header fields:
  - `schema_version: string`
  - `resource_type: 'timeseries/candles'`
  - `instrument_id: 'KRX:<code>'`
  - `interval: 'daily'`
  - `year: number`
  - `meta: { currency, timezone, units.volume }`
  - `source: { name, endpoint, api_id }`
  - `retrieved_at: ISO8601`
  - `data: { 'YYYY-MM-DD': { o,h,l,c,v, as_of } }`
- Writer: `upsert_daily_candles_store` (`apps/stock_analysis_app_v3.py:159`)
- Reader: `load_daily_candles_from_store` (`apps/stock_analysis_app_v3.py:236`)

## Timeseries: Investor Flows (KRW)
- Path: `@data/timeseries/<code>/investor_flows-YYYY.json`
- Header fields:
  - `schema_version: string`
  - `resource_type: 'timeseries/investor_flows'`
  - `instrument_id: string`
  - `year: number`
  - `meta: { currency: 'KRW', unit: 'KRW', scale: number }`
  - `source: { name, endpoint, api_id }`
  - `retrieved_at: ISO8601`
  - `data: { 'YYYY-MM-DD': { individual, foreign, institution, as_of } }`
- Writer: `upsert_investor_flows_store` (`apps/stock_analysis_app_v3.py:199`)
- Reader: `load_investor_flows_from_store` (`apps/stock_analysis_app_v3.py:262`)

## Manifest per Instrument
- Path: `@data/manifests/<code>.json`
- Fields:
  - `schema_version`
  - `instrument_id`
  - `datasets: { <dataset_name>: [ { year, start, end, last_updated } ] }`
- Maintainer: `_upsert_manifest` (called from upserts)

## Instrument Metadata
- Path: `@data/instruments/<code>.json`
- Fields:
  - `schema_version`, `resource_type: 'instrument'`
  - `instrument_id`, `code`, optional `name`, `market`, `currency`
  - `retrieved_at`
- Maintainer: `upsert_instrument_metadata`

## Similarity Index (Vector‑based)
- Path: `@data/similarity/index-<window>-<feature>.json`
- Fields:
  - `schema_version`, `window`, `feature`
  - `vectors: { '<code>': { dates: [..], vec: [..] } }`
  - `updated_at`
- Writers/Readers: `build_similarity_index`, `_load_similarity_index`

## Multiview Similarity Index (Proposed)
- Path: `@data/similarity/multiview-index-d<ND>-w<NW>-m<NM>.json`
- Fields:
  - `schema_version: '2.0.0'`
  - `windows: { daily: ND, weekly: NW, monthly: NM, investor: [20,60] }`
  - `vectors:
      { '<code>':
          { daily: { dates:[..], zret:[..] },
            weekly: { dates:[..], zret:[..] },
            monthly: { dates:[..], zret:[..], pos: number },
            investor: { ind20, ind60, for20, for60, org60, for_recent_buy: bool } } }`
  - `stats: { investor: { mean/std per feature }, optional cross‑sectional refs }`
  - `updated_at`

## Clusters Manifest (Proposed)
- Path: `@data/clusters/charts-<YYYYMMDD>.json`
- Fields:
  - `schema_version: '1.0.0'`
  - `params: { windows, weights, method, universe_filter }`
  - `clusters: { '<cluster_id>': [ '<code>', ... ] }`
  - `labels: { '<code>': '<cluster_id>' }`
  - `medoids: { '<cluster_id>': '<code>' }`
  - `reports_root: 'reports/charts_clusters/<date>/'`
  - `images: { '<code>': 'reports/charts/<filename>.jpg' }` (optional convenience map used by the viewer)

## Data Similarity (Feature Map)
- In‑memory result persisted ad‑hoc when needed
- Fields:
  - `schema_version`, `window_days`
  - `features: { '<code>': { k:v, ... } }`
  - `stats: { '<feature>': { mean, std } }`
  - `updated_at`
- Builders: `_compute_features_from_store`, `build_data_similarity_index`

## Reports
- Combined chart exports: `reports/charts/<company>_<timestamp>.jpg`
- Export helpers: `save_plotly_fig_as_jpg`, `save_combined_charts_as_jpg`
