# Similarity & Clustering v2 (Reports/Charts)

## Goals
- Improve image grouping under `reports/charts/` by using a richer, multi‑view similarity:
  1) Capture whether foreigners are recently net‑buying and factor it in.
  2) Consider monthly chart relative price position (within a multi‑year window).
  3) Vectorize Daily/Weekly/Monthly/Investor views separately; compute per‑view similarities and integrate.

## Universe & Windows
- Universe: from `stock_list.csv`, optionally filtered by market and market‑cap rank.
- Suggested windows (configurable):
  - Daily: 60 trading days (returns)
  - Weekly: 52 weeks (returns)
  - Monthly: last 36 months (closes; plus position)
  - Investor: last 20 and 60 trading days (KRW net buys by type)

## Feature Representation
- Daily view (shape)
  - Input: daily close series; compute log returns; z‑score; tail N=60
  - Vector: `zret_daily[60]`
  - Distance: correlation distance `1 - corr(x,y)` with overlap ≥ 0.8*N

- Weekly view (shape)
  - Input: weekly resampled OHLC closes; log returns; z‑score; tail N=52
  - Vector: `zret_weekly[52]`
  - Distance: correlation distance with overlap ≥ 0.8*N

- Monthly view (shape + position)
  - Shape vector: monthly close returns, z‑score, tail N=36 → `zret_monthly[36]`
  - Position scalar: last close percentile within rolling 3Y min/max, `pos ∈ [0,1]`
  - Distance: `d_monthly = α*(1-corr) + (1-α)*|posA - posB|`, default α=0.7

- Investor view (flows and trend)
  - Features (KRW): sum net buys over 20/60 days per type; optional EWMA(10) trend
    - `[ind20, ind60, for20, for60, org60]` (scale in millions)
  - Standardize by cross‑sectional mean/std
  - Distance: cosine (or Euclidean) on standardized vector
  - Recent foreign net‑buy signal: `for20 > 0` → boolean. Agreement adds a small bonus by reducing the integrated distance (see fusion).

## Distance Fusion (Integrated Similarity)
- Compute per‑view distances: `dD, dW, dM, dI ∈ [0,2]` (bounded by correlation distance and normalized position/standardization).
- Combine with weights: `D_total = wD*dD + wW*dW + wM*dM + wI*dI`
  - Defaults: `wD=0.35, wW=0.25, wM=0.25, wI=0.15`
- Foreigners recent net‑buy bonus: if both have `for20 > 0`, apply `D_total = max(0, D_total - β)` with β=0.05; if signs disagree, optionally add +0.03 (tunable).
- Missing view handling: re‑normalize weights over available views; enforce minimum overlap checks per view.

## Clustering
- Build pairwise distance matrix `D` over the universe.
- Algorithm options:
  - Agglomerative (average linkage) with precomputed distance; choose K via silhouette over grid K∈{6,8,10,12,14}.
  - Fallback: DBSCAN on normalized distances for noise isolation (eps/grid search with small validation sample).
- Output
  - `clusters: { cluster_id: [codes...] }`
  - `labels: { code: cluster_id }`
  - `centroids`: optional medoid per cluster (lowest average distance to others)

## Batch Pipeline
1) Build per‑view vectors using local store; fetch from Kiwoom only when coverage is insufficient.
2) Persist multiview index with stats to `@data/similarity/multiview-index-<spec>.json`.
3) Compute distance matrix and clustering result; save to `@data/clusters/charts-<date>.json`.
4) Organize report images by cluster under `reports/charts_clusters/<date>/<cluster_id>/` (copy or symlink) and emit an index manifest linking image paths.

## Cluster Visualization UI (Streamlit)
- Panel: “유사도 기반 차트 군집 생성 (Multiview)”
  - Tabs: [Parameters], [Build], [Clusters], [Preview]
  - Parameters
    - Windows: daily(60), weekly(52), monthly(36), investor(20/60)
    - Weights: daily/weekly/monthly/investor sliders; auto re-normalize
    - Universe filter: market, rank range, max universe size (e.g., 200)
  - Build
    - Buttons: [Build Index], [Build Clusters], [Materialize Folders]
    - Show progress, silhouette score, K chosen, and cluster size summary
  - Clusters
    - Selector for manifest file (default latest date)
    - Display table: cluster_id, size, medoid code/name, sample images
  - Preview (Group Viewer)
    - Controls: cluster id select, sort by [code, distance-to-medoid], filter top-K by distance, columns per row (2–5), image size
    - Render: side-by-side images from `reports/charts/` or clustered folder; fallback to regenerate missing via `generate_and_save_combined_chart_headless`
    - Option: open image path, copy to clipboard (path), or download

### Rendering Details
- Use `st.columns` to layout a responsive grid; batch `st.image` calls for performance.
- Prefer prebuilt combined JPGs; if none, either skip or regenerate on demand (guarded by a toggle).
- Provide medoid and nearest neighbors preview per cluster for quick sanity checks.

## API Design (to implement)
- Vectorization
  - `vectorize_daily_returns(code, ndays=60) -> dates, vec`
  - `vectorize_weekly_returns(code, nweeks=52) -> dates, vec`
  - `vectorize_monthly_shape_position(code, nmonths=36) -> dates, vec_shape, pos_scalar`
  - `vectorize_investor_features(code, windows=(20,60)) -> feature_dict`

- Indices & Stats
  - `build_multiview_index(universe, params) -> {vectors, stats, meta}`
  - `compute_distance(a, b, index, weights, params) -> {total, components}`
  - `build_distance_matrix(index, weights, params) -> ndarray`

- Clustering
  - `cluster_from_distance_matrix(D, method='agglomerative', params) -> {labels, clusters, medoids}`
  - `save_clusters_manifest(path, clusters, labels, meta)`

- UI/Batch
  - Streamlit batch panel: “유사도 기반 차트 군집 생성” with controls for windows, weights, market filter, top‑N by market‑cap.
  - Progress and logs; outputs link to clustered folders and JSON manifest.
  - Cluster Viewer: interactive grid for a selected cluster with sorting, filtering and image sizing.

## Persistence & Schemas
- See `data-schemas.md` additions for `multiview-index` and `clusters` manifests.
  - For visualization, the clusters manifest may include an optional `images` map `{ code: reports/charts/<file>.jpg }`.

## Risks & Considerations
- Quality depends on store coverage; ensure prechecks and degrade gracefully.
- Distance fusion weights need tuning; expose via UI and persist last used.
- Keep computation time manageable: restrict universe by filters and ranks; cache per‑view vectors.
