# ADR-0001: Local JSON Data Store

Context
- Market/time‑series data and investor flows are reused frequently across sessions and batch jobs.
- External API quotas/latency require caching/persistence.

Decision
- Persist per‑instrument JSON files under `@data/` for daily candles and investor flows.
- Maintain a per‑instrument manifest for coverage by dataset and year.

Status
- Accepted — implemented in `apps/stock_analysis_app_v3.py:159,199,236,262`.

Consequences
- Pros: Offline reuse, reduced API calls, deterministic batch ops
- Cons: Local disk footprint, eventual consistency vs. live API

