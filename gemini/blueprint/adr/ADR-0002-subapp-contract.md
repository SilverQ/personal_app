# ADR-0002: Sub‑App Contract

Context
- `main_app.py` routes to multiple domain apps; v3 stock analysis is the primary target.

Decision
- Standardize sub‑app entrypoint to `def main()`.
- The main app calls `stock_analysis_app_v3.main()` (`main_app.py:28`).

Status
- Accepted — implemented in v3; other apps may expose `run()` historically.

Consequences
- Pros: Simple integration, predictable navigation coupling
- Cons: Minor refactors for legacy sub‑apps exposing `run()`

