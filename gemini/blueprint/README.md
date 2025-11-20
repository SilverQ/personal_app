# SilverQ Blueprint

- Purpose: Document current plan, build, and function implementations for the existing app stack.
- Scope: `main_app.py` entry and `apps/stock_analysis_app_v3.py` sub‑app, shared data store and integrations.

## Contents
- architecture.md — C4 context/container and data flows
- modules/stock_analysis_app_v3.md — responsibilities, key classes/functions, UI flow
- data-schemas.md — local JSON store and index schemas
- configuration.md — required config, env, dependencies
- adr/ADR-0001-data-store-json.md — local JSON data store decision
- adr/ADR-0002-subapp-contract.md — sub‑app entrypoint and boundaries

## Current Entry Points
- main app: `main_app.py:1`
- stock analysis v3: `apps/stock_analysis_app_v3.py:1775` (`def main()`)

