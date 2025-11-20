# ADR-0003: Multiview Similarity and Clustering for Chart Reports

Context
- Existing similarity uses single‑view vectors; chart grouping needs to reflect investor behavior and multi‑horizon price context.

Decision
- Introduce multiview similarity combining Daily/Weekly/Monthly shapes and Investor features, with tunable weights and a foreigner net‑buy agreement bonus.
- Cluster images based on integrated distances and materialize cluster folders for quick review.

Status
- Proposed — documented in `design/similarity-clustering-v2.md` and `data-schemas.md`.

Consequences
- Pros: More meaningful clusters for screening; configurable knobs
- Cons: Extra computation and disk I/O; requires new batch UI and persistence

