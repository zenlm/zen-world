# Commercial Rebase — Zen World

## Status: MODEL CLEAN, repo code migration pending
The shipped weights (HF `zenlm/zen-world`) are based on **Wan2.1-T2V-14B**
(Apache-2.0, unconditionally commercial). The repository inference code under
`hy3dworld/` originated from Tencent HunyuanWorld; it is being migrated to a
Wan2.1/2.2-based world-generation pipeline so code and weights share one
Apache-2.0 lineage.

## Steps
1. Replace `hy3dworld/` scene/pano generation with a Wan2.1/2.2 pipeline.
2. Verify outputs against current model card.
3. Remove residual Tencent-derived code paths.
