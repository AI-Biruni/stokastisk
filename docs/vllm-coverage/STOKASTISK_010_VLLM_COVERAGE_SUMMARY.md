# Stokastisk 0.1.0 — vLLM 0.25.1 coverage summary (2026-11-12)

**This is not complete vLLM coverage.** 1967 vLLM modules were inventoried and classified by **static analysis**; only a small path has Kaggle execution evidence. The goal was the maximum *honestly documented* T4 subset, so unexercised functionality is labelled, not asserted.

## Identity
* Archive: `stage.zip` SHA256 `c1eb62cc622a4b97645467b6b65a3c6e1be56bb30ffaae2d84b792e27f221800` (31,538,686 bytes, integrity OK, identical to the earlier extraction). Audit extraction: `work/stage-audit-20261112/stage/` (2560 files). It is a raw vLLM package tree, **not** the T4-stripped build; the executed vLLM wheel (`cc3bcafb…`) is.
* Scripts/modules inspected: **1967** Python modules (1951 in the wheel + 16 removed from it) and 6 native `.so`; classified: **1967** (100 % of Python modules). 567 JSON tuned-MoE configs and 8 templates were not classified individually.

## Counts (module level)
| Class | Count |
|---|---|
| SUPPORTED | 19 |
| SUPPORTED_WITH_CONDITIONS | 537 |
| NEEDS_ADAPTATION | 10 |
| NOT_SUPPORTED_SM75 | 183 |
| EXCLUDED_FLASH_DEPENDENCY | 35 |
| NOT_IMPLEMENTED | 0 (module level); 1 SDK feature: L4/SM89 |
| UNKNOWN_REQUIRES_EVIDENCE | 1183 |
Caveats: SUPPORTED requires direct Kaggle evidence (kernel cubins / executed path). SUPPORTED_WITH_CONDITIONS means "statically reachable from the executed entry points", not "executed". NOT_SUPPORTED_SM75 partly rests on file-name heuristics (stated per row). Counts will move if the inventory rules are refined.

## Kaggle evidence reused (verified against the actual outputs in `notebooks-executed-with-output/`)
* Notebook 13: 7/7 stages PASS; TP=1/TP=2/PP=2 each 12/12 checks; peak GPU memory TP=2 13751/13751 MiB, PP=2 12895/13353 MiB, TP=1 13249/3 MiB; deterministic; Triton cache has `kernel_unified_attention`; no Flash library mapped.
* Notebook 14: 50/50 checks; both replicas received work for batches 2–64 × 3 rounds.
* Decode tok/s medians (Qwen2.5-1.5B, batch 8, 64 out): TP=1 241.3, TP=2 132.8, PP=2 145.7, replicas 225.8; replicas/TP=1 = 0.91–0.96 up to batch 64. Matches the previous report; nothing extrapolated.
* 127 (notebook, kernel) cubin records across the 14 acceptance ZIPs contain 24 distinct named Triton kernels (plus 3 torch-inductor generated `triton_poi_*` kernels); the 24 named kernels were mapped by `def <name>` to 14 vLLM source files (sampling, logprobs, penalties, block tables, input batch, grammar bitmask, unified attention, KV-cache write).

## Verification status of what exists
| Group | Status |
|---|---|
| Offline generation, batches, TP=2, PP=2 (experimental), replicas (SDK), Triton attention, telemetry, topology, NCCL | **previously Kaggle verified** |
| Worker-failure recovery | previously CPU tested |
| Sampling/logprob/penalty/grammar kernels | Triton kernels compiled and launched on Kaggle; output not asserted |
| `stokastisk.compat`, `stokastisk.serving`, `LLM.chat`, dtype pre-check | **implemented, not executed**; `check_model` was applied once statically to the real wheel tree (346 pass / 10 blocked of 356) |
| OpenAI server, streaming, metrics, structured outputs, prefix caching, chunked prefill, CUDA graphs, LoRA, multimodal, pooling, speculative decoding, MLA, quantization kernels, other model families | **require future Kaggle validation** |

## SDK changes
Modified: `sdk/stokastisk/__init__.py`, `engine.py`, `errors.py`. New: `compat.py`, `serving.py`. Preserved: every existing public API. New public API: `check_model`, `check_feature`, `check_dtype`, `check_quantization`, `feature_matrix`, `StokastiskCompatibilityError`, `ServerProcess`, `build_serve_command`, `LLM.chat`.
Other files: `tests/test_compat_serving.py` (written, not run), `scripts/audit/vllm_inventory.py`, `docs/VLLM_T4_COVERAGE.md`, `README.md`, notebooks removed (14) with README/MIGRATION-MANIFEST updated.

## Unsupported / conditional
Unsupported: bfloat16, FlashAttention, FlashInfer, FlashMLA, FlashInfer all-reduce fusion, native FP8/FP4/MXFP4, DeepEP/NIXL/FlashInfer all2all, Qwen Omni/ASR (module removed; needs a new vLLM wheel), OpenPangu (imports a removed backend), L4/SM89. Conditional: vision-encoder attention uses PyTorch SDPA on SM75; fp8 KV cache not expected on T4; server needs third-party packages that Stokastisk does not install.

## Release artifacts
**No change.** No SDK wheel was built; the published SDK wheel (`29edaa55…`), the vLLM wheel (`cc3bcafb…`), the release ZIP (`46439612…`), the manifests and the executed bundle (`dist-rc/_kaggle-executed-efe492a6/`, `…a6b8596b/`) are untouched. The new code is **not** in any published wheel and has not been Kaggle tested.

## Git and remotes (read-only check at the end of this work)
* As of that check, GitHub `main`: `af81fc11eb89f7197df77d7c97c7c7f092c7145d` (unchanged then). Tag `v0.1.0` peels to `1acc49ccf60f5051eb5be16165c3ed4d2b67076a` (unchanged, not moved). Work was later published (see Resolution).
* Hugging Face model repo head: `1ae37fafc20e48973b5c63938d2fb3858c5e59d5` (unchanged by this work).
* Before any push: the changed-file list is in this report and `git diff main --stat`.

## Resolution (2026-11-12, later)
The 14 notebooks were restored byte-identical from `af81fc1` because CI, the validator, the tests and the assembler require them. The branch was then published to `main` without moving `v0.1.0`. The published SDK wheel, vLLM wheel, ZIP and manifests were **not** changed; the new SDK code is repository source only, implemented but not executed.
