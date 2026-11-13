# Kaggle verification — build 3 (manual guide)

**Status: Verification pending.** Build 3 is built, independently verified and CPU-tested. It has not run on Kaggle. Gate E passes only when the owner provides the executed notebook and acceptance ZIP.

## Setup (Kaggle)
1. New notebook → *Accelerator: GPU T4 × 2*, *Internet: ON*, *Language: Python*, Python 3.13 image.
2. Upload `stokastisk-0.1.0-build3-kaggle-t4-sm75-kaggle-input.zip` as a Kaggle **Dataset** and attach it to the notebook (Add input). A ZIP holding only a wheel is rejected on purpose.
3. Import `15_stokastisk_candidate_sdk_integration.ipynb` — use the **standalone** copy shipped next to the ZIP: it additionally pins the ZIP's own SHA256 (a ZIP cannot contain its own hash). The copy inside the ZIP pins only the wheel's SHA256.
4. *Run all* (or *Save & Run All*). Do not edit cells. The first cell prints the attached bundle, the wheel name/SHA256, the manifest SHA256 and the number of verified files — compare them with the release page.

## What each stage proves
| Stage | Script | Proves (on the real runtime) |
|---|---|---|
| bundle (cell 1) | `bundle.py` | the attached ZIP is a complete release, archive-safe, every file matches `SHA256SUMS`, the wheel equals the manifest and the notebook's pinned hash; **no download** |
| installer | `02_install_integrated.py` | protected baseline (torch/triton/CUDA/NCCL) unchanged; locks hash-verified and installed `--no-deps --no-index`; only planned packages added; **no new `pip check` line**; all feature imports work; installed files identical to the wheel; installed tree has 0 Flash-family files |
| static | `candidate_15_static.py` | inventory identical live vs installer (PEP 503, dotted names present), configuration/compat/serving-command validation, backend registry + SM7.5 auto-selection, Flash guard, dependency-lock completeness, CPU structured-output stack |
| engine | `candidate_15_engine.py` | FP16 `generate/chat/complete`, determinism, invalid-input rejection, **structured output with the pinned xgrammar backend** (schema-valid, invalid detected, unsupported schema *rejected* — no hidden fallback), serialisation + timeout, `TRITON_ATTN` in config **and** Triton cache, no Flash library mapped, close/cleanup, upstream `vllm.LLM` never selects Flash |
| server | `candidate_15_server.py` | `vllm serve` start/ready, `/v1/models`, chat, **streaming**, HTTP errors, structured output over HTTP, metrics, startup-failure report (exit code, command, log, excerpt), port collision, restart, graceful shutdown, no surviving process, GPU memory released |
| parallel | `candidate_15_parallel.py` | TP=2, PP=2 (experimental) and `ReplicaPool` on build 3: real generation, both GPUs engaged, `TRITON_ATTN`, no Flash library, clean shutdown and GPU memory released; topology + recommender last |
| summary | `candidate_15_summary.py` | all stages and the installer describe the **same wheel SHA256** and runtime; a missing report is `NOT_RUN`; only `PASS` passes |

The acceptance archive `/kaggle/working/15-stokastisk-build3-acceptance.zip` is created **even if a stage fails**; the last cell then fails the notebook unless every stage is `PASS`.

## After the run
Send back: the executed notebook (`.ipynb` with outputs) and `15-stokastisk-build3-acceptance.zip`. Expected files inside: `15c_static.json`, `15c_engine.json`, `15c_server.json`, `15c_summary.json`, `provenance.json`, `prereq-install/02_install_integrated.json` (+ before/after inventories, plan, pip lists), `15c_*.log`, `15c_package_inventory.txt`, `15c_structured_cpu.json`, `ACCEPTANCE-SHA256SUMS`.

## Result vocabulary
`PASS`, `FAIL`, `ERROR` (the check itself crashed), `WARNING` (a documented limitation — not a pass), `SKIPPED` (not performed — not a pass), `NOT_RUN` (no evidence — not a pass).

## What this does not cover (remaining limitations)
`ReplicaPool` *failure recovery* is CPU-tested with fake workers only (the `parallel` stage covers the healthy path). Models other than Qwen2.5-0.5B-Instruct, quantisation, LoRA, multimodal, CUDA graphs and speculative decoding are unverified. See `docs/PRODUCTION_READINESS.md`.
