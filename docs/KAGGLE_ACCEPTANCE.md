# Kaggle acceptance (maintainers)

## Verification record (Kaggle 2 x Tesla T4, Python 3.13.16, PyTorch 2.11.0+cu130, Triton 3.6.0) — build 3
| Notebook | Date | What ran | Result |
|---|---|---|---|
| 15 - acceptance (maintainers) | 2026-11-13 | full GPU suite on a locally built bundle | installer 32/32, static 79/79, engine 44/44, parallel 23/23, server 21/21, summary 15/15 |
| 16 - quickstart | 2026-11-13 | `pip install` from the published index + `demo.py` | `TRITON_ATTN`; completion, chat, structured JSON; clean shutdown |
| 18 - Qwen2.5, plain cells | 2026-11-14 | 0.5B-Instruct TP=1; 3B-Instruct TP=2; 3B-Instruct PP=2 | all generated; `TRITON_ATTN`; GPUs released (14.46 of 14.56 GiB free) |
| 19 - Qwen2.5-Coder-7B-Instruct | 2026-11-14 | original FP16 checkpoint, TP=2 then PP=2 | both generated; `TRITON_ATTN`; GPUs released |

Raw artifacts are in `evidence/kaggle/` (executed notebooks with outputs, the notebook-15 reports and logs) and indexed, with sizes, SHA256 values, per-run versions, models, modes, memory readings and the forced-cleanup warning, in `evidence/kaggle/evidence-manifest.json` (rebuild: `python3.13 scripts/audit/build_evidence_manifest.py`). `stokastisk.compat` and the tests are checked against that manifest. Not in the repository: the two original output archives (their hashes are in the manifest).

**Build 4 was not run on Kaggle.** It differs from build 3 only in compatibility reporting; the member-by-member audit proves the vLLM tree, native binaries, dependency pins and executable SDK code unchanged (see [PRODUCTION_READINESS.md](PRODUCTION_READINESS.md)), so this record is inherited for those paths.

Notebooks 18 and 19 (in `notebooks/`, outputs stripped) are the end-user pattern: one `pip install` cell, then ordinary cells. They are not throughput benchmarks; on two PCIe-connected T4s TP=1 is fastest for any model that fits one card.

## What the notebook 15 stages proved (build 3; the harness is no longer part of the repository, its reports are in `evidence/kaggle/`)

| Stage | Proves |
|---|---|
| installer | protected packages untouched; hash-pinned locks installed `--no-deps --no-index`; no new `pip check` line; feature imports; installed tree has 0 Flash files |
| static | one inventory implementation (live vs installer), configuration/compat/serving validation, backend registry on SM7.5, Flash import guard, dependency completeness, CPU structured-output stack |
| engine | FP16 generate/chat/complete, determinism, input validation, pinned-xgrammar structured output (valid, invalid detected, unsupported schema rejected), serialisation + timeout, `TRITON_ATTN` in config and as a compiled kernel, no Flash library, close/cleanup, direct `vllm.LLM` never selects Flash |
| parallel | TP=2, PP=2 (experimental), ReplicaPool: real generation, both GPUs engaged, clean shutdown, memory released; topology + recommender |
| server | `vllm serve` start/ready, models, chat, streaming, HTTP errors, structured output over HTTP, metrics, startup-failure report, port check, restart, graceful shutdown |
| summary | all stages describe the same wheel and runtime; a missing report is `NOT_RUN`; only `PASS` passes |

Result vocabulary: `PASS`, `FAIL`, `ERROR`, `WARNING`, `SKIPPED`, `NOT_RUN` - only `PASS` passes.
