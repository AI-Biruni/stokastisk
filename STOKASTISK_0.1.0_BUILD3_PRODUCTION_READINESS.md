# Stokastisk 0.1.0 build 3 — production readiness

**Verdict: NOT production-verified. Ready for final Kaggle verification (Gate E).** Gates A–D below are met locally; Gate E needs the owner's run of notebook 15.

| Gate | State | Basis |
|---|---|---|
| A — source quality | **Met (CPU)** | full CPU suite green (counts in the final report); lifecycle, config, concurrency, replica regressions; no silent `except: pass` (AST test) |
| B — dependency integrity | **Met (resolution + CPU)** | 4 locks resolved against the real Kaggle inventory, 0 conflicts; every pin hash-locked; static import closure of the LLM and serve paths fully covered; xgrammar conflict resolved by version choice; protected packages never in a lock. *Install behaviour on Kaggle: verification pending* |
| C — package integrity | **Met** | tags cp313/linux_x86_64, version 0.1.0, build 3, one vLLM tree, RECORD valid, 0 Flash-family files/markers/bindings/`__pycache__`, natives byte-identical, byte-equal to pinned wheel + documented transform, reproducible build |
| D — Kaggle input reproducibility | **Met** | complete bundle ZIP, embedded discovery, no silent fallback, manifest + wheel hash enforced, build printed |
| E — actual Kaggle verification | **NOT MET — pending** | requires the executed notebook and acceptance ZIP from the owner |

## Status labels per feature
| Feature | Label |
|---|---|
| FP16 `generate`/`chat`/`complete`, TRITON_ATTN, determinism, cleanup | Kaggle-tested on **build 2** (same engine code path); build 3 re-run pending |
| Structured output (xgrammar 0.2.3, pinned) | Implemented, CPU-tested (vLLM's `XgrammarBackend` on CPU); **Verification pending** (build 2 failed on the missing package) |
| OpenAI server, streaming, metrics, restart, startup-failure report | Implemented, CPU-tested against fake servers; **Verification pending** (build 2 failed on the missing package) |
| Flash-free tree | Implemented, independently verified, CPU import sweep; GPU run **pending** |
| TP=2, PP=2, ReplicaPool | Kaggle-tested on the legacy bundle (notebooks 13, 14); not re-run on build 3; PP/replica **Experimental**; slower than TP=1 for models that fit one T4 |
| ReplicaPool worker-failure recovery | CPU-tested with fake workers only — **not** a GPU claim |
| Observability (OpenTelemetry exporters) | **Unsupported** on this Kaggle image |
| Direct `vllm.LLM` / `vllm serve` | **Unsupported** entry point (cannot select Flash; not verified as supported) |
| `models` profile (quantisation, Mistral tokenizers) | Implemented as an optional lock; not exercised |
| L4/SM89, bf16 | **Unsupported** |

## Remaining limitations / risks
1. **Gate E.** Everything GPU-side for build 3 is unproven until the run.
2. **fastapi 0.141.1 vs upstream's `<0.137` cap** (route-tree change affecting model-hosting-container-standards handler overrides). CPU: the app builds and `/metrics`, `/version` answer. SageMaker `/ping`/`/invocations` are not advertised; not Kaggle-verified.
3. **xgrammar 0.2.3 is older than what vLLM was developed against** (0.2.5.post1). It is inside vLLM's declared range and passes vLLM's own backend on CPU for JSON-schema, regex, EBNF and structural-tag compilation; GPU decoding is pending.
4. **Removed modules.** Code paths that need a removed Flash backend fail with an explicit `ProhibitedBackendError` instead of importing it (DeepSeek V4 sparse MLA is reported by `check_model`). Whisper's causal-attention path, TurboQuant and the ViT FlashAttention wrappers reference helpers that now resolve to placeholders; they were never exercised on a T4 and are not claimed to work.
5. **Offline `LLM` cannot cancel a generation**: `timeout=` raises but the request runs on; later calls wait. Documented, tested.
6. **Orphans after a parent crash**: `ReplicaPool` workers are non-daemon processes; a hard-killed parent leaves them (Kaggle kills the session anyway).
7. **Unresolved upstream facts**: `numpy 1.26.4` on Python 3.13 and Kaggle's 8 pre-existing `pip check` lines are properties of the image; the installer tolerates only lines identical before/after.
8. Models other than Qwen2.5-0.5B-Instruct, LoRA, multimodal, CUDA graphs, speculative decoding, prefix caching claims: unverified.
9. The Hub/CDN download path was not exercised for build 3 (the notebook uses the attached dataset).
