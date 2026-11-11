# Stokastisk 0.1.0 — vLLM SDK integration report (2026-11-12)

Branch `feature/vllm-t4-coverage` (local only; not pushed; `main` is unchanged at `af81fc1`). Project version 0.1.0, vLLM 0.25.1, no dependency pin changed.

## Design decision
The audit shows that most of vLLM's surface is either already reachable through `vllm.LLM` (which Stokastisk wraps) or untestable without GPU runs that are not allowed now. Re-wrapping 1967 modules would create unstable duplicates, so the SDK adds only what was missing and is checkable from source:
1. a **static compatibility layer** so unsupported/unknown things fail early with a reason;
2. an **OpenAI-compatible server launcher** (the one major vLLM entry point with no SDK path) with streaming and metrics clients;
3. a `chat()` pass-through on the existing `LLM`.
Standalone upstream commands (`vllm bench`, `run-batch`, `collect-env`) are not wrapped; they remain `vllm …` commands.

## New and changed SDK files
| File | Change | Status tag |
|---|---|---|
| `sdk/stokastisk/compat.py` (new) | `FEATURES` matrix, `feature_matrix`, `get_feature`, `check_feature`, `check_dtype`, `check_quantization` (minimums read from the shipped source), `read_registry`, `find_missing_imports`, `check_model` (parses the installed vllm registry and import chain with `ast`; imports nothing) | implemented; **not executed**; logic exercised once, statically, against the real wheel tree: Qwen2 passes, 10 of 356 architectures blocked (Qwen Omni/ASR, OpenPangu) |
| `sdk/stokastisk/serving.py` (new) | `build_serve_command`, `ServerProcess` (start/wait_ready/stop, context manager), `check_server_requirements`, `chat`, `stream_chat` (SSE), `list_models`, `fetch_metrics` | implemented; **not executed**; requires future Kaggle validation |
| `sdk/stokastisk/errors.py` | + `StokastiskCompatibilityError(ConfigurationError)` | implemented |
| `sdk/stokastisk/engine.py` | import `check_dtype`; explicit-bfloat16 pre-check before engine start; `chat()`; PP warning text updated (it still said "not Kaggle-verified") | implemented; **not executed**; no change to the existing engine call |
| `sdk/stokastisk/__init__.py` | exports `StokastiskCompatibilityError`, `check_model/feature/dtype/quantization`, `feature_matrix`; lazy `ServerProcess`, `build_serve_command`; every previous export kept | implemented |
| `tests/test_compat_serving.py` (new) | 31 CPU tests with a synthetic vllm tree and mocks | **written, not executed** (no test runs were permitted) |
| `scripts/audit/vllm_inventory.py` (new) | static inventory/classification of the shipped wheel | run once (static analysis; produced the matrix) |
| `docs/VLLM_T4_COVERAGE.md`, `README.md` | coverage and evidence-tier documentation; no universal-support claim | docs |

## Preserved
All previous public names (`LLM`, `StokastiskConfig`, `ReplicaPool`, `recommend_parallelism`, `discover_topology`, `validate_environment`, `validate_kaggle_t4`, `SamplingParams`, error classes, `LIMITS`, …) remain in `__all__`; no signature changed; the engine kwargs, NCCL defaults, `TRITON_ATTN` enforcement, TP/PP/replica logic, native binaries and vLLM wheel are untouched.

## Behaviour changes to be aware of
* `LLM(...)` now raises `StokastiskCompatibilityError` (a `ConfigurationError`/`ValueError`) for an explicit `dtype="bfloat16"` before starting vLLM; previously vLLM raised its own `ValueError` later. `dtype="auto"` and `"half"` are unchanged.
* `import stokastisk` now also imports `stokastisk.compat` (standard library only).

## Requested capabilities versus outcome
| Requested | Outcome |
|---|---|
| Offline inference, batch, sampling, TP/PP, replicas, telemetry, topology | already present; Kaggle-verified earlier (notebooks 05–14) |
| Streaming, OpenAI server, metrics | `stokastisk.serving` (implemented, not executed) |
| Chat | `LLM.chat` (implemented, not executed) |
| Model/GPU compatibility inspection | `stokastisk.compat` |
| Async engine, scheduler controls, KV/block internals, preemption | upstream internals; reachable via `LLM.engine`; deliberately **not** promoted to public API |
| Quantization, LoRA, multimodal, pooling, speculative decoding, prefix caching, chunked prefill, structured outputs, CUDA graphs | present in the wheel, **not exposed by new SDK APIs** and not run; classified in the matrix |
| L4/SM89 | not implemented, not claimed |
