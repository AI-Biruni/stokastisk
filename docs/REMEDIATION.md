# Stokastisk 0.1.0 — build 3 remediation report

Date: 2026-11-13 · Project version: **0.1.0** (unchanged) · Build tag: **3** · Status: **built, independently verified, CPU-tested — Kaggle verification pending**.

## 1. Evidence reproduced (nothing assumed)
* The uploaded build-1 wheel is `stokastisk-0.1.0-1-…` — 38,148,975 bytes, SHA256 `815f9783…cf6b` — verified against the local file.
* The executed notebook 15 **did not test it**: its setup cell found no directory with `manifests/SHA256SUMS` and printed `not attached (attached: []); downloading from Hugging Face`; it then installed **build 2** (`7e4b7009…4369`, 38,151,399 bytes). All Notebook-15 evidence is therefore build-2 evidence. The original notebook and acceptance ZIP are preserved byte-for-byte in `(removed from the repository)` (plus JSON reports, logs, inventory, environment versions, `SHA256SUMS`).
* Installer stage: 30 checks, PASS with one WARNING. `static`, `engine`, `server`, `summary`: FAIL.
* What **passed** on build 2 and is therefore not in doubt: Python/torch/Triton/CUDA/T4 checks, FP16 engine start, `TRITON_ATTN` in the live config, `generate`, `chat`, `complete`, determinism, `generate([])`, invalid-input rejection, one-GPU residency, compiled `kernel_unified_attention`, no Flash library mapped, `close()` semantics, no live process and GPU memory released.

## 2. Problems, root causes, fixes
| # | Problem | Root cause | Fix (files) | Evidence |
|---|---|---|---|---|
| 1 | `engine`: structured output → `No module named 'lmformatenforcer'` | vLLM's `backend_lm_format_enforcer.py` evaluates `lmformatenforcer.TokenEnforcer` at import time (no postponed annotations) and `SamplingParams` validation imports it unconditionally → **every** structured request needs it, regardless of backend. Build 2 demoted it to an extra and never installed it. | `requirements/structured-output.lock`; hard `Requires-Dist`; backend pinned to xgrammar (`config.py`, `engine.py`, `serving.py`); `LLM.generate_structured` + `structured.py` | reproduced locally (import fails without the package, succeeds with it); `tests/test_dependency_locks.py::StructuredOutputRootCause` |
| 2 | `server`: `No module named 'model_hosting_container_standards'` | `build_app` bootstraps SageMaker routes unconditionally; `SERVER_REQUIREMENTS` was a hand-written list that omitted the `server` extra. | `requirements/serving.lock` (7 packages); `dependencies.py` registry shared by SDK/installer/tests; static import-closure test | `tests/test_dependency_locks.py::ImportClosure`; CPU: `build_app` builds under Kaggle's fastapi 0.141.1 |
| 3 | Server failure discarded the child's error | `wait_ready` raised a generic message; the log path was optional | `serving.py`: persistent log always, `ServerStartupError(exit_code, command, log_path, log_excerpt, timeout)` | `tests/test_serving_runtime.py` |
| 4 | "8 packages missing" | name-normalisation mismatch (see `docs/DEPENDENCY_COMPATIBILITY.md` §1) — **no package was removed** | `scripts/kaggle/_inventory.py` | `tests/test_inventory.py` reproduces the old bug and the fix |
| 5 | Flash `.pyc` in the installed tree | 28 Flash-family modules shipped in the vLLM tree | `scripts/release/t4_flash_free.py`, `sdk/stokastisk/prohibited.py` | §3 below |
| 6 | Wrong wheel under test | input discovery ignored the attached ZIP and fell back to the network | `scripts/kaggle/bundle.py` embedded in cell 1; opt-in-only fallback | `tests/test_bundle_discovery.py` |
| 7 | `pip check` unclean | `xgrammar 0.2.5.post1` → `transformers<5` | xgrammar **0.2.3** (no cap, tvm-ffi `>=0.1.9`, inside vLLM's range) — a version choice, not a patch | closure JSON, CPU structured-output check |
| 8 | A warning could count as a pass | `finish()` treated WARNING/SKIPPED as OK when ≥1 PASS | `stage_status()`: ERROR > FAIL > NOT_RUN > WARNING > SKIPPED > PASS | `tests/test_stage_status.py` |
| 9 | Structured test could not prove the backend | no backend assertion; `auto` falls back silently | pinned `xgrammar`; test of an xgrammar-unsupported schema that must be **rejected**; native-library evidence | `candidate_15_engine.py` |
| 10 | SDK lifecycle gaps | `close()` logged and swallowed engine shutdown errors; no timeout; empty prompt accepted; `bool` accepted as int in `replica_devices`; stale-result id reuse in `ReplicaPool`; worker close problems only logged | `engine.py`, `config.py`, `parallel/replica.py`; 14 best-effort `except: pass` → logged | `tests/test_engine_lifecycle.py`, `test_config_strict.py`, `test_replica_regressions.py` |
| 11 | Prior analysis error | the build-2 dependency page claimed no xgrammar release satisfied both constraints | corrected in `docs/DEPENDENCY_COMPATIBILITY.md` | PyPI metadata table |

## 3. FlashAttention / FlashInfer / FlashMLA
Audit of the byte-pinned vLLM wheel (`cc3bcafb…`): **28** Flash-family modules — your 25 name-pattern files plus `fa_utils.py` (FlashAttention version selection / `flash_attn_varlen_func` re-export), `rocm_aiter_fa.py` and `warmup/fa4_cutedsl_config.py`. No Flash native object (6 `.so`, no cubin/ptx), no kernel binding. 15 `.py` files hold function-level `import flashinfer`-style *references* (reported, not implementations). Six "flash"-named files are not FlashAttention and are kept deliberately (`triton_reshape_and_cache_flash` is the Triton KV-cache write kernel `TRITON_ATTN` uses; `longcat_flash*`; `*dflash*`).
Removal was done by a deterministic transform, not by hand: 54 import sites in 28 files redirected to one generated placeholder, 5 files patched (registries, `AttentionConfig`, CUDA selector message, `vllm/__init__`), 2 files added (`t4_backend_policy.py`, `t4_prohibited_stubs.py`). Verified: independent byte-equality re-derivation (`verify_build3_wheel.py`), 1,783-module import sweep — **0 regressions**, auto-selection on SM7.5 → `[TRITON_ATTN, FLEX_ATTENTION]` with `FLASH_ATTN`/`FLASHINFER` refused by name. **Not** verified until Kaggle: the same on the GPU, and any model family whose module imported a removed backend (they now fail with an explicit `ProhibitedBackendError`; `check_model` reports them).

## 4. Tests (exact)
`CUDA_VISIBLE_DEVICES= python3.13 -m pytest tests -q -rs` — see `RELEASE.md` for the final counts. CPU only; the local GeForce 940M was hidden and never used. Mocks do not establish GPU behaviour: ReplicaPool failure recovery, structured generation, serving and Flash-free runtime behaviour are **Verification pending**.

## 5. What was deliberately not done
* No metadata patching; no downgrade of Transformers; no change to torch/Triton/CUDA/NCCL; no `--no-deps` removal; no weakening of any check; the version stays 0.1.0; builds 1/2 and the `v0.1.0` tag are untouched.
* Observability (OpenTelemetry exporters) is **Unsupported** on this image (protobuf 7.x vs `protobuf<7`) rather than forced.

## 6. Findings from the first Kaggle runs of build 3 (2026-11-13, 2 × T4)
| Run | Result | Cause | Fix |
|---|---|---|---|
| Owner's run | installer 31/32 PASS, rest NOT_RUN | the installer imported `lmformatenforcer.integrations.vllm`; lm-format-enforcer 0.11.3's adapter needs `vllm.transformers_utils.tokenizer.MistralTokenizer`, absent in vLLM 0.25.1. Only the unused lm-format-enforcer *backend* needs it (xgrammar is pinned) | import removed from the installer list; documented |
| Maintainer run 1 | input not found | `find_candidates` stopped at depth 4; Kaggle mounts at `/kaggle/input/datasets/<user>/<slug>/<release>/manifests/…` (depth 5) | depth 8 + regression test with the real layout; the error now lists what is mounted |
| Maintainer run 2 | installer 32/32, **static 79/79**, engine and server ERROR | `kernel_warmup()` calls `flashinfer_sparse_mla_decode_autotune_warmup()` and `deepseek_v4_sparse_mla_attention_warmup()` for every model; my placeholders raised instead of doing nothing | `*_warmup` placeholders are no‑ops; regression test pins the unconditional calls in `kernel_warmup()` |
| (same run) | stale scripts mounted | Kaggle served the previous dataset version though `datasets status` said ready | standalone notebook pins the bundle's `SHA256SUMS` hash; the runner waits for the new files |
Mistral: `mistral_common` and `ijson` removed from every profile; `check_model` refuses Mistral‑family architectures.
Lesson recorded: a CPU import sweep cannot prove call‑time behaviour of placeholders — only the GPU run can; the GPU stages for the final wheel remain **Verification pending** (owner to run).
