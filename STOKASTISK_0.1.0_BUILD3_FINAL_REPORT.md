# Stokastisk 0.1.0 — build 3 final report

**Result: ready for final Kaggle verification (Gate E). NOT verified on Kaggle. Not production-verified.** Project version unchanged: **0.1.0**.

## Artifacts
| | |
|---|---|
| Wheel | `stokastisk-0.1.0-3-cp313-cp313-linux_x86_64.whl` — 38,060,994 bytes — SHA256 `1d43089a85187469913719e52771387087821695efbc7c0132b0732f09877ad3` |
| Kaggle bundle | `stokastisk-0.1.0-build3-kaggle-t4-sm75-kaggle-input.zip` — 36,977,553 bytes — SHA256 `5a223beff26d28095369e17ed078d2ed8413bd7974e6062a840794c67a1e4f3a` |
| Standalone notebook | `15_stokastisk_candidate_sdk_integration.ipynb` (pins both hashes) |
| Source commit of the build | `3a73b36439b714a1440dee3885cc1e5dd3d7dca8` (tracked tree clean) |
| Build 1 / build 2 (preserved, untouched) | `815f978346cfc6d85b7b47f9b96ce6805b5630ea2e0d3c0070c45f538c11cf6b` / `7e4b700929ec7f3489570e93252ef5a49da2d1682c111c5feefb55efead84369` |
| GitHub / Hugging Face commits | recorded in section "Publication" below |

Provenance: the failed Notebook 15 tested **build 2** (downloaded from Hugging Face), never the uploaded build 1. Build-2 evidence is preserved byte-for-byte in `evidence/kaggle/build-2-failed-notebook-15/`; success evidence of that run (FP16 inference, chat, complete, cleanup) belongs to build 2.

## Problems found, root causes, fixes
See `STOKASTISK_0.1.0_BUILD3_REMEDIATION_REPORT.md` §2 (11 items) and `docs/DEPENDENCY_COMPATIBILITY.md`. Headlines: (1) vLLM imports `lmformatenforcer` for every structured request; (2) the server extra was never installed; (3) the "8 missing packages" were a name-normalisation mismatch, nothing was removed; (4) 28 Flash-family modules shipped; (5) the notebook silently downloaded another release; (6) xgrammar 0.2.5.post1's `transformers<5` cap — resolved by xgrammar 0.2.3 (and a correction of the earlier analysis that claimed no such release existed).

## Source files modified / added
SDK: `engine.py`, `serving.py`, `config.py`, `errors.py`, `compat.py`, `diagnostics.py`, `parallel/replica.py`, `__init__.py`, `limits.py`, `telemetry/{nvml,report}.py`, `topology/{discovery,gpu_capabilities,nvml}.py`; new `prohibited.py`, `dependencies.py`, `structured.py`.
Kaggle scripts: `02_install_integrated.py` (rewritten), `_common.py`, new `_inventory.py`, `bundle.py`, `prohibited.py`, `candidate_15_{common,static,engine,server,summary}.py`, dispatcher `candidate_15_integration.py`.
Release tooling: new `scripts/release/{lock_closure,t4_flash_free,build_integrated_wheel,make_notebook15_build3,assemble_build3_release}.py`, `scripts/audit/{import_closure,import_sweep,structured_output_cpu_check,verify_build3_wheel}.py`.
Dependencies: new `requirements/{core,structured-output,serving,models}.{roots,lock}`, `closure/*.closure.json`, `kaggle-baseline-2026-11-13.json`.
Tests: 11 new/rewritten files (below).

## Dependency changes (vs build 2)
Added to the default install: lm-format-enforcer 0.11.3, outlines-core 0.2.14, diskcache 5.6.3, interegular 0.3.3 (structured output); fastapi-cli 0.0.33, model-hosting-container-standards 0.1.16, prometheus-fastapi-instrumentator 8.1.0, pydantic-settings 2.16.2, rich-toolkit 0.20.6, sse-starlette 3.7.0, supervisor 4.3.0 (serving). Changed: xgrammar 0.2.5.post1 → **0.2.3**. Unchanged: apache-tvm-ffi 0.1.9, llguidance 1.7.6, cbor2, msgspec, openai-harmony, partial-json-parser, pybase64. Optional (`--with-models`): compressed-tensors, mistral-common, depyf, ijson, setproctitle (+4 transitive). Unsupported: OpenTelemetry exporters (protobuf 7.x conflict). All hash-pinned; none touches the protected baseline.

## Flash removal and verification
28 modules removed (25 by your patterns + `fa_utils.py`, `rocm_aiter_fa.py`, `fa4_cutedsl_config.py`); 54 import sites redirected to one placeholder; 5 files patched; 2 added. Checked by: strict scan of the wheel (0 path hits across .py/.pyc/.so/.cubin/.ptx, 0 native markers, 0 bindings, 0 `__pycache__`); natives byte-identical to the pinned wheel; independent re-derivation of every `vllm/**` byte; CPU import sweep of 1,783 modules → 0 regressions (24 more import than before); selector on SM7.5 → `[TRITON_ATTN, FLEX_ATTENTION]`; `pip --target` install of the final wheel → 0 prohibited files in the installed tree. Not verified: the GPU run.

## Test commands and exact results
```
CUDA_VISIBLE_DEVICES= python3.13 -m pytest tests -p no:cacheprovider -q -rs
377 passed, 1 warning, 158 subtests passed in 341.32s   (final run, on the committed source 3a73b36)
```
Baseline before the work: 253 passed (re-run, 86 s). Skipped: 0 (the transform/closure tests need the private pinned vLLM wheel, which is present locally). Intermediate full runs showed two failures that were real and fixed: README label test (literal `KAGGLE_T4_VERIFIED` removed by my rewording) and graceful server shutdown degrading to SIGTERM when the parent ignores SIGINT (fixed in `ServerProcess.start`). The one warning is a deliberate duplicate-ZIP-member test. The local GeForce 940M was hidden (`CUDA_VISIBLE_DEVICES=`); no GPU test ran locally.
Other checks: `verify_build3_wheel.py` 20/20 PASS; two consecutive builds → identical wheel bytes; `pip download --require-hashes` of all 19 pins OK; `locate_and_verify` against a simulated `/kaggle/input` with the final ZIP OK; installer `--dry-run` on this host: bundle, wheel audit and lock plan PASS, baseline FAIL only because this machine lacks transformers (expected).

## Gates
A source quality — met (CPU) · B dependency integrity — met (resolution + CPU; Kaggle install pending) · C package integrity — met · D input reproducibility — met · **E Kaggle verification — NOT MET (owner to run)**.

## Remaining blockers / limitations
Gate E; fastapi 0.141.1 vs upstream's `<0.137` cap (CPU-built app only); xgrammar 0.2.3 older than the 0.2.5.post1 vLLM was developed against; direct `vllm.LLM`/`vllm serve` unsupported; ReplicaPool recovery CPU-tested only; observability unsupported. Full list: `STOKASTISK_0.1.0_BUILD3_PRODUCTION_READINESS.md`.

## Manual Kaggle steps (exact)
1. Download `stokastisk-0.1.0-build3-kaggle-t4-sm75-kaggle-input.zip` and `15_stokastisk_candidate_sdk_integration.ipynb` from the GitHub pre-release `integrated-0.1.0-build3` (or Hub `releases/stokastisk-0.1.0-build3-kaggle-t4-sm75/`); check the ZIP's SHA256 above.
2. Kaggle → New Dataset → upload the ZIP **unchanged** (a wheel-only ZIP is rejected).
3. New Notebook → Import the notebook → Accelerator **GPU T4 × 2**, Internet **ON**, Python 3.13.
4. Add the dataset as input → **Run All**. Cell 1 must print `RELEASE: stokastisk-0.1.0-build3-kaggle-t4-sm75 source: attached-zip`, the wheel SHA256 `1d43089a…7ad3` and `VERIFIED: 42 files`.
5. Download `/kaggle/working/15-stokastisk-build3-acceptance.zip` (written even on failure) and save the executed notebook with outputs; send both back. The notebook fails unless every stage is `PASS`.
