# Dependency compatibility — Stokastisk 0.1.0 builds 3 and 4 (vLLM 0.25.1, Kaggle 2 × T4)

Builds 4 and 5 have exactly the same dependency pins, locks and vLLM tree as build 3 (proved by the member delta audits); the sections below describe all of them. The Kaggle runs cited were executed on build 3.

Status labels used here: **Implemented**, **CPU-tested**, **Kaggle-tested**, **Verification pending**, **Unsupported**. Build 3 was run on Kaggle 2 × T4 on 2026-11-13 (notebook 15: installer 32/32, static 79/79, engine 44/44, parallel 23/23, server 21/21, summary 15/15) and the published install on 2026-11-13/14 (notebooks 16, 18, 19) — see [KAGGLE_ACCEPTANCE.md](KAGGLE_ACCEPTANCE.md). Build 2's Kaggle run is cited only as the evidence of what failed.

## 1. What failed in build 2 (the owner's executed notebook 15; the raw outputs were removed from the repository when it was slimmed down — the findings are recorded here)
| # | Failure | Root cause (verified) | Build-3 fix |
|---|---|---|---|
| 1 | `engine`: `No module named 'lmformatenforcer'` | `vllm/v1/structured_output/backend_lm_format_enforcer.py` has no `from __future__ import annotations`; its dataclass annotation `lmformatenforcer.TokenEnforcer` is evaluated at import time, and `SamplingParams._validate_structured_outputs` imports that module **before looking at the backend** → every structured-output request, even with `backend="xgrammar"`, needs `lmformatenforcer`. Upstream lists `lm-format-enforcer`, `outlines_core`, `diskcache` as hard requirements; build 2 had demoted them to a `models` extra and the installer never installed them. | `requirements/structured-output.lock` (lm-format-enforcer 0.11.3, outlines_core 0.2.14, diskcache 5.6.3, interegular 0.3.3), installed by default and declared as hard `Requires-Dist`. Backend **pinned** to xgrammar. |
| 2 | `server`: `No module named 'model_hosting_container_standards'` | `vllm serve` builds its app with `sagemaker_standards_bootstrap(app)` unconditionally. Build 2's `SERVER_REQUIREMENTS` was a hand-picked list of 10 names and did not include the `server` extra. | `requirements/serving.lock` (7 packages incl. `model-hosting-container-standards`, `prometheus-fastapi-instrumentator`, `sse-starlette`, `fastapi-cli`, `pydantic-settings`, `rich-toolkit`, `supervisor`); `stokastisk.dependencies` is the single registry that `serving`, the installer and the tests share. |
| 3 | `static`: "8 packages missing" | **Not a removal.** The installer normalised names with PEP 503 (`[-_.]+`→`-`); the static stage used `lower().replace("_","-")`, which keeps dots. All eight names contain a dot (`jaraco.*`, `lazr.*`, `ruamel.yaml`, `path.py`, `keyrings.google-artifactregistry-auth`). Duplicate dist-infos (`Mako`, `jaraco.context`) made a dict-comprehension pick one arbitrarily. | One implementation, a single inventory helper (since removed with the bundle tooling; `pip list --format=json`, PEP 503 names, duplicates reported); installer and static stage both use it. |
| 4 | `static`: Flash `.pyc` in the installed tree | The byte-pinned vLLM wheel carried 28 Flash-family modules (25 by your name patterns + `fa_utils.py`, `rocm_aiter_fa.py`, `fa4_cutedsl_config.py`); Python compiled them on install. | Removed from the wheel; see §4. |
| 5 | provenance | The notebook found no `manifests/SHA256SUMS` under `/kaggle/input` and silently downloaded **build 2** from Hugging Face. Build 2's results therefore describe build 2, never the uploaded build-1 ZIP. | the bundle verifier (since removed); no remote fallback unless `STOKASTISK_ALLOW_HF_FALLBACK=1`. |
| 6 | `pip check` not clean | `xgrammar 0.2.5.post1` declares `transformers<5`. | xgrammar 0.2.3, §3. |

## 2. Method
`scripts/release/lock_closure.py` walks PyPI `Requires-Dist` metadata for each profile's pinned roots against **the real Kaggle inventory** (`requirements/kaggle-baseline-2026-11-13.json`, 937 packages,
the installer's own "before" snapshot from the failed run). A requirement satisfied by the baseline is kept and not walked; one the baseline does *not* satisfy is reported as a **conflict** (the baseline is
protected — the tool never proposes changing it); an absent package is resolved to the highest version that has a CPython 3.13 / Linux x86_64 / glibc ≤ 2.35 wheel. The lock lists the sha256 of **every**
compatible wheel of the chosen version. Audit trail: `requirements/closure/*.closure.json`. All four profiles resolve with **0 conflicts, 0 unresolvable**.

| Profile | Installed by default | New packages (absent from Kaggle) |
|---|---|---|
| `core` | yes | apache-tvm-ffi 0.1.9, cbor2 6.2.0.post2, llguidance 1.7.6, msgspec 0.22.0, openai-harmony 0.0.8, partial-json-parser 0.2.1.1.post7, pybase64 1.5.1, xgrammar 0.2.3 |
| `structured-output` | yes | diskcache 5.6.3, interegular 0.3.3, lm-format-enforcer 0.11.3, outlines-core 0.2.14 |
| `serving` | yes | fastapi-cli 0.0.33, model-hosting-container-standards 0.1.16, prometheus-fastapi-instrumentator 8.1.0, pydantic-settings 2.16.2, rich-toolkit 0.20.6, sse-starlette 3.7.0, supervisor 4.3.0 |
| `models` | **no** (`--with-models`) | astor, compressed-tensors 0.17.0, depyf 0.20.0, loguru, setproctitle. `mistral_common` and `ijson` (Mistral tool parser) are deliberately **not** included — Mistral-family models are unsupported on this build |
| observability | **Unsupported** | `opentelemetry-exporter-otlp` needs `protobuf<7`; Kaggle has protobuf 7.36.2 (protected). The resolver reports the conflict; it is not hidden. vLLM treats the exporters as optional. |

Why `models` is optional: a full-tree import sweep (1,781 modules) with and without those packages shows that **no** module on the `vllm.LLM`, `vllm serve`, Qwen2, Triton-attention or structured-output
paths needs them; off-path modules (quantisation; Mistral/Pixtral/Voxtral tokenizers and models) do. They stay a documented opt-in.

## 3. xgrammar / transformers / apache-tvm-ffi — resolved without patching metadata
PyPI metadata (queried 2026-11-13):

| xgrammar | `transformers` | `apache-tvm-ffi` | vs vLLM's `apache-tvm-ffi==0.1.9` and Kaggle's transformers 5.20.2 |
|---|---|---|---|
| 0.2.1, 0.2.2, **0.2.3** | `>=4.38.0` (no cap) | `>=0.1.9` | **compatible** |
| 0.2.5.post1 (build 2) | **`<5`**, `>=4.38.0` | `>=0.1.9` | transformers conflict |
| 0.2.6, 0.2.7 | `>=4.38.0` | `>=0.1.10` | tvm-ffi conflict |
| 0.2.8, 0.2.9 | `>=4.38.0` | `>=0.1.11` | tvm-ffi conflict |

(The build-2 version of this page said no release removes the transformers cap *and* accepts tvm-ffi 0.1.9. That was wrong: it listed only 0.2.5.post1 and newer. 0.2.1–0.2.3 do both.)
vLLM 0.25.1 declares `xgrammar >=0.2.1,<1.0.0`, so **0.2.3 is inside vLLM's supported range** — this is a version choice, not a patch. No distribution is modified, no metadata is edited, `pip check` is expected clean.
CPU evidence (`scripts/audit/structured_output_cpu_check.py`, transformers 5.20.2, apache-tvm-ffi 0.1.9, torch 2.11.0): **vLLM's own** `XgrammarBackend` + `XgrammarGrammar` drive a simulated constrained decode with the Qwen2.5 tokenizer —
valid JSON that satisfies the schema, prose tokens masked out, `validate_tokens` stops at the first schema violation, rollback/reset work; `Grammar.from_regex/from_ebnf/from_json_schema/from_structural_tag` compile.
**Kaggle-tested:** the same on the T4 through `stokastisk.LLM.generate_structured` (notebook 15 `engine` stage 44/44, notebook 16 quickstart) and over HTTP (`server` stage 21/21). Not claimed: any xgrammar CUDA kernel behaviour.

## 4. FlashAttention / FlashInfer / FlashMLA
* **Shipped:** none. `scripts/release/t4_flash_free.py` removes 28 modules from the byte-pinned vLLM tree (the list is `flash-free-transform-report.json`, written into the local acceptance bundle by `assemble_build3_release.py`) and leaves every native object byte-identical (the wheel has no Flash `.so`, `.cubin`, `.ptx`).
* **Not removed by design** (names contain "flash" but they are not FlashAttention): `triton_reshape_and_cache_flash.py` (a Triton KV-cache write kernel used by `TRITON_ATTN`), `longcat_flash*.py` (a model family), `*dflash*` (a speculative-decoding method). The policy records this (`stokastisk.prohibited.ALLOWED_FLASH_NAMED`).
* **References that remain** (reported, never counted as implementations): 15 `.py` files contain function-level `import flashinfer` statements in code that cannot run on a T4. A runtime guard in `vllm/t4_backend_policy.py` sets `sys.modules["flashinfer"] = None` (and `flash_attn`, `flash_mla`, `vllm_flash_attn`) at `import vllm`, so those imports raise `ModuleNotFoundError` and `importlib.util.find_spec` returns `None` even if a Flash package were installed.
* **Imports of removed modules** (49 sites, e.g. the MoE oracles and the linear-kernel registry that `_custom_ops.py` pulls in) now come from one generated placeholder, `vllm/utils/t4_prohibited_stubs.py`: predicates return `False`, kernel classes answer *not supported*, functions raise `ProhibitedBackendError` (an `ImportError`). It is not an implementation.
* **Selection:** `AttentionBackendEnum`/`MLAPrefillBackendEnum.get_path()/get_class()` refuse every Flash-family member (even if a user registers an override); `AttentionConfig` rejects them at construction; the CUDA platform's selector (which treats `ImportError` as "invalid candidate") skips them. CPU-evaluated for compute capability 7.5: valid = `[TRITON_ATTN, FLEX_ATTENTION]`, `FLASH_ATTN`/`FLASHINFER` invalid with the explicit reason.
* **Direct `vllm.LLM` / `vllm serve`:** *Unsupported entry point* (`compat.FEATURES["direct_upstream_vllm"]`). With this tree they cannot select Flash and auto-selection resolves to `TRITON_ATTN` on SM75 (CPU-evaluated; notebook 15 `engine` checks it on the GPU), but they bypass the SDK's dtype, NCCL and structured-output policy and are not verified as a supported path.
* **Equality proof:** `scripts/audit/verify_integrated_wheel.py` re-runs the transform on the pinned wheel and compares every `vllm/**` byte; a 1,783-module import sweep (CPU) shows **0 modules that imported before and fail now** (24 more modules import than before: the dangling DeepSeek imports).

## 5. Known version facts about the Kaggle image (protected, never changed)
`torch 2.11.0+cu130`, `triton 3.6.0`, `transformers 5.20.2`, `fastapi 0.141.1` (**upstream vLLM caps `fastapi<0.137`** because a newer route tree "breaks model-hosting-container-standards handler overrides"; CPU check: vLLM's `build_app` builds, `/metrics` and `/version` answer under fastapi 0.141.1 / starlette 1.9.0 / mhcs 0.1.16 — the SageMaker `/ping`/`/invocations` routes are not part of Stokastisk's advertised API; the server stage passed on Kaggle (app built, models, chat, streaming, metrics), the SageMaker routes were not exercised), `protobuf 7.36.2`, `numpy 1.26.4` (no cp313 wheel on PyPI — never reinstalled), 8 pre-existing `pip check` lines (unrelated: google-colab, gradio, moviepy …) that the installer records and tolerates only if they are identical before and after.

## 6. Protected packages
The wheel is installed with `--no-deps --no-index`; pins are downloaded with `pip download --no-deps --require-hashes --only-binary=:all:` and installed with `--no-deps --no-index`. No resolver runs on Kaggle.
During the build-3 acceptance an inventory check (`02_install_integrated`, archived in `evidence/kaggle/`) showed that no pre-installed distribution changed or disappeared and that exactly the planned packages were added.
