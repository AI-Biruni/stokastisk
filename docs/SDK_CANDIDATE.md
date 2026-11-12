# Stokastisk 0.1.0 — integrated SDK candidate (build tag 1) — SUPERSEDED

> **Superseded 2026-11-12** by the canonical integrated wheel with build tag 2 (`docs/INTEGRATED_RELEASE.md`), which additionally contains the SDK hardening, a hash-verified installer and notebook 15. This page documents the first candidate and is kept as history; its wheel (`sdk-candidate/stokastisk-0.1.0-build1/`) is unchanged and was never executed on Kaggle.

**Status: `BUILT AND PACKAGED — STATICALLY INSPECTED — NOT EXECUTED ON KAGGLE DUAL T4 GPUs`**

This page describes a *candidate* distribution. It is **not** the Kaggle-verified release. Project version is exactly `0.1.0`; the candidate is told apart from the historical wheel only by a PEP 427 build tag.

## A. What was Kaggle-verified (unchanged, still published)
| Artifact | SHA-256 | Notes |
|---|---|---|
| `stokastisk-0.1.0-py3-none-any.whl` | `29edaa55…ee49` | SDK only. Built **before** `stokastisk.compat`, `stokastisk.serving`, `LLM.chat` and the bfloat16 pre-check existed — installing it does **not** give you those modules. |
| `vllm-0.25.1-cp38-abi3-linux_x86_64.whl` | `cc3bcafb…f5e` | T4-only vLLM 0.25.1 (Triton attention, `sm_75` cubins, CUDA 13.0). Executed in notebooks 01–14. |
| `stokastisk-0.1.0-kaggle-t4-sm75-kaggle-input.zip` | `46439612…1210` | the release bundle (installer, notebooks, manifests). |
| Git tag `v0.1.0` | → `1acc49c` | not moved. |

## B. The new candidate
`stokastisk-0.1.0-1-cp313-cp313-linux_x86_64.whl` — **one wheel containing everything**:

| Content | Source | Byte-identical to verified artifact? |
|---|---|---|
| `stokastisk/**` — incl. `compat.py`, `serving.py`, `LLM.chat`, `check_dtype` bfloat16 pre-check, topology/parallel/telemetry/cuda/communication/benchmarks | fresh setuptools build of `sdk/` | n/a (new SDK code) |
| `vllm/**` (2543 files, incl. the six native objects) | the verified vLLM wheel | **yes, every file** |
| `vllm-0.25.1.dist-info/` (installed through `*.data/platlib/`) | the verified vLLM wheel's METADATA / entry points | yes |
| console script `vllm` + vLLM `vllm.general_plugins` entry points | the verified vLLM wheel | yes |

Why a second dist-info: vLLM 0.25.1 resolves its own version with `importlib.metadata.version("vllm")` (`vllm/platforms/__init__.py`, `vllm/triton_utils/importing.py`). A wheel that shipped the `vllm/` tree without a `vllm` distribution would make platform detection warn/raise and could silently disable Triton. The candidate therefore also registers `vllm 0.25.1`. Consequences: `pip list` shows both `stokastisk 0.1.0` and `vllm 0.25.1`; `pip uninstall stokastisk` removes both; **do not install the separate vLLM wheel at the same time** (both own `vllm/`).

Dependencies (`Requires-Dist`): vLLM 0.25.1's own list with the Kaggle-resolved core pins (`apache-tvm-ffi==0.1.9`, `xgrammar==0.2.5.post1`, `llguidance==1.7.6`, `cbor2==6.2.0.post2`, `msgspec==0.22.0`, `openai-harmony==0.0.8`, `partial-json-parser==0.2.1.1.post7`, `pybase64==1.5.1`), `transformers>=5.5.3,<6`, `torch==2.11.0+cu130`, `triton==3.6.0`. `vllm==0.25.1` is not a dependency (it is built in). Extras `server`, `observability`, `models` are carried over.

### Version locks
Stokastisk 0.1.0 · Python 3.13 (`>=3.13,<3.14`) · vLLM 0.25.1 · CUDA Toolkit 13.0 · PyTorch 2.11.0+cu130 · Triton 3.6.0 · NVIDIA SM75 · 2 × Tesla T4 · attention `TRITON_ATTN`.

### Install (Kaggle, Internet ON, Accelerator *GPU T4 × 2*, Python 3.13)
Kaggle already has PyTorch 2.11.0+cu130 and Triton 3.6.0; **do not let pip replace them**. The candidate's dependency list pins them, so a plain `pip install` of a conflicting torch is refused rather than silently swapped; use `--no-deps` and the release's dependency plan (below) to stay safe.
```bash
URL=https://huggingface.co/waqasm86/stokastisk/resolve/main/sdk-candidate/stokastisk-0.1.0-build1/stokastisk-0.1.0-1-cp313-cp313-linux_x86_64.whl
curl -L -o /tmp/stokastisk-0.1.0-1-cp313-cp313-linux_x86_64.whl "$URL"
sha256sum /tmp/stokastisk-0.1.0-1-cp313-cp313-linux_x86_64.whl            # compare with SHA256SUMS next to the wheel
python3.13 -m pip uninstall -y vllm stokastisk                              # if the historical release is installed
python3.13 -m pip install --no-deps --no-index /tmp/stokastisk-0.1.0-1-cp313-cp313-linux_x86_64.whl
```
Third-party dependencies (everything in `Requires-Dist` other than torch/triton) are installed exactly as in the historical release by `scripts/kaggle/02_install.py` (its plan lists the pins and refuses protected-package changes) — run its dependency step, then install the candidate instead of the two historical wheels. This combined flow has **not** been executed.

### Use
```python
import stokastisk
stokastisk.check_dtype("half")                                   # bfloat16 raises StokastiskCompatibilityError before an engine starts
config = stokastisk.StokastiskConfig(tensor_parallel_size=2)     # attention_backend is fixed to "TRITON_ATTN"
llm = stokastisk.LLM("Qwen/Qwen2.5-1.5B-Instruct", config=config)
print(llm.verify_attention_backend())                           # live VllmConfig must report TRITON_ATTN
out = llm.chat([{"role": "user", "content": "Hello"}])           # LLM.chat -> vllm.LLM.chat, same engine and backend
```
OpenAI-compatible server: `stokastisk.serving.ServerProcess(...)` / `build_serve_command(...)` launch `vllm serve … --attention-backend TRITON_ATTN` (an override in `extra_args` is refused). Equivalent console command: `vllm serve MODEL --dtype half --attention-backend TRITON_ATTN`.

Environment variables: `STOKASTISK_*` (read by `StokastiskConfig.from_env()`); the SDK sets (never overrides) `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1`, `NCCL_CUMEM_ENABLE=0`, `VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_ALLREDUCE_USE_FLASHINFER=0`, and thread-count variables.

## TRITON_ATTN-only: what enforces it
1. **Source level.** The vLLM tree is the Flash-free derivative (patches `0001–0003`): no `vllm_flash_attn`, no FlashInfer, no FlashMLA, no FA2/FA3 `.so`; `fa_utils.py` defines the flash entry points as `None`; `FlashAttentionBackend.supports_compute_capability` is `>= 8.0`, so it is invalid on SM75 anyway.
2. **SDK level.** `StokastiskConfig` accepts only `TRITON_ATTN` and rejects any name containing FLASH/FA2/FA3/FA4; the engine always passes `attention_config={"backend": "TRITON_ATTN"}` (explicit selection — not an environment variable); `LLM.verify_attention_backend()` raises `AttentionBackendError` if the live config differs; `serving` refuses overrides.
3. **Honest boundary (unchanged vLLM code, not edited here).** Without explicit selection upstream auto-selection probes `FLASH_ATTN` first and would fall to `TRITON_ATTN` after it is rejected; and the vision-tower attention of multimodal models falls to `TORCH_SDPA` on SM75. The SDK never relies on auto-selection, but a user who calls `vllm.LLM(...)` directly without `attention_config` is outside the SDK guard. The vLLM bytes were deliberately **not** edited, to keep them identical to the executed wheel.

## C. NOT executed — exact boundary
Nothing in this candidate has run on a GPU or on Kaggle. In particular **unverified**: installing it on Kaggle; `import vllm` from the integrated wheel; `importlib.metadata` resolution inside the Kaggle image; loading the `_C_stable_libtorch` ABI against torch 2.11.0+cu130 (the identical bytes did load in the verified runs, but the integrated install path did not); `LLM.chat`, `check_model/dtype/quantization/feature`, `stokastisk.serving`, the server's SSE streaming and metrics; any model other than Qwen2.5-0.5B/1.5B; CUDA graphs; fp8/bfloat16; speculative decoding; LoRA; multimodal. CPU unit tests (223) pass locally, with mocked GPUs — they are not GPU evidence.

## Static inspection performed
See `evidence/` in the candidate directory on the Hub (`candidate-static-verification.json`, `native-audit.json`) and `scripts/audit/verify_sdk_candidate.py`.
