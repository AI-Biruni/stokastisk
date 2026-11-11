# Stokastisk 0.1.0 — TRITON_ATTN compatibility audit (vLLM 0.25.1, Tesla T4 / SM75)

Evidence classes: **[K]** executed on Kaggle 2×T4 by the owner; **[S]** static source inspection of the executed wheel; **[I]** inference, not verified. No code was run for this report.

## 1. How the backend is selected (source: `vllm/platforms/cuda.py`, `vllm/v1/attention/backends/registry.py`) [S]
* `AttentionBackendEnum.TRITON_ATTN = vllm.v1.attention.backends.triton_attn.TritonAttentionBackend`.
* **Explicit selection** (`attention_config={"backend": ...}` / `--attention-backend`): the backend is validated for the device and configuration; if invalid, the platform raises `ValueError("Selected backend … is not valid for this configuration. Reason: …")`. There is no silent substitution of an explicitly selected backend.
* **No selection**: the priority list for a CUDA device that is not SM10x is `FLASH_ATTN, FLASHINFER, TRITON_ATTN, FLEX_ATTENTION, TURBOQUANT`; each candidate is validated and the best valid one wins. On a T4 `FLASH_ATTN` is invalid (see §3), so auto-selection would also end on `TRITON_ATTN` — but only after probing Flash first. For MLA models the list starts with `CUTLASS_MLA`/`FLASH_ATTN_MLA`/`FLASHMLA`/`FLASHINFER_MLA` before `TRITON_MLA`.
* **Stokastisk therefore never relies on auto-selection**: `StokastiskConfig(attention_backend="TRITON_ATTN")` is the only accepted value (`FORBIDDEN_ATTENTION_MARKERS = FLASH, FA2, FA3, FA4`); the engine kwargs always contain `attention_config={"backend": "TRITON_ATTN"}`; `LLM.verify_attention_backend()` compares the live `VllmConfig` with the request and raises `AttentionBackendError`; `serving.build_serve_command` always emits `--attention-backend TRITON_ATTN` and refuses an `--attention-backend`/`--attention-config` override in `extra_args`.

## 2. Executed evidence [K]
* Notebooks 05, 13 (and 08, 09, 11, 14 through the same engine path): the live engine reported `TRITON_ATTN`.
* Notebook 13: for TP=1, TP=2 and PP=2 the Triton cache held a compiled `kernel_unified_attention` and the process trees mapped **no** FlashAttention/FlashInfer/FlashMLA library.
* Triton kernels compiled and launched on the T4s (cubins in the executed acceptance ZIPs): `kernel_unified_attention`, `reduce_segments` (both `v1/attention/ops/triton_unified_attention.py`), `reshape_and_cache_kernel_flash` (`v1/attention/ops/triton_reshape_and_cache_flash.py` — a plain Triton KV-cache write; the name contains "flash" but it has no FlashAttention dependency), plus sampling/input-batch kernels under `v1/worker/gpu/…`.
* Model/shapes actually run: Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct, fp16, `enforce_eager=True`, `max_model_len` 2048–4096, batches 1–64, 16–1024 prompt tokens, greedy decoding.

## 3. Why Flash backends cannot be selected on SM75 [S]
* `flash_attn.py`: `supports_compute_capability(capability) = capability >= DeviceCapability(8, 0)` → invalid on a 7.5 device. The vendored `vllm_flash_attn` package and the FA2/FA3 `.so` are not shipped; `fa_utils.py` defines the flash entry points as `None` so the module still imports.
* `flashinfer.py`, `mla/flashinfer_mla.py`, `mla/flashattn_mla*.py`, `ops/flashmla.py`, `third_party/flashmla`: removed from the wheel (see the archive audit). Modules that referenced them are patched to a guarded `None` and a `RuntimeError` that says FlashInfer is absent (`pass_manager.py`, topk/topp sampler).
* Remaining flash-named modules are inert adapters; the inventory classifies 35 as `EXCLUDED_FLASH_DEPENDENCY`.

## 4. What the Triton backend supports (`v1/attention/backends/triton_attn.py`) [S]
| Property | Source fact | Consequence on T4 |
|---|---|---|
| `supports_compute_capability` | always `True` | no architecture gate in the backend itself |
| head size | `head_size >= 32` | Qwen2.5-0.5B/1.5B (64/128) fine; models with head size < 32 get a backend validity error from vLLM |
| dtypes | float16, bfloat16, float32 | bfloat16 is rejected earlier by the platform (`check_if_supports_dtype`, capability ≥ 8.0) → use `dtype="half"` |
| KV-cache dtypes | auto, float16, bfloat16, fp8 … | `CudaPlatform.supports_fp8()` is `has_device_capability(89)`; fp8 KV cache is **not** expected to work on T4 (SDK default `kv_cache_dtype="auto"`); not run |
| block size | multiple of 16 | default KV block size accepted in all runs |
| attention types | decoder, encoder, … ("supports all attention types") | [I] encoder-decoder/ViT paths not run |
| sinks, mm-prefix | supported | not run |
| prefill / decode / causal / varlen / paged KV | one unified Triton kernel (`kernel_unified_attention`) with a segment-reduction kernel | prefill+decode of Qwen2 verified [K] (prompts 16–1024 tokens); other shapes [I] |
| CUDA graphs | not used: every run was `enforce_eager=True` | graph capture on T4 unverified |

## 5. Where TRITON_ATTN is *not* what runs (honest boundary) [S]/[I]
* **Vision-encoder attention (multimodal)**: `CudaPlatform.get_vit_attn_backend` tries `FLASH_ATTN` first (invalid on SM75), then returns `TORCH_SDPA` for capability < 8.0. Multimodal models on a T4 would therefore use PyTorch SDPA for the vision tower, not Triton. No multimodal model was run. This is a deviation from "Triton only" for that component and is reported, not hidden.
* **MLA models (DeepSeek family)**: Flash/FlashInfer/FlashMLA MLA backends are excluded; `TRITON_MLA` exists in the list but was never run. Classified UNKNOWN_REQUIRES_EVIDENCE.
* **Linear-attention / Mamba / hybrid models**: use their own attention modules (`mamba*_attn.py`, `gdn_attn.py`, `linear_attn.py`); not run.
* **Sampler**: the FlashInfer sampler is guarded; the native/Triton top-k/top-p path is used (`_topk_topp_kernel` compiled [K]).

## 6. Clear compatibility errors added by the SDK (implemented, not executed)
`stokastisk.check_model` (module/imports missing from the T4 wheel), `check_dtype` (explicit bfloat16), `check_quantization` (declared minimum capability), `check_feature` (Flash, FP8/FP4-native, expert-parallel all2all, Qwen Omni, L4) all raise `StokastiskCompatibilityError` **before** an engine starts.

## 7. Not claimed
Speculative decoding, LoRA kernels, pooling/embedding attention, encoder-decoder attention, sliding-window or sink models, fp8 KV cache and CUDA-graph decoding were **not** exercised and are not claimed to work.
