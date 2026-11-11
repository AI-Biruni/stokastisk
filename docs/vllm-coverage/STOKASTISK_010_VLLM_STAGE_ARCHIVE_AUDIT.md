# Stokastisk 0.1.0 — vLLM stage archive audit

Date 2026-11-12. Source inspection only: nothing was imported, run on a GPU or tested.

## Identity
| Item | Value |
|---|---|
| File | `/media/waqasm86/External1/Waqas-Projects/Project-Stokastisk/stage.zip` (git-ignored; never modified) |
| SHA256 | `c1eb62cc622a4b97645467b6b65a3c6e1be56bb30ffaae2d84b792e27f221800` |
| Size / mtime | 31,538,686 bytes / 2026-11-07 17:00 |
| Integrity | `unzip -tq`: no errors; 2813 entries = 2560 files + directories |
| Audit extraction (new, separate) | `work/stage-audit-20261112/stage/` (2560 files, 137 MB) |
| Earlier extraction | `work/stage-extract-20261107/stage/` — **per-file SHA256 of all 2560 files identical** to the audit extraction (aggregate listing hash `166dbde78665…`). The attached archive and the local archive are the same file; there is one source tree, not two. Neither extraction was overwritten. |

## What the archive is — and is not
* It is an **installed vLLM 0.25.1 package tree** (`stage/vllm/`, 1967 `.py`, 567 `.json` tuned-kernel configs, 8 `.jinja`, 5 compiled `.so`), plus one stale file `stage/stochastisk/__init__.py` (an old facade labelled version 0.3.0 with the misspelt package name; **not used** by the 0.1.0 SDK).
* It is **not** a source checkout: there is no `setup.py`, `pyproject.toml`, `CMakeLists.txt`; the only C++ file is `hf3fs_utils.cpp` (a KV-connector helper). Native code exists only as five prebuilt `.so`. Native build configuration is therefore not in the archive (it lives in `work/upstream/` and `scripts/build_vllm_sm75.sh`).
* It is **not** the T4-stripped implementation. It still contains FlashAttention (`vllm/vllm_flash_attn/`), FlashMLA (`vllm/third_party/flashmla/`) and FlashInfer modules.
* **The T4-stripped implementation is the vLLM wheel that Kaggle executed** (`vllm-0.25.1-cp38-abi3-linux_x86_64.whl`, SHA256 `cc3bcafb6516791990ca302073ae153a08465d3df1e460d666e3ec09b4e98f5e`, byte-identical in the published release and in the executed bundle).

## Stage tree versus the executed wheel (file-level diff)
| Difference | Count | Detail |
|---|---|---|
| Only in the stage tree | 18 | `stochastisk/__init__.py` (stale facade) and 17 vllm files = the 16 removed `.py` modules listed below + `vllm/vllm_flash_attn/.gitkeep` |
| Only in the wheel | 1 | `vllm/_rust_tool_parser.abi3.so` (not in the stage tree) |
| Present in both, different bytes | 12 | 7 patched `.py` (patch 0003 and related guards) + 5 `.so` (rebuilt for SM75; see below) |
| vllm `.py` in stage / in wheel | 1967 / 1951 | 16 removed |

Removed from the wheel (module-level flash import rule in `scripts/release/build_t4_tree.py`):
* `compilation/passes/fusion/allreduce_rms_fusion.py`
* `distributed/device_communicators/all2all.py`
* `distributed/device_communicators/flashinfer_all_reduce.py`
* `distributed/device_communicators/mnnvl_compat.py`
* `model_executor/models/qwen2_5_omni_thinker.py`
* `models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py`
* `third_party/flashmla/__init__.py`
* `v1/attention/backends/flash_attn_diffkv.py`
* `v1/attention/backends/flashinfer.py`
* `v1/attention/backends/mla/flashattn_mla.py`
* `v1/attention/backends/mla/flashattn_mla_sparse.py`
* `v1/attention/backends/mla/flashinfer_mla.py`
* `v1/attention/backends/mla/prefill/flashinfer.py`
* `v1/attention/ops/flashmla.py`
* `vllm_flash_attn/__init__.py`
* `vllm_flash_attn/flash_attn_interface.py`

Patched `.py` (wheel vs stage): `compilation/passes/pass_manager.py` (FlashInfer fusion pass replaced by a guarded `None` and a clear `RuntimeError`), `config/compilation.py` (no allreduce-fusion size table), `distributed/device_communicators/cuda_communicator.py` (FlashInfer all-reduce import made lazy), `model_executor/layers/fused_moe/all2all_utils.py` (FlashInfer NVLink all2all names set to `None`), `model_executor/layers/rotary_embedding/common.py` (`forward_native` instead of the vendored flash rotary), `v1/attention/backends/fa_utils.py` (flash names defined as `None`), `v1/sample/ops/topk_topp_sampler.py` (FlashInfer sampler guarded, native sampler used).

## Native extensions
| `.so` | GPU code (cuobjdump, static) |
|---|---|
| `_C_stable_libtorch.abi3.so` | `sm_75` only, no PTX (both stage and wheel builds) |
| `_moe_C_stable_libtorch.abi3.so` | `sm_75` only, no PTX (both) |
| `cumem_allocator`, `fs_io_C`, `spinloop` | no device code |
| `_rust_tool_parser.abi3.so` (wheel only) | no device code |
The five common `.so` differ in bytes between the stage tree and the wheel (rebuilt/repacked); both contain `sm_75` cubins only. FlashAttention extensions (`_vllm_fa2_C`, `_vllm_fa3_C`) are absent from both.

## Structure retained
33 top-level packages under `vllm/`: `config`, `engine`, `entrypoints` (cli, openai, serve, generate, pooling, anthropic, grpc, mcp, scale_out, speech_to_text), `v1` (engine, core, worker, executor, attention, sample, spec_decode …), `model_executor` (models, layers, kernels, warmup), `distributed`, `lora`, `multimodal`, `platforms`, `transformers_utils`, `tokenizers`, `reasoning`, `tool_parsers`, `renderers`, `profiler`, `tracing`, `usage`, `benchmarks`, `compilation`, `plugins`, `ray`, `triton_utils`, `utils`, `logging_utils`, `inputs`, `ir`, `kernels`, `parser`, `assets`, `connections.py` and the root modules (`envs.py`, `sampling_params.py`, `outputs.py`, `scripts.py` …).

Original upstream entry points (`vllm-0.25.1.dist-info/entry_points.txt`): console script `vllm = vllm.entrypoints.cli.main:main`; plugins `lora_filesystem_resolver`, `lora_hf_hub_resolver`. CLI subcommands present: `serve`, `chat`/`complete` (openai.py), `run-batch`, `launch`, `collect-env`, `bench` (latency, throughput, serve, startup, sweep, mm_processor).

## Attention, Triton and CUDA sources
* Attention backends retained in `v1/attention/backends/`: `triton_attn`, `triton_attn_diffkv`, `flex_attention`, `turboquant_attn`, `cpu_attn`, `flash_attn` (inert: needs `vllm_flash_attn`, absent; its `supports_compute_capability` is `>= 8.0`), `rocm_*`, `mla/`, `linear_attn`, `mamba*_attn`, `gdn_attn`, `hpc_attn`, `short_conv_attn`.
* Triton source: **140 files with 371 `@triton.jit`/autotune/triton_op decorators** (AST count).
* CUDA source: none in the archive (only the five prebuilt `.so`).
* GPU architecture assumptions: 114 modules contain a capability gate (`has_device_capability`, `get_min_capability`, …); the gates are tabulated per module in the compatibility matrix.
* Dependencies (wheel METADATA): `transformers>=5.5.3`, `fastapi`, `starlette`, `aiohttp`, `openai>=2.0.0`, `pydantic`, `prometheus_client`, `pyzmq`, `xgrammar`, `llguidance`, `opentelemetry-*`, `apache-tvm-ffi==0.1.9`, and others. Optional GPU packages (`flashinfer`, `flash_attn`) are referenced by 44 modules (any depth); none is installed or imported on the T4 path.

## Conclusion
The attached and local archives are identical. The T4-stripped implementation is the executed wheel, not the stage tree. All later classification uses the **wheel's** 1951 modules plus the 16 removed modules from the stage tree.
