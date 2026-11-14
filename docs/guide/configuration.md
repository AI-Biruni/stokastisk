# 2. Configuration

`StokastiskConfig` is a validated dataclass. Invalid values raise `ConfigurationError` at construction, before any GPU work.

```python
from stokastisk import StokastiskConfig

cfg = StokastiskConfig(tensor_parallel_size=2, max_model_len=2048)
print(cfg.strategy, cfg.world_size)      # tensor_parallel 2
print(cfg.as_dict())
```

## Fields

| Field | Default | Meaning |
|---|---|---|
| `tensor_parallel_size` | `1` | 1 or 2. TP=2 splits each layer over both T4s (traffic over PCIe, no NVLink). |
| `pipeline_parallel_size` | `1` | 1 or 2. PP=2 puts half the layers on each GPU. **Experimental.** |
| `execution_strategy` | `None` | `single_gpu`, `tensor_parallel`, `pipeline_parallel`, `replica_parallel`, `auto`. `None` is inferred from TP/PP. `auto` only works with `recommend_parallelism`. |
| `replica_devices` | `None` | GPU indices for `replica_parallel` (set by `ReplicaPool`). |
| `dtype` | `"half"` | FP16. bfloat16 is refused (T4 has no bf16). |
| `kv_cache_dtype` | `"auto"` | Passed to vLLM. |
| `attention_backend` | `"TRITON_ATTN"` | The only allowed value. Any Flash-style name raises. |
| `enforce_eager` | `True` | No CUDA graphs / `torch.compile`. |
| `disable_custom_all_reduce` | `True` | Custom all-reduce needs peer access, which varies between Kaggle VMs. |
| `gpu_memory_utilization` | `0.88` | Fraction of each GPU vLLM may use, within 0.1–0.98. |
| `max_model_len` | `4096` | Context length, `None` or an int ≥ 16. Lower it to save KV-cache memory. |
| `structured_output_backend` | `"xgrammar"` | Pinned. `"auto"` is refused because it silently falls back between backends. |
| `apply_nccl_env` / `nccl_env` | `True` / see below | NCCL settings that do not rely on peer access. |

Combinations are checked: `TP × PP` may not exceed 2; `single_gpu` needs TP=1/PP=1; `tensor_parallel` needs TP=2; `pipeline_parallel` needs PP=2. Violations raise `UnsupportedParallelismError`.

## Properties and methods

| Name | Returns |
|---|---|
| `strategy` | the effective strategy string |
| `world_size` | `tensor_parallel_size × pipeline_parallel_size` |
| `is_experimental` | `True` for `pipeline_parallel` and `replica_parallel` |
| `validate()` | re-checks the object (raises on error) |
| `as_dict()` | plain dict of all fields |
| `engine_kwargs()` | the keyword arguments handed to the engine |
| `apply_environment()` | exports NCCL, thread-count and FlashInfer-off variables; **never overrides** a variable you already set; returns what it set |
| `StokastiskConfig.from_env(prefix="STOKASTISK_")` | builds a config from environment variables |

## Environment variables

`from_env()` reads (all optional):

`STOKASTISK_TENSOR_PARALLEL_SIZE`, `STOKASTISK_PIPELINE_PARALLEL_SIZE`, `STOKASTISK_EXECUTION_STRATEGY`, `STOKASTISK_GPU_MEMORY_UTILIZATION`, `STOKASTISK_MAX_MODEL_LEN`, `STOKASTISK_DTYPE`, `STOKASTISK_ATTENTION_BACKEND`, `STOKASTISK_STRUCTURED_OUTPUT_BACKEND`, `STOKASTISK_ENFORCE_EAGER`, `STOKASTISK_DISABLE_CUSTOM_ALL_REDUCE`.

Booleans accept `1/true/yes/on` and `0/false/no/off`. `LLM(model)` without a `config=` uses `from_env()`.

```python
import os
os.environ["STOKASTISK_MAX_MODEL_LEN"] = "2048"
cfg = StokastiskConfig.from_env()
```

## What `apply_environment()` sets

* NCCL: `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1`, `NCCL_CUMEM_ENABLE=0`
* `VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_ALLREDUCE_USE_FLASHINFER=0`
* `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS` = 4 CPU cores ÷ `world_size`

## Presets

```python
one_gpu  = StokastiskConfig(max_model_len=2048)                              # small model
two_gpu  = StokastiskConfig(tensor_parallel_size=2, max_model_len=2048)      # 7B
pipeline = StokastiskConfig(pipeline_parallel_size=2, max_model_len=2048)    # experimental
```

Next: [The LLM engine](engine.md).
