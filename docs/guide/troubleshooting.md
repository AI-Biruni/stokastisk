# 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `No matching distribution` / `is not a supported wheel` | the notebook is not Python 3.13, or Internet is OFF, or an old install line with the retired index address is used — use the pinned command in [Getting started](getting-started.md) |
| `THESE PACKAGES DO NOT MATCH THE HASHES` | the downloaded file is not the one the command names; do not use it. Retry; if it persists, report it with the full log |
| pip wants to change torch / transformers | should never happen; report it with the full `pip install` log |
| Engine start fails, no GPU | Accelerator is not *GPU T4 × 2* |
| `ProhibitedBackendError` (an `ImportError` raised by the bundled vLLM) or `ConfigurationError` mentioning Flash | a Flash backend was requested; only `TRITON_ATTN` exists |
| `bfloat16` error | use `dtype="half"` |
| Out of memory at start-up | lower `max_model_len` (e.g. 2048), lower `gpu_memory_utilization`, or use `tensor_parallel_size=2` |
| Out of memory after an earlier experiment | the previous engine is still alive: `close()` it, `gc.collect()`, check `torch.cuda.mem_get_info(i)` |
| Engine code hangs or re-runs in a script | add `if __name__ == "__main__":` (vLLM spawns processes that re-import the script) |
| `Triton kernel JIT compilation during inference` | one-time warm-up warning on the first request; harmless |
| `unauthenticated requests to the HF Hub` | informational; add an `HF_TOKEN` Kaggle secret for faster downloads |
| `ProcessGroupNCCL … unbatched P2P op` (PP=2) | harmless warning |
| `UNKNOWN_REQUIRES_EVIDENCE` from `check_model` | that exact model id was never run on Kaggle (or its architecture could not be determined: pass `architectures=[...]`). It is not an error — see [Checks](diagnostics.md) |
| `Process manager: force killing remaining processes` at shutdown | seen after every TP=2/PP=2 engine in the Kaggle runs; the GPUs were fully released each time (see [The LLM engine](engine.md)) |
| `StructuredOutputError: truncated` | bound the schema fields and/or raise `max_tokens` |
| `ServerStartupError` | read `e.log_excerpt` and `e.log_path`; a second server on the same port also causes this |
| Slow with TP=2 / PP=2 | expected for a model that fits one T4 (PCIe, no NVLink); use TP=1 or `ReplicaPool` |
| Disk full on `/kaggle/working` | quota is 20 GiB; check `stokastisk.working_dir_status()` |

To start clean: `llm.close()`, then restart the Kaggle session.

Report problems at <https://github.com/stokastisk-ai/stokastisk/issues> with the output of `stokastisk.report()`.
