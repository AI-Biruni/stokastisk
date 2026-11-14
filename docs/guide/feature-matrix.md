# Feature matrix and model evidence

Generated from `stokastisk.compat` (`FEATURES`, `MODEL_EVIDENCE`) by `scripts/release/build_docs_site.py`; do not edit by hand. These are the records the installed SDK reports through `stokastisk.feature_matrix()` and `stokastisk.check_model()`.

**Evidence classes.** `kaggle`: executed on Kaggle 2 × T4 (the *notebooks* column says which: 15 = acceptance suite on build 3, 16/18/19 = end-user notebooks, 01–14 = earlier notebooks). `kaggle-kernel`: a Triton kernel compiled and launched, output not asserted. `cpu-tests`: CPU unit tests only. `static`: source inspection only. Builds 4 and 5 were not executed on Kaggle; their inference code is byte-identical to build 3.

## Features

| Feature | Status | Evidence | Notebooks | Note |
|---|---|---|---|---|
| `offline_generation` | SUPPORTED | kaggle | 15,16,18,19 | vllm.LLM generate() in FP16, enforce_eager, TRITON_ATTN. Exact models run: Qwen/Qwen2.5-0.5B-Instruct (TP=1; TP=2, PP=2 and 2 replicas in the notebook 15 parallel stage), Qwen/Qwen2.5-3B-Instruct (TP=2, PP=2) and Qwen/Qwen2.5-Coder-7B-Instruct (TP=2, PP=2); max_model_len 2048. See MODEL_EVIDENCE; other checkpoints are not covered |
| `batch_generation` | SUPPORTED | kaggle | 09,11,14 | batches of 1..64 requests, 64 output tokens |
| `triton_attention` | SUPPORTED | kaggle | 05,13 | TRITON_ATTN selected and confirmed in the live VllmConfig; kernel_unified_attention compiled; no Flash library mapped in any process |
| `tensor_parallel_2` | SUPPORTED | kaggle | 05,06,13,15,18,19 | both GPUs hold a shard; slower than TP=1 for a model that fits one T4 (about 0.55x decode, earlier notebooks); generated with Qwen2.5-0.5B (nb 15), Qwen2.5-3B (nb 18) and Qwen2.5-Coder-7B (nb 19) |
| `pipeline_parallel_2` | SUPPORTED_WITH_CONDITIONS | kaggle | 08,13,15,18,19 | experimental in this SDK; both GPUs hold stages; about 0.6x of TP=1 decode for a model that fits one T4 (earlier notebooks); generated with Qwen2.5-0.5B (nb 15), Qwen2.5-3B (nb 18) and Qwen2.5-Coder-7B (nb 19) |
| `replica_parallel` | SUPPORTED_WITH_CONDITIONS | kaggle | 09,14,15 | SDK feature (ReplicaPool): one TP=1 engine per GPU, both receive work; no throughput gain over one engine up to batch 64; also the notebook 15 parallel stage (two workers on two GPUs, both received work, clean close) |
| `gpu_telemetry` | SUPPORTED | kaggle | 09,11,14 | nvidia-smi sampling during benchmarks |
| `topology_discovery` | SUPPORTED | kaggle | 01,07 | PHB; CUDA peer access measured per session (true in one Kaggle VM, false in another) |
| `nccl_collectives` | SUPPORTED | kaggle | 04 | all_reduce/all_gather/reduce_scatter/broadcast on both ranks |
| `sampling_temperature_top_k_top_p` | SUPPORTED_WITH_CONDITIONS | kaggle-kernel | — | kernels compiled during runs; only greedy (temperature 0) output was asserted |
| `logprobs` | SUPPORTED_WITH_CONDITIONS | kaggle-kernel | — | logprob / prompt-logprob kernels compiled; output not asserted |
| `penalties_min_p_bad_words_logit_bias` | SUPPORTED_WITH_CONDITIONS | kaggle-kernel | — | kernels compiled; output not asserted |
| `chat_generation` | SUPPORTED | kaggle | 15,16,18,19 | LLM.chat and LLM.complete returned text on 2 x T4: notebook 15 engine stage (build 3), the notebook 16 quickstart, notebook 18 (0.5B TP=1; 3B TP=2 and PP=2) and notebook 19 (Coder-7B TP=2 and PP=2) |
| `openai_server` | SUPPORTED_WITH_CONDITIONS | kaggle | 15 | the real `vllm serve` launched by stokastisk.serving.ServerProcess was started and exercised on 2 x T4 (notebook 15, server stage, 21/21: /health, /v1/models, chat completions, SSE streaming, HTTP error mapping, structured output over HTTP, /metrics, restart on the same port, graceful SIGINT shutdown, GPU memory released). Conditions: exercised with Qwen/Qwen2.5-0.5B-Instruct, TP=1 only; Kaggle's fastapi 0.141.1 is newer than upstream vLLM's <0.137 cap; the SageMaker /ping and /invocations routes were NOT exercised and are not claimed. Calling upstream `vllm serve` directly is unsupported (see direct_upstream_vllm) |
| `streaming` | SUPPORTED | kaggle | 15 | stream_chat: SSE through the real server yielded several non-empty deltas and ended with [DONE] (notebook 15, server stage, Qwen2.5-0.5B, TP=1); the truncated-stream error path is CPU-tested against a fake server only |
| `prometheus_metrics` | SUPPORTED_WITH_CONDITIONS | kaggle | 15 | GET /metrics on the running server exposed vllm: samples (notebook 15, server stage). Only the presence of those samples was asserted, not their values; the fetch_metrics text parser is CPU-tested |
| `structured_outputs` | SUPPORTED_WITH_CONDITIONS | kaggle | 15,16,18 | JSON-schema output with the pinned xgrammar 0.2.3 backend (never 'auto'): notebook 15 engine stage (every generate_structured output satisfied the bounded schema; not-JSON, wrong type, missing key, extra key and truncated outputs were each detected; a schema xgrammar does not support was rejected with no fallback; xgrammar's native library was mapped and outlines_core's was not), the server stage (structured output over HTTP) and notebooks 16 and 18 (generate_structured). Conditions: bound open-ended schema fields (a grammar cannot make a model stop); acceptance model Qwen2.5-0.5B |
| `direct_upstream_vllm` | UNSUPPORTED_ENTRY_POINT | static | — | calling vllm.LLM or `vllm serve` directly bypasses the SDK's dtype, NCCL, structured-output and backend checks and is NOT a supported Stokastisk entry point (use stokastisk.LLM / ServerProcess / ReplicaPool). The Flash-free tree makes upstream auto-selection resolve to TRITON_ATTN on SM75 (CPU-evaluated), and FlashAttention/FlashInfer/FlashMLA are refused with ProhibitedBackendError, but only the SDK paths are verified |
| `prefix_caching` | UNKNOWN_REQUIRES_EVIDENCE | static | — | no run asserts it |
| `chunked_prefill` | UNKNOWN_REQUIRES_EVIDENCE | static | — | no run asserts it |
| `cuda_graphs` | UNKNOWN_REQUIRES_EVIDENCE | static | — | every Kaggle run used enforce_eager=True (the SDK default); graph capture on T4 is unverified |
| `custom_all_reduce` | UNKNOWN_REQUIRES_EVIDENCE | static | — | disabled by the SDK default; needs CUDA peer access, which differs between Kaggle VMs |
| `lora` | UNKNOWN_REQUIRES_EVIDENCE | static | — | present in the source; no run |
| `multimodal` | UNKNOWN_REQUIRES_EVIDENCE | static | — | present in the source; no run |
| `embedding_pooling_reranking` | UNKNOWN_REQUIRES_EVIDENCE | static | — | present in the source; no run |
| `speculative_decoding` | UNKNOWN_REQUIRES_EVIDENCE | static | — | present in the source; no run |
| `mla_models_deepseek` | UNKNOWN_REQUIRES_EVIDENCE | static | — | FlashMLA/FlashInfer-MLA backends are excluded; other MLA backends were not run |
| `worker_failure_recovery` | SUPPORTED_WITH_CONDITIONS | cpu-tests | — | ReplicaPool restarts/retries covered by CPU tests with fake workers only |
| `bfloat16` | NOT_SUPPORTED_SM75 | static | — | vllm/platforms/cuda.py rejects bfloat16 below compute capability 8.0; use dtype='half' |
| `flash_attention` | EXCLUDED_FLASH_DEPENDENCY | static | — | FlashAttention is not shipped and not allowed |
| `flashinfer` | EXCLUDED_FLASH_DEPENDENCY | static | — | FlashInfer attention, sampler and all-reduce fusion are not shipped |
| `flashmla` | EXCLUDED_FLASH_DEPENDENCY | static | — | FlashMLA is not shipped |
| `allreduce_rms_fusion` | EXCLUDED_FLASH_DEPENDENCY | static | — | needs FlashInfer comm; pass_manager raises a clear error if requested |
| `fp8_mxfp4_nvfp4_native` | NOT_SUPPORTED_SM75 | static | — | native FP8/FP4 kernels need SM89/SM90/SM100 (see check_quantization for the per-method minimum declared in the source) |
| `expert_parallel_all2all` | NOT_SUPPORTED_SM75 | static | — | DeepEP / NIXL / FlashInfer all2all backends are not shipped or need newer GPUs |
| `qwen_omni_models` | NEEDS_ADAPTATION | static | — | qwen2_5_omni_thinker was removed from the T4 wheel by the module-level flash-import rule (its flash_attn import is guarded); restoring it needs a new vLLM wheel |
| `l4_sm89` | NOT_IMPLEMENTED | static | — | planned only; no SM89 binaries, no support claim |

## Models that were run

| Model id | Architecture | Dtype | Max tested `max_model_len` | Modes | Evidence | Limitations |
|---|---|---|---|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | Qwen2ForCausalLM | half | 2048 | TP=1, TP=2, PP=2, replicas=2 | notebook 15 (build 3 acceptance): engine 44/44 and server 21/21 at TP=1; parallel 23/23 with TP=2, PP=2 and two replicas; notebook 16 (quickstart script, TP=1); notebook 18 (plain cells, TP=1) | FP16, eager execution, max_model_len 2048 only; TP=2, PP=2 and replicas are slower than TP=1 for a model this small |
| `Qwen/Qwen2.5-3B-Instruct` | Qwen2ForCausalLM | half | 2048 | TP=2, PP=2 | notebook 18 (build 3): completion and chat with TP=2, and with PP=2 | TP=1 was not tested for this model; free memory during inference: about 1.09 GiB per GPU at TP=2, 1.96 and 1.48 GiB at PP=2 (gpu_memory_utilization 0.88); PP=2 is experimental |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | Qwen2ForCausalLM | half | 2048 | TP=2, PP=2 | notebook 19 (build 3): original unquantized checkpoint (BF16 weights cast to FP16); one Python-coding chat answer with TP=2 and one with PP=2 | does not fit one T4 in FP16, so TP=1 is not an option; memory is tight: about 1.34 GiB free per GPU during inference at TP=2, 2.42 and 1.93 GiB at PP=2 (max_model_len 2048, gpu_memory_utilization 0.88); a single prompt per mode; this is not a throughput measurement and does not show that other 7B models fit; PP=2 is experimental |

Any other checkpoint — including other `Qwen2ForCausalLM` models — is `UNKNOWN_REQUIRES_EVIDENCE` in `check_model()`; see [Checks and diagnostics](diagnostics.md).
