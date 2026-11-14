# API reference

Every Python module of the `stokastisk` package and its public names. Generated from the source by `scripts/release/build_docs_site.py`.

Names exported at the top level (`import stokastisk`) are in `stokastisk.__all__`; submodule names are imported as shown, e.g. `from stokastisk.serving import ServerProcess`.

## `stokastisk`

Stokastisk: dual-Tesla-T4 (SM75, CUDA 13.0) SDK around the bundled vLLM 0.25.1 engine.

## `stokastisk.benchmarks`

Reproducible dual-GPU benchmarks (import lazily: they need vLLM / CUDA only when run).

## `stokastisk.benchmarks.baseline`

Shared engine benchmark: identical workload for every strategy, ALL runs recorded (median/p95/min/max, never only the best).

| Name | Summary |
|---|---|
| <code>class Workload</code> |  |
| <code>&nbsp;&nbsp;Workload.prompts(self)</code> |  |
| <code>def measure(llm, workload: Workload, sampler_interval: float=0.5)</code> | Time ``workload`` on an already-loaded engine. Warm-up runs are discarded; every measured repeat is kept. |
| <code>def run_engine(config, model: str, workloads: 'Workload &#124; list[Workload]', *, label: str &#124; None=None, sampler_interval: float=0.5)</code> | Load ONE in-process stokastisk.LLM and measure every workload on it. Call from a ``__main__``-guarded script (vLLM spawns). |

## `stokastisk.benchmarks.communication`

Transfer-size sweep: host-staged (torch / native C++) vs NCCL send/recv. NCCL runs inside a 2-rank torchrun job.

| Name | Summary |
|---|---|
| <code>def bench_host_staged(src: int=0, dst: int=1, *, sizes=SIZES, iters: int=20, warmup: int=5, chunk_bytes: int=4 << 20, n_buffers: int=2, native=None)</code> | Verifies the payload (byte-exact) on every size, then times ``iters`` transfers. ``native`` = NativeHostStaged class to use C++ instead. |
| <code>def bench_nccl(rank: int, *, sizes=SIZES, iters: int=20, warmup: int=5)</code> | Inside an initialised 2-rank NCCL group. Each rank returns its own view; rank 1 (receiver) completion time is the transfer latency. |
| <code>def compare(host_staged: dict, nccl: dict)</code> | Per size: ratio of median latencies and which transport wins. Decision rule: host-staged is integrated nowhere unless it wins. |

## `stokastisk.benchmarks.gpu_balance`

Compute / communication / idle accounting and imbalance for replica runs.

| Name | Summary |
|---|---|
| <code>def from_replica_record(rec: dict[str, Any], workload: str)</code> |  |

## `stokastisk.benchmarks.pipeline_parallel`

PP=2 (experimental). Stage-level compute/communication times are NOT exposed by vLLM's public API: they are reported as NOT_MEASURED

| Name | Summary |
|---|---|
| <code>def config_for(**kw)</code> |  |
| <code>def run(model: str, workloads, **kw)</code> |  |

## `stokastisk.benchmarks.replica_parallel`

Independent TP=1 replicas vs one engine. Offline batch generation reports BATCH completion latency (individual request completion

| Name | Summary |
|---|---|
| <code>def run(model: str, workloads, devices=(0, 1), **kw)</code> |  |

## `stokastisk.benchmarks.report`

Combine benchmark records into one JSON + a markdown table. Missing strategies are listed as NOT_RUN, never omitted silently.

| Name | Summary |
|---|---|
| <code>def comparison_table(records: dict[str, dict[str, Any]], workload: str)</code> |  |
| <code>def relative_to(records: dict[str, dict[str, Any]], workload: str, base: str='tp1')</code> |  |

## `stokastisk.benchmarks.tensor_parallel`

TP=1 vs TP=2 on an identical workload.

| Name | Summary |
|---|---|
| <code>def config_for(tp: int, **kw)</code> |  |
| <code>def run(model: str, workloads, tp: int, **kw)</code> |  |

## `stokastisk.communication`

EXPERIMENTAL communication layer: host-staged transport, NCCL timing helpers, transport registry. Not used by the vLLM engine.

## `stokastisk.communication.cuda_transport`

EXPERIMENTAL host-staged GPU->GPU transfer for platforms without CUDA peer access.

| Name | Summary |
|---|---|
| <code>class HostStagedTransport</code> |  |
| <code>&nbsp;&nbsp;HostStagedTransport.__init__(self, src_device: int, dst_device: int, *, chunk_bytes: int=4 << 20, n_buffers: int=2)</code> |  |
| <code>&nbsp;&nbsp;HostStagedTransport.transfer(self, src_tensor, dst_tensor)</code> | Copy ``src_tensor`` (on src device) into ``dst_tensor`` (on dst device); returns timing. Blocks until complete. |

## `stokastisk.communication.native`

Optional ctypes loader for ``native/host_staged_xfer.cpp`` (C++ / CUDA Runtime). Built on demand with nvcc; absence is not an error.

| Name | Summary |
|---|---|
| <code>def build_native(src: str &#124; os.PathLike, out_dir: str &#124; os.PathLike, arch: str='sm_75')</code> |  |
| <code>def load_native(path: str &#124; os.PathLike &#124; None=None)</code> |  |
| <code>class NativeHostStaged</code> |  |
| <code>&nbsp;&nbsp;NativeHostStaged.__init__(self, src_device: int, dst_device: int, *, chunk_bytes: int=4 << 20, n_buffers: int=2, lib=None)</code> |  |
| <code>&nbsp;&nbsp;NativeHostStaged.transfer(self, src_tensor, dst_tensor)</code> |  |
| <code>&nbsp;&nbsp;NativeHostStaged.close(self)</code> |  |

## `stokastisk.communication.nccl`

NCCL timing helpers for a torchrun/``torch.distributed`` process group (2 ranks). NCCL is the reference transport; it is never replaced.

| Name | Summary |
|---|---|
| <code>def time_p2p(nbytes: int, rank: int, *, iters: int=20, warmup: int=5)</code> | rank 0 sends ``nbytes`` to rank 1 (NCCL send/recv). Returns per-iteration seconds measured on this rank (barrier-aligned). |
| <code>def time_allreduce(nbytes: int, *, iters: int=20, warmup: int=5)</code> |  |

## `stokastisk.communication.pinned_memory`

A small ring of page-locked host buffers (cudaHostAlloc via PyTorch's pinned allocator).

| Name | Summary |
|---|---|
| <code>class PinnedBufferRing</code> |  |
| <code>&nbsp;&nbsp;PinnedBufferRing.__init__(self, n_buffers: int, nbytes: int)</code> |  |
| <code>&nbsp;&nbsp;PinnedBufferRing.host_bytes(self)</code> |  |

## `stokastisk.communication.synchronization`

Buffer ownership tracking for the host-staged pipeline.

| Name | Summary |
|---|---|
| <code>class BufferTicket</code> |  |
| <code>&nbsp;&nbsp;BufferTicket.__init__(self)</code> |  |

## `stokastisk.communication.transport_registry`

Registry of known transports with an honest status. Nothing is selected automatically.

| Name | Summary |
|---|---|
| <code>def list_transports()</code> |  |

## `stokastisk.compat`

Static Tesla T4 / SM75 / TRITON_ATTN compatibility inspection of the bundled vLLM 0.25.1.

| Name | Summary |
|---|---|
| <code>def feature_matrix()</code> | A copy of the feature table (safe to mutate). |
| <code>def get_feature(name: str)</code> |  |
| <code>def check_feature(name: str)</code> | Return the feature record, or raise if the feature cannot work on this build. UNKNOWN_REQUIRES_EVIDENCE is returned, not raised: it is untested, not forbidden. |
| <code>def check_dtype(dtype: str &#124; None, capability: tuple[int, int]=SM75)</code> | Reject an explicit bfloat16 below SM80 with the same reasoning as vllm/platforms/cuda.py, but before an engine is started. |
| <code>def check_quantization(method: str &#124; None, capability: tuple[int, int]=SM75)</code> | Compare a vLLM quantization method with the minimum capability declared in the source. Unknown methods are reported UNKNOWN_REQUIRES_EVIDENCE, never assumed to work. |
| <code>def read_registry(package_root: str &#124; pathlib.Path &#124; None=None)</code> | architecture name -> (module, class) from vllm/model_executor/models/registry.py, read with ast (no import). |
| <code>def uses_removed_backend(module_file: pathlib.Path, root: pathlib.Path)</code> | True if the architecture's OWN module imports names from the Flash-family placeholder module at import time, i.e. the model is built on a removed backend. |
| <code>def find_missing_imports(module_file: pathlib.Path, root: pathlib.Path, *, max_depth: int=6)</code> | Chains of unconditional vllm imports from ``module_file`` that end in a module absent from the shipped package. |
| <code>def check_model(model: str &#124; None=None, *, architectures: Iterable[str] &#124; None=None, package_root: str &#124; pathlib.Path &#124; None=None, capability: tuple[int, int]=SM75, dtype: str &#124; None='half', quantization: str &#124; None=None, mode: str &#124; None=None)</code> | Static, OFFLINE pre-flight for a model on the T4 build (nothing is downloaded and no GPU is touched). |

## `stokastisk.config`

Typed configuration with T4-safe defaults and environment overrides.

| Name | Summary |
|---|---|
| <code>class StokastiskConfig</code> |  |
| <code>&nbsp;&nbsp;StokastiskConfig.strategy(self)</code> | Effective strategy. ``None`` keeps the baseline meaning: TP=2 -> tensor_parallel, otherwise single_gpu (PP>1 -> pipeline_parallel). |
| <code>&nbsp;&nbsp;StokastiskConfig.world_size(self)</code> |  |
| <code>&nbsp;&nbsp;StokastiskConfig.is_experimental(self)</code> |  |
| <code>&nbsp;&nbsp;StokastiskConfig.validate(self)</code> |  |
| <code>&nbsp;&nbsp;StokastiskConfig.from_env(cls, prefix: str='STOKASTISK_')</code> |  |
| <code>&nbsp;&nbsp;StokastiskConfig.apply_environment(self)</code> | Export NCCL settings (never overriding values the user already set). |
| <code>&nbsp;&nbsp;StokastiskConfig.engine_kwargs(self)</code> |  |
| <code>&nbsp;&nbsp;StokastiskConfig.as_dict(self)</code> |  |

## `stokastisk.cuda`

Optional CUDA stream/event/memory helpers (PyTorch-backed). Not used by the vLLM engine path.

## `stokastisk.cuda.device`

Thin, optional CUDA device helpers on top of PyTorch. They NEVER touch vLLM's internal streams or contexts.

| Name | Summary |
|---|---|
| <code>def device_count()</code> |  |
| <code>def on_device(index: int)</code> |  |
| <code>def synchronize(index: int &#124; None=None)</code> |  |

## `stokastisk.cuda.events`

Event helpers: dependency edges between streams (also across devices) and GPU-side timing.

| Name | Summary |
|---|---|
| <code>def new_event(*, timing: bool=False)</code> |  |
| <code>def record(event, stream=None)</code> |  |
| <code>def make_wait(stream, event)</code> | cudaStreamWaitEvent: ``stream`` will not run later work until ``event`` completes (event may belong to another device). |
| <code>def elapsed_ms(start, end)</code> |  |

## `stokastisk.cuda.memory`

Memory helpers: free/total per device, pinned allocation, timed async copies.

| Name | Summary |
|---|---|
| <code>def mem_info(device: int)</code> |  |
| <code>def pinned_empty(nbytes: int)</code> |  |
| <code>def timed_copy(dst, src, stream)</code> | Async copy on ``stream`` timed with CUDA events on the stream's device. |

## `stokastisk.cuda.streams`

Stream helpers. Create your own streams; do not enqueue onto vLLM's streams from outside vLLM.

| Name | Summary |
|---|---|
| <code>def create_stream(device: int, *, priority: int=0)</code> |  |
| <code>def stream_synchronize(stream)</code> |  |

## `stokastisk.dependencies`

Feature -> third-party import names. The single registry that the SDK (`serving`, `engine`), the Kaggle installer and the unit tests share.

| Name | Summary |
|---|---|
| <code>def missing(names: Iterable[str])</code> | Import names that cannot be found (``find_spec`` returning None, or raising for a half-installed namespace). |
| <code>def required_imports(*profiles: str)</code> | Locked + baseline import names for the given profiles (order preserved, no duplicates). |
| <code>def missing_for(*profiles: str)</code> |  |

## `stokastisk.diagnostics`

Runtime diagnostics. torch is imported lazily so this module works on CPU-only hosts.

| Name | Summary |
|---|---|
| <code>def discover_gpus()</code> |  |
| <code>def peer_access_matrix()</code> |  |
| <code>def check_native_extensions()</code> | Import every bundled extension; value is 'ok' or the error text. |
| <code>def flash_attn_native_loaded()</code> | FlashAttention native modules present in this process (expected: none). |
| <code>def prohibited_report()</code> | Detect FlashAttention / FlashInfer / FlashMLA distributions, importable modules and shipped files (expected: all empty). |
| <code>def validate_environment(*, require_gpus: int=2, strict_torch: bool=True)</code> | Validate the runtime against the SDK build target (CUDA 13.0 / SM75 / torch 2.11 cu130). |
| <code>def report()</code> | Non-raising environment report (safe on CPU hosts). |

## `stokastisk.engine`

LLM facade over the bundled vLLM engine (Tesla T4 / SM75 / TRITON_ATTN / fp16).

| Name | Summary |
|---|---|
| <code>class LLM</code> | Thin wrapper: applies the T4 defaults (NCCL configured not to depend on peer access, TRITON_ATTN, pinned structured-output backend), then delegates to ``vllm.LLM``. |
| <code>&nbsp;&nbsp;LLM.__init__(self, model: str, *, config: StokastiskConfig &#124; None=None, **kwargs: Any)</code> |  |
| <code>&nbsp;&nbsp;LLM.selected_attention_backend(self)</code> | Backend recorded in the live engine's VllmConfig (config-level evidence; the worker's own log line and the Triton cache are the kernel-level evidence). |
| <code>&nbsp;&nbsp;LLM.verify_attention_backend(self)</code> |  |
| <code>&nbsp;&nbsp;LLM.structured_output_backend(self)</code> | The structured-output backend recorded in the live engine's VllmConfig (pinned to 'xgrammar'; never 'auto'). |
| <code>&nbsp;&nbsp;LLM.verify_structured_output_backend(self)</code> |  |
| <code>&nbsp;&nbsp;LLM.engine(self)</code> |  |
| <code>&nbsp;&nbsp;LLM.closed(self)</code> | True once :meth:`close` has run, or when the engine failed to start. |
| <code>&nbsp;&nbsp;LLM.generate(self, prompts, sampling_params=None, *, timeout: float &#124; None=None, **kwargs)</code> | Pass-through to ``vllm.LLM.generate``. An empty prompt list returns ``[]`` without touching the engine; non-str or empty prompts raise ``TypeError`` / |
| <code>&nbsp;&nbsp;LLM.chat(self, messages, sampling_params=None, *, timeout: float &#124; None=None, **kwargs)</code> | Pass-through to ``vllm.LLM.chat`` (the chat template is applied by vLLM). ``messages`` is a list of ``{"role", "content"}`` dicts or a list of such lists. |
| <code>&nbsp;&nbsp;LLM.complete(self, prompt: str, *, max_tokens: int=64, temperature: float=0.0, timeout: float &#124; None=None)</code> | Generate one completion for ``prompt`` and return its text. |
| <code>&nbsp;&nbsp;LLM.generate_structured(self, prompts, schema, *, max_tokens: int=64, temperature: float=0.0, timeout: float &#124; None=None)</code> | Generate JSON constrained to ``schema`` by the engine's pinned structured-output backend, then validate every output against the schema. |
| <code>&nbsp;&nbsp;LLM.close(self)</code> | Shut the engine (and, for TP>1, its worker processes) down explicitly, then release cached CUDA memory. Idempotent. |

## `stokastisk.errors`

Stokastisk exception hierarchy.

| Name | Summary |
|---|---|
| <code>class StokastiskError</code> | Base class for all SDK errors. |
| <code>class EnvironmentValidationError</code> | The runtime does not match the SDK build target. |
| <code>class GPUDiscoveryError</code> | Required GPUs are missing or have the wrong architecture. |
| <code>class EngineError</code> | vLLM engine failed to initialise or generate. |
| <code>class ConfigurationError</code> | Invalid Stokastisk configuration (e.g. a FlashAttention backend was requested). |
| <code>class AttentionBackendError</code> | The engine did not end up with the requested Triton attention backend. |
| <code>class EngineClosedError</code> | The engine was closed (explicitly, or because it never started); create a new ``stokastisk.LLM``. |
| <code>class EngineTimeoutError</code> | A generation call did not finish within the caller's timeout. The request may still be running inside the engine. |
| <code>class EngineCloseError</code> | The engine shut down with an error. The SDK still released everything it could; ``__cause__`` carries the original failure. |
| <code>class StructuredOutputError</code> | A structured-output request failed, or its output violated the schema that was requested. |
| <code>class MissingDependencyError</code> | A third-party package needed by the requested feature is not importable. ``missing`` lists the import names; ``feature`` names the feature. |
| <code>&nbsp;&nbsp;MissingDependencyError.__init__(self, feature: str, missing, hint: str='')</code> |  |
| <code>class ServerError</code> | Base class for failures of the OpenAI-compatible server process. |
| <code>class ServerStartupError</code> | The server process exited, or did not become ready, before ``wait_ready`` returned. |
| <code>&nbsp;&nbsp;ServerStartupError.__init__(self, reason: str, *, exit_code, command, log_path, log_excerpt: str, timeout=None, waited=None)</code> |  |
| <code>class ServerHTTPError</code> | The server answered an API request with an error status or an unparsable body. ``status`` is the HTTP status (None for a transport failure). |
| <code>&nbsp;&nbsp;ServerHTTPError.__init__(self, message: str, *, status=None, body: str='')</code> |  |
| <code>class ServerShutdownError</code> | The server process could not be stopped cleanly (it had to be killed, or children survived). |
| <code>class UnsupportedParallelismError</code> | The requested TP/PP/replica combination is not valid for the two-T4 target. |
| <code>class InsufficientMemoryError</code> | The model (weights + KV cache) does not fit the requested GPU layout. |
| <code>class TopologyError</code> | GPU topology could not be determined or does not match the requested layout. |
| <code>class ReplicaError</code> | A replica worker failed to start, died, or returned an error. |
| <code>class StokastiskCompatibilityError</code> | A model, dtype, quantization method or vLLM feature cannot work on the Tesla T4 / SM75 / TRITON_ATTN build. |

## `stokastisk.limits`

Hard resource limits of the Kaggle target and helpers that enforce / report them.

| Name | Summary |
|---|---|
| <code>class KaggleLimits</code> |  |
| <code>def available_cpus()</code> | CPUs this process may really use (affinity-aware; equals `nproc`), never more than the Kaggle budget of 4 for planning. |
| <code>def cpu_budget()</code> |  |
| <code>def threads_per_worker(tensor_parallel_size: int)</code> | Split the CPU budget across the TP worker processes (4 cores, TP=2 -> 2 threads each) so CPU-side libraries never oversubscribe. |
| <code>def cpu_thread_env(tensor_parallel_size: int)</code> |  |
| <code>def system_ram_gib()</code> |  |
| <code>def dir_size_bytes(path: str &#124; os.PathLike)</code> |  |
| <code>def working_dir_status(path: str &#124; None=None)</code> | Usage of the (quota-limited) output directory against the 20 GiB Kaggle limit. |
| <code>def check_resources()</code> | Compare the host with the Kaggle limits. Never raises. Statuses: PASS (within the limit) / WARNING (host differs from the Kaggle design). |

## `stokastisk.nccl`

NCCL validation helpers (launch with ``python -m torch.distributed.run --standalone --nproc_per_node=2``).

| Name | Summary |
|---|---|
| <code>def run_collective_selftest(expected_world: int=2)</code> | Run all-reduce / all-gather / reduce-scatter / broadcast / barrier over the torchrun world. Never raises. |

## `stokastisk.parallel`

Application-level parallelism: independent GPU replicas, token-aware scheduling, rule-based recommendations.

## `stokastisk.parallel.memory`

Weight / KV-cache memory estimation from a model's config.json (an ESTIMATE, labelled as such everywhere it is shown).

| Name | Summary |
|---|---|
| <code>class ModelSpec</code> |  |
| <code>&nbsp;&nbsp;ModelSpec.head_dim_(self)</code> |  |
| <code>&nbsp;&nbsp;ModelSpec.params(self)</code> |  |
| <code>&nbsp;&nbsp;ModelSpec.weight_mib(self)</code> |  |
| <code>&nbsp;&nbsp;ModelSpec.kv_mib(self, tokens: int)</code> |  |
| <code>&nbsp;&nbsp;ModelSpec.unsupported_reason(self)</code> |  |
| <code>&nbsp;&nbsp;ModelSpec.to_dict(self)</code> |  |
| <code>&nbsp;&nbsp;ModelSpec.from_hf_config(cls, name: str, cfg: dict[str, Any], **kw)</code> |  |
| <code>def load_model_spec(model: 'str &#124; ModelSpec', *, local_files_only: bool=False)</code> | ModelSpec from a ModelSpec, a local model directory, or a Hub repo id (downloads ONLY config.json). |

## `stokastisk.parallel.queue`

Request records and a deterministic FIFO queue.

| Name | Summary |
|---|---|
| <code>def estimate_tokens(text: str)</code> | Cheap, deterministic input-token estimate (~4 characters per token). Replace with a tokenizer when exactness matters. |
| <code>class Request</code> |  |
| <code>&nbsp;&nbsp;Request.cost(self)</code> | Token-aware work estimate: prompt tokens (prefill) + requested output tokens (decode). |
| <code>class RequestQueue</code> |  |
| <code>&nbsp;&nbsp;RequestQueue.__init__(self, maxsize: int &#124; None=None)</code> |  |
| <code>&nbsp;&nbsp;RequestQueue.put(self, r: Request, *, front: bool=False)</code> |  |
| <code>&nbsp;&nbsp;RequestQueue.get(self)</code> |  |
| <code>&nbsp;&nbsp;RequestQueue.pending_tokens(self)</code> |  |

## `stokastisk.parallel.recommender`

Rule-based, deterministic parallelism recommendation. It only RECOMMENDS: the caller decides (``Recommendation.to_config()``).

| Name | Summary |
|---|---|
| <code>class Recommendation</code> |  |
| <code>&nbsp;&nbsp;Recommendation.to_dict(self)</code> |  |
| <code>&nbsp;&nbsp;Recommendation.to_config(self, **kw)</code> | A validated StokastiskConfig for this recommendation. Raises for non-runnable outcomes. |
| <code>def default_recommendation(topology)</code> | Model-independent summary used by ``Topology.recommended_parallelism``. |
| <code>def recommend_parallelism(model, topology=None, workload: str='latency', *, max_model_len: int=4096, gpu_memory_utilization: float=0.88, min_concurrent_seqs: int=1)</code> |  |

## `stokastisk.parallel.replica`

ReplicaPool: independent TP=1 vLLM engines, one per GPU, behind a deterministic token-aware scheduler.

| Name | Summary |
|---|---|
| <code>class ReplicaPool</code> |  |
| <code>&nbsp;&nbsp;ReplicaPool.__init__(self, model: str, devices: list[int] &#124; tuple[int, ...]=(0, 1), *, config: StokastiskConfig &#124; None=None, ready_timeout: float=900.0, max_attempts: int=2, engine_kwargs: dict[str, Any] &#124; None=None)</code> |  |
| <code>&nbsp;&nbsp;ReplicaPool.restart_dead(self, timeout: float=900.0)</code> | Restart every worker that has exited. Never called implicitly. |
| <code>&nbsp;&nbsp;ReplicaPool.generate(self, prompts: list[str], *, max_tokens: int=64, temperature: float=0.0, timeout: float=1800.0)</code> |  |
| <code>&nbsp;&nbsp;ReplicaPool.close(self, timeout: float=60.0)</code> | Stop every worker. Always completes the whole cleanup (shutdown message, join, terminate, kill, queue close); then, if any worker had to be terminated, exited |

## `stokastisk.parallel.routing`

Weighted least-loaded routing (deterministic; ties go to the lowest worker id).

| Name | Summary |
|---|---|
| <code>class WorkerLoad</code> |  |
| <code>&nbsp;&nbsp;WorkerLoad.available(self)</code> |  |
| <code>def effective_speed(w: WorkerLoad, reference_speed: float &#124; None=None)</code> | Speed used for routing. An observed speed is tokens/s; the default 1.0 is unit-less, so an unobserved worker is |
| <code>def projected_load(w: WorkerLoad, extra_tokens: int, reference_speed: float &#124; None=None)</code> | Estimated time-to-drain (arbitrary units): queued tokens + the new work, divided by the worker's speed. |
| <code>def choose_worker(workers: list[WorkerLoad], cost_tokens: int)</code> |  |

## `stokastisk.parallel.scheduler`

Deterministic, token-aware, weighted least-loaded scheduler. Pure Python: no CUDA, no processes (the pool owns those).

| Name | Summary |
|---|---|
| <code>class Decision</code> |  |
| <code>class Scheduler</code> |  |
| <code>&nbsp;&nbsp;Scheduler.__init__(self, workers: list[WorkerLoad], *, max_attempts: int=2, speed_alpha: float=0.3)</code> |  |
| <code>&nbsp;&nbsp;Scheduler.assign(self, req: Request)</code> |  |
| <code>&nbsp;&nbsp;Scheduler.assign_batch(self, reqs: list[Request], *, longest_first: bool=True)</code> | Greedy LPT (longest-processing-time-first) placement; deterministic. Returns worker id -> requests; unplaceable ones under key -1. |
| <code>&nbsp;&nbsp;Scheduler.complete(self, req_id: int, *, seconds: float &#124; None=None)</code> |  |
| <code>&nbsp;&nbsp;Scheduler.mark_dead(self, worker_id: int)</code> | Mark a worker dead and return its in-flight requests that may still be retried elsewhere. |
| <code>&nbsp;&nbsp;Scheduler.revive(self, worker_id: int)</code> |  |

## `stokastisk.parallel.worker`

One replica worker process: one physical GPU, one TP=1 vLLM engine, one full copy of the weights.

| Name | Summary |
|---|---|
| <code>def gpu_identity(physical_index: int)</code> | uuid + memory.used of a PHYSICAL GPU via nvidia-smi (independent of CUDA_VISIBLE_DEVICES, no CUDA context). |
| <code>def worker_main(worker_id: int, device: int, model: str, config_dict: dict, engine_kwargs: dict, in_q, out_q)</code> |  |

## `stokastisk.prohibited`

Policy and scanners for the Stokastisk "no FlashAttention / FlashInfer / FlashMLA" rule.

| Name | Summary |
|---|---|
| <code>def is_prohibited_path(path: str)</code> | True if any component of ``path`` (a ``/``-separated path inside a wheel or tree) names a Flash-family module or package. |
| <code>def is_binary(path: str)</code> |  |
| <code>def native_markers(data: bytes)</code> | Flash-family markers present in a native object's bytes. |
| <code>def scan_members(names: Iterable[str], read)</code> | Scan a set of archive/tree members. ``read(name) -> bytes`` is only called for files whose content must be inspected. |
| <code>def scan_wheel(path: str &#124; pathlib.Path)</code> |  |
| <code>def scan_tree(root: str &#124; pathlib.Path)</code> |  |
| <code>def flash_in_maps(lines: Iterable[str])</code> | Flash-family shared objects in the lines of ``/proc/<pid>/maps`` (path column only). |
| <code>def loaded_flash_libraries(pid: int &#124; str='self')</code> | Flash-family shared objects mapped by process ``pid`` (empty list = none; unreadable maps raise OSError, never silently pass). |

## `stokastisk.serving`

OpenAI-compatible serving for the T4 build: launch the bundled ``vllm serve`` with the Stokastisk T4 defaults, wait for /health, read /metrics, stream tokens.

| Name | Summary |
|---|---|
| <code>def check_server_requirements(names: Sequence[str] &#124; None=None)</code> | Return the import names that cannot be found (empty list = nothing missing). Uses find_spec only (no import side effects). |
| <code>def validate_model_id(model: Any)</code> | A Hugging Face id (``org/name``) or a path. Rejects empty strings, whitespace/control characters and anything that vLLM's argument parser would read as an option. |
| <code>def validate_host(host: Any)</code> |  |
| <code>def validate_port(port: Any)</code> |  |
| <code>def port_is_free(host: str, port: int)</code> | True if nothing is listening on (host, port) and the address can be bound. |
| <code>def find_free_port(host: str='127.0.0.1')</code> |  |
| <code>def build_serve_command(model: str, config: StokastiskConfig &#124; None=None, *, host: str='127.0.0.1', port: int=8000, served_model_name: str &#124; None=None, quantization: str &#124; None=None, extra_args: Sequence[str]=(), python: str &#124; None=None)</code> | The exact argv that ServerProcess runs. Pure function: no process, no import of vllm. |
| <code>class ServerProcess</code> | Supervise one ``vllm serve`` child. Use as a context manager. Never started implicitly, never left running by ``with``. |
| <code>&nbsp;&nbsp;ServerProcess.__init__(self, model: str, config: StokastiskConfig &#124; None=None, *, host: str='127.0.0.1', port: int=8000, served_model_name: str &#124; None=None, quantization: str &#124; None=None, extra_args: Sequence[str]=(), cuda_visible_devices: str &#124; None=None, log_path: str &#124; None=None, python: str &#124; None=None)</code> |  |
| <code>&nbsp;&nbsp;ServerProcess.base_url(self)</code> |  |
| <code>&nbsp;&nbsp;ServerProcess.pid(self)</code> |  |
| <code>&nbsp;&nbsp;ServerProcess.running(self)</code> |  |
| <code>&nbsp;&nbsp;ServerProcess.log_tail(self, max_lines: int=40)</code> |  |
| <code>&nbsp;&nbsp;ServerProcess.start(self)</code> |  |
| <code>&nbsp;&nbsp;ServerProcess.health(self, timeout: float=5.0)</code> | True if ``GET /health`` answers 200 right now. |
| <code>&nbsp;&nbsp;ServerProcess.wait_ready(self, timeout: float=600.0, interval: float=2.0)</code> | Poll ``GET /health`` until HTTP 200. Raises :class:`ServerStartupError` if the child exits first or the timeout passes (with the exit code, the command, |
| <code>&nbsp;&nbsp;ServerProcess.start_and_wait(self, timeout: float=600.0)</code> |  |
| <code>&nbsp;&nbsp;ServerProcess.restart(self, timeout: float=600.0)</code> | Stop the server (graceful first) and start a fresh one on the same port; returns once it is healthy. |
| <code>&nbsp;&nbsp;ServerProcess.stop(self, grace: float=30.0)</code> | SIGINT the process group (vLLM's own graceful shutdown), then SIGTERM, then SIGKILL; then make sure no process of the group survives. |
| <code>def chat(base_url: str, model: str, messages: list[dict[str, str]], *, max_tokens: int=64, temperature: float=0.0, timeout: float=120.0)</code> | Non-streaming /v1/chat/completions; returns the assistant text. |
| <code>def stream_chat(base_url: str, model: str, messages: list[dict[str, str]], *, max_tokens: int=64, temperature: float=0.0, timeout: float=120.0)</code> | Yield content deltas from /v1/chat/completions with stream=true (server-sent events: ``data: {...}`` lines, terminated by ``data: [DONE]``). |
| <code>def list_models(base_url: str, timeout: float=10.0)</code> |  |
| <code>def fetch_metrics(base_url: str, timeout: float=10.0)</code> | Parse the Prometheus text at /metrics into {sample_name{labels}: value} (comment lines skipped; non-numeric samples ignored). |

## `stokastisk.structured`

Structured (JSON-schema) output helpers: pure validation that needs no GPU, plus the sampling-parameter builder.

| Name | Summary |
|---|---|
| <code>def check_schema(schema: Any)</code> | Return ``schema`` (a dict, or a JSON string parsed to one) if it is a valid JSON Schema, else raise :class:`StructuredOutputError`. |
| <code>def validate_json_output(text: str, schema: Any)</code> | Parse ``text`` as JSON and validate it against ``schema``. Returns the parsed object; raises :class:`StructuredOutputError` (never returns invalid data). |
| <code>def json_schema_params(schema: Any, *, max_tokens: int=64, temperature: float=0.0)</code> | ``vllm.SamplingParams`` that constrain the output to ``schema``. The backend is the engine's (pinned) one; a request cannot choose another. |

## `stokastisk.telemetry`

Optional GPU / communication / worker telemetry. Every probe degrades to an explicit 'unavailable'.

## `stokastisk.telemetry.communication_profiler`

Wall-clock + CUDA-event timing for communication/compute segments. Segments are recorded; nothing is inferred.

| Name | Summary |
|---|---|
| <code>class SegmentTimer</code> |  |
| <code>&nbsp;&nbsp;SegmentTimer.__init__(self)</code> |  |
| <code>&nbsp;&nbsp;SegmentTimer.wall(self, name: str)</code> |  |
| <code>&nbsp;&nbsp;SegmentTimer.cuda(self, name: str, stream=None)</code> | GPU-side duration via CUDA events (synchronises the stream when the block exits). |
| <code>&nbsp;&nbsp;SegmentTimer.totals(self)</code> |  |
| <code>&nbsp;&nbsp;SegmentTimer.report(self)</code> |  |

## `stokastisk.telemetry.cuda_profiler`

Nsight availability probe. Nsight is never required: an unusable profiler is an explicit SKIPPED.

| Name | Summary |
|---|---|
| <code>def nsight_status()</code> |  |

## `stokastisk.telemetry.metrics`

Pure statistics + latency helpers (no GPU needed).

| Name | Summary |
|---|---|
| <code>def percentile(values: Sequence[float], q: float)</code> | Linear-interpolated percentile (q in [0,100]); raises on empty input. |
| <code>def summarize(values: Sequence[float])</code> | min / median / p95 / max / mean over ALL runs (never only the best one). |
| <code>def imbalance_percent(t0: float, t1: float)</code> | abs(T0 - T1) / max(T0, T1) * 100. 0 = perfectly balanced; 0.0 when both are 0. |
| <code>def throughput(tokens: int, seconds: float)</code> |  |
| <code>def token_latencies(first_token_s: float &#124; None, total_s: float, n_tokens: int)</code> | TTFT and mean inter-token latency from a single request's timing. |

## `stokastisk.telemetry.nvml`

GPU sampling: NVML if present, else nvidia-smi, else an explicit 'unavailable'. Counters Kaggle does not expose stay ``None``.

| Name | Summary |
|---|---|
| <code>def sample_once()</code> |  |
| <code>def pcie_throughput_kib_s()</code> | PCIe RX/TX in KiB/s per GPU via NVML, or None when NVML / the counter is unavailable. |
| <code>class GpuSampler</code> | Background sampler: ``with GpuSampler(0.5) as s: ...; s.summary()``. Overhead = one nvidia-smi call per interval. |
| <code>&nbsp;&nbsp;GpuSampler.__init__(self, interval: float=0.5)</code> |  |
| <code>&nbsp;&nbsp;GpuSampler.summary(self)</code> |  |

## `stokastisk.telemetry.report`

Local JSON evidence + optional OpenTelemetry spans. All metrics are available as JSON without any network telemetry.

| Name | Summary |
|---|---|
| <code>def write_json(path: str &#124; pathlib.Path, obj: Any)</code> |  |
| <code>def span(name: str, **attrs: Any)</code> | OpenTelemetry span if ``opentelemetry`` is installed AND a provider is configured; otherwise a no-op. Never required. |
| <code>def provenance()</code> | Versions + identity every benchmark records. |

## `stokastisk.telemetry.worker_metrics`

Per-worker accounting used by the balance benchmark: compute vs communication vs idle, and queue wait.

| Name | Summary |
|---|---|
| <code>class WorkerMetrics</code> |  |
| <code>&nbsp;&nbsp;WorkerMetrics.idle_s(self)</code> |  |
| <code>&nbsp;&nbsp;WorkerMetrics.to_dict(self)</code> |  |
| <code>def balance_report(w0: WorkerMetrics, w1: WorkerMetrics)</code> | Imbalance on compute time and on total busy time, reported separately (utilisation alone does not prove balance). |

## `stokastisk.topology`

GPU topology discovery (torch / NVML / nvidia-smi / sysfs; all but torch optional).

## `stokastisk.topology.discovery`

discover_topology(): torch -> NVML -> nvidia-smi -> sysfs, every layer optional. Nothing is hardcoded to Kaggle's PHB.

| Name | Summary |
|---|---|
| <code>def discover_topology(*, with_recommendation: bool=True)</code> |  |

## `stokastisk.topology.gpu_capabilities`

Basic GPU properties from PyTorch (the primary provider). cuda-python is an optional, never-required refinement.

| Name | Summary |
|---|---|
| <code>def torch_devices()</code> |  |
| <code>def cuda_python_version()</code> |  |

## `stokastisk.topology.models`

Plain, JSON-serialisable topology records. A value that could not be measured is ``None`` — never invented.

| Name | Summary |
|---|---|
| <code>class DeviceInfo</code> |  |
| <code>class PeerAccess</code> | CUDA peer access per ordered pair. ``matrix[src][dst]`` is ``torch.cuda.can_device_access_peer(src, dst)`` (None = not measured). |
| <code>&nbsp;&nbsp;PeerAccess.can_access(self, src: int, dst: int)</code> |  |
| <code>&nbsp;&nbsp;PeerAccess.all_pairs_available(self)</code> |  |
| <code>&nbsp;&nbsp;PeerAccess.any_pair_available(self)</code> |  |
| <code>class Topology</code> |  |
| <code>&nbsp;&nbsp;Topology.gpu_count(self)</code> |  |
| <code>&nbsp;&nbsp;Topology.pair_link(self, a: int, b: int)</code> |  |
| <code>&nbsp;&nbsp;Topology.to_dict(self)</code> |  |

## `stokastisk.topology.nvml`

Optional NVML (``pynvml`` / ``nvidia-ml-py``) and ``nvidia-smi`` helpers. Neither is required: every function degrades to ``None``.

| Name | Summary |
|---|---|
| <code>def nvml_devices()</code> | Per-GPU dict from NVML, or None when NVML is unusable. |
| <code>def smi_query(fields: str)</code> | Rows of ``nvidia-smi --query-gpu=<fields>``, or None when nvidia-smi is missing/fails. |
| <code>def smi_topo_matrix()</code> |  |

## `stokastisk.topology.pcie`

PCIe / link classification from ``nvidia-smi topo -m`` text, and PCIe link speed from sysfs.

| Name | Summary |
|---|---|
| <code>def classify_cell(cell: str)</code> |  |
| <code>def parse_topo_matrix(text: str)</code> | ``{"0-1": "PHB"}`` (lower index first). Unknown / unparsable cells become UNKNOWN; an unparsable matrix yields ``{}``. |
| <code>def pcie_sysfs(pci_bus_id: str &#124; None)</code> | Current/max PCIe generation + width from /sys (no root needed). Missing files -> missing keys. |

## `stokastisk.topology.peer_access`

CUDA peer access, measured with ``torch.cuda.can_device_access_peer`` in BOTH directions. Never forced, never assumed.

| Name | Summary |
|---|---|
| <code>def measure_peer_access(device_count: int &#124; None=None)</code> |  |

