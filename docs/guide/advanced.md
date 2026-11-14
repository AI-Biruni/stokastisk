# 9. Advanced modules

These modules are optional. They are import-light, degrade to an explicit "unavailable" instead of failing, and are not needed for normal inference.

!!! `communication` is **experimental** and is not used by the engine. The benchmarks are research tools, not throughput promises: on two PCIe-connected T4s, TP=2/PP=2/replicas are slower than TP=1 for models that fit one T4.

## `stokastisk.telemetry`

| Name | Purpose |
|---|---|
| `nvml.sample_once()` | one GPU sample (NVML if present, else `nvidia-smi`, else "unavailable") |
| `nvml.GpuSampler(interval=0.5)` | background sampler: `with GpuSampler() as s: ...; s.summary()` |
| `metrics.percentile / summarize / throughput / imbalance_percent / token_latencies` | pure statistics |
| `communication_profiler.SegmentTimer` | `wall(name)` and `cuda(name)` context managers, `totals()`, `report()` |
| `worker_metrics.WorkerMetrics`, `balance_report` | compute vs communication vs idle per worker |
| `report.write_json(path, obj)`, `report.provenance()`, `report.span(name)` | JSON evidence; optional OpenTelemetry span (a no-op unless configured) |
| `cuda_profiler.nsight_status()` | whether Nsight is usable (never required) |

```python
from stokastisk.telemetry.nvml import GpuSampler
with GpuSampler(1.0) as s:
    ...                       # run something
print(s.summary())
```

## `stokastisk.cuda`

Thin PyTorch-backed helpers that never touch vLLM's own streams: `device.device_count/on_device/synchronize`, `streams.create_stream/stream_synchronize`, `events.new_event/record/make_wait/elapsed_ms`, `memory.mem_info/pinned_empty/timed_copy`.

```python
from stokastisk.cuda.memory import mem_info
print(mem_info(0))            # free / total for GPU 0
```

## `stokastisk.nccl` and `stokastisk.communication`

* `nccl.run_collective_selftest()` — all-reduce, all-gather, reduce-scatter, broadcast, barrier over a 2-rank `torch.distributed.run` job; never raises, returns a report.
* `communication.nccl.time_p2p / time_allreduce` — NCCL timing helpers.
* `communication.cuda_transport.HostStagedTransport`, `pinned_memory.PinnedBufferRing`, `synchronization.BufferTicket`, `native.build_native/load_native/NativeHostStaged` — an experimental host-staged GPU→GPU copy for machines without peer access (the optional C++ file is `native/host_staged_xfer.cpp`; needs `nvcc`).
* `transport_registry.list_transports()` — known transports with an honest status; nothing is selected automatically.

## `stokastisk.benchmarks`

Identical workloads for every strategy; all runs are recorded (median/p95/min/max).

| Module | Entry points |
|---|---|
| `baseline` | `Workload`, `measure`, `run_engine` |
| `tensor_parallel` | `config_for(tp)`, `run(model, workloads, tp)` |
| `pipeline_parallel` | `config_for`, `run` |
| `replica_parallel` | `run(model, workloads, devices=(0, 1))` |
| `communication` | `bench_host_staged`, `bench_nccl`, `compare` |
| `gpu_balance` | `from_replica_record` |
| `report` | `comparison_table`, `relative_to` — combine records into JSON + a Markdown table; missing strategies are listed as `NOT_RUN` |

## `stokastisk.parallel` internals

`scheduler.Scheduler`, `routing.choose_worker/WorkerLoad`, `queue.RequestQueue`, `memory.load_model_spec/ModelSpec`, `worker.worker_main` implement `ReplicaPool` and the recommender. They are public but you rarely need them. `memory.load_model_spec(model, local_files_only=False)` estimates weight and KV-cache MiB from `config.json`.
