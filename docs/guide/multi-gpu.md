# 6. Using both GPUs

The two T4s share a PCIe host bridge (no NVLink). **For a model that fits one T4, TP=1 is fastest.** Use two GPUs when the model does not fit one.

| Strategy | Config | Use for | Status |
|---|---|---|---|
| Single GPU | `StokastiskConfig()` | models up to about 3B in FP16 | Kaggle-tested |
| Tensor parallel | `tensor_parallel_size=2` | a model too large for one T4 (e.g. 7B) | Kaggle-tested |
| Pipeline parallel | `pipeline_parallel_size=2` | same, layer-split | Kaggle-tested, **experimental** |
| Replicas | `ReplicaPool(model, devices=[0, 1])` | throughput: two independent copies | Kaggle-tested |

TP and PP are alternatives, not a combination: TP=2 × PP=2 would need four GPUs and raises `UnsupportedParallelismError`.

## Tensor parallel (7B example)

```python
cfg = StokastiskConfig(tensor_parallel_size=2, max_model_len=2048, gpu_memory_utilization=0.88)
with LLM("Qwen/Qwen2.5-Coder-7B-Instruct", config=cfg) as llm:
    out = llm.chat([{"role": "user", "content": "Write a Python function that parses JSON Lines."}])
    print(out[0].outputs[0].text)
```

Between experiments, close the previous engine first (the `with` block does this) so GPU memory is free.

## Pipeline parallel

```python
cfg = StokastiskConfig(pipeline_parallel_size=2, max_model_len=2048)
```

A warning says it is experimental and slower than TP=1 for models that fit one T4. NCCL may print an "unbatched P2P op" warning; it is harmless.

## `ReplicaPool`

Two independent TP=1 engines, one per GPU, behind a token-aware scheduler. Each GPU holds a full copy of the weights.

```python
from stokastisk import ReplicaPool

pool = ReplicaPool("Qwen/Qwen2.5-0.5B-Instruct", devices=[0, 1],
                   config=StokastiskConfig(max_model_len=2048))
try:
    texts = pool.generate(["Hello", "Bonjour", "Hola", "Ciao"], max_tokens=16)
    print(texts, pool.last_stats["output_tok_per_s"])
finally:
    pool.close()
```

| Member | Purpose |
|---|---|
| `ReplicaPool(model, devices=(0, 1), *, config=None, ready_timeout=900, max_attempts=2, engine_kwargs=None)` | spawns one worker per device; `config` must be TP=1/PP=1 |
| `generate(prompts, *, max_tokens=64, temperature=0.0, timeout=1800)` | list of texts in prompt order |
| `restart_dead()` | restarts exited workers (never done implicitly) |
| `close(timeout=60)` | stops all workers, raises `ReplicaError` if any had to be killed |
| `info`, `last_stats`, `close_report`, `config` | state; `last_stats` has wall time, tokens/s, per-GPU busy time |

Failure recovery has been exercised with fake workers on CPU only, not on real GPUs.

## Topology

```python
topo = stokastisk.discover_topology()
print(topo.gpu_count, topo.pair_link(0, 1), topo.peer_access.all_pairs_available)
print(topo.to_dict())
```

`Topology` fields include `devices` (name, UUID, compute capability, VRAM, PCIe generation/width), `peer_access`, `link_class` (`NVLINK`, `PIX`, `PXB`, `PHB`, `NODE`, `SYS`, `UNKNOWN`), `nccl_available`, `cuda_runtime_version`, `driver_version`, `cpu_count`, RAM, `notes`, `recommended_parallelism`. A value that could not be measured is `None`, never invented. Peer access varies between Kaggle VMs, so the SDK does not rely on it.

## Recommender

```python
rec = stokastisk.recommend_parallelism("Qwen/Qwen2.5-Coder-7B-Instruct", workload="latency", max_model_len=2048)
print(rec.strategy, rec.reason, rec.warnings, rec.estimates)
cfg = rec.to_config()                 # a validated StokastiskConfig; raises if nothing can run
```

`workload` is `"latency"` or `"throughput"`. Strategies include `single_gpu`, `tensor_parallel`, `pipeline_parallel`, `replica_parallel`, and the non-runnable `insufficient_memory`, `unsupported_architecture`, `no_gpu`. Memory numbers are **estimates** from the model's `config.json`. It only recommends; you decide.

Next: [Checks and diagnostics](diagnostics.md).
