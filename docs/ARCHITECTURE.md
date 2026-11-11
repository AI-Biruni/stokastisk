# Stokastisk 0.1.0 — T4/Triton-only architecture

**Status: `KAGGLE_T4_VERIFIED` — all 14 notebooks ran on Kaggle 2 × T4 (see `RELEASE.md` and `docs/KAGGLE_T4_VERIFICATION.md`). Features with CPU coverage only are called out below. The vLLM wheel and native binaries are byte-identical to the earlier baseline build.**

## Target (single supported configuration)
Kaggle, Python 3.13, PyTorch 2.11.0+cu130, CUDA 13.0, Triton 3.6.0, glibc 2.39, driver 580.x, **4 CPU cores, ~30 GiB RAM, 20 GiB `/kaggle/working`, 2 × Tesla T4 (SM75, ~15 GiB each), PHB topology; CUDA peer access varies between Kaggle VMs** (`stokastisk.LIMITS`). NCCL: torch is compiled against 2.28.9; the runtime library on Kaggle is whatever `nvidia/nccl/lib/libnccl.so.2` is (observed 2.32.3, see `RELEASE.md`).
Other GPUs are unsupported on purpose: the native extensions contain `sm_75` cubins only (no PTX).

## Components
| Piece | What it is |
|---|---|
| `stokastisk-0.1.0-py3-none-any.whl` | pure-Python SDK (`import stokastisk`): config validation, T4 checks, NCCL helpers, prohibited-package diagnostics, thin wrapper over `vllm.LLM` |
| `vllm-0.25.1-cp38-abi3-linux_x86_64.whl` | vLLM 0.25.1 **derived** T4-only distribution (see below) |
| `requirements/` | exact `--no-deps` pins + constraints + prohibited list |
| `scripts/kaggle` | installer and the six acceptance stages (run by the project owner on Kaggle) |

### How the vLLM wheel was made (honest provenance)
1. Source: vLLM `v0.25.1`, commit `752a3a504485790a2e8491cacbb35c137339ad34` (verified against the tarball header and the GitHub tag API).
2. Native extensions (`_C_stable_libtorch`, `_moe_C_stable_libtorch`, `cumem_allocator`, `spinloop`, `fs_io_C`, `_rust_tool_parser`) come from the **existing 5.5 h local build** (NVCC 13.0.48, `TORCH_CUDA_ARCH_LIST=7.5`). They were *not* rebuilt: the previous build was audited and found CUDA-13/sm_75-only. That build also compiled FlashAttention 2/3 binaries, which are **not included**. A from-scratch rebuild that never compiles them was not performed (≈5.5 h) and remains an open item.
3. Python tree: `stage.zip` (stripped vLLM Python tree, 2,560 files) — but `stage.zip`'s own `.so` files link **`libcudart.so.12`** and were therefore discarded and replaced by the CUDA-13 build outputs above.
4. FlashAttention/FlashInfer/FlashMLA removal (`scripts/release/build_t4_tree.py`): `vllm_flash_attn/`, `third_party/flashmla/` removed wholesale; 15 further modules removed because they import FlashInfer/FlashAttention/FlashMLA at module level (AST-detected, listed in the release `evidence/`).
5. Patches (`patches/`): 5 files changed so that the core still imports — `fa_utils.py` (names stubbed), `rotary_embedding/common.py` (`forward_cuda` → native torch implementation instead of the Triton rotary kernel that lived inside `vllm_flash_attn`), `fused_moe/all2all_utils.py`, `compilation/passes/pass_manager.py`, `config/compilation.py` (FlashInfer allreduce fusion absent).
6. Metadata rewritten: `Requires-Dist` is only the import-proven core; the upstream `cuda.txt` dependencies (flashinfer, cutlass-dsl, humming-kernels, …) are gone. One deliberate deviation: `fastapi>=0.133.0` instead of `<0.137` (Kaggle ships 0.141.1; the import sweep passes with it).

`vllm/utils/flashinfer.py` and about 30 other `*flash*`-named upstream modules are **retained** because core modules import them; they only probe with `importlib.util.find_spec` or contain lazy imports and never import a FlashInfer/FlashAttention package at module level (AST-verified). They are inert unless the FlashInfer package is installed — which the installer, the SDK and the audits all forbid.

## Attention
`StokastiskConfig(attention_backend="TRITON_ATTN")` is the only accepted value. It is passed as `attention_config={"backend": "TRITON_ATTN"}` (verified against `EngineArgs`/`AttentionConfig` in 0.25.1: dict → `AttentionConfig`, string → `AttentionBackendEnum.TRITON_ATTN`; verified on CPU that `EngineArgs(...).attention_config.backend` is `TRITON_ATTN`). vLLM 0.25.1 raises (`Selected backend … is not valid`) rather than falling back when an explicitly chosen backend is invalid, and logs `Using AttentionBackendEnum.TRITON_ATTN backend.` on success. Acceptance evidence (notebooks 05/06) is three-fold: engine config, that log line per worker, and compiled `kernel_unified_attention` artifacts in a fresh `TRITON_CACHE_DIR`. MLA / hybrid / sparse-attention model families select other backends and are unsupported here.

## CUDA components
| Component | Where it comes from |
|---|---|
| `libcudart.so.13` | dynamic; Kaggle's CUDA 13.0 runtime (NEEDED by the two big extensions) |
| `libcuda.so.1` | dynamic; NVIDIA driver |
| `libtorch*`, `libc10*` | dynamic; Kaggle's PyTorch |
| Triton attention kernels | JIT-compiled by Triton 3.6.0 at run time (no cubin shipped) |
| device code in `.so` | `sm_75` cubins only (`manifests/CUDA-ARCH-MANIFEST.json`) |
| NCCL | the `libnccl.so.2` found by torch's RUNPATH in Kaggle's image (observed 2.32.3+cuda12.9) |
No CUDA library is bundled. Static-vs-dynamic of individual CUDA symbols was not analysed beyond `DT_NEEDED`. glibc requirement of the native files: max GLIBC 2.38 (Kaggle has 2.39).

## NCCL / tensor parallel
No NVLink (T4); CUDA peer access is a per-VM property (measured true in one Kaggle session, false in another), so the defaults do not depend on it. Defaults set before engine start (never overriding user values): `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1`, `NCCL_CUMEM_ENABLE=0`; vLLM `disable_custom_all_reduce=True`; `enforce_eager=True`. Notebook 04 runs real all-reduce/all-gather/reduce-scatter and saves `NCCL_DEBUG` logs; TP=2 speed is measured, not assumed.


---
# Topology-aware orchestration layer (CPU-tested and Kaggle-verified; exceptions listed below)

Stokastisk 0.1.0 includes a **pure-Python** orchestration layer above the unchanged baseline engine. The vLLM wheel and all native CUDA extensions are byte-identical to the baseline build; the SDK wheel is the only rebuilt artifact.
Everything below is labelled with what has actually been verified: *CPU-tested* (mocked, runs anywhere), *locally run* (the dev machine has one GeForce 940M, SM50 — it proves nothing about T4/SM75), *Kaggle-verified* (executed by the owner on 2 × T4). Worker-failure recovery and the C++ transport's speed are **not** Kaggle-verified in the sense of a benefit: recovery has CPU tests only, and the transport is experimental.

## The physical constraint (measured on Kaggle, never assumed by the code)
Two Tesla T4 behind a PCIe host bridge: topology `PHB`, no NVLink. `torch.cuda.can_device_access_peer` was true in both directions in notebook 01 and false in both in notebook 07 (different Kaggle VMs), so it is measured each time and never assumed. NCCL works in both cases.
Nothing in this SDK can create NVLink bandwidth on a machine without NVLink: pinned memory, NCCL SHM, CUDA streams and host staging are *different* communication techniques with different costs, not substitutes.
The topology detector reports what the machine has; it does not hardcode PHB.

## 1. GPU topology detection (`stokastisk.topology`, CPU-tested)
```
 torch (always)  ──►  name, UUID, cc, VRAM, free/used
 NVML / pynvml   ──►  PCI bus id, PCIe gen/width, NVLink state        (optional)
 nvidia-smi      ──►  same fields if NVML is absent; `topo -m` link classes (optional)
 /sys/bus/pci    ──►  PCIe gen/width without root                     (optional)
 torch.cuda.can_device_access_peer(a,b) and (b,a) ──► P2P matrix, each direction measured separately
                       │
                       ▼
  Topology{devices, peer_access, link_class[NVLINK|PIX|PXB|PHB|NODE|SYS|UNKNOWN], nvlink_available, nccl_*, cpu, ram, notes}
  A value that could not be measured is None (never invented); UNKNOWN link classes never imply "no NVLink".
```

## 2. Execution strategies (`StokastiskConfig`, CPU-tested)
| strategy | TP | PP | processes | status |
|---|---|---|---|---|
| `single_gpu` | 1 | 1 | 1 engine | **Kaggle-verified** (notebooks 03, 05, 06, 08, 11, 13, 14) |
| `tensor_parallel` | 2 | 1 | 1 engine, 2 workers | **Kaggle-verified** (notebooks 05, 06, 11, 13): both GPUs hold a shard; ≈0.55× of TP=1 decode for a model that fits one GPU |
| `pipeline_parallel` | 1 | 2 | 1 engine, 2 workers | **experimental, Kaggle-verified** (notebooks 08, 13): both GPUs hold a shard, deterministic output equal to TP=1; ≈0.6× of TP=1 decode |
| `replica_parallel` | 1 | 1 | 1 TP=1 engine per GPU | **experimental, Kaggle-verified** (notebooks 09, 14): both GPUs receive work at every batch size; 0.91–0.96× of one TP=1 engine up to batch 64 |
| `auto` | – | – | – | recommendation only: `LLM()` refuses it; call `recommend_parallelism()` |

TP=2 × PP=2 is rejected (4 GPUs on a 2-GPU target). `execution_strategy=None` keeps the baseline meaning, and at PP=1 the engine call is byte-for-byte the baseline call.

### TP=2 — every layer is split, an all-reduce per layer
```
 GPU0 ┌─ layer i (half of every weight matrix) ─┐   NCCL all-reduce   ┌─ layer i+1 (half) ─┐
      └──────────────────────────────────────────┘◄══════ SHM ═══════►└────────────────────┘
 GPU1 ┌─ layer i (other half) ──────────────────┐   (twice per layer; over the host bridge when peer access is unavailable, which differs between Kaggle VMs)
```
### PP=2 — layers are split, one activation tensor crosses per stage boundary
```
 GPU0 [embed, layers 0..N/2-1] ──activations (NCCL send/recv)──► GPU1 [layers N/2..N-1, norm, lm_head] ──► token
 vLLM decides the partition (`make_layers`); stages are idle part of the time unless several micro-batches are in flight.
```
**What was verified about PP in the vLLM 0.25.1 source and bundled wheel (static, no execution):** `EngineArgs.pipeline_parallel_size` exists; `ParallelConfig` multiplies TP×PP into `world_size`;
the `mp` executor builds PP groups (`get_pp_group`) and names workers `Worker_PP<rank>`; `Qwen2ForCausalLM` is `SupportsPP` (`make_layers`, `PPMissingLayer`, `IntermediateTensors`); inter-stage tensors go through `GroupCoordinator.send_tensor_dict/recv_tensor_dict`.
**Verified on Kaggle (notebooks 08, 13):** PP=2 runs on the Flash-free wheel on two T4, with both GPUs holding stages. Measured decode (Qwen2.5-1.5B, batch 8): PP=2 ≈146–150 tok/s vs TP=2 ≈133 tok/s vs TP=1 ≈241 tok/s, so PP=2 is slightly faster than TP=2 here and both are slower than TP=1. A model too large for one T4 was not run.

### Replicas — no inter-GPU traffic at all
```
            ReplicaPool.generate(prompts)
                     │   weighted least-loaded, token-aware, deterministic (LPT)
          ┌──────────┴──────────┐
   worker 0 (CUDA_VISIBLE_DEVICES=0)   worker 1 (CUDA_VISIBLE_DEVICES=1)     separate processes, weights, KV cache, lifecycle
   vLLM TP=1, full model                vLLM TP=1, full model                parent never initialises CUDA; distinct UUIDs required
```
The scheduler is pure Python (`stokastisk.parallel.{queue,routing,scheduler}`): cost = estimated prompt tokens + `max_tokens`; speed weights are learned from observed latency (a worker that has not completed a request yet is weighted at the mean observed speed, so it still receives work; fixed after notebook 09 and verified by notebook 14); a dead worker's in-flight requests are retried on a survivor within a budget; `restart_dead()` is explicit. No Redis/Ray. CPU-tested with real spawned (fake-GPU) workers.

## 3. NCCL vs host-staged transfer (`stokastisk.communication`, EXPERIMENTAL)
```
 NCCL (used by vLLM):   GPU0 ══ NCCL SHM/direct transport (shared host memory, managed by NCCL) ══► GPU1
 Host-staged (this repo, benchmark only, not wired into vLLM):
   GPU0 mem ──cudaMemcpyAsync D2H──► pinned buf[0] ──cudaMemcpyAsync H2D──► GPU1 mem
                 chunk i+1 D2H overlaps chunk i H2D via buf[1]; cudaEvent hands each buffer from stage to stage (cudaStreamWaitEvent)
```
Implemented twice: PyTorch pinned tensors + streams/events (`cuda_transport.py`) and C++ on the CUDA Runtime API (`native/host_staged_xfer.cpp`, built on demand with nvcc, `sm_75`, links `libcudart.so.13`; it is *not* in the wheel). The local machine has one GPU, so neither has moved a byte between two devices yet — notebook 10 does that.
Decision rule: if NCCL is not slower at every measured size the custom transport stays experimental and is never integrated. NVSHMEM is not integrated (peer access is not guaranteed on Kaggle, no verified IB path).

## 4. Intelligent parallelism selection (`recommend_parallelism`, CPU-tested)
```
 model config.json ─► ModelSpec ─► weights + KV(max_model_len) ─► fits one GPU?  fits two?
 topology (P2P, NVLink, NCCL, VRAM, RAM) ───────────────────────────────┐
 workload (latency | throughput) ───────────────────────────────────────┤
                                                                        ▼
 unsupported arch ► unsupported_architecture │ fits none ► insufficient_memory
 fits one GPU:  latency ► single_gpu (TP=1) │ throughput ► replica_parallel (experimental flag + warnings)
 needs two:     tensor_parallel (TP=2), with a warning when there is no NVLink/P2P; PP=2 listed as an experimental alternative
```
Rule-based and deterministic; memory numbers are estimates from `config.json` and say so. It never changes anything by itself: `Recommendation.to_config()` returns a config the caller chooses to use.

## 5. Telemetry and benchmarks (`stokastisk.telemetry`, `stokastisk.benchmarks`, CPU-tested maths)
nvidia-smi/NVML sampling (utilisation, memory, clocks, temperature, power, PCIe throughput where exposed — unavailable counters stay `None`), segment timers, per-worker compute/communication/idle accounting, optional OpenTelemetry spans (no-op without the package), local JSON for everything, Nsight probe (SKIPPED when absent).
Benchmarks record provenance, model revision, every repeat (min/median/p95/max), TTFT, inter-token latency, throughput and imbalance `|T0−T1|/max(T0,T1)·100`. Per-stage PP compute/communication time is **not exposed** by vLLM's public API and is reported `NOT_MEASURED`; the PP balance number is a labelled utilisation proxy, never stage time.

## Decisions
* No native rebuild: nothing in 0.1.0 needs a new vLLM kernel. The only new native source is an optional, separately built CUDA-Runtime-only transport.
* NCCL is never replaced; protected packages are never touched; `TRITON_ATTN`, fp16 and SM75 stay the only supported combination.
* New features are opt-in and default off; the baseline call `StokastiskConfig(tensor_parallel_size=1)` is unchanged.
