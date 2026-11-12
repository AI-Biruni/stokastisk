# Stokastisk 0.1.0

**A Python 3.13 SDK and standalone CUDA 13.0 binary build of vLLM 0.25.1 for Kaggle's 2 × Tesla T4 (SM75).**

[![Release](https://img.shields.io/badge/release-0.1.0-blue)](RELEASE.md)
[![Kaggle](https://img.shields.io/badge/Kaggle%202xT4%20verification-VERIFIED-brightgreen)](RELEASE.md)
[![CUDA](https://img.shields.io/badge/CUDA-13.0%20only-76b900)](docs/ARCHITECTURE.md)
[![Python](https://img.shields.io/badge/python-3.13-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-lightgrey)](LICENSE)

| | |
|---|---|
| **Python** | 3.13 only |
| **Engine** | vLLM 0.25.1 (T4-only derived wheel) |
| **PyTorch / Triton** | 2.11.0+cu130 / 3.6.0 (Kaggle's own — never replaced) |
| **CUDA** | Toolkit 13.0 only (`libcudart.so.13`); no CUDA 12.x in any binary or script |
| **GPU code** | `sm_75` cubins only, no PTX |
| **Attention** | `TRITON_ATTN` only. No FlashAttention, FlashInfer or FlashMLA code is shipped |
| **Target** | Kaggle, 2 × Tesla T4 |
| **Project version** | `0.1.0` (permanently locked, see [VERSION_POLICY.md](VERSION_POLICY.md)) |
| **Download** | Hugging Face CDN: <https://huggingface.co/waqasm86/stokastisk> |

## Status — `KAGGLE_T4_VERIFIED`

All fourteen Kaggle notebooks were executed by the project owner on **2 × Tesla T4** (results, hashes and limits: [docs/KAGGLE_T4_VERIFICATION.md](docs/KAGGLE_T4_VERIFICATION.md)).
Tensor parallelism (TP=2), pipeline parallelism (PP=2) and independent dual-GPU replicas all run on the Flash-free vLLM 0.25.1 wheel with `TRITON_ATTN`, with both GPUs holding model memory and memory released afterwards.
**Measured fact: for a model that fits one T4, TP=1 is the fastest configuration.** TP=2 ran at about 0.55× and PP=2 about 0.6× of TP=1 (Qwen2.5-1.5B), and two replicas at 0.91–0.96× up to batch 64. TP/PP are for models that do not fit one 15 GiB GPU; that case was not measured.
The development machine has one GeForce 940M (SM50) and is never used for GPU claims; the CPU tests use mocked GPUs and fake workers.

| Level | Meaning |
|---|---|
| Implemented | code exists in this repository |
| CPU-tested | covered by `tests/` using mocks / fake workers (no GPU) |
| CUDA-compiled | native C++ compiles for `sm_75` locally and links only `libcudart.so.13` |
| Kaggle-tested | executed on 2 × T4 by the owner (see [docs/KAGGLE_T4_VERIFICATION.md](docs/KAGGLE_T4_VERIFICATION.md)) |
| Experimental | no evidence of a benefit; may change or be removed |

| Feature | API | Status |
|---|---|---|
| Core SDK: validated config, T4/SM75 checks, TP=1/TP=2 engine wrapper, NCCL helpers, Kaggle limits | `stokastisk.LLM`, `StokastiskConfig` | Implemented, CPU-tested, **Kaggle-tested** (notebooks 03, 05, 06, 13) |
| GPU topology (UUIDs, PCIe, P2P per direction, link class, NVLink, NCCL) | `stokastisk.discover_topology()` | CPU-tested, **Kaggle-tested** (notebook 07; peer access varies per VM) |
| Execution strategies, validated | `StokastiskConfig(execution_strategy=..., pipeline_parallel_size=...)` | CPU-tested, **Kaggle-tested** (13) |
| Rule-based parallelism recommendation (never applied automatically) | `stokastisk.recommend_parallelism(model, topology, workload)` | CPU-tested; Kaggle: run in notebooks 12 and 14 |
| Independent dual-GPU replicas + token-aware scheduler, worker lifecycle and failure recovery | `stokastisk.parallel.ReplicaPool(model, devices=[0, 1])` | CPU-tested (fake workers), **Kaggle-tested** (notebooks 09, 14: both GPUs receive work); failure recovery CPU-only; no speed-up measured |
| Pipeline parallelism PP=2 | `StokastiskConfig(pipeline_parallel_size=2, ...)` | **Kaggle-tested** (notebooks 08, 13): both GPUs hold a shard; slower than TP=1 for models that fit one T4; **experimental** |
| Telemetry (GPU, communication, worker metrics, Nsight probe, optional OTel) | `stokastisk.telemetry` | CPU-tested, nvidia-smi sampling **Kaggle-tested** (09, 11, 14) |
| Host-staged CUDA transport (pinned memory, streams, events; Python + C++) vs NCCL | `stokastisk.communication`, `native/` | CUDA-compiled; run on Kaggle (notebook 10, with a warning); **experimental**, not integrated into vLLM |
| Reproducible benchmarks (TP/PP/replicas, imbalance %, communication sweep) | `stokastisk.benchmarks`, `scripts/kaggle/dual_*.py` | CPU-tested maths; **Kaggle-tested** (06, 08, 09, 11, 13, 14) |

```python
import stokastisk
topology = stokastisk.discover_topology()
rec = stokastisk.recommend_parallelism("Qwen/Qwen2.5-1.5B-Instruct", topology, workload="throughput")
print(rec.strategy, rec.reason, rec.warnings)       # e.g. a TP=1 recommendation for a model that fits one T4 (PHB; peer access varies per VM)

from stokastisk.parallel import ReplicaPool
with ReplicaPool("Qwen/Qwen2.5-0.5B-Instruct", devices=[0, 1]) as pool:      # Kaggle-tested in notebooks 09 and 14
    print(pool.generate(["Explain CUDA streams.", "Explain Tensor Core architecture."]))
```
No software here replaces NVLink: the T4 pair has none, and pinned memory / NCCL SHM / host staging are different techniques with different costs. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the diagrams and the exact verification status of each piece.

## Why this exists

Kaggle's T4s are SM75 (Turing). Upstream vLLM wheels ship FlashAttention 2/3/4 and FlashInfer kernels that target newer GPUs, and they
pull in CUDA-12 and CUDA-13.2 packages that would replace Kaggle's preinstalled stack. Stokastisk gives you a vLLM that is built and audited for exactly
this machine, and an installer that cannot break the Kaggle image.

## Kaggle hard limits baked into the SDK (`stokastisk.LIMITS`)

| Resource | Limit | How the SDK uses it |
|---|---|---|
| CPU | 4 cores (`nproc`) | `OMP/MKL/OPENBLAS/NUMEXPR` threads are split across tensor-parallel workers |
| System RAM | ~30 GiB | checked by `check_resources()` |
| `/kaggle/working` | 20 GiB | evidence directories are measured against the quota; models and wheels live in `/tmp` and the HF cache |
| GPUs | 2 × Tesla T4, SM75, ~15 GiB VRAM each | `gpu_memory_utilization` is bounded; `dtype="half"` (T4 has no bf16 tensor cores) |

## Quick start (Kaggle: Internet ON, Accelerator *GPU T4 × 2*, Python 3.13)

```python
# 1. Download the release from the Hugging Face CDN (stdlib only; a single HTTP stream (parallel ranges are opt-in via `workers`; 4 parallel ranges stalled on the Hub CDN in a workstation test); SHA256-verified; ~37 MB)
import urllib.request, runpy, subprocess, sys
url = "https://huggingface.co/waqasm86/stokastisk/resolve/main/scripts/fetch_release.py"
open("/tmp/fetch_release.py", "wb").write(urllib.request.urlopen(url).read())
bundle = runpy.run_path("/tmp/fetch_release.py")["fetch_release"]("stokastisk-0.1.0-kaggle-t4-sm75")

# 2. Install. Never runs a dependency resolver, never touches torch / triton / NCCL / CUDA / transformers.
subprocess.run([sys.executable, f"{bundle}/scripts/kaggle/02_install.py", str(bundle)], check=True)
```

```python
# 3. Use it (top-level notebook statements; each cell runs in order).
import stokastisk

config = stokastisk.StokastiskConfig(tensor_parallel_size=1)       # 1 = fastest on this machine, 2 = both GPUs
llm = stokastisk.LLM("Qwen/Qwen2.5-0.5B-Instruct", config=config)
print(llm.complete("The capital of France is", max_tokens=32))
print(llm.verify_attention_backend())                              # "TRITON_ATTN"
llm.close()
```

**About this snippet:** vLLM 0.25.1 starts engine processes with `spawn`. The Kaggle acceptance scripts (`scripts/kaggle/05_inference.py`, `dual_15_tp_pp_fulfilment.py`, used by notebooks 05, 06, 13) run each engine in a separate Python process, and that is the pattern that was validated on Kaggle 2 × T4; running `stokastisk.LLM` directly in a notebook cell was **not** part of the acceptance runs. If it fails there, use the notebook-05 pattern and report the error.

### Public API

| Name | Purpose |
|---|---|
| `stokastisk.LLM(model, config=…)` | thin wrapper over `vllm.LLM`; `.generate()`, `.complete()`, `.verify_attention_backend()`, `.close()` |
| `stokastisk.StokastiskConfig` | validated settings: `tensor_parallel_size`, `dtype="half"`, `gpu_memory_utilization=0.88`, `max_model_len=4096`, `attention_backend="TRITON_ATTN"`; `from_env()` reads `STOKASTISK_*` |
| `stokastisk.validate_environment()` / `report()` | T4/SM75, CUDA 13.0, Python 3.13, prohibited-package and native-extension checks |
| `stokastisk.LIMITS`, `check_resources()`, `cpu_budget()`, `working_dir_status()` | the Kaggle limits above |
| `stokastisk.SamplingParams` | re-exported from the bundled vLLM |

Errors derive from `StokastiskError` (`ConfigurationError`, `EnvironmentValidationError`, `GPUDiscoveryError`, `EngineError`, `AttentionBackendError`).

## Historical measurements (earlier internal baseline build, same vLLM wheel — NOT validation of 0.1.0)

Measured on Kaggle 2 × T4 with Qwen2.5-1.5B-Instruct (4096 new tokens, notebook 06): TP=1 ≈ 1003 tok/s best of 3 (median ≈ 984), TP=2 ≈ 545 tok/s (0.55×). The two T4s sit behind the host bridge (`PHB`); peer access differs from one Kaggle VM to the next.
**Use TP=1 when the model fits one T4**; use TP=2 or PP=2 only for models that do not fit in 15 GiB. Two independent TP=1 replicas split work evenly (notebook 14) but did not exceed one engine up to batch 64. Full tables: [docs/KAGGLE_T4_VERIFICATION.md](docs/KAGGLE_T4_VERIFICATION.md).

## vLLM 0.25.1 coverage on the T4

> **Two distinct artifact sets — do not confuse them.**
> **A. Kaggle-verified release (unchanged):** `stokastisk-0.1.0-py3-none-any.whl` (`29edaa55…`) + `vllm-0.25.1-cp38-abi3-linux_x86_64.whl` (`cc3bcafb…`) + the release ZIP. The SDK wheel was built before `stokastisk.compat`, `stokastisk.serving`, `LLM.chat` and the bfloat16 pre-check existed and does **not** contain them.
> **B. New integrated SDK candidate (build tag 1):** `stokastisk-0.1.0-1-cp313-cp313-linux_x86_64.whl` — one wheel with the complete SDK (including `compat`, `serving`, `LLM.chat`, the bfloat16 pre-check), the bundled vLLM 0.25.1 Triton-attention tree (byte-identical to the verified vLLM wheel) and its CUDA 13.0 `sm_75` native objects. Project version stays `0.1.0`; only the PEP 427 build tag differs. Details: [docs/SDK_CANDIDATE.md](docs/SDK_CANDIDATE.md).
> **C. Not executed:** everything in B beyond what A already ran. Status of B: `BUILT AND PACKAGED — STATICALLY INSPECTED — NOT EXECUTED ON KAGGLE DUAL T4 GPUs`. A successful wheel build is not GPU inference.

Not every vLLM feature works on a Tesla T4. [docs/VLLM_T4_COVERAGE.md](docs/VLLM_T4_COVERAGE.md) lists, per feature, whether it was **executed on Kaggle**, only compiled, CPU-tested, statically inspected or not run — and what cannot work on SM75 (bfloat16, Flash*, native FP8/FP4, Qwen Omni).
In the repository source and in candidate B, `stokastisk.check_model(...)`, `check_dtype`, `check_quantization` and `check_feature` raise a clear `StokastiskCompatibilityError` before an engine starts; `stokastisk.serving` launches the bundled OpenAI-compatible server (implemented and packaged in candidate B; not yet run on Kaggle).

## Install options

| Option | When |
|---|---|
| **Hub CDN** (above) | default; no dataset needed; needs Internet ON |
| **Kaggle dataset** | offline runs: upload `releases/stokastisk-0.1.0-kaggle-t4-sm75/stokastisk-0.1.0-kaggle-t4-sm75-kaggle-input.zip` from the Hub as a dataset |
| **Manual** | see [INSTALL.md](INSTALL.md) |

The installer plans before it changes anything, and it refuses if a protected package would change. It installs the two wheels with `--force-reinstall --no-deps` (so any other installed Stokastisk version is replaced by exactly 0.1.0), then eight exact pins,
and verifies the after-state equals the before-state plus those eight packages.

## Repository layout

| Path | Contents |
|---|---|
| `sdk/stokastisk/` | the SDK (pure Python): core, `topology/`, `parallel/`, `telemetry/`, `cuda/`, `communication/`, `benchmarks/` |
| `native/` | optional C++ (CUDA Runtime API) host-staged transport, built on demand with nvcc |
| `scripts/kaggle/` | installer, stages `01_env_audit` … `07_summary`, and the `dual_*` stages for notebooks 07–14 |
| `scripts/fetch_release.py` | stdlib CDN downloader |
| `scripts/release/`, `scripts/build_vllm_sm75.sh`, `scripts/audit/` | build, repack, patch and audit tooling |
| `patches/` | wheel patches that keep the Flash-free engine importable (0003 fixes the TP=1/TP=2 FlashInfer default-path imports) |
| `requirements/` | exact pins, constraints, prohibited list |
| `tests/` | CPU-only unit tests (they do not prove GPU behaviour — Kaggle runs do) |
| `docs/` | [ARCHITECTURE](docs/ARCHITECTURE.md), [DEPENDENCIES](docs/DEPENDENCIES.md), [KAGGLE_T4_VERIFICATION](docs/KAGGLE_T4_VERIFICATION.md) |
| `notebooks/kaggle-t4-sm75/notebooks/` | the 14 Kaggle T4 notebooks (unexecuted sources); executed copies stay local and are not published |
| `notebooks/kaggle-l4-sm89/` | `PLANNED_NOT_IMPLEMENTED` — no L4 binaries, no L4 support claimed |
| `VERSION`, `VERSION_POLICY.md`, `scripts/release/verify_project_version.py` | the permanent project-version lock |

## Known limitations

* **NCCL runtime is the CUDA 12.9 build (2.32.3).** Kaggle's own image contains both `nvidia-nccl-cu12` and `nvidia-nccl-cu13`; both install
  `nvidia/nccl/lib/libnccl.so.2`, and the cu12 file wins. This predates the installer and Stokastisk's binaries do not link it. Notebook 04 reports it as a WARNING while all collectives pass.
  Forcing the cu13 build would mean changing Kaggle's protected baseline, so this project does not.
* **Triton attention only.** MLA, hybrid and sparse-attention model families select other backends and are unsupported.
* **T4 / SM75 only.** There is no PTX, so other GPUs cannot run the native extensions.
* **Not covered by the Kaggle runs:** worker-failure recovery (CPU unit tests only), models too large for one T4 (so the memory-capacity benefit of TP/PP is not demonstrated), and the Hub CDN downloader under Kaggle's network (the notebooks fetch the release from the Hub or use the uploaded dataset).

## License

Apache-2.0 (see [LICENSE](LICENSE)). vLLM is Apache-2.0 © the vLLM project.
