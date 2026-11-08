# Stokastisk 0.1.0

**A Python 3.13 SDK and standalone CUDA 13.0 binary build of vLLM 0.25.1 for Kaggle's 2 × Tesla T4 (SM75).**

[![Release](https://img.shields.io/badge/release-0.1.0-blue)](RELEASE.md)
[![Kaggle](https://img.shields.io/badge/Kaggle%20GPU%20verification-PENDING-orange)](RELEASE.md)
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

## Status — `KAGGLE_GPU_VERIFICATION_PENDING`

Release **`stokastisk-0.1.0-kaggle-t4-sm75`** is **CPU-tested only** (mocked GPUs and fake workers; the development machine has one GeForce 940M, SM50 — not a T4). Notebooks 01–12 are **NOT_RUN** for this release:
nothing below is Kaggle-verified until the project owner runs them on 2 × Tesla T4. The vLLM wheel and the native CUDA binaries are byte-identical to an earlier internal baseline build
whose TP=1 / TP=2 inference was run on Kaggle; those historical measurements (in [RELEASE.md](RELEASE.md)) are engineering context, **not validation of this release**.

| Level | Meaning |
|---|---|
| Implemented | code exists in this repository |
| CPU-tested | covered by `tests/` using mocks / fake workers (no GPU) |
| CUDA-compiled | native C++ compiles for `sm_75` locally and links only `libcudart.so.13` |
| Kaggle-tested | executed on 2 × T4 by the owner — **nothing in 0.1.0 yet** |
| Experimental | no evidence of a benefit; may change or be removed |

| Feature | API | Status |
|---|---|---|
| Core SDK: validated config, T4/SM75 checks, TP=1/TP=2 engine wrapper, NCCL helpers, Kaggle limits | `stokastisk.LLM`, `StokastiskConfig` | Implemented, CPU-tested; Kaggle-tested: no |
| GPU topology (UUIDs, PCIe, P2P per direction, link class, NVLink, NCCL) | `stokastisk.discover_topology()` | CPU-tested |
| Execution strategies, validated | `StokastiskConfig(execution_strategy=..., pipeline_parallel_size=...)` | CPU-tested |
| Rule-based parallelism recommendation (never applied automatically) | `stokastisk.recommend_parallelism(model, topology, workload)` | CPU-tested |
| Independent dual-GPU replicas + token-aware scheduler, worker lifecycle and failure recovery | `stokastisk.parallel.ReplicaPool(model, devices=[0, 1])` | CPU-tested (fake workers); **experimental** |
| Pipeline parallelism PP=2 | `StokastiskConfig(pipeline_parallel_size=2, ...)` | source-verified in vLLM 0.25.1; **experimental, unproven on the Flash-free wheel** |
| Telemetry (GPU, communication, worker metrics, Nsight probe, optional OTel) | `stokastisk.telemetry` | CPU-tested |
| Host-staged CUDA transport (pinned memory, streams, events; Python + C++) vs NCCL | `stokastisk.communication`, `native/` | CUDA-compiled; **experimental**, never run on two GPUs |
| Reproducible benchmarks (TP/PP/replicas, imbalance %, communication sweep) | `stokastisk.benchmarks`, `scripts/kaggle/dual_*.py` | CPU-tested maths; not run on Kaggle |

```python
import stokastisk
topology = stokastisk.discover_topology()
rec = stokastisk.recommend_parallelism("Qwen/Qwen2.5-1.5B-Instruct", topology, workload="throughput")
print(rec.strategy, rec.reason, rec.warnings)       # e.g. replica_parallel (experimental) on 2x T4 / PHB / no P2P

if __name__ == "__main__":                           # required: vLLM and the replica workers use spawn
    from stokastisk.parallel import ReplicaPool
    with ReplicaPool("Qwen/Qwen2.5-0.5B-Instruct", devices=[0, 1]) as pool:
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
# 1. Download the release from the Hugging Face CDN (stdlib only; 4 parallel range requests; SHA256-verified; ~37 MB)
import urllib.request, runpy, subprocess, sys
url = "https://huggingface.co/waqasm86/stokastisk/resolve/main/scripts/fetch_release.py"
open("/tmp/fetch_release.py", "wb").write(urllib.request.urlopen(url).read())
bundle = runpy.run_path("/tmp/fetch_release.py")["fetch_release"]("stokastisk-0.1.0-kaggle-t4-sm75")

# 2. Install. Never runs a dependency resolver, never touches torch / triton / NCCL / CUDA / transformers.
subprocess.run([sys.executable, f"{bundle}/scripts/kaggle/02_install.py", str(bundle)], check=True)
```

```python
# 3. Use it. vLLM 0.25.1 starts worker processes with `spawn`, so a script must keep the __main__ guard.
import stokastisk

def main():
    cfg = stokastisk.StokastiskConfig(tensor_parallel_size=1)      # 1 = fastest on this machine, 2 = both GPUs
    llm = stokastisk.LLM("Qwen/Qwen2.5-0.5B-Instruct", config=cfg)
    print(llm.complete("The capital of France is", max_tokens=32))
    print(llm.verify_attention_backend())                          # "TRITON_ATTN"
    llm.close()

if __name__ == "__main__":
    main()
```

In a Kaggle notebook cell, put the code in a `.py` file and run it with `python`, or run it as a subprocess: vLLM's `spawn` re-imports the launching script, which a notebook kernel cannot satisfy.
The validated pattern is `scripts/kaggle/05_inference.py`.

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

On an earlier internal build with the same vLLM wheel, the owner measured Qwen2.5-1.5B-Instruct (4096 new tokens) on Kaggle 2 × T4: TP=1 949 tok/s, TP=2 521 tok/s (0.55×);
the two T4s sit behind the host bridge (`PHB`) with no peer-to-peer. **Use TP=1 when the model fits one T4**; use TP=2 only for models that do not fit in 15 GiB.
Two independent TP=1 engines (replicas) are implemented but unmeasured. Re-measure with notebooks 06–12.

## Install options

| Option | When |
|---|---|
| **Hub CDN** (above) | default; no dataset needed; needs Internet ON |
| **Kaggle dataset** | offline runs: upload `releases/<name>/<name>-kaggle-input.zip` from the Hub as a dataset |
| **Manual** | see [INSTALL.md](INSTALL.md) |

The installer plans before it changes anything, and it refuses if a protected package would change. It installs the two wheels with `--force-reinstall --no-deps` (so any other installed Stokastisk version is replaced by exactly 0.1.0), then eight exact pins,
and verifies the after-state equals the before-state plus those eight packages.

## Repository layout

| Path | Contents |
|---|---|
| `sdk/stokastisk/` | the SDK (pure Python): core, `topology/`, `parallel/`, `telemetry/`, `cuda/`, `communication/`, `benchmarks/` |
| `native/` | optional C++ (CUDA Runtime API) host-staged transport, built on demand with nvcc |
| `scripts/kaggle/` | installer, stages `01_env_audit` … `07_summary`, and the `dual_*` stages for notebooks 07–12 |
| `scripts/fetch_release.py` | stdlib CDN downloader |
| `scripts/release/`, `scripts/build_vllm_sm75.sh`, `scripts/audit/` | build, repack, patch and audit tooling |
| `patches/` | wheel patches that keep the Flash-free engine importable (0003 fixes the TP=1/TP=2 FlashInfer default-path imports) |
| `requirements/` | exact pins, constraints, prohibited list |
| `tests/` | CPU-only unit tests (they do not prove GPU behaviour — Kaggle runs do) |
| `docs/` | [ARCHITECTURE](docs/ARCHITECTURE.md), [DEPENDENCIES](docs/DEPENDENCIES.md) |
| `notebooks/` | Kaggle notebooks 01–12, all unexecuted (`NOT_RUN`) |
| `VERSION`, `VERSION_POLICY.md`, `scripts/release/verify_project_version.py` | the permanent project-version lock |

## Known limitations

* **NCCL runtime is the CUDA 12.9 build (2.32.3).** Kaggle's own image contains both `nvidia-nccl-cu12` and `nvidia-nccl-cu13`; both install
  `nvidia/nccl/lib/libnccl.so.2`, and the cu12 file wins. This predates the installer and Stokastisk's binaries do not link it. Notebook 04 reports it as a WARNING while all collectives pass.
  Forcing the cu13 build would mean changing Kaggle's protected baseline, so this project does not.
* **Triton attention only.** MLA, hybrid and sparse-attention model families select other backends and are unsupported.
* **T4 / SM75 only.** There is no PTX, so other GPUs cannot run the native extensions.
* **No Kaggle run of this release yet**: the Hub download path is verified by SHA256 on a workstation, not on Kaggle.

## License

Apache-2.0 (see [LICENSE](LICENSE)). vLLM is Apache-2.0 © the vLLM project.
