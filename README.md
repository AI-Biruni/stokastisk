# Stokastisk 0.1.0

**A Python 3.13 SDK with a bundled, T4-only build of vLLM 0.25.1 (CUDA 13.0, Triton attention) for Kaggle's 2 × Tesla T4 (SM75).**

Stokastisk wraps vLLM 0.25.1 for one machine — two Tesla T4 GPUs behind a PCIe host bridge — and adds validated configuration, topology discovery, tensor-/pipeline-parallel and independent-replica execution, a parallelism recommender, telemetry, communication helpers, Kaggle resource limits, static T4 compatibility checks and an OpenAI-compatible server launcher. It exists because upstream vLLM wheels ship FlashAttention/FlashInfer kernels for newer GPUs and pull CUDA-12/13.2 packages that would replace Kaggle's preinstalled stack.

[![Release](https://img.shields.io/badge/release-0.1.0-blue)](RELEASE.md)
[![Kaggle](https://img.shields.io/badge/Kaggle%202xT4%20verification-VERIFIED-brightgreen)](RELEASE.md)
[![CUDA](https://img.shields.io/badge/CUDA-13.0%20only-76b900)](docs/ARCHITECTURE.md)
[![Python](https://img.shields.io/badge/python-3.13-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-lightgrey)](LICENSE)

| | |
|---|---|
| **Python** | 3.13 only |
| **Engine** | vLLM 0.25.1, T4-only derivative, bundled inside the integrated wheel |
| **Canonical distribution** | one integrated, **Flash-free** wheel `stokastisk-0.1.0-3-cp313-cp313-linux_x86_64.whl` (build 3; hash in the release manifest, GitHub release and Hub README) — **Kaggle verification pending (notebook 15, build 3)**. Build 2 *failed* notebook 15 and is superseded (see Status) |
| **Legacy distribution** | two wheels `stokastisk-0.1.0-py3-none-any.whl` + `vllm-0.25.1-cp38-abi3-linux_x86_64.whl` — Kaggle-verified (notebooks 01–14) |
| **PyTorch / Triton** | 2.11.0+cu130 / 3.6.0 (Kaggle's own — never replaced) |
| **CUDA** | Toolkit 13.0 only (`libcudart.so.13`); no CUDA 12.x in any binary or script |
| **GPU code** | `sm_75` cubins only, no PTX |
| **Attention** | `TRITON_ATTN` only. No FlashAttention, FlashInfer or FlashMLA code is shipped |
| **Target** | Kaggle, 2 × Tesla T4 |
| **Project version** | `0.1.0` (permanently locked, see [VERSION_POLICY.md](VERSION_POLICY.md)) |
| **Download** | Hugging Face CDN: <https://huggingface.co/waqasm86/stokastisk> |

## Status

| Distribution | State |
|---|---|
| Legacy two-wheel bundle `stokastisk-0.1.0-kaggle-t4-sm75` | `KAGGLE_T4_VERIFIED` — **Kaggle-tested** (notebooks 01–14), unchanged, still published |
| Integrated wheel **build 3** `stokastisk-0.1.0-3-…` | Implemented · independently verified (RECORD, byte-equality with the pinned vLLM wheel + documented transform, strict Flash scan) · **CPU-tested** · **Verification pending** — becomes *Kaggle-tested* only when the owner runs notebook 15 and provides the executed notebook and acceptance ZIP |
| Integrated wheel build 2 `stokastisk-0.1.0-2-…` | **Failed** Kaggle notebook 15 (2026-11-13): structured output (`No module named 'lmformatenforcer'`), HTTP serving (`No module named 'model_hosting_container_standards'`), a misreported package-inventory comparison and Flash `.pyc` files. Basic FP16 inference, chat, complete, determinism, input validation and cleanup **passed** on it. Evidence: `evidence/kaggle/build-2-failed-notebook-15/`. Kept published, superseded |
| Build 1 `stokastisk-0.1.0-1-…` | Never executed on Kaggle (the build-2 run downloaded build 2 instead of the uploaded build 1 — an input-discovery defect fixed in build 3). Superseded |

What build 3 changes, with root causes: [STOKASTISK_0.1.0_BUILD3_REMEDIATION_REPORT.md](STOKASTISK_0.1.0_BUILD3_REMEDIATION_REPORT.md) · remaining limits: [STOKASTISK_0.1.0_BUILD3_PRODUCTION_READINESS.md](STOKASTISK_0.1.0_BUILD3_PRODUCTION_READINESS.md) · dependencies: [docs/DEPENDENCY_COMPATIBILITY.md](docs/DEPENDENCY_COMPATIBILITY.md) · how to run notebook 15: [docs/KAGGLE_BUILD3_VERIFICATION.md](docs/KAGGLE_BUILD3_VERIFICATION.md).

The Kaggle evidence below was produced with the legacy bundle; it is evidence for the engine, the installer-protected environment and the parallel execution paths whose vLLM native objects the integrated wheels contain byte-identically, not for build 3's installation, structured output, serving or Flash-free tree.

### Legacy bundle evidence

All fourteen Kaggle notebooks were executed by the project owner on **2 × Tesla T4** (results, hashes and limits: [docs/KAGGLE_T4_VERIFICATION.md](docs/KAGGLE_T4_VERIFICATION.md)).
Tensor parallelism (TP=2), pipeline parallelism (PP=2) and independent dual-GPU replicas all run on the Flash-free vLLM 0.25.1 wheel with `TRITON_ATTN`, with both GPUs holding model memory and memory released afterwards.
**Measured fact: for a model that fits one T4, TP=1 is the fastest configuration.** TP=2 ran at about 0.55× and PP=2 about 0.6× of TP=1 (Qwen2.5-1.5B), and two replicas at 0.91–0.96× up to batch 64. TP/PP are for models that do not fit one 15 GiB GPU; that case was not measured.
The development machine has one GeForce 940M (SM50) and is never used for GPU claims; the CPU tests use mocked GPUs and fake workers.

| Label | Meaning |
|---|---|
| Implemented | code exists in this repository |
| CPU-tested | covered by `tests/` using mocks / fake workers (no GPU) |
| CUDA-compiled | native C++ compiles for `sm_75` locally and links only `libcudart.so.13` |
| Kaggle-tested | executed on 2 × T4 by the owner (see [docs/KAGGLE_T4_VERIFICATION.md](docs/KAGGLE_T4_VERIFICATION.md)); applies to the legacy bundle's code unless stated |
| Unsupported | cannot work on this build (bfloat16, FlashAttention/FlashInfer/FlashMLA, native FP8/FP4, L4/SM89) and is rejected with a clear error |
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

### Canonical: integrated wheel, build 3 (verification pending — notebook 15)
Attach the dataset containing `stokastisk-0.1.0-build3-kaggle-t4-sm75-kaggle-input.zip` and run notebook 15 (`docs/KAGGLE_BUILD3_VERIFICATION.md`). The first cell finds the attached bundle, verifies it and prints the exact build; there is **no silent download fallback**. Manual install of an extracted bundle:
```python
import subprocess, sys
subprocess.run([sys.executable, "scripts/kaggle/02_install_integrated.py", "<bundle dir>"], check=True)      # --dry-run plans only; --with-models adds the optional models lock
```
The installer verifies every bundle file against `SHA256SUMS`, the wheel against the manifest, Python 3.13, PyTorch 2.11.0+cu130, Triton 3.6.0, CUDA 13.0 and two T4 GPUs; installs the wheel with `--no-deps --no-index` and the hash-pinned locks (`core`, `structured-output`, `serving`) with `pip download --require-hashes --no-deps` then `--no-deps --no-index`; proves that no pre-installed package changed (one inventory implementation, PEP 503 names); and requires `pip check` to show **no new line**. It never runs a resolver and never touches torch/triton/CUDA/NCCL. **Never install the integrated wheel together with the separate vLLM wheel** — both own `vllm/`.

```python
import stokastisk
llm = stokastisk.LLM("Qwen/Qwen2.5-0.5B-Instruct", config=stokastisk.StokastiskConfig(max_model_len=2048))
print(llm.chat([{"role": "user", "content": "Hello"}], stokastisk.SamplingParams(max_tokens=32))[0].outputs[0].text)   # Kaggle-tested on build 2; build 3 pending
print(llm.generate_structured(["Describe Paris."], {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}))   # pinned xgrammar backend + schema validation; verification pending
llm.close()          # raises EngineCloseError if the engine's own shutdown failed (after finishing the cleanup)
```
OpenAI-compatible server (CPU-tested; Kaggle verification pending for build 3): `with stokastisk.serving.ServerProcess("Qwen/Qwen2.5-0.5B-Instruct") as srv: srv.wait_ready(); print(stokastisk.serving.chat(srv.base_url, "Qwen/Qwen2.5-0.5B-Instruct", [{"role": "user", "content": "Hi"}]))`.
Parallelism: `StokastiskConfig(tensor_parallel_size=2)` (TP=2), `StokastiskConfig(pipeline_parallel_size=2, execution_strategy="pipeline_parallel")` (PP=2, experimental), `stokastisk.parallel.ReplicaPool(model, devices=[0, 1])` (two independent replicas); `stokastisk.discover_topology()` and `stokastisk.recommend_parallelism(...)` choose between them. For a model that fits one T4, **TP=1 is fastest**.

### Legacy two-wheel bundle (Kaggle-verified; used by notebooks 01–14)

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

> **Distributions.** (1) *Integrated wheel* — canonical, one file with the complete SDK (`compat`, `serving`, `LLM.chat`, bfloat16 pre-check) and the bundled vLLM 0.25.1 tree and CUDA 13.0 `sm_75` objects; Kaggle verification **pending (notebook 15)**. (2) *Legacy two-wheel bundle* — the Kaggle-verified release; its SDK wheel (`29edaa55…`) was built before `compat`/`serving`/`LLM.chat` existed and does not contain them. Project version is exactly `0.1.0` in both; only the PEP 427 build tag distinguishes them. Details, hashes and the exact unverified boundary: [docs/INTEGRATED_RELEASE.md](docs/INTEGRATED_RELEASE.md), [docs/KAGGLE_T4_VERIFICATION.md](docs/KAGGLE_T4_VERIFICATION.md).

Not every vLLM feature works on a Tesla T4. [docs/VLLM_T4_COVERAGE.md](docs/VLLM_T4_COVERAGE.md) lists, per feature, whether it was **executed on Kaggle**, only compiled, CPU-tested, statically inspected or not run — and what cannot work on SM75 (bfloat16, Flash*, native FP8/FP4, Qwen Omni).
`stokastisk.check_model(...)`, `check_dtype`, `check_quantization` and `check_feature` raise a clear `StokastiskCompatibilityError` before an engine starts; `stokastisk.serving` launches the bundled OpenAI-compatible server (implemented, CPU-tested and packaged in the integrated wheel; not yet run on Kaggle).

## Install options

| Option | When |
|---|---|
| **Hub CDN** (above) | default; no dataset needed; needs Internet ON; integrated wheel or legacy bundle |
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
| `notebooks/kaggle-t4-sm75/notebooks/` | the 14 legacy-bundle Kaggle notebooks (sources without outputs; executed copies in `evidence/kaggle-legacy-two-wheel/`) |
| `notebooks/kaggle-t4-sm75/integrated/` | notebook 15, the integrated-wheel verification (UNEXECUTED) |
| `evidence/` | executed Kaggle notebooks 13/14 and their acceptance reports (legacy bundle), install and dependency evidence |
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
