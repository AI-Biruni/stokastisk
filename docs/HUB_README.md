---
license: apache-2.0
library_name: stokastisk
tags: [vllm, kaggle, tesla-t4, sm75, cuda-13, triton, tensor-parallel, inference, python-3.13]
---
# Stokastisk 0.1.0 — vLLM 0.25.1 for Kaggle's 2 × Tesla T4 (SM75)

A Python 3.13 SDK with a **bundled, T4-only vLLM 0.25.1 (Triton attention, CUDA 13.0 `sm_75` binaries)**. It installs next to Kaggle's own PyTorch 2.11.0+cu130 / Triton 3.6.0 and never replaces them. Stokastisk adds validated configuration, topology discovery, TP / PP / independent-replica execution, a parallelism recommender, telemetry, communication helpers, static T4 compatibility checks and an OpenAI-compatible server launcher.
Source, docs and tooling: <https://github.com/AI-Biruni/stokastisk>

| | |
|---|---|
| Project / version | Stokastisk **0.1.0** (permanently locked — `VERSION_POLICY.md` on GitHub) |
| **Canonical distribution** | **integrated wheel** `stokastisk-0.1.0-2-cp313-cp313-linux_x86_64.whl` · release `stokastisk-0.1.0-integrated-kaggle-t4-sm75` · sha256 `@@WHEEL_SHA256@@` · **GPU verification pending (notebook 15)** |
| Legacy distribution | two wheels, release `stokastisk-0.1.0-kaggle-t4-sm75` — `KAGGLE_T4_VERIFIED` |
| Python / PyTorch / Triton | 3.13 / 2.11.0+cu130 (Kaggle's) / 3.6.0 (Kaggle's) |
| CUDA | Toolkit 13.0 only — binaries link `libcudart.so.13`; `sm_75` cubins, no PTX |
| vLLM | 0.25.1 |
| Attention | `TRITON_ATTN` only — no FlashAttention / FlashInfer / FlashMLA code |
| Hardware | Kaggle, 2 × Tesla T4 (SM75, ~15 GiB each) |

## Status
**Integrated wheel: built, statically inspected, CPU-tested — Kaggle verification PENDING (notebook 15).** **Legacy two-wheel bundle: `KAGGLE_T4_VERIFIED`.** The evidence below was produced with the legacy bundle; it covers the engine, the protected environment and the TP/PP/replica paths that the integrated wheel contains byte-identically (vLLM tree and native objects), not the integrated wheel's own installation, `LLM.chat`, `stokastisk.compat` or `stokastisk.serving`.

### Legacy bundle evidence (`KAGGLE_T4_VERIFIED`)
All 14 notebooks were executed by the owner on Kaggle 2 × Tesla T4 (PASS, or WARNING for 05 and 10; none failed). TP=2, PP=2 and 2-replica inference run on the Flash-free wheel with both GPUs in use.
For Qwen2.5-1.5B-Instruct (decode, batch 8): TP=1 241 tok/s, TP=2 133, PP=2 146, 2 replicas 226. **Use TP=1 whenever the model fits one T4**; TP/PP are for models that do not fit 15 GiB (not measured here). Peer access differs between Kaggle VMs.
Notebooks 13 and 14 were run twice: first with SDK wheel `e17b99b4…` (2026-11-11), then again on 2026-11-12 with **the released SDK wheel `29edaa55…` itself** (13: 7/7 stages, TP=1/TP=2/PP=2 12/12 checks each; 14: 50/50 checks, 15 evenly split rounds; 2026-11-12 decode, batch 8: TP=1 238, TP=2 133, PP=2 148 tok/s; two replicas 0.91–0.95× of TP=1). Notebooks 01–12 ran the earlier wheel `a6b8596b…`; the vLLM wheel is identical in all runs. One informational WARNING: TP=2 text is not byte-identical to TP=1. Details and the original executed notebooks: `docs/KAGGLE_T4_VERIFICATION.md`, `evidence/kaggle-legacy-two-wheel/`. The host-staged CUDA transport remains experimental. L4/SM89 is **not** supported.
Implemented in the SDK: GPU topology and peer-access detection, PCIe reporting, TP/PP strategies, independent dual-GPU replicas with a token-aware scheduler and failure recovery, a parallelism recommender, GPU telemetry, CUDA streams/events, pinned-memory transport vs NCCL, and a benchmark framework.
Kaggle's own image resolves NCCL to the cu12 build 2.32.3 (two NCCL packages share one file); our binaries do not link it. Details in the GitHub `RELEASE.md`.

## Integrated wheel (canonical) — `BUILT, STATICALLY INSPECTED, CPU-TESTED — KAGGLE VERIFICATION PENDING`
One file — `releases/stokastisk-0.1.0-integrated-kaggle-t4-sm75/wheels/stokastisk-0.1.0-2-cp313-cp313-linux_x86_64.whl` (@@WHEEL_SIZE@@ bytes, sha256 `@@WHEEL_SHA256@@`) — contains the complete SDK (`stokastisk.compat`, `stokastisk.serving`, `LLM.chat`, the explicit-bfloat16 pre-check, topology, parallel, telemetry, CUDA, communication, benchmarks) **and** the bundled vLLM 0.25.1 tree (every file byte-identical to the Kaggle-verified vLLM wheel `cc3bcafb…`) with its six native objects (`sm_75` cubins only, no PTX, `libcudart.so.13`, no Flash code). Project version is exactly 0.1.0; build tag `2` is the only thing that distinguishes it from the earlier build-tag-1 candidate (`sdk-candidate/`, superseded, kept) and from the legacy wheels. Requires Python `>=3.13,<3.14`; pins `torch==2.11.0+cu130`, `triton==3.6.0`, `transformers>=5.5.3,<6`.

**Install (Kaggle, Internet ON, GPU T4 × 2):** the installer verifies the SDK/torch/triton/CUDA/GPU preconditions and the wheel hash, installs the wheel `--no-deps --no-index`, hash-verifies and installs the eight missing pins, proves no preinstalled package changed, and prints every check as PASS / FAIL / WARNING / SKIPPED.
```python
import urllib.request, runpy, subprocess, sys
open("/tmp/fetch_release.py", "wb").write(urllib.request.urlopen("https://huggingface.co/waqasm86/stokastisk/resolve/main/scripts/fetch_release.py").read())
bundle = runpy.run_path("/tmp/fetch_release.py")["fetch_release"]("stokastisk-0.1.0-integrated-kaggle-t4-sm75")
subprocess.run([sys.executable, f"{bundle}/scripts/kaggle/02_install_integrated.py", str(bundle)], check=True)
```
Direct wheel: `https://huggingface.co/waqasm86/stokastisk/resolve/main/releases/stokastisk-0.1.0-integrated-kaggle-t4-sm75/wheels/stokastisk-0.1.0-2-cp313-cp313-linux_x86_64.whl`. Never install it together with the separate `vllm` wheel (both own `vllm/`); the installer removes a legacy two-wheel install first.

| Feature | Label |
|---|---|
| Core SDK, TP=1 / TP=2 / PP=2 engine paths, replicas, topology, recommender, telemetry | Implemented, CPU-tested; **Kaggle-tested on the legacy bundle** (notebooks 13, 14 and earlier); integrated install **pending** |
| `LLM.chat`, `stokastisk.compat`, `stokastisk.serving` (OpenAI-compatible server, SSE streaming, metrics), bfloat16 pre-check, hardening of `LLM`/`ReplicaPool` lifecycle | Implemented, CPU-tested, **not Kaggle-tested** |
| Host-staged CUDA transport | CUDA-compiled, Experimental (notebook 10 WARNING), not integrated into vLLM |
| bfloat16, FlashAttention / FlashInfer / FlashMLA, native FP8/FP4, L4 / SM89 | Unsupported — rejected with a clear error |

**Known limitations.** `pip check` is not clean: `xgrammar 0.2.5.post1` declares `transformers<5` (documented, CPU-functionally verified, GPU structured decoding unverified — `docs/DEPENDENCY_COMPATIBILITY.md`). TP=2, PP=2 and two replicas are **slower** than TP=1 for a model that fits one T4. Peer access varies between Kaggle VMs. Calling `vllm.LLM` directly (without the SDK) bypasses the SDK's explicit `TRITON_ATTN` selection; multimodal vision towers use `TORCH_SDPA` on SM75. Details: `docs/INTEGRATED_RELEASE.md`, `docs/SDK_CANDIDATE.md` (history of the build-tag-1 candidate).

## Kaggle hard limits the SDK is built for (`stokastisk.LIMITS`)
4 CPU cores (thread budget split across TP workers) · ~30 GiB RAM · 20 GiB `/kaggle/working` · 2 × T4 (SM75, ~15 GiB VRAM each).

## Fast install (Hub CDN)
Kaggle: Internet ON, Accelerator *GPU T4 × 2*. `fetch_release.py` is stdlib-only: one ~37 MB archive, a single HTTP stream (parallel ranges are opt-in via `workers`; 4 parallel ranges stalled on the Hub CDN in a workstation test) over the CDN, verified against its `.sha256`.
```python
import urllib.request, runpy, subprocess, sys
open("/tmp/fetch_release.py", "wb").write(urllib.request.urlopen("https://huggingface.co/waqasm86/stokastisk/resolve/main/scripts/fetch_release.py").read())
bundle = runpy.run_path("/tmp/fetch_release.py")["fetch_release"]("stokastisk-0.1.0-kaggle-t4-sm75")
subprocess.run([sys.executable, f"{bundle}/scripts/kaggle/02_install.py", str(bundle)], check=True)   # --no-deps, protected packages untouched
```
Then, as top-level notebook statements:
```python
import stokastisk

config = stokastisk.StokastiskConfig(tensor_parallel_size=1)
llm = stokastisk.LLM("Qwen/Qwen2.5-0.5B-Instruct", config=config)
print(llm.complete("The capital of France is"))
print(llm.verify_attention_backend())      # "TRITON_ATTN"
llm.close()
```
vLLM 0.25.1 uses `spawn`; the acceptance notebooks run each engine in a separate process (`scripts/kaggle/05_inference.py`, `dual_15_tp_pp_fulfilment.py`), which is the pattern that was validated. Running `stokastisk.LLM` directly in a notebook cell was not part of the acceptance runs.

## Files
| Path | What |
|---|---|
| `latest.json` | machine-readable pointer: release name, status, sizes, SHA256 |
| `scripts/fetch_release.py` | CDN downloader (stdlib) |
| `releases/stokastisk-0.1.0-kaggle-t4-sm75/stokastisk-0.1.0-kaggle-t4-sm75-kaggle-input.zip` (+ `.sha256`) | the complete release (installer, scripts, wheels, requirements, docs) — also usable as a Kaggle dataset |
| `releases/stokastisk-0.1.0-kaggle-t4-sm75/wheels/` | `stokastisk-0.1.0-py3-none-any.whl` and `vllm-0.25.1-cp38-abi3-linux_x86_64.whl` for direct access |
| `releases/stokastisk-0.1.0-kaggle-t4-sm75/manifests/` | `SHA256SUMS` and `RELEASE-MANIFEST.json` |

`latest.json` carries the per-notebook results (`kaggle_acceptance`). The documentation inside the archive carries the same `KAGGLE_T4_VERIFIED` status and results as this card.

## License
Apache-2.0. vLLM is Apache-2.0 © the vLLM project.
