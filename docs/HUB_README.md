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
| **Canonical distribution** | **integrated Flash-free wheel, build 3** `stokastisk-0.1.0-3-cp313-cp313-linux_x86_64.whl` · release `stokastisk-0.1.0-build3-kaggle-t4-sm75` · sha256 `1d43089a85187469913719e52771387087821695efbc7c0132b0732f09877ad3` · **Kaggle verification pending (notebook 15, build 3)**. Build 2 (`7e4b7009…`) *failed* notebook 15 and is superseded; build 1 (`815f9783…`) was never executed |
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

## Integrated wheel, build 3 (canonical) — `BUILT, INDEPENDENTLY VERIFIED, CPU-TESTED — KAGGLE VERIFICATION PENDING`
* `releases/stokastisk-0.1.0-build3-kaggle-t4-sm75/wheels/stokastisk-0.1.0-3-cp313-cp313-linux_x86_64.whl` — 38,060,994 bytes, sha256 `1d43089a85187469913719e52771387087821695efbc7c0132b0732f09877ad3`.
* `releases/stokastisk-0.1.0-build3-kaggle-t4-sm75/stokastisk-0.1.0-build3-kaggle-t4-sm75-kaggle-input.zip` — 36,977,553 bytes, sha256 `5a223beff26d28095369e17ed078d2ed8413bd7974e6062a840794c67a1e4f3a` (the **complete** bundle: wheel, installer, stage scripts, hash-pinned locks, manifests, docs, notebook 15).
* `releases/stokastisk-0.1.0-build3-kaggle-t4-sm75/15_stokastisk_candidate_sdk_integration.ipynb` — the standalone notebook (pins both hashes above). Instructions: `docs/KAGGLE_BUILD3_VERIFICATION.md`.

Why a new build: **build 2 failed Kaggle notebook 15** (structured output: `No module named 'lmformatenforcer'`; HTTP serving: `No module named 'model_hosting_container_standards'`; a package-inventory comparison with two name normalisers; Flash `.pyc` files; and the notebook silently downloaded build 2 instead of the uploaded wheel). Basic FP16 inference, chat, complete and cleanup had passed. Build 3 removes the 28 Flash-family modules from the vLLM tree, ships complete hash-pinned locks (`core`, `structured-output`, `serving`, optional `models`), pins the xgrammar backend (xgrammar 0.2.3, metadata-compatible, nothing patched), and requires an attached, verified bundle (no silent download). Project version stays **0.1.0**.

Use it on Kaggle by attaching the ZIP as a dataset and running the standalone notebook; there is deliberately no one-line download install for build 3 yet (`fetch_release.py` serves the legacy and build-2 releases).

| Feature | Label |
|---|---|
| Core SDK, TP=1 / TP=2 / PP=2 engine paths, replicas, topology, recommender, telemetry | Implemented, CPU-tested; **Kaggle-tested on the legacy bundle** (notebooks 13, 14 and earlier); build-3 install **verification pending** |
| FP16 `generate` / `chat` / `complete`, `TRITON_ATTN`, cleanup | **Kaggle-tested on build 2** (same code path); build 3 re-run pending |
| Structured output (pinned xgrammar) | Implemented, CPU-tested (vLLM's own `XgrammarBackend` on CPU); **Verification pending** (failed on build 2 for a missing package) |
| OpenAI-compatible server, SSE streaming, metrics, restart | Implemented, CPU-tested against fake servers; **Verification pending** (failed on build 2 for a missing package) |
| Flash-free tree (28 modules removed, byte-equal re-derivation, 0 import regressions in a 1,783-module CPU sweep) | Implemented, independently verified; GPU run **pending** |
| `ReplicaPool` worker-failure recovery | CPU-tested with fake workers only |
| Observability (OpenTelemetry exporters), direct `vllm.LLM`/`vllm serve`, bfloat16, FlashAttention / FlashInfer / FlashMLA, native FP8/FP4, L4 / SM89 | Unsupported |

Remaining limitations: `STOKASTISK_0.1.0_BUILD3_PRODUCTION_READINESS.md` on GitHub.

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
