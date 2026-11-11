---
license: apache-2.0
library_name: stokastisk
tags: [vllm, kaggle, tesla-t4, sm75, cuda-13, triton, tensor-parallel, inference, python-3.13]
---
# Stokastisk 0.1.0 — vLLM 0.25.1 for Kaggle's 2 × Tesla T4 (SM75)

A Python 3.13 SDK plus a **T4-only vLLM 0.25.1 build with standalone CUDA 13.0 binaries**. It installs next to Kaggle's own PyTorch 2.11.0+cu130 and never replaces it.
Source, docs and tooling: <https://github.com/AI-Biruni/stokastisk>

| | |
|---|---|
| Project / version | Stokastisk **0.1.0** (permanently locked — `VERSION_POLICY.md` on GitHub) |
| Release | `stokastisk-0.1.0-kaggle-t4-sm75` |
| Python / PyTorch / Triton | 3.13 / 2.11.0+cu130 (Kaggle's) / 3.6.0 (Kaggle's) |
| CUDA | Toolkit 13.0 only — binaries link `libcudart.so.13`; `sm_75` cubins, no PTX |
| vLLM | 0.25.1 |
| Attention | `TRITON_ATTN` only — no FlashAttention / FlashInfer / FlashMLA code |
| Hardware | Kaggle, 2 × Tesla T4 (SM75, ~15 GiB each) |

## Status: `KAGGLE_T4_VERIFIED`
All 14 notebooks were executed by the owner on Kaggle 2 × Tesla T4 (PASS, or WARNING for 05 and 10; none failed). TP=2, PP=2 and 2-replica inference run on the Flash-free wheel with both GPUs in use.
For Qwen2.5-1.5B-Instruct (decode, batch 8): TP=1 241 tok/s, TP=2 133, PP=2 146, 2 replicas 226. **Use TP=1 whenever the model fits one T4**; TP/PP are for models that do not fit 15 GiB (not measured here). Peer access differs between Kaggle VMs.
Notebooks 13–14 ran SDK wheel `e17b99b4…`; 01–12 ran the earlier wheel `a6b8596b…`; the vLLM wheel is identical in all runs. The released SDK wheel (`29edaa55…`) differs from the executed one only in comments, note strings and the packaged README (no code change), and was not re-executed itself. Details: `docs/KAGGLE_T4_VERIFICATION.md` on GitHub. The host-staged CUDA transport remains experimental. L4/SM89 is **not** supported.
Implemented in the SDK: GPU topology and peer-access detection, PCIe reporting, TP/PP strategies, independent dual-GPU replicas with a token-aware scheduler and failure recovery, a parallelism recommender, GPU telemetry, CUDA streams/events, pinned-memory transport vs NCCL, and a benchmark framework.
Kaggle's own image resolves NCCL to the cu12 build 2.32.3 (two NCCL packages share one file); our binaries do not link it. Details in the GitHub `RELEASE.md`.

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
