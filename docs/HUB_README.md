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

## Status: `KAGGLE_GPU_VERIFICATION_PENDING`
This release is **CPU-tested only** (mocked GPUs, fake workers). Kaggle notebooks 01–12 are **NOT_RUN** for it; PP=2, replica inference and the host-staged CUDA transport are **experimental and not Kaggle-verified**.
The vLLM wheel and native binaries are byte-identical to an earlier internal baseline build that ran TP=1/TP=2 inference on Kaggle (Qwen2.5-1.5B-Instruct: TP=1 949 tok/s, TP=2 521 tok/s, 0.55× — historical, **not validation of 0.1.0**).
**Use TP=1 whenever the model fits one T4**: the two GPUs connect through the host bridge with no peer-to-peer.
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
Then, in a script (vLLM uses `spawn`, so keep the `__main__` guard):
```python
import stokastisk
def main():
    llm = stokastisk.LLM("Qwen/Qwen2.5-0.5B-Instruct", config=stokastisk.StokastiskConfig(tensor_parallel_size=1))
    print(llm.complete("The capital of France is"), llm.verify_attention_backend()); llm.close()
if __name__ == "__main__":
    main()
```

## Files
| Path | What |
|---|---|
| `latest.json` | machine-readable pointer: release name, status, sizes, SHA256 |
| `scripts/fetch_release.py` | CDN downloader (stdlib) |
| `releases/<name>/<name>-kaggle-input.zip` (+ `.sha256`) | the complete release (installer, scripts, wheels, requirements, docs) — also usable as a Kaggle dataset |
| `releases/<name>/wheels/` | `stokastisk-0.1.0-py3-none-any.whl` and `vllm-0.25.1-cp38-abi3-linux_x86_64.whl` for direct access |
| `releases/<name>/manifests/` | `SHA256SUMS` and `RELEASE-MANIFEST.json` |

The archive's documentation, this card and `latest.json` all state the same status: `KAGGLE_GPU_VERIFICATION_PENDING`.

## License
Apache-2.0. vLLM is Apache-2.0 © the vLLM project.
