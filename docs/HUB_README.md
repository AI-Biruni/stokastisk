---
license: apache-2.0
library_name: stokastisk
tags: [vllm, kaggle, tesla-t4, sm75, cuda-13, triton, inference, python-3.13]
---
# Stokastisk 0.1.0 — vLLM 0.25.1 (Triton attention) for Kaggle's 2 × Tesla T4 (SM75)

A Python 3.13 SDK with a bundled, **Flash‑free** vLLM 0.25.1 (`TRITON_ATTN`, CUDA 13.0 `sm_75` binaries) that installs next to Kaggle's own PyTorch 2.11.0+cu130 / Triton 3.6.0 and never replaces them. Source and docs: <https://github.com/AI-Biruni/stokastisk>

**Status: verification pending.** Built, independently verified and CPU‑tested; the Kaggle GPU stages (`engine`, `parallel`, `server`) of notebook 15 must be run on 2 × T4 by the owner for this wheel before it is called production‑verified. Details: `docs/PRODUCTION_READINESS.md` on GitHub.

| | |
|---|---|
| Project version | **0.1.0** (locked; the wheel build tag is 3) |
| Runtime | Python 3.13 · vLLM 0.25.1 · PyTorch 2.11.0+cu130 · Triton 3.6.0 · CUDA 13.0 · 2 × Tesla T4 |
| Attention | `TRITON_ATTN` only — no FlashAttention / FlashInfer / FlashMLA code is shipped |
| Structured output | xgrammar 0.2.3, pinned (never `auto`), outputs validated against the schema |
| Unsupported | bfloat16, Flash*, FP8/FP4, L4/SM89, Mistral‑family models, OpenTelemetry exporters, direct `vllm.LLM` |

## Files (`releases/stokastisk-0.1.0-build3-kaggle-t4-sm75/`)
| File | SHA256 |
|---|---|
| `wheels/stokastisk-0.1.0-3-cp313-cp313-linux_x86_64.whl` (38,056,415 B) | `2c25da2eadab1ebfcf781f4b63b27a972fc911f60999ee87cce5bd167cfdd316` |
| `stokastisk-0.1.0-build3-kaggle-t4-sm75-kaggle-input.zip` (36,968,171 B) — the complete bundle | `97a377559f5b3ec0609f0d2d526d4cbbf96991e56e07597fd494c7fdf3d9517b` |
| `15_stokastisk_candidate_sdk_integration.ipynb` | pins the hashes above |
| `manifests/` | `RELEASE-MANIFEST.json`, `BUILD-MANIFEST.json`, `DEPENDENCY-MANIFEST.json`, `SHA256SUMS` |

## Use on Kaggle (GPU T4 × 2, Internet ON, Python 3.13)
Upload the ZIP **unchanged** as a Kaggle Dataset, attach it, import the notebook and *Run All*. Cell 1 prints the exact release, wheel SHA256 and number of verified files; nothing is downloaded silently. The notebook fails unless every stage is `PASS` and always writes `/kaggle/working/15-stokastisk-build3-acceptance.zip`.

## License
Apache‑2.0. vLLM is Apache‑2.0 © the vLLM project.
