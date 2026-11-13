---
license: apache-2.0
library_name: stokastisk
tags: [vllm, kaggle, tesla-t4, sm75, cuda-13, triton, inference, python-3.13]
---
# Stokastisk 0.1.0 — vLLM 0.25.1 (Triton attention) for Kaggle's 2 × Tesla T4 (SM75)

A Python 3.13 SDK with a bundled, **Flash‑free** vLLM 0.25.1 (`TRITON_ATTN`, CUDA 13.0 `sm_75` binaries) that installs next to Kaggle's own PyTorch 2.11.0+cu130 / Triton 3.6.0 and never replaces them. Source and docs: <https://github.com/AI-Biruni/stokastisk>

**Status: verification pending.** Built, independently verified and CPU‑tested. On the owner's 2 × T4 run of the previous revision: installer 32/32, static 79/79, parallel (TP=2, PP=2, replicas) 23/23, server 20/21; the remaining failures (structured output truncated by an unbounded schema) are fixed here. The `engine` and `server` stages must be re‑run on 2 × T4 for this wheel before it is called production‑verified. Details: `docs/PRODUCTION_READINESS.md` on GitHub.

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
| `wheels/stokastisk-0.1.0-3-cp313-cp313-linux_x86_64.whl` (38,056,842 B) | `b31588c9b5f0d5ae103903283c7cfc25a9d7ef6c00112c3d46bb7b9ee84b023a` |
| `stokastisk-0.1.0-build3-kaggle-t4-sm75-kaggle-input.zip` (36,969,662 B) — the complete bundle | `eb9198788afd93413c457ddd074ece16977ae19aa8e7487ba7f3a4c3f8b5c92a` |
| `15_stokastisk_candidate_sdk_integration.ipynb` | pins the hashes above |
| `manifests/` | `RELEASE-MANIFEST.json`, `BUILD-MANIFEST.json`, `DEPENDENCY-MANIFEST.json`, `SHA256SUMS` |

## Use on Kaggle (GPU T4 × 2, Internet ON, Python 3.13)
Upload the ZIP **unchanged** as a Kaggle Dataset, attach it, import the notebook and *Run All*. Cell 1 prints the exact release, wheel SHA256 and number of verified files; nothing is downloaded silently. The notebook fails unless every stage is `PASS` and always writes `/kaggle/working/15-stokastisk-build3-acceptance.zip`.

## License
Apache‑2.0. vLLM is Apache‑2.0 © the vLLM project.
