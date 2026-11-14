# Stokastisk 0.1.0 — User Guide

Stokastisk is a Python 3.13 SDK for **Kaggle's 2 × Tesla T4 GPUs** (SM75, CUDA 13.0). It bundles vLLM 0.25.1, runs attention only through the Triton backend (`TRITON_ATTN`), and installs with one `pip` line.

!!! This guide is for people running notebooks on Kaggle. Maintainer material (building the wheel, acceptance suite) is in [BUILD.md](../../BUILD.md) and [KAGGLE_ACCEPTANCE.md](../KAGGLE_ACCEPTANCE.md).

## Contents

| # | Page | What you learn |
|---|---|---|
| 1 | [Getting started](getting-started.md) | Notebook settings, install, first generation |
| 2 | [Configuration](configuration.md) | `StokastiskConfig`, defaults, `STOKASTISK_*` environment variables |
| 3 | [The LLM engine](engine.md) | `LLM`: `generate`, `chat`, `complete`, `close` |
| 4 | [Structured output](structured-output.md) | JSON-schema constrained generation and validation |
| 5 | [OpenAI-compatible server](server.md) | `ServerProcess`, `chat`, `stream_chat`, metrics |
| 6 | [Using both GPUs](multi-gpu.md) | TP, PP, `ReplicaPool`, topology, recommender |
| 7 | [Checks and diagnostics](diagnostics.md) · [Feature matrix](feature-matrix.md) | `check_model` (which exact models were run), environment report, Kaggle limits, backend policy, what each feature's evidence is |
| 8 | [Errors](errors.md) | Exception hierarchy and what to do |
| 9 | [Advanced modules](advanced.md) | Benchmarks, telemetry, CUDA helpers, NCCL (experimental) |
| 10 | [Troubleshooting](troubleshooting.md) | Symptoms and fixes |
| — | [API reference](api-reference.md) | Every module and public name |

## Verified on Kaggle

| Run | Result |
|---|---|
| Qwen2.5-0.5B-Instruct, TP=1 | completion, chat, structured JSON; also the notebook 15 acceptance suite (server, streaming, metrics, TP=2, PP=2, replicas) |
| Qwen2.5-3B-Instruct, TP=2 and PP=2 | completion and chat on both GPUs |
| Qwen2.5-Coder-7B-Instruct (FP16, original checkpoint), TP=2 and PP=2 | Python code generated on both GPUs |
| All runs | `TRITON_ATTN` confirmed; both T4s fully released after `close()` |

Details: [KAGGLE_ACCEPTANCE.md](../KAGGLE_ACCEPTANCE.md). Ready-made notebooks: [18](../../notebooks/18_stokastisk_qwen25_t4.ipynb) and [19](../../notebooks/19_stokastisk_qwen25_coder_7b_dual_t4.ipynb).

## Not supported

bfloat16 · FlashAttention / FlashInfer / FlashMLA · native FP8/FP4 · L4 / SM89 · Mistral-family models · OpenTelemetry exporters · calling `vllm.LLM` or `vllm serve` directly.
