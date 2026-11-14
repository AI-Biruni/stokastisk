# Production readiness - Stokastisk 0.1.0

**Verdict: production-ready for Kaggle 2 x Tesla T4 (SM75, Python 3.13).** All Kaggle GPU stages passed on 2026-11-13 (installer 32/32, static 79/79, engine 44/44, parallel 23/23, server 21/21, summary 15/15). The published one-line `pip install` itself was then run on Kaggle (notebooks 16, 18, 19, 2026-11-13/14): quickstart; Qwen2.5-0.5B TP=1; Qwen2.5-3B TP=2 and PP=2; Qwen2.5-Coder-7B-Instruct TP=2 and PP=2 - all completed with `TRITON_ATTN`, and both GPUs were fully released after every engine.

| Gate | State |
|---|---|
| A source quality | met - CPU suite passes; lifecycle, configuration, concurrency, replica and serving regressions; no silent `except: pass` |
| B dependency integrity | met - four hash-pinned locks resolved against Kaggle's real inventory, 0 conflicts; the wheel pins all 19 new packages exactly; the resolver simulation changes nothing in the image |
| C package integrity | met - tags, RECORD, byte-equality with the pinned vLLM wheel + documented transform, 0 Flash files / markers / bindings, natives unchanged, reproducible |
| D delivery | met - one wheel on the Hugging Face CDN, PEP 503 index on GitHub Pages (SHA256 in the link) |
| E Kaggle GPU verification | **met** (wheel `b31588c9...`; the published wheel differs only in metadata) |

## Feature labels
| Feature | Label |
|---|---|
| FP16 generate / chat / complete, `TRITON_ATTN`, determinism, timeouts, serialised calls, cleanup | Kaggle-tested |
| Structured output (xgrammar 0.2.3 pinned, schema-validated, no fallback) | Kaggle-tested |
| OpenAI-compatible server, streaming, metrics, restart, startup-failure report | Kaggle-tested |
| TP=2, ReplicaPool | Kaggle-tested (slower than TP=1 for models that fit one T4); TP=2 also with Qwen2.5-3B and Qwen2.5-Coder-7B |
| PP=2 | Kaggle-tested (also Qwen2.5-3B and Qwen2.5-Coder-7B), **Experimental** |
| `ReplicaPool` worker-failure recovery | CPU-tested with fake workers only |
| Flash-free tree | Kaggle-tested (no Flash backend or library in any process) |
| One-line `pip install` | Kaggle-tested (notebooks 16, 18, 19) |
| `models` extra (compressed-tensors, depyf, setproctitle) | Implemented, not exercised |
| Observability (OpenTelemetry), Mistral-family models, bfloat16, FP8/FP4, L4/SM89, direct `vllm.LLM`/`vllm serve` | **Unsupported** |

## Limitations
1. Kaggle's `fastapi 0.141.1` is newer than upstream vLLM's `<0.137` cap (SageMaker handler overrides); the app built and served on Kaggle, the SageMaker routes are not advertised.
2. xgrammar 0.2.3 is older than the 0.2.5.post1 vLLM was developed against (inside vLLM's declared range; JSON-schema constrained decoding Kaggle-tested). A grammar cannot make a model stop: bound open-ended schema fields.
3. Offline generation cannot be cancelled: `timeout=` raises, the request runs on and later calls wait for it.
4. `ReplicaPool` workers are non-daemon; a hard-killed parent leaves them (Kaggle ends the session anyway).
5. Kaggle-exercised models are Qwen2.5-0.5B/3B-Instruct and Qwen2.5-Coder-7B-Instruct (FP16, `max_model_len` 2048); `check_model()` without `architectures=` returns `UNKNOWN_REQUIRES_EVIDENCE` for a model it cannot inspect offline, and its evidence note still names only the 0.5B/1.5B runs. Other models, LoRA, multimodal, CUDA graphs, speculative decoding are unverified.
6. `--extra-index-url` also searches PyPI; the name `stokastisk` is unregistered there today.
