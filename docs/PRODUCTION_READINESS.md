# Production readiness — Stokastisk 0.1.0

## Release assessment

| Item | Value |
|---|---|
| Project version | **0.1.0** (locked; `v0.1.0` tag and GitHub release unchanged) |
| Current wheel | build tag **4** — `stokastisk-0.1.0-4-cp313-cp313-linux_x86_64.whl`, 38,057,512 bytes, SHA256 `052156d71adc6ad9966901565f86dbded9ff41c620c782db4ff377a1a2b4f62e` |
| Earlier build | build tag 3 (SHA256 `72039ca4044e48fdf5edca8051dcb48b42fe61a69e07715fa8803e029581fe12`) was the build the Kaggle runs used; it has been **withdrawn** and is not an install target |
| Bundled vLLM | 0.25.1, from the byte-pinned base wheel SHA256 `cc3bcafb6516791990ca302073ae153a08465d3df1e460d666e3ec09b4e98f5e` (natives compiled once for `sm_75`, CUDA 13.0) |
| Runtime target | Python 3.13 · CUDA 13.0 · PyTorch 2.11.0+cu130 · Triton 3.6.0 · 2 × Tesla T4 (SM75) · FP16 · `TRITON_ATTN` only |
| Machine-readable identities | [RELEASE.json](../RELEASE.json) |

**Verdict: production-ready for Kaggle 2 × Tesla T4, within the verified scope below.** "Verified scope" means: the models and modes in the table below, FP16, `max_model_len` 2048, and the features marked Kaggle-tested in the [feature matrix](guide/feature-matrix.md). Everything else is untested, experimental or unsupported and is labelled so.

## Evidence — three separate classes
1. **Build 3, Kaggle acceptance (notebook 15, 2026-11-13).** installer 32/32, static 79/79, engine 44/44, parallel 23/23, server 21/21, summary 15/15, run on the acceptance wheel `b31588c9b5f0d5ae103903283c7cfc25a9d7ef6c00112c3d46bb7b9ee84b023a` with Qwen2.5-0.5B-Instruct. Raw reports and the executed notebook: `evidence/kaggle/2026-11-13-notebook-15-build3-acceptance/` (the two original ZIP archives are not in the repository; their SHA256 values are in `evidence/kaggle/evidence-manifest.json`). The published build-3 wheel differs from the acceptance wheel only in `METADATA`/`RECORD` (2,580 of 2,582 members byte-identical).
2. **Build 3, end-user notebooks 16, 18, 19 (2026-11-13/14).** The published install, then inference. Python 3.13.16, PyTorch 2.11.0+cu130, CUDA 13.0, Triton 3.6.0, two Tesla T4 (SM75). Executed notebooks with outputs: `evidence/kaggle/`.
3. **Build 4, CPU only.** Independent wheel audit (21 checks), member-by-member delta against build 3 (when it was still published), reproducible rebuild, unit and documentation-consistency tests. **Build 4 was not separately run on Kaggle.**

### Exact verified models (what `stokastisk.check_model()` reports)
| Model id | Dtype | Modes run | Max `max_model_len` | Evidence |
|---|---|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | FP16 | TP=1; TP=2, PP=2, two replicas | 2048 | notebook 15 (engine, server, parallel stages), 16, 18 |
| `Qwen/Qwen2.5-3B-Instruct` | FP16 | TP=2, PP=2 | 2048 | notebook 18 |
| `Qwen/Qwen2.5-Coder-7B-Instruct` (original unquantized checkpoint) | FP16 | TP=2, PP=2 | 2048 | notebook 19 |

Not claimed: that every `Qwen2ForCausalLM` checkpoint works (any other checkpoint is `UNKNOWN_REQUIRES_EVIDENCE`); that other 7B models fit (the Coder-7B run left about 1.34 GiB free per GPU at TP=2 and 2.42 / 1.93 GiB at PP=2); that PP=2 is faster than TP=1 (it is not, for models that fit one T4); any throughput figure from these single runs. Qwen2.5-1.5B-Instruct appears in older notes but its raw evidence is no longer part of the project, so it is not listed.

Lifecycle: after every engine in notebooks 18 and 19 both GPUs returned to 14.46 of 14.56 GiB free. At the end of each TP=2 and PP=2 engine vLLM's process manager logged `force killing remaining processes count=1`; `close()` still completed and memory was recovered. The two facts are recorded separately in the evidence manifest.

## Build 4 — what changed and how it was verified
| | |
|---|---|
| Defect corrected | The shipped SDK carried build-time evidence: the OpenAI server, streaming, metrics and structured output were labelled "unknown / pending", chat cited a failed build-2 run, offline generation named only the two smallest Qwen2.5 models, `check_model()` marked **any** Qwen2 checkpoint `SUPPORTED`, and the `serving` docstring said server verification was pending. |
| Change | `stokastisk/compat.py`: evidence records updated from the recorded runs; new exact-model registry `MODEL_EVIDENCE`; `check_model()` answers per model id (`model_status`, `architecture_status`, `tested_modes`, `mode_status`, …). `stokastisk/serving.py`: docstring only. `pyproject.toml`: the wheel description is a hash-free `README.wheel.md`. |
| Unchanged | all 2,517 `vllm/**` files, all 6 native binaries, the bundled vLLM dist-info, every `Requires-Dist` pin, the lock files, `engine.py`, `config.py`, parallel/ReplicaPool, CUDA/NCCL code. |
| Delta audit | of 2,582 members, 2,577 byte-identical; the 5 that differ are `stokastisk/compat.py`, `stokastisk/serving.py`, `METADATA`, `WHEEL`, `RECORD`. `serving.py` has an identical AST once docstrings are removed; the `compat` names the runtime imports (`check_dtype`, `check_quantization`, `check_feature`, `get_feature`, `BLOCKING`, `SM75`, `QUANT_MIN_CAPABILITY`) are AST-identical. |
| Reproducibility | a second build produced the identical SHA256. |
| Independent audit | RECORD hashes, strict Flash policy (0 hits), natives equal to the pinned vLLM wheel, `vllm/**` equals pinned wheel + documented transform, dependency pins — 21/21. |
| Not done | no GPU test, no Kaggle run, no model inference, no CUDA initialisation. Build 3's Kaggle evidence is **inherited** for the unchanged implementation paths, and only because the delta audit proves them unchanged. |

## Gates
| Gate | State |
|---|---|
| A source quality | met — CPU suite passes (see [RELEASE.md](../RELEASE.md) for the current counts); evidence/documentation consistency tests |
| B dependency integrity | met — four hash-pinned locks resolved against Kaggle's real inventory, 0 conflicts; the wheel pins all 19 new packages exactly; unchanged in build 4 |
| C package integrity | met — tags, RECORD, byte-equality with the pinned vLLM wheel + documented transform, 0 Flash files / markers / bindings, natives unchanged, reproducible |
| D delivery | met — wheels on the Hugging Face CDN, SHA256-pinned install command, optional PEP 503 index on GitHub Pages |
| E Kaggle GPU verification | **met for build 3** (acceptance + notebooks 16/18/19); **inherited, not re-run, for build 4** |
| F Documentation | met — [user guide](guide/index.md) (also <https://stokastisk-ai.github.io/stokastisk/docs/>), generated [feature matrix](guide/feature-matrix.md), maintainer notes here; tests fail on broken links, stale API reference, missing referenced files, or a document that contradicts the SDK's evidence |

## Feature labels (the authoritative table is the generated [feature matrix](guide/feature-matrix.md))
| Feature | Label |
|---|---|
| FP16 generate / chat / complete, `TRITON_ATTN`, determinism, timeouts, serialised calls, cleanup | Kaggle-tested |
| Structured output (xgrammar 0.2.3 pinned, schema-validated, no fallback) | Kaggle-tested, with conditions (bounded schemas; acceptance model 0.5B) |
| `ServerProcess` (OpenAI-compatible, streaming, restart, startup-failure report) | Kaggle-tested, with conditions (Qwen2.5-0.5B, TP=1) |
| `/metrics` | Kaggle-tested, with conditions (presence of `vllm:` samples, not their values) |
| TP=2, ReplicaPool | Kaggle-tested (slower than TP=1 for models that fit one T4); TP=2 also with Qwen2.5-3B and Coder-7B |
| PP=2 | Kaggle-tested, **experimental** (also 3B and Coder-7B) |
| `ReplicaPool` worker-failure recovery | CPU-tested with fake workers only |
| Flash-free tree | Kaggle-tested (no Flash backend or library in any process) |
| One-line install | Kaggle-tested for build 3 (notebooks 16, 18, 19); the build-4 install command was verified by downloading the published file and checking its hash, not by a Kaggle run |
| `models` extra (compressed-tensors, depyf, setproctitle) | Implemented, not exercised |
| Prefix caching, chunked prefill, CUDA graphs, custom all-reduce, LoRA, multimodal, speculative decoding | **Unknown** — never run |
| SageMaker `/ping`, `/invocations` | **Not exercised, not claimed** |
| OpenTelemetry exporters, Mistral-family models, bfloat16, FP8/FP4, L4/SM89, Flash*, direct `vllm.LLM` / `vllm serve` | **Unsupported** |

## Limitations
1. Kaggle's `fastapi 0.141.1` is newer than upstream vLLM's `<0.137` cap (SageMaker handler overrides); the app built and served on Kaggle; the SageMaker routes are not advertised and were not exercised.
2. xgrammar 0.2.3 is older than the 0.2.5.post1 vLLM was developed against (inside vLLM's declared range; JSON-schema constrained decoding Kaggle-tested). A grammar cannot make a model stop: bound open-ended schema fields.
3. Offline generation cannot be cancelled: `timeout=` raises, the request runs on and later calls wait for it.
4. `ReplicaPool` workers are non-daemon; a hard-killed parent leaves them (Kaggle ends the session anyway).
5. Only the three models above were run. LoRA, multimodal, CUDA graphs, speculative decoding and other checkpoints are unverified; the 7B run leaves little GPU memory.
6. The observability exporters conflict with Kaggle's protobuf 7.x and are unsupported.
7. `--extra-index-url` (the optional convenience index) also searches PyPI; the recommended command does not use it.
8. The raw original Kaggle archives (about 17 MB, mostly Triton caches) are kept outside the repository; only their reports and logs, plus their hashes, are archived.
9. Build 4 has not been executed on a GPU; if a defect is ever found in a path build 4 shares with build 3, it is in both.
