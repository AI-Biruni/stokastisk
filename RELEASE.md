# Release status — Stokastisk 0.1.0

Project version **0.1.0** (locked — [VERSION_POLICY.md](VERSION_POLICY.md)). GitHub release / tag: `v0.1.0`. The single distribution is the **build‑tag‑3** integrated wheel; earlier integrated builds (1, 2) and the legacy two‑wheel release were removed (build 2 failed Kaggle notebook 15; see [CHANGELOG.md](CHANGELOG.md)).

| Artifact | Size | SHA256 |
|---|---|---|
| `stokastisk-0.1.0-3-cp313-cp313-linux_x86_64.whl` | 38,056,842 | `b31588c9b5f0d5ae103903283c7cfc25a9d7ef6c00112c3d46bb7b9ee84b023a` |
| `stokastisk-0.1.0-build3-kaggle-t4-sm75-kaggle-input.zip` (complete Kaggle input bundle) | 36,969,662 | `eb9198788afd93413c457ddd074ece16977ae19aa8e7487ba7f3a4c3f8b5c92a` |
| bundle `manifests/SHA256SUMS` (pinned by the notebook) | | `624359336e8c54eaa6f01d3e83eea4e7332a10bb92c2b3be545392715b57a659` |
| `vllm-0.25.1-cp38-abi3-linux_x86_64.whl` (byte‑pinned base wheel, build input only) | | `cc3bcafb6516791990ca302073ae153a08465d3df1e460d666e3ec09b4e98f5e` |

Source commit of the build: `c08956f`. Notebook: `15_stokastisk_candidate_sdk_integration.ipynb` (pins the wheel, ZIP and `SHA256SUMS` hashes above).

## Status: verification pending
| Item | State |
|---|---|
| Wheel: tags, RECORD, byte‑equality with the pinned vLLM wheel + documented transform, 0 Flash files / markers / bindings, natives unchanged, reproducible | **verified** (`scripts/audit/verify_build3_wheel.py`) |
| Dependency locks vs Kaggle's real inventory | 0 conflicts; all pins hash‑verified by `pip download --require-hashes` |
| CPU test suite | passes (`CUDA_VISIBLE_DEVICES= python3.13 -m pytest tests`) |
| Kaggle 2 × T4 (owner's run of the previous wheel, 2026-11-13): input discovery, installer **32/32**, static **79/79**, **parallel 23/23** (TP=2, PP=2, ReplicaPool: both GPUs engaged, `TRITON_ATTN`, no Flash library, clean shutdown, memory released), server **20/21** (start, health, models, chat, streaming, HTTP errors, metrics, startup‑failure report, port check, restart, graceful shutdown) | **Kaggle‑tested** |
| The two failures of that run — structured output cut off at `max_tokens` because the acceptance schema had an unbounded integer (the SDK correctly raised `StructuredOutputError`; the stage did not survive it) — | fixed in this wheel/bundle: bounded schemas, explained truncation error, resilient stages |
| Kaggle 2 × T4: `engine` and `server` stages (and the full notebook) for **this** wheel | **Verification pending** |

Do not call this release production‑verified until the executed notebook 15 and its acceptance ZIP show every stage `PASS`. Remaining limits: [docs/PRODUCTION_READINESS.md](docs/PRODUCTION_READINESS.md).

## Known facts
* TP=2, PP=2 and replicas are **slower** than TP=1 for a model that fits one T4 (PCIe host bridge, no NVLink; peer access varies between Kaggle VMs).
* Kaggle resolves NCCL to its cu12 build 2.32.3 (two NCCL packages share one file); our binaries do not link it.
* Unsupported: bfloat16, Flash*, native FP8/FP4, L4/SM89, Mistral‑family models, OpenTelemetry exporters, direct `vllm.LLM`/`vllm serve`.
