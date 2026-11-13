# Release status — Stokastisk 0.1.0

Project version **0.1.0** (locked — [VERSION_POLICY.md](VERSION_POLICY.md)). GitHub release / tag: `v0.1.0`. The single distribution is the **build‑tag‑3** integrated wheel; earlier integrated builds (1, 2) and the legacy two‑wheel release were removed (build 2 failed Kaggle notebook 15; see [CHANGELOG.md](CHANGELOG.md)).

| Artifact | Size | SHA256 |
|---|---|---|
| `stokastisk-0.1.0-3-cp313-cp313-linux_x86_64.whl` | 38,056,415 | `2c25da2eadab1ebfcf781f4b63b27a972fc911f60999ee87cce5bd167cfdd316` |
| `stokastisk-0.1.0-build3-kaggle-t4-sm75-kaggle-input.zip` (complete Kaggle input bundle) | 36,968,171 | `97a377559f5b3ec0609f0d2d526d4cbbf96991e56e07597fd494c7fdf3d9517b` |
| bundle `manifests/SHA256SUMS` (pinned by the notebook) | | `10e9a57084fd2233508ee0706ed046f86682424751ee524bded96cfd989b938b` |
| `vllm-0.25.1-cp38-abi3-linux_x86_64.whl` (byte‑pinned base wheel, build input only) | | `cc3bcafb6516791990ca302073ae153a08465d3df1e460d666e3ec09b4e98f5e` |

Source commit of the build: `70acd9a`. Notebook: `15_stokastisk_candidate_sdk_integration.ipynb` (pins the wheel, ZIP and `SHA256SUMS` hashes above).

## Status: verification pending
| Item | State |
|---|---|
| Wheel: tags, RECORD, byte‑equality with the pinned vLLM wheel + documented transform, 0 Flash files / markers / bindings, natives unchanged, reproducible | **verified** (`scripts/audit/verify_build3_wheel.py`) |
| Dependency locks vs Kaggle's real inventory | 0 conflicts; all pins hash‑verified by `pip download --require-hashes` |
| CPU test suite | passes (`CUDA_VISIBLE_DEVICES= python3.13 -m pytest tests`) |
| Kaggle 2 × T4: input discovery, installer (32/32), static stage (79/79) | **Kaggle‑tested** (earlier bundle revisions; same installer/static logic) |
| Kaggle 2 × T4: `engine`, `parallel`, `server` stages for **this** wheel | **Verification pending** — the first full run exposed a Flash placeholder that raised during vLLM's unconditional kernel warm‑up; fixed in this wheel, not yet re‑run on a GPU |

Do not call this release production‑verified until the executed notebook 15 and its acceptance ZIP show every stage `PASS`. Remaining limits: [docs/PRODUCTION_READINESS.md](docs/PRODUCTION_READINESS.md).

## Known facts
* TP=2, PP=2 and replicas are **slower** than TP=1 for a model that fits one T4 (PCIe host bridge, no NVLink; peer access varies between Kaggle VMs).
* Kaggle resolves NCCL to its cu12 build 2.32.3 (two NCCL packages share one file); our binaries do not link it.
* Unsupported: bfloat16, Flash*, native FP8/FP4, L4/SM89, Mistral‑family models, OpenTelemetry exporters, direct `vllm.LLM`/`vllm serve`.
