# Kaggle 2 × Tesla T4 verification — Stokastisk 0.1.0

Status: **`KAGGLE_T4_VERIFIED`** (machine-readable copy: [`kaggle-t4-verification.json`](kaggle-t4-verification.json)).
All fourteen notebooks in [`notebooks/kaggle-t4-sm75/notebooks/`](../notebooks/kaggle-t4-sm75/notebooks) were run manually by the project owner on Kaggle with 2 × Tesla T4, Python 3.13.16, PyTorch 2.11.0+cu130, Triton 3.6.0, vLLM 0.25.1, `TRITON_ATTN`.
Nothing here comes from the development machine's GeForce 940M (SM50), which is never used for GPU claims.

## Which bytes were executed
| Notebooks | SDK wheel | vLLM wheel | Bundle |
|---|---|---|---|
| 13, 14 — **re-run 2026-11-12** | **`29edaa55…ee49` (the released SDK wheel)** — printed by the executed notebooks' provenance cell | `cc3bcafb…8f5e` | legacy two-wheel bundle, `SHA256SUMS` sha256 `97ad6660…09b7` |
| 13, 14 — first run 2026-11-11 | `e17b99b4…6cae` | `cc3bcafb…8f5e` | manifest `efe492a6…56ca` |
| 01–12 (2026-11-11) | `a6b8596b…8585` (earlier build) | `cc3bcafb…8f5e` (identical) | earlier bundle |
| **integrated wheel** (SDK + vLLM in one file) | — | — | **never executed: notebook 15 pending** |

The vLLM wheel and the native CUDA binaries are byte-identical in all runs and in the integrated wheel. The 2026-11-12 runs of 13 and 14 executed the released SDK wheel itself, which supersedes the earlier statement that it had not been executed. That wheel was built **before** `stokastisk.compat`, `stokastisk.serving`, `LLM.chat` and the bfloat16 pre-check existed, so **none of the runs below verifies them or the integrated wheel's installation path**. The source delta between the executed wheel and the final SDK is listed in [SOURCE_DIFF_VS_KAGGLE_EXECUTED.md](SOURCE_DIFF_VS_KAGGLE_EXECUTED.md). Original executed notebooks, acceptance reports and logs: [`evidence/kaggle-legacy-two-wheel/`](../evidence/kaggle-legacy-two-wheel/).

## Results
| # | Notebook | Result | Note |
|---|---|---|---|
| 01 | environment audit | PASS | |
| 02 | installation | PASS | protected baseline untouched, no Flash* |
| 03 | single GPU | PASS | cubins exactly `sm_75` |
| 04 | dual-GPU NCCL | PASS | all collectives on both ranks |
| 05 | vLLM inference | WARNING | TP=1 and TP=2 work; vLLM force-killed one TP=2 process at shutdown |
| 06 | TP benchmark | PASS | |
| 07 | topology | PASS | PHB; peer access differs between Kaggle VMs |
| 08 | pipeline parallel | PASS | experimental |
| 09 | replicas | PASS (validation) | scaling runs exposed the routing defect, fixed and re-verified in 14 |
| 10 | CUDA communication | WARNING | experimental transport, not integrated into vLLM |
| 11 | load balancing | PASS | imbalance < 2 % |
| 12 | consolidated | PASS | shallow; see 13/14 for strict checks |
| 13 | TP / PP fulfilment | **PASS** (2026-11-12, released SDK wheel) | 7/7 stages; TP=1, TP=2, PP=2 each 12/12 checks; one informational WARNING (TP=2 text not byte-identical to TP=1) |
| 14 | replica balance | **PASS** (2026-11-12, released SDK wheel) | 50/50 checks; 15 rounds, each split exactly evenly |
| 15 | **integrated wheel** | **NOT RUN — pending** | install, versions, `compat`, `serving`, `LLM.chat`, FP16 inference, TRITON_ATTN, shutdown |

## Tensor and pipeline parallelism on 2 × T4 (notebook 13)
| Mode | vLLM `parallel_config` | GPU memory while generating (MiB) | Deterministic | Memory freed |
|---|---|---|---|---|
| TP=1 | TP 1 / PP 1 | 13249 / 3 (one GPU) | yes | yes |
| TP=2 | TP 2 / PP 1 | 13751 / 13751 | yes | yes |
| PP=2 | TP 1 / PP 2 | 12895 / 13353 | yes, output equals TP=1 | yes |

TP=2 differs from TP=1 in the second prompt's later tokens (fp16 reduction order); the first prompt agrees. Each mode also confirmed `TRITON_ATTN`, a compiled `kernel_unified_attention`, and no Flash library mapped in any process.

## Throughput (Qwen2.5-1.5B-Instruct, fp16, decode tokens/s, median)
| Workload | TP=1 | TP=2 | PP=2 | 2 replicas |
|---|---|---|---|---|
| batch 8, 64 output tokens, 2026-11-11 | 241.3 | 132.8 | 145.7 | 225.8 |
| batch 8, 64 output tokens, 2026-11-12 (notebook 13; replicas from notebook 14 scaling run, batch 8) | 238.4 | 133.1 | 148.1 | 191.2 (TP=1 in the same scaling run: 209.2) |

Replica pool relative to one TP=1 engine (notebook 14), batch 1/8/16/32/64: 0.96× / 0.92× / 0.91× / 0.94× / 0.96× (2026-11-11) and 0.95× / 0.91× / 0.93× / 0.91× / 0.94× (2026-11-12), with an even split of requests (for example 32/32 at batch 64).

**Conclusion supported by the data:** for a model that fits one T4, TP=1 is fastest. TP and PP cost throughput; they exist to fit models that do not fit one 15 GiB GPU, which was not measured. Two replicas did not beat one engine up to batch 64 (a single engine's throughput was still growing almost linearly with batch size, so a larger batch may change this — untested).

## Limitations
* Peer access differs between Kaggle VMs (`true` in 01, `false` in 07); never assume it.
* Worker-failure recovery has CPU unit tests only.
* The host-staged CUDA transport is experimental and not wired into vLLM.
* No claim is made for L4 / SM89: see [`notebooks/kaggle-l4-sm89/`](../notebooks/kaggle-l4-sm89/README.md).

## Dependency finding from the 2026-11-12 installer report
The legacy installer printed `PASS` for "no conflicts involving vllm/stokastisk" while its recorded `pip check` output contained `xgrammar 0.2.5.post1 has requirement transformers<5,>=4.38.0, but you have transformers 5.20.2.` (the check only looked at lines beginning with `vllm`/`stokastisk`). Analysis and resolution: [DEPENDENCY_COMPATIBILITY.md](DEPENDENCY_COMPATIBILITY.md). The integrated installer reports this as a WARNING and fails on any other new conflict.


## Integrated wheel — build 2 FAILED, build 3 pending (added 2026-11-13)
Notebook 15 was executed on build 2 and failed in `static`, `engine` (structured output) and `server`; the installer passed and basic FP16 inference/chat/complete/cleanup passed. Raw evidence: `evidence/kaggle/build-2-failed-notebook-15/`. Build 3 (`docs/KAGGLE_BUILD3_VERIFICATION.md`) has **not** run on Kaggle; nothing on this page is evidence for it.
