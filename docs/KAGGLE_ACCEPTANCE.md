# Kaggle acceptance (maintainers)

Two notebooks, two purposes:
* **Notebook 16 - quickstart** (`16_stokastisk_quickstart.ipynb`): what an end user does - `pip install stokastisk --extra-index-url ...`, then a short run. This is the first-run check of the published one-line install.
* **Notebook 15 - acceptance** (`15_stokastisk_candidate_sdk_integration.ipynb`): the full GPU test suite against a locally built bundle.

## Running notebook 15
1. `python3.13 scripts/release/assemble_build3_release.py` builds the wheel and `dist/wheel-build3/stokastisk-0.1.0-build3-kaggle-t4-sm75-kaggle-input.zip` and regenerates the notebook pinned to the wheel, ZIP and `SHA256SUMS` hashes.
2. Upload the ZIP **unchanged** as a Kaggle Dataset and attach it (Kaggle extracts it). Import the notebook; *GPU T4 x 2*, *Internet ON*; *Run All*.
3. Cell 1 finds and verifies the attached bundle (no silent download; a stale or different bundle is refused). The notebook prints a per-stage table and fails unless every stage is `PASS`; `/kaggle/working/15-stokastisk-build3-acceptance.zip` is written even on failure.

| Stage | Proves |
|---|---|
| installer | protected packages untouched; hash-pinned locks installed `--no-deps --no-index`; no new `pip check` line; feature imports; installed tree has 0 Flash files |
| static | one inventory implementation (live vs installer), configuration/compat/serving validation, backend registry on SM7.5, Flash import guard, dependency completeness, CPU structured-output stack |
| engine | FP16 generate/chat/complete, determinism, input validation, pinned-xgrammar structured output (valid, invalid detected, unsupported schema rejected), serialisation + timeout, `TRITON_ATTN` in config and as a compiled kernel, no Flash library, close/cleanup, direct `vllm.LLM` never selects Flash |
| parallel | TP=2, PP=2 (experimental), ReplicaPool: real generation, both GPUs engaged, clean shutdown, memory released; topology + recommender |
| server | `vllm serve` start/ready, models, chat, streaming, HTTP errors, structured output over HTTP, metrics, startup-failure report, port check, restart, graceful shutdown |
| summary | all stages describe the same wheel and runtime; a missing report is `NOT_RUN`; only `PASS` passes |

Result vocabulary: `PASS`, `FAIL`, `ERROR`, `WARNING`, `SKIPPED`, `NOT_RUN` - only `PASS` passes.
