# Install on Kaggle (2 × T4, Python 3.13)
1. Notebook settings: *Accelerator: GPU T4 × 2*, *Internet: ON*.
2. Upload `stokastisk-0.1.0-build3-kaggle-t4-sm75-kaggle-input.zip` (from the GitHub release `v0.1.0` or the Hub folder `releases/stokastisk-0.1.0-build3-kaggle-t4-sm75/`) as a Kaggle **Dataset**, **unchanged**, and attach it. A wheel‑only archive is rejected on purpose.
3. Import `15_stokastisk_candidate_sdk_integration.ipynb` (same place) and *Run All*. Cell 1 prints the exact release, wheel SHA256 and number of verified files; there is **no** silent download fallback (`STOKASTISK_ALLOW_HF_FALLBACK=1` opts in to a pinned copy).
4. Send back / keep the executed notebook and `/kaggle/working/15-stokastisk-build3-acceptance.zip`. The notebook fails unless every stage is `PASS`.

Manual: `python3.13 scripts/kaggle/02_install_integrated.py <extracted bundle> [--dry-run] [--with-models]` (locks: `core`, `structured-output`, `serving` by default; `models` = compressed‑tensors, depyf, setproctitle only). Never install the integrated wheel next to a standalone `vllm` wheel — both own `vllm/`.
