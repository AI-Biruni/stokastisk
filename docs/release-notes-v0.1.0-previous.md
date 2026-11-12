# GitHub release `v0.1.0` notes before the integrated-wheel update (preserved 2026-11-12)

Release name: Stokastisk 0.1.0; target 4ac17f64d64cc780393f03de8660341788d233e4; assets: SHA256SUMS (11975 B), stokastisk-0.1.0-kaggle-t4-sm75-kaggle-input.zip (37205093 B), stokastisk-0.1.0-kaggle-t4-sm75-kaggle-input.zip.sha256 (115 B), stokastisk-0.1.0-py3-none-any.whl (69006 B), vllm-0.25.1-cp38-abi3-linux_x86_64.whl (38067501 B)

---

# Stokastisk 0.1.0

Release `stokastisk-0.1.0-kaggle-t4-sm75` — Python 3.13 · CUDA Toolkit 13.0 · vLLM 0.25.1 · PyTorch 2.11.0+cu130 · Triton 3.6.0 · 2 × Tesla T4 (SM75) · `TRITON_ATTN`.

**Validation status: `KAGGLE_T4_VERIFIED`.** All 14 notebooks were executed by the project owner on Kaggle 2 × T4 (PASS; WARNING for 05 and 10; none failed). TP=2, PP=2 and two-replica inference run with both GPUs in use. Full results and limits: [docs/KAGGLE_T4_VERIFICATION.md](https://github.com/AI-Biruni/stokastisk/blob/main/docs/KAGGLE_T4_VERIFICATION.md).

**Measured:** for a model that fits one T4, TP=1 is fastest (Qwen2.5-1.5B decode, batch 8: TP=1 241 tok/s, TP=2 133, PP=2 146, 2 replicas 226). TP/PP are for models that do not fit one 15 GiB GPU (not measured). L4/SM89 is not supported.

Changed since the first upload: the replica scheduler now routes work to both GPUs, and the bundled documentation and SDK comments were corrected (CUDA peer access varies between Kaggle VMs). Notebooks 13–14 executed SDK wheel `e17b99b4…`; the released SDK wheel differs from it only in comments, note strings and the packaged README (no code change) and was not itself re-executed. The vLLM wheel is byte-identical (`cc3bcafb…`). Git tag `v0.1.0` marks the verified commit `1acc49c`; the corrected documentation is on `main`.

SHA256: ZIP `4643961225d0cd6156906249703ce7a32a8cf5db97160d0ce3f4240b97b91210` · SDK wheel `29edaa550f49a1cd343a8f043ddd2b5c98c9154ab1da0172fc3fc8725aa3ee49` · vLLM wheel `cc3bcafb6516791990ca302073ae153a08465d3df1e460d666e3ec09b4e98f5e`.
Download: https://huggingface.co/waqasm86/stokastisk
