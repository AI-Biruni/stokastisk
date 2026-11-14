# Stokastisk 0.1.0

A Python 3.13 SDK for Kaggle's 2 × Tesla T4 GPUs (SM75, CUDA 13.0) with a bundled vLLM 0.25.1, Triton attention (`TRITON_ATTN`), FP16 inference, structured output (xgrammar), an OpenAI-compatible server and two-GPU parallelism.

This file is the package description embedded in the wheel. It deliberately contains no download hash (a wheel cannot contain its own hash). The SHA256-pinned install command for the current wheel is in
[INSTALL.md](https://github.com/stokastisk-ai/stokastisk/blob/main/INSTALL.md); the user guide is at <https://stokastisk-ai.github.io/stokastisk/docs/>.

* Source and documentation: <https://github.com/stokastisk-ai/stokastisk>
* Wheels: <https://huggingface.co/waqasm86/stokastisk>
* Supported: Python 3.13 · vLLM 0.25.1 · PyTorch 2.11.0+cu130 · Triton 3.6.0 · 2 × Tesla T4 (FP16). Not supported: bfloat16, FlashAttention/FlashInfer/FlashMLA, FP8/FP4 native kernels, Mistral-family models.
* Which models were actually run on Kaggle: `stokastisk.check_model(...)` and the user guide (Checks and diagnostics).

Apache-2.0. vLLM is Apache-2.0 © the vLLM project.
