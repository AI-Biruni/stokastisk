---
license: apache-2.0
library_name: stokastisk
tags: [vllm, kaggle, tesla-t4, sm75, cuda-13, triton, inference, python-3.13]
---
# Stokastisk 0.1.0 — vLLM 0.25.1 + Triton attention + CUDA binaries for Kaggle's 2 × Tesla T4

This repository is the **CDN for the Stokastisk wheel** (SDK + bundled vLLM 0.25.1 + CUDA 13.0 `sm_75` binaries, no FlashAttention/FlashInfer/FlashMLA). Source, docs and tests: <https://github.com/stokastisk-ai/stokastisk> · User guide: <https://stokastisk-ai.github.io/stokastisk/docs/>

## Install (one line, Kaggle: GPU T4 × 2, Internet ON, Python 3.13)
```bash
python -m pip install --index-url https://pypi.org/simple "stokastisk @ https://huggingface.co/waqasm86/stokastisk/resolve/main/wheels/stokastisk-0.1.0-5-cp313-cp313-linux_x86_64.whl#sha256=fe80d7c36fa78cf8a26feb1e00bb5f1f006ca45a027450353603976320ae0b3e"
```
pip downloads `wheels/stokastisk-0.1.0-5-cp313-cp313-linux_x86_64.whl` from this repository (38 MB), **refuses it unless its SHA256 matches**, and installs 19 pinned dependencies from PyPI (the only index used). It never changes Kaggle's PyTorch 2.11.0+cu130 / Triton 3.6.0 or any other pre‑installed package. The older `--extra-index-url` form is explained, with its security trade-off, in the GitHub [INSTALL.md](https://github.com/stokastisk-ai/stokastisk/blob/main/INSTALL.md).

## Use
```python
%%writefile demo.py
import stokastisk

if __name__ == "__main__":
    with stokastisk.LLM("Qwen/Qwen2.5-0.5B-Instruct", config=stokastisk.StokastiskConfig(max_model_len=2048)) as llm:
        print(llm.verify_attention_backend())          # TRITON_ATTN
        print(llm.complete("The capital of France is", max_tokens=16))
```
then `!python demo.py` (in a script keep engine code under a `__main__` guard; plain notebook cells work too). Structured output, the OpenAI‑compatible server, tensor/pipeline parallelism and replicas: see the GitHub README user guide.

## File
| File | Size | SHA256 | Role |
|---|---|---|---|
| `wheels/stokastisk-0.1.0-5-cp313-cp313-linux_x86_64.whl` | 38,057,515 B | `fe80d7c36fa78cf8a26feb1e00bb5f1f006ca45a027450353603976320ae0b3e` | **current** (build 4) |

The wheel belongs to project version 0.1.0 (locked); its PEP 427 build tag is 5. Builds 3 and 4 have been withdrawn; build 3 is the one the Kaggle runs below were made on.

## Status — what was run where
* **Build 3 on Kaggle 2 × T4:** notebook 15 acceptance (installer 32/32, static 79/79, engine 44/44, parallel 23/23, server 21/21, summary 15/15) on a wheel that differs from the published build‑3 file only in `METADATA`/`RECORD`; end‑user notebooks 16, 18, 19 ran the published install with Qwen2.5‑0.5B‑Instruct (TP=1), Qwen2.5‑3B‑Instruct (TP=2, PP=2) and Qwen2.5‑Coder‑7B‑Instruct (TP=2, PP=2), FP16, `TRITON_ATTN`.
* **Build 5 (this install):** build 4's compatibility‑reporting correction (`stokastisk.compat`) plus one corrected error message in `stokastisk.serving`. 2,579 of its 2,582 files are byte‑identical to build 4 — every vLLM file, every CUDA binary, every dependency pin; build 4 in turn differed from build 3 only in the compatibility reporting. Verified on CPU; **not itself executed on Kaggle**, so build 3's evidence is inherited for the unchanged code.
* `stokastisk.check_model(model_id)` tells you whether *your* model id was among those runs; any other checkpoint, even of the same architecture, is `UNKNOWN_REQUIRES_EVIDENCE`.
* Unsupported: bfloat16, Flash*, FP8/FP4, L4/SM89, Mistral‑family models, OpenTelemetry exporters, SageMaker endpoints.

## License
Apache‑2.0. vLLM is Apache‑2.0 © the vLLM project.
