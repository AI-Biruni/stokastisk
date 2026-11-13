---
license: apache-2.0
library_name: stokastisk
tags: [vllm, kaggle, tesla-t4, sm75, cuda-13, triton, inference, python-3.13]
---
# Stokastisk 0.1.0 — vLLM 0.25.1 + Triton attention + CUDA binaries for Kaggle's 2 × Tesla T4

This repository is the **CDN for the Stokastisk wheel** (SDK + bundled vLLM 0.25.1 + CUDA 13.0 `sm_75` binaries, no FlashAttention/FlashInfer/FlashMLA). Source, docs and tests: <https://github.com/AI-Biruni/stokastisk>

## Install (one line, Kaggle: GPU T4 × 2, Internet ON, Python 3.13)
```bash
pip install stokastisk --extra-index-url https://ai-biruni.github.io/stokastisk/simple/
```
pip reads the index page, downloads `wheels/stokastisk-0.1.0-3-cp313-cp313-linux_x86_64.whl` from this repository (38 MB), verifies its SHA256 and installs 19 pinned dependencies. It never changes Kaggle's PyTorch 2.11.0+cu130 / Triton 3.6.0 or any other pre‑installed package.

## Use
```python
%%writefile demo.py
import stokastisk

if __name__ == "__main__":
    with stokastisk.LLM("Qwen/Qwen2.5-0.5B-Instruct", config=stokastisk.StokastiskConfig(max_model_len=2048)) as llm:
        print(llm.verify_attention_backend())          # TRITON_ATTN
        print(llm.complete("The capital of France is", max_tokens=16))
```
then `!python demo.py` (vLLM starts its engine in a separate process, so keep engine code in a script with a `__main__` guard). Structured output, the OpenAI‑compatible server, tensor/pipeline parallelism and replicas: see the GitHub README user guide.

## File
| File | SHA256 |
|---|---|
| `wheels/stokastisk-0.1.0-3-cp313-cp313-linux_x86_64.whl` (38,056,869 B) | `d0b4b5fcfde3bb0af1ef1431b9508cf1a7fb44392f964b733f15f77e90c62fbc` |

## Status
Project version **0.1.0** (locked; the wheel build tag is 3). Kaggle 2 × T4 acceptance (installer 32/32, static 79/79, engine 44/44, parallel 23/23, server 21/21) passed on a wheel that differs from this one only in `METADATA`/`RECORD` (two extra exact dependency pins); every code and CUDA‑binary member is byte‑identical. The one‑line install itself has been verified by a resolver simulation against Kaggle's real package inventory (installs exactly `stokastisk` + 19 pinned packages, changes nothing else); its first run on Kaggle is pending. Unsupported: bfloat16, Flash*, FP8/FP4, L4/SM89, Mistral‑family models, OpenTelemetry exporters.

## License
Apache‑2.0. vLLM is Apache‑2.0 © the vLLM project.
