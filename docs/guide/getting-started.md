# 1. Getting started

## Notebook settings

| Setting | Value |
|---|---|
| Accelerator | **GPU T4 × 2** |
| Internet | **ON** |
| Python | 3.13 image |

## Install

One cell:

```python
!pip install stokastisk --extra-index-url https://stokastisk-ai.github.io/stokastisk/simple/
```

pip reads a small index page, downloads the 38 MB wheel from the Hugging Face CDN, verifies its SHA256, and installs 19 pinned dependencies. Kaggle's own PyTorch 2.11.0+cu130 and Triton 3.6.0 are never touched.

!!! `--extra-index-url` also searches PyPI. The name `stokastisk` is not registered there today.

## Check the install

```python
import stokastisk, json
print(stokastisk.__version__)                           # 0.1.0
print(json.dumps(stokastisk.prohibited_report(), indent=1))   # "clean": true
```

`clean: true` means no FlashAttention, FlashInfer or FlashMLA code is present.

## First generation

Engine code runs in an ordinary notebook cell:

```python
from stokastisk import LLM, StokastiskConfig

config = StokastiskConfig(max_model_len=2048)           # one T4, FP16, TRITON_ATTN

with LLM("Qwen/Qwen2.5-0.5B-Instruct", config=config) as llm:
    print(llm.verify_attention_backend())               # TRITON_ATTN
    print(llm.complete("The capital of France is", max_tokens=16))
    print(llm.chat([{"role": "user", "content": "Say hello in five words."}])[0].outputs[0].text)
```

`with` releases the GPUs on exit. If you do not use `with`, call `llm.close()`.

### Running as a script

vLLM starts its engine in a separate (spawned) process. In a **script** (`!python demo.py`), keep engine code under a guard:

```python
%%writefile demo.py
import stokastisk

if __name__ == "__main__":
    with stokastisk.LLM("Qwen/Qwen2.5-0.5B-Instruct", config=stokastisk.StokastiskConfig(max_model_len=2048)) as llm:
        print(llm.complete("Hello", max_tokens=8))
```

Both forms ran successfully on Kaggle (plain cells in notebooks 18 and 19, script in notebook 16).

## Which model fits?

Each T4 has about 15 GiB. FP16 weights take about 2 bytes per parameter, plus KV cache.

| Model size | Use |
|---|---|
| up to about 3B | one GPU, `tensor_parallel_size=1` (fastest) |
| about 7B (14 GiB of weights) | both GPUs: `tensor_parallel_size=2` |

Ask the SDK: `stokastisk.recommend_parallelism("Qwen/Qwen2.5-Coder-7B-Instruct")` — see [Using both GPUs](multi-gpu.md).

Next: [Configuration](configuration.md).
