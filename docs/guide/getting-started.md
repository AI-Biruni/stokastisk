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
!python -m pip install --index-url https://pypi.org/simple "stokastisk @ https://huggingface.co/waqasm86/stokastisk/resolve/main/wheels/stokastisk-0.1.0-5-cp313-cp313-linux_x86_64.whl#sha256=fe80d7c36fa78cf8a26feb1e00bb5f1f006ca45a027450353603976320ae0b3e"
```

pip downloads the 38 MB wheel from the Hugging Face CDN, **refuses it unless its SHA256 matches**, and installs 19 pinned dependencies from PyPI. Kaggle's own PyTorch 2.11.0+cu130 and Triton 3.6.0 are never touched. The URL names the build‑4 wheel (project version 0.1.0); the current file and hash are in [RELEASE.json](../../RELEASE.json).

!!! Why this form and not `pip install stokastisk --extra-index-url …`? That also works, but `--extra-index-url` makes pip consider PyPI as well as the extra index, so a different project with the same name could be chosen. The pinned command above names the exact file and uses PyPI only for ordinary dependencies. To go back to build 3, see [INSTALL.md](../../INSTALL.md).

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

Ask the SDK: `stokastisk.recommend_parallelism("Qwen/Qwen2.5-Coder-7B-Instruct")` — see [Using both GPUs](multi-gpu.md) — and `stokastisk.check_model("Qwen/Qwen2.5-Coder-7B-Instruct", mode="TP=2")` to see whether that exact model and mode were run on Kaggle ([Checks and diagnostics](diagnostics.md)).

Next: [Configuration](configuration.md).
