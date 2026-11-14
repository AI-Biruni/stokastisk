# 3. The LLM engine

`stokastisk.LLM` wraps the bundled vLLM engine with the T4 defaults and adds lifecycle safety.

```python
from stokastisk import LLM, StokastiskConfig
llm = LLM("Qwen/Qwen2.5-0.5B-Instruct", config=StokastiskConfig(max_model_len=2048))
```

* `model`: a Hugging Face id or a local path (non-empty string).
* `config`: a `StokastiskConfig`; default `StokastiskConfig.from_env()`.
* Extra keyword arguments go to vLLM and override config defaults, **except** a different attention backend or structured-output backend, which raise `StokastiskCompatibilityError`.
* `execution_strategy="auto"` and `"replica_parallel"` are refused here (use `recommend_parallelism` / [`ReplicaPool`](multi-gpu.md)).
* After start-up the SDK confirms the attention backend; a mismatch raises `AttentionBackendError`.
* Engine start-up failures raise `EngineError`.

Use it as a context manager so the GPUs are always released.

## Methods

| Method | Purpose |
|---|---|
| `generate(prompts, sampling_params=None, *, timeout=None, **kw)` | vLLM `generate`. `prompts` is a list of non-empty strings; an empty list returns `[]`. |
| `chat(messages, sampling_params=None, *, timeout=None, **kw)` | vLLM `chat`; the model's chat template is applied. `messages` is a list of `{"role", "content"}` dicts (or a list of such lists). |
| `complete(prompt, *, max_tokens=64, temperature=0.0, timeout=None)` | One prompt in, text out. |
| `generate_structured(prompts, schema, *, max_tokens=64, temperature=0.0, timeout=None)` | JSON that satisfies `schema`. See [Structured output](structured-output.md). |
| `verify_attention_backend()` | Returns `"TRITON_ATTN"` (raises otherwise). |
| `selected_attention_backend` | The backend recorded in the live engine configuration. |
| `verify_structured_output_backend()` / `structured_output_backend` | Same for the pinned `xgrammar`. |
| `close()` | Stops the engine and its workers, frees cached CUDA memory. Idempotent. |
| `closed` | `True` after `close()`. |
| `engine` | The underlying vLLM object (advanced use). |

## Sampling

`generate` and `chat` return vLLM `RequestOutput` objects; the text is `out[0].outputs[0].text`.

```python
from stokastisk import SamplingParams        # lazy import from the bundled vLLM

params = SamplingParams(temperature=0.2, max_tokens=96)
outs = llm.generate(["Explain TCP in two sentences."], params)
print(outs[0].outputs[0].text)
```

Qwen models ship a `generation_config.json`; vLLM applies its defaults (e.g. `temperature=0.7`, `repetition_penalty`) unless you pass your own `SamplingParams`. Use `temperature=0.0` for repeatable output.

## Timeouts and concurrency

* Calls are serialised: one engine call at a time per `LLM`.
* `timeout=` (seconds) raises `EngineTimeoutError`. Offline generation **cannot be cancelled**: the request keeps running and later calls wait for it.
* Calls after `close()` raise `EngineClosedError`.

## Closing

`close()` always runs every cleanup step. If the engine's own shutdown failed it raises `EngineCloseError` *after* cleaning up; the engine is closed either way.

What the Kaggle runs showed, kept apart on purpose:

* **GPU memory:** after `close()` both T4s returned to 14.46 of 14.56 GiB free in every run (0.5B TP=1; 3B TP=2 and PP=2; Coder‑7B TP=2 and PP=2).
* **Log line:** at the end of every TP=2 and PP=2 engine, vLLM's process manager logged `[shutdown] Process manager: force killing remaining processes count=1`. `close()` still completed without raising and the memory was recovered, so it is a warning about one worker that had to be killed, not a leak. It was not seen in the single‑GPU 0.5B run. Check memory yourself between experiments (`torch.cuda.mem_get_info(i)`).

## Example: chat with a system prompt

```python
with LLM("Qwen/Qwen2.5-0.5B-Instruct", config=StokastiskConfig(max_model_len=2048)) as llm:
    reply = llm.chat([
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Five practical uses of Python for data analysis."},
    ])
    print(reply[0].outputs[0].text)
```

You may see `Triton kernel JIT compilation during inference` warnings on the first request. They are informational (a one-time latency spike).

Next: [Structured output](structured-output.md).
