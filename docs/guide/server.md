# 5. OpenAI-compatible server

`stokastisk.serving` runs the bundled `vllm serve` as a supervised child process with the T4 defaults, and offers small client helpers. Use it from a notebook cell or script.

```python
from stokastisk import serving

with serving.ServerProcess("Qwen/Qwen2.5-0.5B-Instruct") as srv:
    srv.wait_ready()
    print(serving.chat(srv.base_url, srv.served_model_name, [{"role": "user", "content": "Hi"}]))
    print("".join(serving.stream_chat(srv.base_url, srv.served_model_name,
                                      [{"role": "user", "content": "Count to five."}])))
```

Leaving the `with` block stops the server (it is never started implicitly or left running).

## `ServerProcess(model, config=None, *, ...)`

| Argument | Default | Meaning |
|---|---|---|
| `model` | — | model id or path |
| `config` | `StokastiskConfig()` | T4 defaults; `replica_parallel` is refused |
| `host`, `port` | `127.0.0.1`, `8000` | `find_free_port()` gives a free one |
| `served_model_name` | `model` | name clients send |
| `quantization` | `None` | refused if the method needs a GPU newer than SM75 |
| `extra_args` | `()` | additional `vllm serve` arguments |
| `cuda_visible_devices` | `None` | e.g. `"0"` |
| `log_path` | temp file | server output; kept after `stop()` |

Always passed: `--attention-backend TRITON_ATTN`, FP16, pinned xgrammar.

| Member | Purpose |
|---|---|
| `start()` / `wait_ready(timeout=600)` / `start_and_wait()` | start, poll `/health` until 200 |
| `health()` | `True` if `/health` answers now |
| `restart()` | graceful stop then fresh start on the same port |
| `stop(grace=30)` | SIGINT → SIGTERM → SIGKILL on the process group; returns the exit code |
| `base_url`, `pid`, `running`, `command`, `log_path`, `served_model_name` | state |
| `log_tail(max_lines)` | last lines of the server log |
| `shutdown_report` | which signals were needed, whether children survived |

If start-up fails, `ServerStartupError` carries the exit code, the exact command, the log path and the last log lines.

## Client helpers

| Function | Returns |
|---|---|
| `chat(base_url, model, messages, *, max_tokens=64, temperature=0.0, timeout=120)` | assistant text |
| `stream_chat(...)` | generator of text deltas; a truncated stream raises `ServerHTTPError` |
| `list_models(base_url)` | `/v1/models` response |
| `fetch_metrics(base_url)` | Prometheus `/metrics` parsed to `{sample: value}` |

You can also call the server with any OpenAI client using `base_url=f"{srv.base_url}/v1"` and any API key.

## Utilities

`build_serve_command(model, config, ...)` returns the exact argv (pure function, nothing started). `find_free_port()`, `port_is_free(host, port)`, `check_server_requirements()` (lists missing serving packages), `validate_model_id`, `validate_host`, `validate_port`.

Next: [Using both GPUs](multi-gpu.md).
