# 8. Errors

All SDK exceptions derive from `StokastiskError` (a `RuntimeError`). Import from `stokastisk`.

```
StokastiskError
├── EnvironmentValidationError
│   ├── GPUDiscoveryError
│   └── TopologyError
├── EngineError
│   ├── AttentionBackendError
│   ├── EngineClosedError
│   ├── EngineTimeoutError
│   ├── EngineCloseError
│   ├── StructuredOutputError
│   └── ServerError
│       ├── ServerStartupError
│       ├── ServerHTTPError
│       └── ServerShutdownError
├── MissingDependencyError
├── ReplicaError
└── ConfigurationError (also a ValueError)
    ├── UnsupportedParallelismError
    ├── InsufficientMemoryError
    └── StokastiskCompatibilityError
```

| Exception | Typical cause | What to do |
|---|---|---|
| `ConfigurationError` | invalid `StokastiskConfig` value | read the message; it names the field |
| `UnsupportedParallelismError` | TP×PP > 2, strategy/size mismatch | use TP=2 **or** PP=2, not both |
| `StokastiskCompatibilityError` | Flash backend, bfloat16, unsupported model/quantization | use `TRITON_ATTN`, `dtype="half"` |
| `InsufficientMemoryError` | estimate exceeds GPU memory | smaller `max_model_len`, TP=2, smaller model |
| `EngineError` | vLLM failed to start or rejected a request | read `__cause__` and the log above it |
| `AttentionBackendError` | engine is not on `TRITON_ATTN` | should never happen; report it |
| `EngineClosedError` | method called after `close()` | create a new `LLM` |
| `EngineTimeoutError` | `timeout=` expired (request keeps running) | wait, or raise the timeout |
| `EngineCloseError` | engine shutdown raised (cleanup still completed) | check GPU memory with `mem_get_info` |
| `StructuredOutputError` | invalid/truncated/non-conforming JSON | bound the schema, raise `max_tokens` |
| `MissingDependencyError` | a required package is not installed | reinstall `stokastisk`; the message lists names |
| `ServerStartupError` | `vllm serve` exited or timed out | read `.log_excerpt`, `.log_path`, `.command` |
| `ServerHTTPError` | non-200 or truncated stream | read the status/body in the message |
| `ServerShutdownError` | a server process could not be killed | rare; restart the notebook |
| `ReplicaError` | worker died, timed out, or had to be killed on close | `restart_dead()` or recreate the pool |
| `EnvironmentValidationError` / `GPUDiscoveryError` / `TopologyError` | wrong runtime or no GPU | set Accelerator to *GPU T4 × 2* |

Catch broadly with `except stokastisk.StokastiskError`.
