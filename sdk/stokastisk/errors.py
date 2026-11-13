"""Stokastisk exception hierarchy."""


class StokastiskError(RuntimeError):
    """Base class for all SDK errors."""


class EnvironmentValidationError(StokastiskError):
    """The runtime does not match the SDK build target."""


class GPUDiscoveryError(EnvironmentValidationError):
    """Required GPUs are missing or have the wrong architecture."""


class EngineError(StokastiskError):
    """vLLM engine failed to initialise or generate."""


class ConfigurationError(StokastiskError, ValueError):
    """Invalid Stokastisk configuration (e.g. a FlashAttention backend was requested)."""


class AttentionBackendError(EngineError):
    """The engine did not end up with the requested Triton attention backend."""


class EngineClosedError(EngineError):
    """The engine was closed (explicitly, or because it never started); create a new ``stokastisk.LLM``."""


class EngineTimeoutError(EngineError):
    """A generation call did not finish within the caller's timeout. The request may still be running inside the engine."""


class EngineCloseError(EngineError):
    """The engine shut down with an error. The SDK still released everything it could; ``__cause__`` carries the original failure."""


class StructuredOutputError(EngineError):
    """A structured-output request failed, or its output violated the schema that was requested."""


class MissingDependencyError(StokastiskError):
    """A third-party package needed by the requested feature is not importable. ``missing`` lists the import names; ``feature`` names the feature."""

    def __init__(self, feature: str, missing, hint: str = "") -> None:
        self.feature, self.missing = feature, tuple(missing)
        super().__init__(f"{feature} needs packages that are not importable here: {', '.join(self.missing)}." + (f" {hint}" if hint else ""))


class ServerError(EngineError):
    """Base class for failures of the OpenAI-compatible server process."""


class ServerStartupError(ServerError):
    """The server process exited, or did not become ready, before ``wait_ready`` returned.

    Carries everything needed to diagnose it without opening a log: ``exit_code`` (None = still running at timeout), ``command``, ``log_path`` (a persistent file),
    ``log_excerpt`` (the last lines of the child's output) and ``timeout`` (seconds waited, or None when the process exited first)."""

    def __init__(self, reason: str, *, exit_code, command, log_path, log_excerpt: str, timeout=None, waited=None) -> None:
        self.exit_code, self.command, self.log_path, self.log_excerpt, self.timeout, self.waited = exit_code, list(command), str(log_path), log_excerpt, timeout, waited
        wait = f"startup timeout: {timeout}s" + (f" (waited {waited:.1f}s)" if waited is not None else "")
        super().__init__(f"{reason}\n  exit code : {exit_code}\n  command   : {' '.join(map(str, self.command))}\n  log file  : {log_path}\n"
                         f"  {wait}\n  --- last lines of the server output ---\n{log_excerpt}")


class ServerHTTPError(ServerError):
    """The server answered an API request with an error status or an unparsable body. ``status`` is the HTTP status (None for a transport failure)."""

    def __init__(self, message: str, *, status=None, body: str = "") -> None:
        self.status, self.body = status, body
        super().__init__(message + (f" (HTTP {status}: {body[:300]})" if status is not None else ""))


class ServerShutdownError(ServerError):
    """The server process could not be stopped cleanly (it had to be killed, or children survived)."""


class UnsupportedParallelismError(ConfigurationError):
    """The requested TP/PP/replica combination is not valid for the two-T4 target."""


class InsufficientMemoryError(ConfigurationError):
    """The model (weights + KV cache) does not fit the requested GPU layout."""


class TopologyError(EnvironmentValidationError):
    """GPU topology could not be determined or does not match the requested layout."""


class ReplicaError(StokastiskError):
    """A replica worker failed to start, died, or returned an error."""


class StokastiskCompatibilityError(ConfigurationError):
    """A model, dtype, quantization method or vLLM feature cannot work on the Tesla T4 / SM75 / TRITON_ATTN build.

    Raised before any engine start so the user sees the reason instead of an obscure ImportError or CUDA failure."""
