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
