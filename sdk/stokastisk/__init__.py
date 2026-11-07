"""Stokastisk: dual-Tesla-T4 (SM75, CUDA 13.0) SDK around the bundled vLLM 0.25.1 engine."""
from ._version import (CUDA_BUILD_TARGET, GPU_COMPUTE_CAPABILITY, VLLM_BASE_VERSION,
                       __version__)
from .config import StokastiskConfig
from .diagnostics import check_native_extensions, discover_gpus, prohibited_report, report, validate_environment
from .engine import LLM
from .topology import discover_topology, Topology
from .parallel.recommender import recommend_parallelism, Recommendation
from .limits import LIMITS, check_resources, cpu_budget, working_dir_status
from .errors import (AttentionBackendError, ConfigurationError, EngineError, EnvironmentValidationError, GPUDiscoveryError,
                     InsufficientMemoryError, ReplicaError, StokastiskError, TopologyError, UnsupportedParallelismError)

# kept for 0.3.0-stage API compatibility
validate_kaggle_t4 = validate_environment


def __getattr__(name):  # lazy: SamplingParams comes from the bundled vLLM
    if name == "SamplingParams":
        from vllm import SamplingParams
        return SamplingParams
    if name == "ReplicaPool":
        from .parallel.replica import ReplicaPool
        return ReplicaPool
    raise AttributeError(name)


__all__ = ["LLM", "SamplingParams", "StokastiskConfig", "validate_environment", "validate_kaggle_t4",
           "report", "prohibited_report", "discover_gpus", "check_native_extensions", "StokastiskError",
           "EnvironmentValidationError", "GPUDiscoveryError", "EngineError", "ConfigurationError",
           "AttentionBackendError", "__version__", "LIMITS", "check_resources", "cpu_budget", "working_dir_status",
           "discover_topology", "Topology", "recommend_parallelism", "Recommendation", "ReplicaPool", "InsufficientMemoryError",
           "ReplicaError", "TopologyError", "UnsupportedParallelismError", "VLLM_BASE_VERSION", "CUDA_BUILD_TARGET", "GPU_COMPUTE_CAPABILITY"]
