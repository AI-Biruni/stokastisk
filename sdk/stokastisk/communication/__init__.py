"""EXPERIMENTAL communication layer: host-staged transport, NCCL timing helpers, transport registry. Not used by the vLLM engine."""
from .transport_registry import list_transports

__all__ = ["list_transports", "HostStagedTransport", "PinnedBufferRing", "NativeHostStaged", "build_native", "load_native"]


def __getattr__(name):     # torch-dependent classes load lazily so `import stokastisk.communication` works on CPU-only machines
    if name == "HostStagedTransport":
        from .cuda_transport import HostStagedTransport; return HostStagedTransport
    if name == "PinnedBufferRing":
        from .pinned_memory import PinnedBufferRing; return PinnedBufferRing
    if name in ("NativeHostStaged", "build_native", "load_native"):
        from . import native; return getattr(native, name)
    raise AttributeError(name)
