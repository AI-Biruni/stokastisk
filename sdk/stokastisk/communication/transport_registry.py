"""Registry of known transports with an honest status. Nothing is selected automatically."""
from __future__ import annotations

TRANSPORTS = {
    "nccl": {"status": "supported", "used_by_vllm": True, "note": "reference transport; Kaggle 2xT4 uses SHM/direct (no CUDA P2P)"},
    "host_staged": {"status": "experimental", "used_by_vllm": False, "note": "pinned-host double-buffering; kept out of vLLM unless it beats NCCL in measurement"},
    "host_staged_native": {"status": "experimental", "used_by_vllm": False, "note": "same algorithm in C++ (CUDA Runtime API, native/host_staged_xfer.cpp); optional build"},
    "cuda_p2p": {"status": "unavailable_on_target", "used_by_vllm": False, "note": "needs torch.cuda.can_device_access_peer == True; false on Kaggle 2xT4"},
    "nvshmem": {"status": "not_applicable", "used_by_vllm": False, "note": "not integrated: no P2P and no verified IB path on this platform"},
}


def list_transports() -> dict[str, dict]:
    return {k: dict(v) for k, v in TRANSPORTS.items()}
