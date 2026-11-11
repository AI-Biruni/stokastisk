"""Registry of known transports with an honest status. Nothing is selected automatically."""
from __future__ import annotations

TRANSPORTS = {
    "nccl": {"status": "supported", "used_by_vllm": True, "note": "reference transport; on Kaggle 2xT4 it uses whatever path NCCL selects (SHM, or P2P where the VM exposes it)"},
    "host_staged": {"status": "experimental", "used_by_vllm": False, "note": "pinned-host double-buffering; kept out of vLLM unless it beats NCCL in measurement"},
    "host_staged_native": {"status": "experimental", "used_by_vllm": False, "note": "same algorithm in C++ (CUDA Runtime API, native/host_staged_xfer.cpp); optional build"},
    "cuda_p2p": {"status": "unavailable_on_target", "used_by_vllm": False, "note": "needs torch.cuda.can_device_access_peer == True in both directions; this differs between Kaggle VMs (measured true in one session, false in another), so it is not relied on or integrated"},
    "nvshmem": {"status": "not_applicable", "used_by_vllm": False, "note": "not integrated: peer access is not guaranteed on this platform and there is no verified IB path"},
}


def list_transports() -> dict[str, dict]:
    return {k: dict(v) for k, v in TRANSPORTS.items()}
