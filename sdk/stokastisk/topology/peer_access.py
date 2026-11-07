"""CUDA peer access, measured with ``torch.cuda.can_device_access_peer`` in BOTH directions. Never forced, never assumed."""
from __future__ import annotations

from .models import PeerAccess


def measure_peer_access(device_count: int | None = None) -> PeerAccess:
    try:
        import torch
        if not torch.cuda.is_available():
            return PeerAccess(source="torch: CUDA unavailable")
        n = torch.cuda.device_count() if device_count is None else device_count
        m: dict[int, dict[int, bool | None]] = {}
        for s in range(n):
            m[s] = {}
            for d in range(n):
                if s != d:
                    try:
                        m[s][d] = bool(torch.cuda.can_device_access_peer(s, d))
                    except Exception:  # noqa: BLE001
                        m[s][d] = None
        return PeerAccess(matrix=m, source="torch.cuda.can_device_access_peer")
    except Exception as e:  # noqa: BLE001
        return PeerAccess(source=f"unavailable: {type(e).__name__}")
