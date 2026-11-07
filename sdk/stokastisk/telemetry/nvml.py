"""GPU sampling: NVML if present, else nvidia-smi, else an explicit 'unavailable'. Counters Kaggle does not expose stay ``None``."""
from __future__ import annotations

import threading
import time
from typing import Any

from ..topology import nvml as _n

_FIELDS = "index,utilization.gpu,utilization.memory,memory.used,memory.total,clocks.sm,clocks.mem,temperature.gpu,power.draw"
_KEYS = ("index", "gpu_util_pct", "mem_util_pct", "mem_used_mib", "mem_total_mib", "clock_sm_mhz", "clock_mem_mhz", "temp_c", "power_w")


def sample_once() -> dict[str, Any]:
    t = time.time()
    rows = _n.smi_query(_FIELDS)
    if rows is None:
        return {"t": t, "source": "unavailable", "gpus": []}
    gpus = []
    for r in rows:
        d: dict[str, Any] = {}
        for k, v in zip(_KEYS, r):
            try:
                d[k] = float(v) if k != "index" else int(v)
            except ValueError:
                d[k] = None          # "[N/A]" / "[Not Supported]"
        gpus.append(d)
    return {"t": t, "source": "nvidia-smi", "gpus": gpus}


def pcie_throughput_kib_s() -> dict[int, dict[str, int | None]] | None:
    """PCIe RX/TX in KiB/s per GPU via NVML, or None when NVML / the counter is unavailable."""
    nv = _n._pynvml()
    if nv is None:
        return None
    try:
        out = {}
        for i in range(nv.nvmlDeviceGetCount()):
            h = nv.nvmlDeviceGetHandleByIndex(i); d: dict[str, int | None] = {}
            for k, c in (("rx_kib_s", nv.NVML_PCIE_UTIL_RX_BYTES), ("tx_kib_s", nv.NVML_PCIE_UTIL_TX_BYTES)):
                try:
                    d[k] = int(nv.nvmlDeviceGetPcieThroughput(h, c))
                except Exception:  # noqa: BLE001
                    d[k] = None
            out[i] = d
        return out
    except Exception:  # noqa: BLE001
        return None
    finally:
        try:
            nv.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass


class GpuSampler:
    """Background sampler: ``with GpuSampler(0.5) as s: ...; s.summary()``. Overhead = one nvidia-smi call per interval."""

    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval; self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event(); self._th: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(sample_once()); self._stop.wait(self.interval)

    def __enter__(self) -> "GpuSampler":
        self._th = threading.Thread(target=self._run, daemon=True); self._th.start(); return self

    def __exit__(self, *a) -> None:
        self._stop.set()
        if self._th:
            self._th.join(5)

    def summary(self) -> dict[str, Any]:
        from .metrics import summarize
        per: dict[int, dict[str, list[float]]] = {}
        for s in self.samples:
            for g in s["gpus"]:
                for k, v in g.items():
                    if k != "index" and v is not None:
                        per.setdefault(g["index"], {}).setdefault(k, []).append(v)
        return {"n_samples": len(self.samples), "source": self.samples[0]["source"] if self.samples else "unavailable",
                "per_gpu": {i: {k: summarize(v) for k, v in d.items()} for i, d in per.items()}}
