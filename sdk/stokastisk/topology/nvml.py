"""Optional NVML (``pynvml`` / ``nvidia-ml-py``) and ``nvidia-smi`` helpers. Neither is required: every function degrades to ``None``."""
from __future__ import annotations

import logging

import shutil
import subprocess
from typing import Any

log = logging.getLogger("stokastisk")


def _pynvml():
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        return pynvml
    except Exception:  # noqa: BLE001 - missing package, missing driver, any NVML failure
        return None


def nvml_devices() -> list[dict[str, Any]] | None:
    """Per-GPU dict from NVML, or None when NVML is unusable."""
    nv = _pynvml()
    if nv is None:
        return None
    out = []
    try:
        dec = lambda x: x.decode() if isinstance(x, bytes) else x  # noqa: E731
        for i in range(nv.nvmlDeviceGetCount()):
            h = nv.nvmlDeviceGetHandleByIndex(i)
            d: dict[str, Any] = {"index": i, "name": dec(nv.nvmlDeviceGetName(h)), "uuid": dec(nv.nvmlDeviceGetUUID(h))}
            try:
                mi = nv.nvmlDeviceGetMemoryInfo(h); d.update(total_mib=mi.total / 2**20, free_mib=mi.free / 2**20, used_mib=mi.used / 2**20)
            except Exception:  # noqa: BLE001
                log.debug("ignored Exception (best-effort probe)", exc_info=True)
            try:
                d["pci_bus_id"] = dec(nv.nvmlDeviceGetPciInfo(h).busId)
            except Exception:  # noqa: BLE001
                log.debug("ignored Exception (best-effort probe)", exc_info=True)
            for key, fn in (("pcie_gen_current", "nvmlDeviceGetCurrPcieLinkGeneration"), ("pcie_gen_max", "nvmlDeviceGetMaxPcieLinkGeneration"),
                            ("pcie_width_current", "nvmlDeviceGetCurrPcieLinkWidth"), ("pcie_width_max", "nvmlDeviceGetMaxPcieLinkWidth")):
                try:
                    d[key] = int(getattr(nv, fn)(h))
                except Exception:  # noqa: BLE001
                    log.debug("ignored Exception (best-effort probe)", exc_info=True)
            # NVLink: any active link means present. T4 reports none / NOT_SUPPORTED.
            active = 0
            for link in range(6):
                try:
                    if nv.nvmlDeviceGetNvLinkState(h, link):
                        active += 1
                except Exception:  # noqa: BLE001
                    break
            d["nvlink_active_links"] = active
            out.append(d)
        try:
            out and out[0].update(driver_version=(lambda v: v.decode() if isinstance(v, bytes) else v)(nv.nvmlSystemGetDriverVersion()))
        except Exception:  # noqa: BLE001
            log.debug("ignored Exception (best-effort probe)", exc_info=True)
        return out
    except Exception:  # noqa: BLE001
        return None
    finally:
        try:
            nv.nvmlShutdown()
        except Exception:  # noqa: BLE001
            log.debug("ignored Exception (best-effort probe)", exc_info=True)


def smi_query(fields: str) -> list[list[str]] | None:
    """Rows of ``nvidia-smi --query-gpu=<fields>``, or None when nvidia-smi is missing/fails."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    return [[c.strip() for c in line.split(",")] for line in r.stdout.strip().splitlines() if line.strip()]


def smi_topo_matrix() -> str | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "topo", "-m"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout if r.returncode == 0 and r.stdout.strip() else None
