"""Nsight availability probe. Nsight is never required: an unusable profiler is an explicit SKIPPED."""
from __future__ import annotations

import shutil
import subprocess
from typing import Any


def nsight_status() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for tool in ("nsys", "ncu"):
        exe = shutil.which(tool)
        if not exe:
            out[tool] = {"status": "SKIPPED", "reason": f"{tool} not on PATH"}; continue
        try:
            v = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
            out[tool] = {"status": "AVAILABLE", "path": exe, "version": v[0] if v else None}
        except Exception as e:  # noqa: BLE001
            out[tool] = {"status": "SKIPPED", "reason": f"{type(e).__name__}: {e}"}
    # GPU performance counters are gated by NVreg_RestrictProfilingToAdminUsers; /proc shows the driver setting when readable.
    try:
        txt = open("/proc/driver/nvidia/params").read()
        out["profiling_restricted_to_admin"] = next((l.split(":")[1].strip() for l in txt.splitlines() if l.startswith("RmProfilingAdminOnly")), None)
    except OSError:
        out["profiling_restricted_to_admin"] = None
    return out
