"""discover_topology(): torch -> NVML -> nvidia-smi -> sysfs, every layer optional. Nothing is hardcoded to Kaggle's PHB."""
from __future__ import annotations

import os

from . import nvml, pcie
from .gpu_capabilities import torch_devices
from .models import DeviceInfo, Topology
from .peer_access import measure_peer_access


def _merge_nvml(devs: list[DeviceInfo], rows: list[dict], topo: Topology) -> None:
    by = {d.index: d for d in devs}
    for r in rows:
        d = by.setdefault(r["index"], DeviceInfo(index=r["index"]))
        for src_key, attr in (("name", "name"), ("uuid", "uuid"), ("pci_bus_id", "pci_bus_id"), ("total_mib", "total_vram_mib"),
                              ("free_mib", "free_vram_mib"), ("used_mib", "used_vram_mib"), ("pcie_gen_current", "pcie_gen_current"),
                              ("pcie_gen_max", "pcie_gen_max"), ("pcie_width_current", "pcie_width_current"), ("pcie_width_max", "pcie_width_max")):
            if src_key in r and getattr(d, attr) is None:
                setattr(d, attr, r[src_key]); d.sources[attr] = "nvml"
        if r.get("driver_version") and not topo.driver_version:
            topo.driver_version = r["driver_version"]
    devs[:] = [by[k] for k in sorted(by)]
    if rows:
        topo.nvlink_available = any(r.get("nvlink_active_links", 0) > 0 for r in rows)


def _merge_smi(devs: list[DeviceInfo], topo: Topology) -> None:
    rows = nvml.smi_query("index,name,uuid,pci.bus_id,memory.total,memory.free,memory.used,driver_version,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max")
    if not rows:
        return
    by = {d.index: d for d in devs}
    toint = lambda x: int(x) if x.isdigit() else None  # noqa: E731
    tofl = lambda x: float(x) if x.replace(".", "", 1).isdigit() else None  # noqa: E731
    for r in rows:
        if len(r) < 12 or not r[0].isdigit():
            continue
        d = by.setdefault(int(r[0]), DeviceInfo(index=int(r[0]))); devs_changed = d
        for attr, val in (("name", r[1]), ("uuid", r[2]), ("pci_bus_id", r[3]), ("total_vram_mib", tofl(r[4])), ("free_vram_mib", tofl(r[5])),
                          ("used_vram_mib", tofl(r[6])), ("pcie_gen_current", toint(r[8])), ("pcie_gen_max", toint(r[9])),
                          ("pcie_width_current", toint(r[10])), ("pcie_width_max", toint(r[11]))):
            if val not in (None, "") and getattr(d, attr) is None:
                setattr(d, attr, val); d.sources[attr] = "nvidia-smi"
        if not topo.driver_version and r[7]:
            topo.driver_version = r[7]
    devs[:] = [by[k] for k in sorted(by)]


def discover_topology(*, with_recommendation: bool = True) -> Topology:
    t = Topology()
    devs = torch_devices()
    rows = nvml.nvml_devices()
    if rows is not None:
        _merge_nvml(devs, rows, t)
    else:
        t.notes.append("NVML (pynvml) unavailable; using nvidia-smi / torch only")
    _merge_smi(devs, t)
    for d in devs:                                          # sysfs fills PCIe gaps (no root needed)
        for k, v in pcie.pcie_sysfs(d.pci_bus_id).items():
            if getattr(d, k) is None:
                setattr(d, k, v); d.sources[k] = "sysfs"
    t.devices = devs
    t.peer_access = measure_peer_access(len(devs) if devs else None)
    matrix = nvml.smi_topo_matrix()
    if matrix:
        t.link_class = pcie.parse_topo_matrix(matrix); t.link_source = "nvidia-smi topo -m"
    if len(devs) > 1 and not t.link_class:
        t.notes.append("link class unavailable (nvidia-smi topo -m missing or unparsable): reported as UNKNOWN")
        t.link_class = {f"{a}-{b}": "UNKNOWN" for a in range(len(devs)) for b in range(a + 1, len(devs))}
    if t.nvlink_available is None and t.link_class and "UNKNOWN" not in t.link_class.values():
        t.nvlink_available = any(v == "NVLINK" for v in t.link_class.values())   # only claimed when every pair was classified
    try:
        import torch
        t.cuda_runtime_version = torch.version.cuda
        t.nccl_available = bool(torch.distributed.is_available() and torch.distributed.is_nccl_available())
        if t.nccl_available:
            t.nccl_compiled_version = ".".join(map(str, torch.cuda.nccl.version()))
    except Exception as e:  # noqa: BLE001
        t.notes.append(f"torch unavailable: {type(e).__name__}")
    try:
        t.cpu_affinity = sorted(os.sched_getaffinity(0)); t.cpu_count = len(t.cpu_affinity)
    except AttributeError:
        t.cpu_count = os.cpu_count()
    from ..limits import system_ram_gib
    t.system_ram_total_gib = system_ram_gib()
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                t.system_ram_available_gib = int(line.split()[1]) / 2**20
    except OSError:
        pass
    if with_recommendation:
        from ..parallel.recommender import default_recommendation
        t.recommended_parallelism = default_recommendation(t)
    return t
