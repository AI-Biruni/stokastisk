"""Plain, JSON-serialisable topology records. A value that could not be measured is ``None`` — never invented."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

LINK_CLASSES = ("NVLINK", "PIX", "PXB", "PHB", "NODE", "SYS", "UNKNOWN")


@dataclass
class DeviceInfo:
    index: int
    name: str | None = None
    uuid: str | None = None
    compute_capability: tuple[int, int] | None = None
    total_vram_mib: float | None = None
    free_vram_mib: float | None = None
    used_vram_mib: float | None = None
    pci_bus_id: str | None = None
    pcie_gen_current: int | None = None
    pcie_gen_max: int | None = None
    pcie_width_current: int | None = None
    pcie_width_max: int | None = None
    sources: dict[str, str] = field(default_factory=dict)   # field -> where the value came from (torch / nvml / nvidia-smi / sysfs)


@dataclass
class PeerAccess:
    """CUDA peer access per ordered pair. ``matrix[src][dst]`` is ``torch.cuda.can_device_access_peer(src, dst)`` (None = not measured)."""
    matrix: dict[int, dict[int, bool | None]] = field(default_factory=dict)
    source: str = "unavailable"

    def can_access(self, src: int, dst: int) -> bool | None:
        return self.matrix.get(src, {}).get(dst)

    @property
    def all_pairs_available(self) -> bool:
        vals = [v for r in self.matrix.values() for v in r.values()]
        return bool(vals) and all(v is True for v in vals)

    @property
    def any_pair_available(self) -> bool:
        return any(v is True for r in self.matrix.values() for v in r.values())


@dataclass
class Topology:
    devices: list[DeviceInfo] = field(default_factory=list)
    peer_access: PeerAccess = field(default_factory=PeerAccess)
    link_class: dict[str, str] = field(default_factory=dict)       # "0-1" -> one of LINK_CLASSES
    link_source: str = "unavailable"
    nvlink_available: bool | None = None                           # None = not measured
    nccl_available: bool | None = None
    nccl_compiled_version: str | None = None
    nccl_runtime_transport: str | None = None                      # only known from an NCCL log; never guessed
    cuda_runtime_version: str | None = None
    driver_version: str | None = None
    cpu_count: int | None = None
    cpu_affinity: list[int] | None = None
    system_ram_available_gib: float | None = None
    system_ram_total_gib: float | None = None
    notes: list[str] = field(default_factory=list)
    recommended_parallelism: str | None = None                      # filled by discover_topology(); rule-based, see stokastisk.parallel.recommender

    @property
    def gpu_count(self) -> int:
        return len(self.devices)

    def pair_link(self, a: int, b: int) -> str:
        lo, hi = sorted((a, b))
        return self.link_class.get(f"{lo}-{hi}", "UNKNOWN")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        lines = [f"Topology: {self.gpu_count} GPU(s), driver={self.driver_version}, CUDA runtime={self.cuda_runtime_version}"]
        for d in self.devices:
            lines.append(f"  GPU{d.index} {d.name} cc={d.compute_capability} vram={d.total_vram_mib} MiB uuid={d.uuid} pci={d.pci_bus_id}")
        for k, v in sorted(self.link_class.items()):
            lines.append(f"  link {k}: {v}")
        lines.append(f"  P2P matrix: {self.peer_access.matrix}  NVLink: {self.nvlink_available}  NCCL: {self.nccl_available}")
        lines.append(f"  recommended parallelism: {self.recommended_parallelism}")
        return "\n".join(lines)
