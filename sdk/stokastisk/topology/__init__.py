"""GPU topology discovery (torch / NVML / nvidia-smi / sysfs; all but torch optional)."""
from .discovery import discover_topology
from .models import DeviceInfo, LINK_CLASSES, PeerAccess, Topology
from .pcie import classify_cell, parse_topo_matrix

__all__ = ["discover_topology", "DeviceInfo", "PeerAccess", "Topology", "LINK_CLASSES", "parse_topo_matrix", "classify_cell"]
