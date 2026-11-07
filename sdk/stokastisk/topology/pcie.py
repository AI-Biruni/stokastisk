"""PCIe / link classification from ``nvidia-smi topo -m`` text, and PCIe link speed from sysfs."""
from __future__ import annotations

import pathlib
import re

from .models import LINK_CLASSES

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
SYSFS_PCI = pathlib.Path("/sys/bus/pci/devices")
_GEN_BY_GTS = {"2.5": 1, "5.0": 2, "8.0": 3, "16.0": 4, "32.0": 5, "64.0": 6}


def classify_cell(cell: str) -> str:
    c = cell.strip().upper()
    if re.fullmatch(r"NV\d+", c):
        return "NVLINK"
    return c if c in LINK_CLASSES else "UNKNOWN"


def parse_topo_matrix(text: str) -> dict[str, str]:
    """``{"0-1": "PHB"}`` (lower index first). Unknown / unparsable cells become UNKNOWN; an unparsable matrix yields ``{}``."""
    text = _ANSI.sub("", text)
    lines = [l for l in text.splitlines() if l.strip()]
    hdr = next((i for i, l in enumerate(lines) if re.match(r"\s*GPU\d+(\s+GPU\d+)+", l)), None)
    if hdr is None:
        return {}
    cols = re.findall(r"GPU\d+", lines[hdr])
    out: dict[str, str] = {}
    for l in lines[hdr + 1:]:
        m = re.match(r"\s*GPU(\d+)\s+(.*)", l)
        if not m:
            continue
        row = int(m.group(1)); cells = re.split(r"\s+", m.group(2).strip())
        for j, col in enumerate(cols):
            if j >= len(cells):
                break
            col_i = int(col[3:])
            if col_i > row:
                out[f"{row}-{col_i}"] = classify_cell(cells[j])
    return out


def pcie_sysfs(pci_bus_id: str | None) -> dict[str, int]:
    """Current/max PCIe generation + width from /sys (no root needed). Missing files -> missing keys."""
    if not pci_bus_id:
        return {}
    base = SYSFS_PCI / pci_bus_id.lower().replace("00000000:", "0000:", 1)
    out: dict[str, int] = {}
    def rd(name):
        try:
            return (base / name).read_text().strip()
        except OSError:
            return None
    for key, f in (("pcie_gen_current", "current_link_speed"), ("pcie_gen_max", "max_link_speed")):
        v = rd(f)
        m = re.match(r"([\d.]+)\s*GT/s", v or "")
        if m and m.group(1) in _GEN_BY_GTS:
            out[key] = _GEN_BY_GTS[m.group(1)]
    for key, f in (("pcie_width_current", "current_link_width"), ("pcie_width_max", "max_link_width")):
        v = rd(f)
        if v and v.isdigit():
            out[key] = int(v)
    return out
