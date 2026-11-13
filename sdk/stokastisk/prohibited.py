"""Policy and scanners for the Stokastisk "no FlashAttention / FlashInfer / FlashMLA" rule.

Stokastisk is a Tesla T4 / SM75 / Triton-attention (``TRITON_ATTN``) derivative of vLLM. It must not SHIP FlashAttention, FlashInfer or FlashMLA code or binaries, and
it must not LOAD them at runtime. This module is the single source of truth for that rule; it is used by the wheel builder, the installer, the Kaggle static stage and
the unit tests, so the rule is identical everywhere.

Three evidence classes are kept apart so that ordinary text never becomes a false positive and a real implementation is never excused:

* **path hits**   - a file or directory whose NAME is a Flash-family implementation or integration module (``flashinfer*.py``, ``flashmla*.py``, ``flash_attn.py``,
  ``*_flash_attn.py``, ``fa_utils.py`` ...). Source (``.py``), bytecode (``.pyc``), native (``.so``/``.cubin``/``.ptx``/``.fatbin``/``.a``/``.o``) alike. FAIL.
* **native hits** - a native object (ELF/cubin/ptx/fatbin) whose bytes contain a Flash-family kernel/symbol marker, whatever the file is called. FAIL.
* **binding hits**- a ``.py`` file that calls the registered native ops of a Flash library (the ``torch.ops`` namespaces of the vLLM FlashAttention extensions). FAIL.
* **references**  - a ``.py`` file that merely *mentions* or guards-imports a Flash package (``import flashinfer`` inside a function, docs, enum names). Reported for
  transparency, never counted as an implementation (the runtime import guard in ``vllm.t4_backend_policy`` makes those imports fail explicitly).

Names that merely CONTAIN the word "flash" but are not FlashAttention (``triton_reshape_and_cache_flash`` - a Triton KV-cache write kernel;
``longcat_flash`` - a model family; ``*_dflash`` - a speculative-decoding method) are deliberately NOT path hits; ``ALLOWED_FLASH_NAMED`` records them so a reviewer can see the decision.
"""
from __future__ import annotations

import pathlib
import re
import zipfile
from typing import Any, Iterable

# a path COMPONENT (file stem or directory) is a Flash-family module when it matches one of these
STEM_RE = re.compile(r"flashinfer|flashmla|flash_mla|flash_attn|flashattn|vllm_fa|flash_fwd|flash_bwd", re.I)
# Flash-family integration modules that do not carry the family name in a way STEM_RE can see
EXACT_STEMS = frozenset({"fa_utils", "rocm_aiter_fa", "fa4_cutedsl_config", "fa3_utils", "fa2_utils"})
FA_PREFIX_RE = re.compile(r"^fa[234]_")           # fa2_*, fa3_*, fa4_* (FlashAttention 2/3/4 helpers)
ALLOWED_FLASH_NAMED = {
    "triton_reshape_and_cache_flash": "Triton kernel that writes K/V into the paged cache (NHD 'flash' layout); used by TRITON_ATTN; not FlashAttention",
    "longcat_flash": "LongCat-Flash model family", "longcat_flash_mtp": "LongCat-Flash MTP head",
    "dflash": "DFlash speculative-decoding method", "laguna_dflash": "DFlash drafter model", "qwen3_dflash": "DFlash drafter model",
}
BINARY_SUFFIXES = (".so", ".cubin", ".ptx", ".fatbin", ".a", ".o", ".pyd", ".dylib")
# byte markers that identify Flash-family native code regardless of the file name
NATIVE_MARKERS = (b"flash_fwd", b"flash_bwd", b"FlashMLA", b"flash_mla", b"flashinfer", b"FlashInfer", b"_vllm_fa2", b"_vllm_fa3", b"_vllm_fa4",
                  b"vllm_flash_attn", b"flash_attn_2_cuda", b"flash_attn_interface", b"FLASHATTENTION")
BINDING_RE = re.compile(rb"torch\.ops\.(?:_vllm_fa[234]\w*|flashmla\w*|_flashmla\w*)")
REFERENCE_RE = re.compile(rb"(?:^|\n)[ \t]*(?:import|from)[ \t]+(?:flashinfer|flash_attn|flash_mla|vllm\.vllm_flash_attn|flashmla)\b")
# top-level import names that the runtime guard (vllm.t4_backend_policy) replaces with a hard ImportError
BLOCKED_IMPORT_NAMES = ("flashinfer", "flash_attn", "flash_attn_2_cuda", "flash_attn_interface", "flash_mla", "flashmla", "vllm_flash_attn")
# distributions that must never be installed next to Stokastisk (checked by the installer and prohibited_report)
PROHIBITED_DISTS = ("flash-attn", "vllm-flash-attn", "flashinfer", "flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache", "flash-mla",
                    "cuda-tile", "nvidia-cuda-tileiras", "humming-kernels", "nvidia-cutlass-dsl", "nvidia-cutlass-dsl-libs-base", "nvidia-cutlass-dsl-libs-cu13",
                    "quack-kernels", "tokenspeed-mla", "tokenspeed-triton")


def _stem(component: str) -> str:
    return component.split(".", 1)[0]


def is_prohibited_path(path: str) -> bool:
    """True if any component of ``path`` (a ``/``-separated path inside a wheel or tree) names a Flash-family module or package."""
    for comp in path.split("/"):
        if not comp:
            continue
        stem = _stem(comp)
        if stem in ALLOWED_FLASH_NAMED:
            continue
        if STEM_RE.search(stem) or stem in EXACT_STEMS or FA_PREFIX_RE.match(stem):
            return True
    return False


def is_binary(path: str) -> bool:
    return path.endswith(BINARY_SUFFIXES) or ".so." in path


def native_markers(data: bytes) -> list[str]:
    """Flash-family markers present in a native object's bytes."""
    return sorted({m.decode() for m in NATIVE_MARKERS if m in data})


def scan_members(names: Iterable[str], read) -> dict[str, Any]:
    """Scan a set of archive/tree members. ``read(name) -> bytes`` is only called for files whose content must be inspected."""
    names = list(names)
    rep: dict[str, Any] = {"files": 0, "py": 0, "pyc": 0, "native": 0, "pycache_dirs": [], "path_hits": [], "native_hits": {}, "binding_hits": [],
                           "references": {}, "allowed_flash_named": []}
    caches: set[str] = set()
    for n in names:
        if n.endswith("/"):
            continue
        rep["files"] += 1
        if "/__pycache__/" in n or n.startswith("__pycache__/"):
            caches.add(n.rsplit("/", 1)[0])
        if n.endswith(".py"):
            rep["py"] += 1
        elif n.endswith(".pyc"):
            rep["pyc"] += 1
        stem = _stem(n.rsplit("/", 1)[-1])
        if stem in ALLOWED_FLASH_NAMED and "flash" in n.lower():
            rep["allowed_flash_named"].append(n)
        if is_prohibited_path(n):
            rep["path_hits"].append(n)
        if is_binary(n):
            rep["native"] += 1
            hits = native_markers(read(n))
            if hits:
                rep["native_hits"][n] = hits
        elif n.endswith(".py") and not is_prohibited_path(n):
            data = read(n)
            if BINDING_RE.search(data):
                rep["binding_hits"].append(n)
            if REFERENCE_RE.search(data):
                rep["references"][n] = len(REFERENCE_RE.findall(data))
    rep["pycache_dirs"] = sorted(caches)
    rep["path_hits"].sort()
    rep["allowed_flash_named"].sort()
    rep["prohibited_file_count"] = len(rep["path_hits"]) + len(rep["native_hits"]) + len(rep["binding_hits"])
    rep["clean"] = rep["prohibited_file_count"] == 0
    return rep


def scan_wheel(path: str | pathlib.Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as z:
        return scan_members(z.namelist(), z.read)


def scan_tree(root: str | pathlib.Path) -> dict[str, Any]:
    root = pathlib.Path(root)
    names = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()]
    return scan_members(names, lambda n: (root / n).read_bytes())


def flash_in_maps(lines: Iterable[str]) -> list[str]:
    """Flash-family shared objects in the lines of ``/proc/<pid>/maps`` (path column only)."""
    found = set()
    for ln in lines:
        parts = ln.split(None, 5)
        if len(parts) == 6:
            p = parts[5].strip()
            low = p.lower()
            if p.startswith("/") and (any(k in low for k in ("flashinfer", "flash_attn", "flashattn", "flashmla", "flash_mla", "_vllm_fa", "libflash")) or is_prohibited_path(p.lstrip("/"))):
                found.add(p)
    return sorted(found)


def loaded_flash_libraries(pid: int | str = "self") -> list[str]:
    """Flash-family shared objects mapped by process ``pid`` (empty list = none; unreadable maps raise OSError, never silently pass)."""
    with open(f"/proc/{pid}/maps") as f:
        return flash_in_maps(f)
