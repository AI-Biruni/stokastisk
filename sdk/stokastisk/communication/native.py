"""Optional ctypes loader for ``native/host_staged_xfer.cpp`` (C++ / CUDA Runtime). Built on demand with nvcc; absence is not an error."""
from __future__ import annotations

import ctypes
import os
import pathlib
import shutil
import subprocess

_LIB = None


def build_native(src: str | os.PathLike, out_dir: str | os.PathLike, arch: str = "sm_75") -> pathlib.Path | None:
    nvcc = shutil.which("nvcc")
    if not nvcc:
        return None
    out = pathlib.Path(out_dir) / "libstokastisk_xfer.so"; out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [nvcc, "-O2", "-std=c++17", "-shared", "-Xcompiler", "-fPIC", "-gencode", f"arch=compute_{arch[3:]},code={arch}", str(src), "-o", str(out), "-lcudart"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return out if r.returncode == 0 else None


def load_native(path: str | os.PathLike | None = None):
    global _LIB
    if _LIB is not None:
        return _LIB
    p = path or os.environ.get("STOKASTISK_XFER_LIB")
    if not p or not pathlib.Path(p).is_file():
        return None
    lib = ctypes.CDLL(str(p))
    lib.stokastisk_xfer_create.restype = ctypes.c_void_p
    lib.stokastisk_xfer_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_size_t, ctypes.c_int]
    lib.stokastisk_xfer_transfer.restype = ctypes.c_int
    lib.stokastisk_xfer_transfer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_double)]
    lib.stokastisk_xfer_destroy.argtypes = [ctypes.c_void_p]
    lib.stokastisk_xfer_last_error.restype = ctypes.c_char_p
    _LIB = lib
    return lib


class NativeHostStaged:
    def __init__(self, src_device: int, dst_device: int, *, chunk_bytes: int = 4 << 20, n_buffers: int = 2, lib=None) -> None:
        self.lib = lib or load_native()
        if self.lib is None:
            raise RuntimeError("native transport library not available (set STOKASTISK_XFER_LIB or call build_native)")
        self.ctx = self.lib.stokastisk_xfer_create(src_device, dst_device, chunk_bytes, n_buffers)
        if not self.ctx:
            raise RuntimeError(self.lib.stokastisk_xfer_last_error().decode())

    def transfer(self, src_tensor, dst_tensor) -> dict:
        ms = ctypes.c_double(0)
        n = src_tensor.numel() * src_tensor.element_size()
        rc = self.lib.stokastisk_xfer_transfer(self.ctx, ctypes.c_void_p(src_tensor.data_ptr()), ctypes.c_void_p(dst_tensor.data_ptr()), n, ctypes.byref(ms))
        if rc != 0:
            raise RuntimeError(self.lib.stokastisk_xfer_last_error().decode())
        return {"bytes": n, "gpu_ms": ms.value, "gbps": n / (ms.value / 1000) / 1e9 if ms.value else None}

    def close(self) -> None:
        if self.ctx:
            self.lib.stokastisk_xfer_destroy(self.ctx); self.ctx = None
