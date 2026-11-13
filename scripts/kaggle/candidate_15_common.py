"""Constants and helpers shared by the notebook-15 stage scripts (build 3). Standard library + the Kaggle `_common.py`."""
import json, os, pathlib, re, subprocess, sys

from _common import OUT, gpu_used_mib, loaded_libs, process_tree, sha256  # noqa: F401  (re-exported)

MODEL = os.environ.get("STOKASTISK_CANDIDATE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
IDLE_MIB = 500
VERSION, BUILD = "0.1.0", "3"
WHEEL = f"stokastisk-{VERSION}-{BUILD}-cp313-cp313-linux_x86_64.whl"
INSTALL_JSON = OUT / "prereq-install" / "02_install_integrated.json"
ENGINE_SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}, "population": {"type": "integer"}}, "required": ["city", "population"], "additionalProperties": False}
# one JSON-schema feature the pinned xgrammar backend does NOT support (vllm.v1.structured_output.backend_xgrammar.has_xgrammar_unsupported_json_features):
# with the default backend 'auto' vLLM would silently serve this through guidance/outlines; with the pinned backend it must be REJECTED.
UNSUPPORTED_BY_XGRAMMAR = {"type": "object", "properties": {"n": {"type": "integer", "multipleOf": 3}}, "required": ["n"]}


def flash_libs_tree(pids):
    """{pid: [flash-family shared objects mapped]} for the given pids (stokastisk.prohibited policy; unreadable maps are reported, never treated as clean)."""
    from stokastisk import prohibited
    out = {}
    for p in pids:
        try:
            out[str(p)] = prohibited.loaded_flash_libraries(p)
        except OSError as e:
            out[str(p)] = [f"UNREADABLE: {e}"]
    return out


def flat(libmap):
    return sorted({x for v in libmap.values() for x in v})


def tail(path, n=40):
    try:
        return "\n".join(pathlib.Path(path).read_text(errors="replace").splitlines()[-n:])
    except OSError as e:
        return f"(cannot read {path}: {e})"


def triton_artifacts(cache_dir, needle="kernel_unified_attention"):
    return sorted({p.name for p in pathlib.Path(cache_dir).rglob("*") if p.is_file() and needle in p.name})


def raises(fn, *exc):
    try:
        fn()
    except exc:
        return True
    except BaseException:  # noqa: BLE001
        return False
    return False


def no_raise(fn):
    try:
        fn()
    except BaseException:  # noqa: BLE001
        return False
    return True
