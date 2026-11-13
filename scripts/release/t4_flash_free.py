#!/usr/bin/env python3.13
"""Deterministic "Flash-free" transform of the vLLM 0.25.1 tree for the Stokastisk Tesla T4 / SM75 / TRITON_ATTN build.

usage: t4_flash_free.py SRC_DIR OUT_DIR [--report REPORT.json]        (SRC_DIR holds the `vllm/` directory; OUT_DIR must not exist)
       library use: transform(files: dict[str, bytes]) -> (files, report)       (this is what build_integrated_wheel.py calls)

What it does, in order (every step is asserted; an unexpected tree makes the build FAIL instead of producing a half-patched wheel):

 1. REMOVE every file/directory the shared policy (`stokastisk.prohibited`) names as a Flash-family implementation or integration module:
    flashinfer*.py, flashmla*.py, flash_attn.py, *_flash_attn.py, plus fa_utils.py, rocm_aiter_fa.py, fa4_cutedsl_config.py (FlashAttention integration layers).
 2. REWRITE every *unguarded* import of a removed module (module level or function level) to import from ONE generated placeholder module,
    `vllm/utils/t4_prohibited_stubs.py`. Imports inside `try/except ImportError` or `with suppress(ImportError)` are left alone: they are designed to degrade
    and now simply take the "not available" branch. The placeholder is NOT an implementation: predicates answer False, kernel classes answer "not supported"
    to `is_supported`/`can_implement`, and every function raises `ProhibitedBackendError` (an ImportError subclass) naming the removed backend.
 3. ADD `vllm/t4_backend_policy.py` (the policy + runtime import guard) and import it first thing in `vllm/__init__.py`, so that `import flashinfer`,
    `import flash_attn` ... fail explicitly (`sys.modules[name] = None`) and `importlib.util.find_spec` reports them absent.
 4. GUARD the backend registries: `AttentionBackendEnum.get_path()/get_class()` and `MLAPrefillBackendEnum.get_path()/get_class()` raise `ProhibitedBackendError` for
    every Flash-family member (even when a user registered an override), so automatic selection skips them (the CUDA platform already treats ImportError as
    "invalid candidate") and an explicit selection fails with a clear message. `AttentionConfig.backend` / `mla_prefill_backend` reject them at construction.
 5. IMPROVE the CUDA platform's selection error so the ImportError text (the reason) is kept instead of the bare string "ImportError".

The transform never touches a `.so`. It reads the byte-pinned vLLM wheel's tree and returns new bytes; nothing on disk is modified."""
from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import re
import sys
import textwrap

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sdk"))
from stokastisk.prohibited import is_prohibited_path  # noqa: E402

STUB_MODULE = "vllm.utils.t4_prohibited_stubs"
STUB_PATH = "vllm/utils/t4_prohibited_stubs.py"
POLICY_PATH = "vllm/t4_backend_policy.py"
PREDICATE_PREFIXES = ("has_", "is_", "should_use_", "supports_", "can_use_", "flash_attn_supports_")
NONE_RETURNING = frozenset({"get_flash_attn_version"})          # "no FlashAttention version available" is None, not False
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731

POLICY_SOURCE = '''\
# SPDX-License-Identifier: Apache-2.0
"""Stokastisk Tesla T4 (SM75) build: FlashAttention / FlashInfer / FlashMLA policy.   [added by scripts/release/t4_flash_free.py - not part of upstream vLLM]

This build is a Triton-attention (TRITON_ATTN) derivative. The Flash-family implementations are not shipped. This module makes that explicit at runtime:

* ``ProhibitedBackendError`` (an ``ImportError`` subclass, so vLLM's own "backend unavailable" handling keeps working) names what is missing and why.
* ``raise_if_prohibited_backend`` guards the attention-backend registries; ``check_attention_backend`` rejects the backends in configuration objects.
* ``install_import_guard`` makes ``import flashinfer`` / ``import flash_attn`` ... fail with ``ModuleNotFoundError`` and ``importlib.util.find_spec`` return ``None``,
  even if one of those packages were installed next to this build. A package that was imported BEFORE vLLM is refused loudly.
* ``prohibited_class`` / ``prohibited_function`` / ``false_predicate`` / ``none_function`` build the placeholders in ``vllm.utils.t4_prohibited_stubs``.
"""
from __future__ import annotations

import re
import sys

BLOCKED_IMPORT_NAMES = %(blocked)r
_BACKEND_NAME_RE = re.compile(r"FLASH_?(INFER|MLA|ATTN)|AITER_FA", re.IGNORECASE)
_PATH_RE = re.compile(r"flashinfer|flashmla|flash_mla|flash_attn|flashattn|vllm_fa|aiter_fa|\\.fa_utils\\b", re.IGNORECASE)
_WHY = ("it is a FlashAttention/FlashInfer/FlashMLA implementation, which the Stokastisk Tesla T4 (SM75) build does not ship; "
        "this build supports the Triton attention backend (TRITON_ATTN) only")


class ProhibitedBackendError(ImportError, NotImplementedError):
    """A FlashAttention / FlashInfer / FlashMLA backend, kernel or helper was requested; this build does not provide it."""


def is_prohibited_backend(name: str, *paths: str | None) -> bool:
    return bool(_BACKEND_NAME_RE.search(name or "")) or any(p and _PATH_RE.search(p) for p in paths)


def raise_if_prohibited_backend(name: str, *paths: str | None) -> None:
    if is_prohibited_backend(name, *paths):
        raise ProhibitedBackendError(f"attention backend {name!r} is not available: {_WHY}.")


def check_attention_backend(backend) -> None:
    """Configuration-time check (``backend`` is an enum member, a string or None)."""
    name = getattr(backend, "name", backend)
    if name is not None and is_prohibited_backend(str(name)):
        raise ValueError(f"attention backend {name!s} is not available: {_WHY}.")


def install_import_guard() -> None:
    for n in BLOCKED_IMPORT_NAMES:
        if sys.modules.get(n, "absent") not in (None, "absent"):
            raise ProhibitedBackendError(f"{n!r} was imported before vLLM: {_WHY}. Remove it from the environment.")
        sys.modules[n] = None            # import raises ModuleNotFoundError; importlib.util.find_spec(n) returns None


def _reason(what: str) -> str:
    return f"{what} is not available: {_WHY}."


def false_predicate(name: str):
    def predicate(*args, **kwargs) -> bool:
        return False
    predicate.__name__ = predicate.__qualname__ = name
    predicate.__doc__ = f"Placeholder for the removed {name}(): always False on this build."
    return predicate


def none_function(name: str):
    def fn(*args, **kwargs):
        return None
    fn.__name__ = fn.__qualname__ = name
    fn.__doc__ = f"Placeholder for the removed {name}(): always None on this build."
    return fn


def prohibited_function(name: str):
    def fn(*args, **kwargs):
        raise ProhibitedBackendError(_reason(f"{name}()"))
    fn.__name__ = fn.__qualname__ = name
    fn.__doc__ = f"Placeholder for the removed {name}(): raises ProhibitedBackendError."
    return fn


def prohibited_class(name: str):
    """A class that answers 'not supported' to vLLM's kernel/backend capability probes and cannot be instantiated."""
    def _init(self, *args, **kwargs):
        raise ProhibitedBackendError(_reason(name))

    def _is_supported(cls, *args, **kwargs):
        return False, _reason(name)

    def _can_implement(cls, *args, **kwargs):
        return False, _reason(name)

    def _validate_configuration(cls, *args, **kwargs):
        return [_reason(name)]

    ns = {"__init__": _init, "__doc__": f"Placeholder for the removed {name}: not supported on this build.",
          "is_supported": classmethod(_is_supported), "can_implement": classmethod(_can_implement),
          "validate_configuration": classmethod(_validate_configuration), "get_min_capability": classmethod(lambda cls: 10**6)}
    return type(name, (), ns)


install_import_guard()
'''

REG_ANCHOR_IMPORT = "from vllm.utils.import_utils import resolve_obj_by_qualname\n"
EDITS: dict[str, list[tuple[str, str]]] = {
    "vllm/__init__.py": [
        ("from .version import __version__, __version_tuple__  # isort:skip\n",
         "from .version import __version__, __version_tuple__  # isort:skip\n\n"
         "# Stokastisk T4 build: FlashAttention/FlashInfer/FlashMLA import guard + backend policy (must run before any other vllm import)\n"
         "import vllm.t4_backend_policy  # noqa: F401  # isort:skip\n"),
    ],
    "vllm/v1/attention/backends/registry.py": [
        (REG_ANCHOR_IMPORT, REG_ANCHOR_IMPORT + "from vllm.t4_backend_policy import raise_if_prohibited_backend\n"),
        ("        path = _ATTN_OVERRIDES.get(self, self.value)\n",
         "        raise_if_prohibited_backend(self.name, self.value, _ATTN_OVERRIDES.get(self))   # Stokastisk T4: Flash-family backends are not shipped\n"
         "        path = _ATTN_OVERRIDES.get(self, self.value)\n"),
    ],
    "vllm/v1/attention/backends/mla/prefill/registry.py": [
        (REG_ANCHOR_IMPORT, REG_ANCHOR_IMPORT + "from vllm.t4_backend_policy import raise_if_prohibited_backend\n"),
        ("        path = _MLA_PREFILL_OVERRIDES.get(self, self.value)\n",
         "        raise_if_prohibited_backend(self.name, self.value, _MLA_PREFILL_OVERRIDES.get(self))   # Stokastisk T4: Flash-family backends are not shipped\n"
         "        path = _MLA_PREFILL_OVERRIDES.get(self, self.value)\n"),
    ],
    "vllm/config/attention.py": [
        ('    @field_validator("mla_prefill_backend", mode="before")\n',
         '    @field_validator("backend", "mla_prefill_backend", mode="after")\n'
         '    @classmethod\n'
         '    def validate_backend_not_prohibited(cls, value: Any) -> Any:\n'
         '        """Stokastisk T4 build: reject FlashAttention/FlashInfer/FlashMLA backends when the configuration is built, not deep inside engine start-up."""\n'
         '        from vllm.t4_backend_policy import check_attention_backend\n\n'
         '        check_attention_backend(value)\n'
         '        return value\n\n'
         '    @field_validator("mla_prefill_backend", mode="before")\n'),
    ],
    "vllm/platforms/cuda.py": [
        ('            except ImportError:\n                invalid_reasons_i = ["ImportError"]\n',
         '            except ImportError as e:   # Stokastisk T4: keep the reason (e.g. ProhibitedBackendError) instead of the bare word "ImportError"\n'
         '                invalid_reasons_i = [f"{type(e).__name__}: {e}"]\n'),
        ('            except ImportError:\n                invalid_reasons = ["ImportError"]\n',
         '            except ImportError as e:   # Stokastisk T4: keep the reason\n                invalid_reasons = [f"{type(e).__name__}: {e}"]\n'),
    ],
}


# --------------------------------------------------------------------------------------------------------------------------------------------- analysis helpers
def module_name(path: str) -> str:
    p = path[:-3]
    return (p[:-9] if p.endswith("/__init__") else p).replace("/", ".")


def classify(name: str) -> str:
    if name in NONE_RETURNING:
        return "none"
    if name.startswith(PREDICATE_PREFIXES):
        return "false"
    if name[:1].isupper() and not name.isupper():
        return "class"
    if name == name.lower():
        return "function"
    raise SystemExit(f"cannot classify the name {name!r} imported from a removed module; add it to an explicit table in t4_flash_free.py")


def _handler_swallows_import_error(handlers) -> bool:
    for h in handlers:
        t = "BaseException" if h.type is None else ast.unparse(h.type)
        if any(k in t for k in ("ImportError", "ModuleNotFoundError", "Exception", "BaseException")):
            return True
    return False


def guarded(node, parents) -> bool:
    """True if the import is inside try/except ImportError, `with suppress(ImportError)` or `if TYPE_CHECKING`."""
    child = node
    while child in parents:
        parent = parents[child]
        if isinstance(parent, ast.Try) and child in parent.body and _handler_swallows_import_error(parent.handlers):
            return True
        if isinstance(parent, ast.With) and any("suppress" in ast.unparse(i.context_expr) and "ImportError" in ast.unparse(i.context_expr) for i in parent.items):
            return True
        if isinstance(parent, ast.If) and "TYPE_CHECKING" in ast.unparse(parent.test) and child in parent.body:
            return True
        child = parent
    return False


def stub_source(entries: dict[str, str]) -> str:
    lines = ['# SPDX-License-Identifier: Apache-2.0',
             '"""Stokastisk Tesla T4 (SM75) build: placeholders for the FlashAttention / FlashInfer / FlashMLA modules that are not shipped.',
             '',
             '[GENERATED by scripts/release/t4_flash_free.py - do not edit.] This is NOT an implementation of any Flash-family backend: predicates answer False, capability',
             'probes answer "not supported", and every function raises ProhibitedBackendError. It exists so that unrelated vLLM modules (quantised-MoE oracles, linear-kernel',
             'registries, warm-up helpers ...) that import those names at module level still import, and then take their "backend unavailable" branch.',
             '"""',
             'from vllm.t4_backend_policy import false_predicate, none_function, prohibited_class, prohibited_function', '']
    maker = {"false": "false_predicate", "none": "none_function", "class": "prohibited_class", "function": "prohibited_function"}
    for name in sorted(entries):
        lines.append(f"{name} = {maker[entries[name]]}({name!r})")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------------------------------------------------- transform
def transform(files: dict[str, bytes]) -> tuple[dict[str, bytes], dict]:
    out = dict(files)
    report: dict = {"removed": [], "rewritten": [], "patched": [], "added": [], "stub_entries": {}, "guarded_untouched": [], "dangling_prohibited_imports": {}}

    removed = sorted(p for p in out if p.startswith("vllm/") and is_prohibited_path(p))
    removed_mods = {module_name(p) for p in removed if p.endswith(".py")}
    # Flash-family modules that an EARLIER derivation step already dropped from the wheel but that other modules still import (e.g. `vllm.v1.attention.ops.flashmla`):
    # they become explicit ProhibitedBackendError placeholders too, instead of a bare ModuleNotFoundError at import time.
    present = {module_name(p) for p in out if p.endswith(".py") and p.startswith("vllm/")}
    packages = {m.rsplit(".", 1)[0] for m in present if "." in m}
    for path in sorted(out):
        if path.endswith(".py") and path.startswith("vllm/") and path not in removed:
            try:
                tree = ast.parse(out[path].decode())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                cands = []
                if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    cands = [node.module]            # only the module part: `from P import name` cannot be told apart from `from P import submodule` when P.name is absent
                elif isinstance(node, ast.Import):
                    cands = [a.name for a in node.names]
                for c in cands:
                    if c.startswith("vllm.") and is_prohibited_path(c.replace(".", "/") + ".py") and c not in present and c not in packages and c not in removed_mods:
                        removed_mods.add(c); report["dangling_prohibited_imports"].setdefault(c, []).append(f"{path}:{node.lineno}")
    for p in removed:
        report["removed"].append({"path": p, "sha256": sha(out[p]), "bytes": len(out[p])})
        del out[p]

    entries: dict[str, str] = {}
    for path in sorted(out):
        if not path.endswith(".py") or not path.startswith("vllm/"):
            continue
        src = out[path].decode()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        parents = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
        lines = src.splitlines(keepends=True)
        edits: list[tuple[int, int, str, str]] = []                    # (start line idx, end line idx exclusive, new text, old text)
        used_aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                direct = node.module in removed_mods
                as_attr = [a for a in node.names if f"{node.module}.{a.name}" in removed_mods]
                if not direct and not as_attr:
                    continue
                if guarded(node, parents):
                    report["guarded_untouched"].append(f"{path}:{node.lineno}")
                    continue
                old = "".join(lines[node.lineno - 1:node.end_lineno])
                if direct:
                    for a in node.names:
                        entries[a.name] = classify(a.name)
                    new = re.sub(r"(from\s+)" + re.escape(node.module) + r"(\s+import)", r"\1" + STUB_MODULE + r"\2", old, count=1)
                else:
                    assert len(node.names) == 1, f"{path}:{node.lineno} mixes a removed module with other names; handle explicitly"
                    a = node.names[0]
                    alias = a.asname or a.name
                    new = f"{old[:len(old) - len(old.lstrip())]}from vllm.utils import t4_prohibited_stubs as {alias}\n"
                    used_aliases[alias] = alias
                assert new != old, f"{path}:{node.lineno} could not be rewritten"
                edits.append((node.lineno - 1, node.end_lineno, new, old))
            elif isinstance(node, ast.Import):
                hit = [a for a in node.names if a.name in removed_mods]
                if not hit or guarded(node, parents):
                    if hit:
                        report["guarded_untouched"].append(f"{path}:{node.lineno}")
                    continue
                assert len(node.names) == 1 and node.names[0].asname, f"{path}:{node.lineno}: plain `import {hit[0].name}` without alias is not supported by the transform"
                old = "".join(lines[node.lineno - 1:node.end_lineno])
                new = old.replace(hit[0].name, STUB_MODULE, 1)
                used_aliases[node.names[0].asname] = node.names[0].asname
                edits.append((node.lineno - 1, node.end_lineno, new, old))
        if not edits:
            continue
        for alias in used_aliases:                                      # `alias.attr` accesses become placeholder entries
            for n in ast.walk(tree):
                if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == alias:
                    entries[n.attr] = classify(n.attr)
        for start, end, new, old in sorted(edits, reverse=True):
            lines[start:end] = [new]
            report["rewritten"].append({"file": path, "line": start + 1, "old": old.strip().splitlines()[0][:110], "new": new.strip().splitlines()[0][:110]})
        out[path] = "".join(lines).encode()
        ast.parse(out[path].decode())                                   # the rewrite must still be valid Python

    out[STUB_PATH] = stub_source(entries).encode()
    out[POLICY_PATH] = (POLICY_SOURCE % {"blocked": tuple(__import__("stokastisk.prohibited", fromlist=["x"]).BLOCKED_IMPORT_NAMES)}).encode()
    report["added"] = [{"path": STUB_PATH, "sha256": sha(out[STUB_PATH])}, {"path": POLICY_PATH, "sha256": sha(out[POLICY_PATH])}]
    report["stub_entries"] = dict(sorted(entries.items()))

    for path, subs in EDITS.items():
        assert path in out, f"{path} missing from the vLLM tree"
        before = out[path].decode()
        after = before
        for old, new in subs:
            assert after.count(old) == 1, f"{path}: anchor not found exactly once: {old[:60]!r} (found {after.count(old)})"
            after = after.replace(old, new)
        out[path] = after.encode()
        ast.parse(after)
        report["patched"].append({"path": path, "before_sha256": sha(before.encode()), "after_sha256": sha(after.encode())})

    # --- invariants: nothing unguarded still points at a removed module; no prohibited path remains
    leftovers = []
    for path, data in out.items():
        if path.endswith(".py") and path.startswith("vllm/") and path != STUB_PATH:
            t = ast.parse(data.decode())
            par = {c: n for n in ast.walk(t) for c in ast.iter_child_nodes(n)}
            for n in ast.walk(t):
                mods = []
                if isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                    mods = [n.module] + [f"{n.module}.{a.name}" for a in n.names]
                elif isinstance(n, ast.Import):
                    mods = [a.name for a in n.names]
                if any(m in removed_mods for m in mods) and not guarded(n, par):
                    leftovers.append(f"{path}:{n.lineno}")
    assert not leftovers, f"unguarded imports of removed modules remain: {leftovers}"
    assert not [p for p in out if p.startswith("vllm/") and is_prohibited_path(p)], "a prohibited path survived the transform"
    report["summary"] = {"removed_files": len(report["removed"]), "rewritten_import_sites": len(report["rewritten"]), "patched_files": len(report["patched"]),
                         "stub_entries": len(entries), "guarded_imports_left_alone": len(report["guarded_untouched"])}
    return out, report


def main(argv: list[str]) -> int:
    rep_path = None
    if "--report" in argv:
        i = argv.index("--report"); rep_path = argv[i + 1]; del argv[i:i + 2]
    src, dst = map(pathlib.Path, argv[:2])
    assert not dst.exists(), f"{dst} exists; never overwrite"
    files = {p.relative_to(src).as_posix(): p.read_bytes() for p in src.rglob("*") if p.is_file() and p.relative_to(src).as_posix().startswith("vllm/") and "__pycache__" not in p.parts}
    out, rep = transform(files)
    for p, b in out.items():
        t = dst / p
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_bytes(b)
    if rep_path:
        pathlib.Path(rep_path).write_text(json.dumps(rep, indent=1, sort_keys=True))
    print(json.dumps(rep["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
