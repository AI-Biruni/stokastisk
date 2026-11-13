"""The ONE package-inventory implementation used by the installer, the static stage and the summary (standard library only).

Why this exists: in the failed build-2 run the installer snapshot keyed packages with PEP 503 normalisation (``re.sub(r"[-_.]+", "-", name).lower()``) but the static stage's
"live" comparison used ``name.lower().replace("_", "-")``, which leaves dots alone. Eight distributions whose names contain a dot (``jaraco.classes``, ``lazr.uri``,
``ruamel.yaml``, ``path.py`` ...) therefore looked "removed" although nothing had been touched. Two different normalisers on one comparison is the bug; this module
makes it impossible by providing a single ``canon()`` and a single ``snapshot()``.

Method: ``python -m pip list --format=json`` (what pip itself reports as installed; one entry per project, the first on ``sys.path`` wins). The importlib view
(``importlib.metadata``) is kept ONLY as a diagnostic: ``duplicates`` lists every project that has more than one ``*.dist-info`` visible (Kaggle's image has several, e.g.
``Mako 1.3.2.dev0`` and ``1.4.3``), so a version difference caused by a shadowed duplicate can be recognised instead of being mistaken for a change."""
from __future__ import annotations

import importlib.metadata as md
import json
import re
import subprocess
import sys
from typing import Any

_CANON = re.compile(r"[-_.]+")


def canon(name: str) -> str:
    """PEP 503 normalised project name."""
    return _CANON.sub("-", name).lower()


def pip_list() -> dict[str, str]:
    r = subprocess.run([sys.executable, "-m", "pip", "list", "--format=json", "--disable-pip-version-check"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"pip list failed ({r.returncode}): {r.stderr[-500:]}")
    out: dict[str, str] = {}
    for row in json.loads(r.stdout):
        out.setdefault(canon(row["name"]), row["version"])      # pip lists each project once; setdefault keeps the first if that ever changes
    return out


def duplicates() -> dict[str, list[dict[str, str]]]:
    seen: dict[str, list[dict[str, str]]] = {}
    for d in md.distributions():
        n = d.metadata["Name"]
        if not n:
            continue
        seen.setdefault(canon(n), []).append({"version": d.version, "location": str(getattr(d, "_path", "") or "")})
    return {k: sorted(v, key=lambda x: (x["version"], x["location"])) for k, v in seen.items() if len({x["location"] for x in v}) > 1}


def snapshot() -> dict[str, Any]:
    pkgs = pip_list()
    return {"method": "python -m pip list --format=json", "python": sys.version.split()[0], "count": len(pkgs), "packages": dict(sorted(pkgs.items())), "duplicates": duplicates()}


def diff(before: dict[str, str], after: dict[str, str], *, expected_new=(), own=("stokastisk", "vllm")) -> dict[str, Any]:
    """Compare two ``packages`` maps (both canonical). ``own`` distributions may change; ``expected_new`` may appear. Everything else must be identical."""
    b = {canon(k): v for k, v in before.items()}
    a = {canon(k): v for k, v in after.items()}
    own_c = {canon(x) for x in own}
    exp = {canon(x) for x in expected_new}
    changed = {n: [b[n], a[n]] for n in b if n in a and a[n] != b[n] and n not in own_c}
    removed = sorted(n for n in b if n not in a and n not in own_c)
    added = sorted(set(a) - set(b))
    unexpected = sorted(set(added) - exp - own_c)
    missing_expected = sorted(exp - set(a))
    return {"changed": changed, "removed": removed, "added": added, "unexpected_added": unexpected, "expected_but_absent": missing_expected,
            "ok": not changed and not removed and not unexpected and not missing_expected}


def same_inventory(a: dict[str, str], b: dict[str, str]) -> dict[str, Any]:
    """Diff of two snapshots with no allowed changes at all (used for live-vs-after comparisons)."""
    return diff(a, b, expected_new=(), own=())
