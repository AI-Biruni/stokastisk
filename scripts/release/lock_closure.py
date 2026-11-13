#!/usr/bin/env python3.13
"""Compute a deterministic, hash-pinned dependency closure for a Stokastisk feature profile against the PROTECTED Kaggle baseline.

usage: lock_closure.py BASELINE_JSON ROOTS_FILE OUT_PREFIX [--cache DIR] [--offline]

  BASELINE_JSON  {normalised-name: version} of the Kaggle environment BEFORE installation (the installer's 02_packages_before.json).
  ROOTS_FILE     PEP 508 requirements, one per line (`#` comments allowed). Each profile's direct requirements, exactly pinned.
  OUT_PREFIX     writes OUT_PREFIX.lock (pip --require-hashes format, only NEW packages) and OUT_PREFIX.closure.json (full audit trail).

Rules (a violation is reported, never silently "fixed"):
  * A requirement already satisfied by the baseline is kept and its dependencies are NOT walked (the baseline is installed and self-consistent).
  * A requirement that the baseline version does NOT satisfy is a CONFLICT: the baseline is protected, so the tool never proposes an upgrade/downgrade.
  * A requirement whose package is absent is resolved to the highest version that satisfies every specifier collected so far, has a wheel (never an sdist)
    that is compatible with CPython 3.13 / Linux x86_64 / glibc <= 2.35, and whose own dependencies are then walked recursively.
  * The lock lists, per package, the sha256 of EVERY compatible wheel of the chosen version, so `pip --require-hashes --no-deps` accepts exactly those bytes.
Only the standard library and `packaging` are used. Network access: PyPI JSON API (cached on disk; `--offline` uses the cache only)."""
from __future__ import annotations

import json
import pathlib
import sys
import time
import urllib.request

from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import Tag
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

ENV = {"python_version": "3.13", "python_full_version": "3.13.16", "platform_system": "Linux", "platform_machine": "x86_64", "sys_platform": "linux",
       "implementation_name": "cpython", "os_name": "posix", "platform_python_implementation": "CPython", "implementation_version": "3.13.16",
       "platform_release": "", "platform_version": ""}
# tags a CPython 3.13 / Linux x86_64 / glibc 2.35 interpreter accepts (order irrelevant; membership only)
PLATS = ["linux_x86_64", "manylinux1_x86_64", "manylinux2010_x86_64", "manylinux2014_x86_64"] + [f"manylinux_2_{m}_x86_64" for m in range(5, 36)]


def tag_ok(t: Tag) -> bool:
    """True if CPython 3.13 on Linux x86_64 (glibc <= 2.35) can use this tag."""
    if t.platform == "any":
        return t.abi == "none" and (t.interpreter.startswith("py3") or t.interpreter == "py2.py3" or t.interpreter == "cp313")
    if t.platform not in PLATS:
        return False
    if t.abi == "abi3":
        return t.interpreter.startswith("cp3") and int(t.interpreter[3:] or 0) <= 13
    return t.interpreter == "cp313" and t.abi in ("cp313", "none")


PROTECTED_PREFIXES = ("torch", "triton", "nvidia-")


def _get_json(url: str) -> dict:
    last = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except OSError as e:     # URLError, connection reset, timeouts
            last = e
            time.sleep(2 ** attempt)
    raise SystemExit(f"cannot fetch {url}: {last!r}")


def fetch(name: str, cache: pathlib.Path, offline: bool) -> dict:
    p = cache / f"{canonicalize_name(name)}.json"
    if p.is_file():
        return json.loads(p.read_text())
    if offline:
        raise SystemExit(f"offline and {p} is not cached")
    data = _get_json(f"https://pypi.org/pypi/{name}/json")
    p.write_text(json.dumps(data))
    return data


def wheels_ok(files: list[dict]) -> list[dict]:
    out = []
    for f in files:
        if f.get("yanked") or not f["filename"].endswith(".whl"):
            continue
        try:
            _, _, _, tags = parse_wheel_filename(f["filename"])
        except Exception:  # noqa: BLE001
            continue
        if any(tag_ok(t) for t in tags):
            out.append(f)
    return out


def requires_for(meta_info: dict, extras: set[str]) -> list[Requirement]:
    reqs = []
    for line in meta_info.get("requires_dist") or []:
        r = Requirement(line)
        envs = [{**ENV, "extra": e} for e in (extras or {""})] if r.marker else [ENV]
        if r.marker is None or any(r.marker.evaluate(e) for e in envs) or (r.marker and r.marker.evaluate({**ENV, "extra": ""})):
            reqs.append(r)
    return reqs


class Closure:
    def __init__(self, baseline: dict[str, str], cache: pathlib.Path, offline: bool):
        self.base = {canonicalize_name(k): v for k, v in baseline.items()}
        self.cache, self.offline = cache, offline
        self.specs: dict[str, list[tuple[str, SpecifierSet]]] = {}      # name -> [(who, spec)]
        self.chosen: dict[str, dict] = {}                                # name -> {version, files}
        self.satisfied: dict[str, dict] = {}
        self.conflicts: list[dict] = []
        self.unresolvable: list[dict] = []

    def add(self, req: Requirement, who: str, extras_from: set[str] | None = None):
        name = canonicalize_name(req.name)
        if req.marker is not None and not any(req.marker.evaluate({**ENV, "extra": e}) for e in ((extras_from or set()) | {""})):
            return
        self.specs.setdefault(name, []).append((who, req.specifier))
        merged = SpecifierSet(",".join(str(s) for _, s in self.specs[name]))
        if name in self.base:
            v = Version(self.base[name])
            ok = merged.contains(v, prereleases=True)
            if ok:
                self.satisfied[name] = {"baseline": self.base[name], "required_by": sorted({w for w, _ in self.specs[name]}), "spec": str(merged)}
            else:
                self.conflicts.append({"package": name, "baseline_version": self.base[name], "required": str(req), "by": who, "merged_spec": str(merged)})
            return
        if name.startswith(PROTECTED_PREFIXES):
            self.conflicts.append({"package": name, "baseline_version": None, "required": str(req), "by": who, "note": "protected package absent from baseline"}); return
        if name in self.chosen:
            if not merged.contains(Version(self.chosen[name]["version"]), prereleases=True):
                self.conflicts.append({"package": name, "chosen": self.chosen[name]["version"], "required": str(req), "by": who, "merged_spec": str(merged)})
            elif req.extras:
                self.walk(name, self.chosen[name]["version"], set(canonicalize_name(e) for e in req.extras))
            return
        data = fetch(req.name, self.cache, self.offline)
        best = None
        for ver in sorted((v for v in data["releases"] if _v(v) is not None), key=_v, reverse=True):
            vv = _v(ver)
            if vv is None or not merged.contains(vv, prereleases=merged_allows_pre(merged)):
                continue
            files = wheels_ok(data["releases"][ver])
            if files:
                best = (ver, files); break
        if best is None:
            self.unresolvable.append({"package": name, "required": str(req), "by": who, "merged_spec": str(merged)}); return
        ver, files = best
        self.chosen[name] = {"version": ver, "files": [{"filename": f["filename"], "sha256": f["digests"]["sha256"], "size": f["size"]} for f in files], "required_by": [who]}
        self.walk(name, ver, set(canonicalize_name(e) for e in req.extras))

    def walk(self, name: str, ver: str, extras: set[str]):
        meta = self._version_meta(name, ver)
        for r in requires_for(meta, extras):
            self.add(r, f"{name}=={ver}", extras)

    def _version_meta(self, name: str, ver: str) -> dict:
        p = self.cache / f"{name}=={ver}.json"
        if p.is_file():
            return json.loads(p.read_text())["info"]
        if self.offline:
            raise SystemExit(f"offline and {p} is not cached")
        data = _get_json(f"https://pypi.org/pypi/{name}/{ver}/json")
        p.write_text(json.dumps(data))
        return data["info"]


def _v(s: str):
    try:
        return Version(s)
    except InvalidVersion:
        return None


def merged_allows_pre(spec: SpecifierSet) -> bool:
    return any(Version(s.version).is_prerelease for s in spec if s.version.replace(".", "").replace("+", "").isalnum() and _v(s.version))


def parse_roots(text: str) -> list[Requirement]:
    return [Requirement(l.split("#")[0].strip()) for l in text.splitlines() if l.split("#")[0].strip()]


def main(argv: list[str]) -> int:
    offline = "--offline" in argv
    cache = pathlib.Path(argv[argv.index("--cache") + 1]) if "--cache" in argv else pathlib.Path(".lock_cache")
    pos = [a for i, a in enumerate(argv) if not a.startswith("--") and (i == 0 or argv[i - 1] != "--cache")]
    base_json, roots_file, prefix = pos
    cache.mkdir(parents=True, exist_ok=True)
    c = Closure(json.loads(pathlib.Path(base_json).read_text()), cache, offline)
    roots = parse_roots(pathlib.Path(roots_file).read_text())
    for r in roots:          # register every root specifier first, so a transitive requirement can never float past a root pin
        c.specs.setdefault(canonicalize_name(r.name), []).append(("root", r.specifier))
    for r in roots:
        c.add(r, "root")
    lock_lines = ["# GENERATED by scripts/release/lock_closure.py -- DO NOT EDIT. Only packages ABSENT from the Kaggle baseline; every compatible wheel hash of the pinned version.",
                  f"# roots: {', '.join(str(r) for r in roots)}"]
    for name in sorted(c.chosen):
        ch = c.chosen[name]
        lock_lines.append(f"{name}=={ch['version']} \\")
        hs = sorted({f['sha256'] for f in ch["files"]})
        lock_lines += [f"    --hash=sha256:{h}" + (" \\" if i < len(hs) - 1 else "") for i, h in enumerate(hs)]
    pathlib.Path(prefix + ".lock").write_text("\n".join(lock_lines) + "\n")
    rep = {"roots": [str(r) for r in roots], "new_packages": {n: {"version": v["version"], "required_by": sorted(set(v["required_by"])), "files": [f["filename"] for f in v["files"]]} for n, v in sorted(c.chosen.items())},
           "satisfied_by_baseline": dict(sorted(c.satisfied.items())), "conflicts": c.conflicts, "unresolvable": c.unresolvable}
    pathlib.Path(prefix + ".closure.json").write_text(json.dumps(rep, indent=1))
    print(f"new={len(c.chosen)} satisfied_by_baseline={len(c.satisfied)} conflicts={len(c.conflicts)} unresolvable={len(c.unresolvable)}")
    for k in c.conflicts + c.unresolvable:
        print("  PROBLEM:", json.dumps(k))
    return 1 if (c.conflicts or c.unresolvable) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
