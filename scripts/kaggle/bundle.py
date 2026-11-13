"""Locate, verify and extract the Stokastisk release bundle that is attached to a Kaggle notebook (standard library only).

This file is embedded VERBATIM in the first cell of the notebook (the bundle's own scripts cannot be used before the bundle has been found), and is also shipped in the
bundle's ``scripts/kaggle`` directory; ``tests/test_bundle_discovery.py`` asserts that the two are byte-identical.

What failed in build 2: the notebook searched ``/kaggle/input/**/manifests/SHA256SUMS``, found nothing (the attached upload was a ZIP holding only a wheel) and silently downloaded a
DIFFERENT release (build 2) from Hugging Face; every report afterwards described a wheel the user had not supplied. The rules below make that impossible:

1. The attached bundle is found under ``/kaggle/input`` - either as a ZIP file or as an already-extracted directory - and must be a COMPLETE release (it has
   ``manifests/RELEASE-MANIFEST.json`` and ``manifests/SHA256SUMS``). A ZIP that contains only a wheel is rejected with an explicit message.
2. Nothing is downloaded unless ``STOKASTISK_ALLOW_HF_FALLBACK=1`` is set; the fallback is pinned to a fixed release, size and SHA256 and prints loudly what it does.
3. The archive is integrity-tested, unsafe member paths are rejected (absolute paths, ``..``, backslashes, drive letters, symlinks, duplicates, oversize), extraction goes to
   a fresh directory, every file listed in ``SHA256SUMS`` is hashed, no unlisted file may exist, and the wheel's name, size and SHA256 must equal ``RELEASE-MANIFEST.json``
   (and the expected values compiled into the notebook, when given).
4. The result is a provenance record that every later stage embeds in its report.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import stat
import time
import urllib.request
import zipfile

RELEASE_ROOT_NAME = "stokastisk-0.1.0-build3-kaggle-t4-sm75"
MAX_UNCOMPRESSED_BYTES = 4 * 1024 ** 3          # a bundle is ~40 MB; 4 GiB is far beyond any legitimate content and still stops a zip bomb
MAX_MEMBERS = 5000


class BundleError(RuntimeError):
    """The attached input is missing, incomplete, unsafe or does not match the expected release."""


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------------------------------------------ archive safety
def unsafe_member(name: str) -> str | None:
    """Reason a ZIP member name must be refused, or None if it is safe."""
    if not name or name.startswith(("/", "\\")):
        return "absolute path"
    if "\\" in name:
        return "backslash in path"
    if len(name) > 1 and name[1] == ":":
        return "drive letter"
    if "\x00" in name:
        return "NUL byte"
    parts = name.split("/")
    if any(p == ".." for p in parts):
        return "path traversal (..)"
    return None


def inspect_zip(path) -> dict:
    """Integrity-test a ZIP and check every member name; returns {'names', 'total_bytes', 'wheels'}. Raises BundleError."""
    path = pathlib.Path(path)
    try:
        z = zipfile.ZipFile(path)
    except zipfile.BadZipFile as e:
        raise BundleError(f"{path} is not a valid ZIP archive: {e}") from e
    with z:
        bad = z.testzip()
        if bad is not None:
            raise BundleError(f"{path}: CRC error in member {bad!r}")
        infos = z.infolist()
        if len(infos) > MAX_MEMBERS:
            raise BundleError(f"{path}: {len(infos)} members (limit {MAX_MEMBERS})")
        names, seen, total = [], set(), 0
        for i in infos:
            why = unsafe_member(i.filename)
            if why:
                raise BundleError(f"{path}: unsafe member {i.filename!r}: {why}")
            mode = i.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise BundleError(f"{path}: member {i.filename!r} is a symbolic link")
            if i.filename in seen:
                raise BundleError(f"{path}: duplicate member {i.filename!r}")
            seen.add(i.filename)
            total += i.file_size
            names.append(i.filename)
        if total > MAX_UNCOMPRESSED_BYTES:
            raise BundleError(f"{path}: {total} bytes uncompressed (limit {MAX_UNCOMPRESSED_BYTES})")
    files = [n for n in names if not n.endswith("/")]
    return {"names": files, "total_bytes": total, "wheels": [n for n in files if n.endswith(".whl")]}


def is_complete_release(names) -> bool:
    s = set(names)
    return any(n.endswith("manifests/RELEASE-MANIFEST.json") for n in s) and any(n.endswith("manifests/SHA256SUMS") for n in s)


def extract_zip(path, dest) -> pathlib.Path:
    """Safe extraction into a NEW directory; returns the release root (the single top-level directory, or ``dest`` itself)."""
    info = inspect_zip(path)
    if not is_complete_release(info["names"]):
        only = info["wheels"]
        what = f"it contains only {', '.join(only)}" if only and len(info["names"]) <= len(only) + 2 else f"it has no manifests/RELEASE-MANIFEST.json + manifests/SHA256SUMS ({len(info['names'])} files)"
        raise BundleError(f"{path} is NOT a complete Stokastisk release bundle: {what}. Attach {RELEASE_ROOT_NAME}-kaggle-input.zip (the full bundle), not a wheel-only archive.")
    dest = pathlib.Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise BundleError(f"refusing to extract into the non-empty directory {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    root_resolved = dest.resolve()
    with zipfile.ZipFile(path) as z:
        for i in z.infolist():
            target = (dest / i.filename).resolve()
            if root_resolved != target and root_resolved not in target.parents:
                raise BundleError(f"member {i.filename!r} would be written outside {dest}")
        z.extractall(dest)
    tops = sorted({n.split("/", 1)[0] for n in info["names"]})
    root = dest / tops[0] if len(tops) == 1 and (dest / tops[0]).is_dir() else dest
    if not (root / "manifests" / "RELEASE-MANIFEST.json").is_file():
        raise BundleError(f"{path}: manifests/RELEASE-MANIFEST.json is not at the release root {root}")
    return root


# ------------------------------------------------------------------------------------------------------------------------------------------------ manifest checks
def read_sums(root: pathlib.Path) -> dict[str, str]:
    sums: dict[str, str] = {}
    for ln in (root / "manifests" / "SHA256SUMS").read_text().splitlines():
        if not ln.strip():
            continue
        parts = ln.split(None, 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise BundleError(f"malformed line in SHA256SUMS: {ln!r}")
        rel = parts[1].strip().lstrip("*")
        why = unsafe_member(rel)
        if why:
            raise BundleError(f"SHA256SUMS lists an unsafe path {rel!r}: {why}")
        if rel in sums:
            raise BundleError(f"SHA256SUMS lists {rel!r} twice")
        sums[rel] = parts[0].lower()
    return sums


def verify_release(root, *, expected_wheel_name: str, expected_wheel_sha256: str | None = None, expected_sums_sha256: str | None = None, expected_version: str = "0.1.0", expected_build: str = "3") -> dict:
    """Verify an extracted (or pre-extracted) release directory completely. Returns the provenance record. Raises BundleError on any mismatch."""
    root = pathlib.Path(root)
    for need in ("manifests/RELEASE-MANIFEST.json", "manifests/SHA256SUMS"):
        if not (root / need).is_file():
            raise BundleError(f"{root}: {need} is missing - not a complete release bundle")
    manifest = json.loads((root / "manifests" / "RELEASE-MANIFEST.json").read_text())
    if manifest.get("version") != expected_version:
        raise BundleError(f"release version {manifest.get('version')!r} != expected {expected_version!r}")
    w = manifest.get("integrated_wheel") or {}
    if str(w.get("build_tag")) != expected_build:
        raise BundleError(f"wheel build tag {w.get('build_tag')!r} != expected build {expected_build!r} (this notebook verifies build {expected_build} only)")
    if w.get("filename") != expected_wheel_name:
        raise BundleError(f"manifest wheel {w.get('filename')!r} != expected {expected_wheel_name!r}")
    if expected_sums_sha256 and sha256_file(root / "manifests" / "SHA256SUMS") != expected_sums_sha256.lower():
        raise BundleError(f"manifests/SHA256SUMS hash {sha256_file(root / 'manifests' / 'SHA256SUMS')} != the bundle this notebook was generated for ({expected_sums_sha256}): "
                          "the attached input is a different (stale or newer) bundle - re-attach the matching dataset version")
    sums = read_sums(root)
    on_disk = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.relative_to(root).as_posix() != "manifests/SHA256SUMS")
    unlisted = sorted(set(on_disk) - set(sums))
    missing = sorted(set(sums) - set(on_disk))
    if unlisted or missing:
        raise BundleError(f"file set differs from SHA256SUMS: unlisted={unlisted[:5]} missing={missing[:5]}")
    bad = [rel for rel, h in sums.items() if sha256_file(root / rel) != h]
    if bad:
        raise BundleError(f"SHA256 mismatch for {bad[:5]}")
    wheels = sorted(p.name for p in (root / "wheels").glob("*.whl"))
    if wheels != [expected_wheel_name]:
        raise BundleError(f"wheels/ holds {wheels}, expected exactly [{expected_wheel_name!r}]")
    wp = root / "wheels" / expected_wheel_name
    wsha = sha256_file(wp)
    if wsha != w.get("sha256"):
        raise BundleError(f"wheel SHA256 {wsha} != RELEASE-MANIFEST {w.get('sha256')}")
    if wp.stat().st_size != w.get("size"):
        raise BundleError(f"wheel size {wp.stat().st_size} != RELEASE-MANIFEST {w.get('size')}")
    if expected_wheel_sha256 and wsha != expected_wheel_sha256.lower():
        raise BundleError(f"wheel SHA256 {wsha} != the value compiled into this notebook ({expected_wheel_sha256})")
    return {"release_root": str(root), "version": expected_version, "build": expected_build, "wheel": expected_wheel_name, "wheel_sha256": wsha, "wheel_size": wp.stat().st_size,
            "manifest_sha256": sha256_file(root / "manifests" / "RELEASE-MANIFEST.json"), "sha256sums_sha256": sha256_file(root / "manifests" / "SHA256SUMS"),
            "files_verified": len(sums), "source_commit": manifest.get("source_commit"), "created": manifest.get("created")}


# ------------------------------------------------------------------------------------------------------------------------------------------------ discovery
def find_candidates(input_root="/kaggle/input", max_depth: int = 8) -> dict[str, list[pathlib.Path]]:
    """Everything under ``input_root`` that could be the bundle: ``zips`` (any .zip), ``dirs`` (directories holding manifests/RELEASE-MANIFEST.json), ``wheels`` (loose .whl)."""
    base = pathlib.Path(input_root)
    out: dict[str, list[pathlib.Path]] = {"zips": [], "dirs": [], "wheels": []}
    if not base.is_dir():
        return out
    for dirpath, dirnames, filenames in os.walk(base):
        depth = len(pathlib.Path(dirpath).relative_to(base).parts)
        if depth > max_depth:
            dirnames[:] = []
            continue
        for fn in filenames:
            p = pathlib.Path(dirpath) / fn
            if fn.endswith(".zip"):
                out["zips"].append(p)
            elif fn.endswith(".whl"):
                out["wheels"].append(p)
            elif fn == "RELEASE-MANIFEST.json" and p.parent.name == "manifests":
                out["dirs"].append(p.parent.parent)
    for k in out:
        out[k].sort()
    return out


def hf_fallback_enabled() -> bool:
    return os.environ.get("STOKASTISK_ALLOW_HF_FALLBACK", "") == "1"


def download_pinned(url: str, dest, *, expected_sha256: str, expected_size: int) -> pathlib.Path:
    """The ONLY network path: a fixed URL, with the size and SHA256 compiled into the notebook. Used only when STOKASTISK_ALLOW_HF_FALLBACK=1."""
    dest = pathlib.Path(dest)
    print(f"!! STOKASTISK_ALLOW_HF_FALLBACK=1: downloading the PINNED release {url}\n!! expected sha256 {expected_sha256}, {expected_size} bytes. This is NOT the attached input.")
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "stokastisk-notebook/1"}), timeout=120) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    if part.stat().st_size != expected_size or sha256_file(part) != expected_sha256:
        part.unlink(missing_ok=True)
        raise BundleError(f"the downloaded file does not match the pinned size/SHA256 ({url})")
    os.replace(part, dest)
    return dest


def locate_and_verify(*, input_root="/kaggle/input", work_dir="/kaggle/working/stokastisk-release", expected_wheel_name: str, expected_wheel_sha256: str | None = None,
                      expected_bundle_zip_sha256: str | None = None, expected_sums_sha256: str | None = None, fallback: dict | None = None) -> dict:
    """Find the attached bundle, verify everything and return the provenance record (with ``release_root``). Never substitutes a different release silently."""
    c = find_candidates(input_root)
    print(f"attached input under {input_root}: {len(c['zips'])} zip(s), {len(c['dirs'])} extracted release dir(s), {len(c['wheels'])} loose wheel(s)")
    chosen_zip = None
    prov: dict = {"source": None, "input_root": str(input_root), "candidates": {k: [str(x) for x in v] for k, v in c.items()}, "started": time.strftime("%FT%T%z")}
    complete_dirs = [d for d in c["dirs"]]
    complete_zips, rejected = [], []
    for z in c["zips"]:
        try:
            info = inspect_zip(z)
        except BundleError as e:
            rejected.append(f"{z}: {e}")
            continue
        if is_complete_release(info["names"]):
            complete_zips.append(z)
        else:
            rejected.append(f"{z}: not a complete release (wheel-only or unrelated archive: {info['names'][:3]})")
    if len(complete_dirs) + len(complete_zips) > 1:
        raise BundleError(f"more than one release bundle is attached ({[str(x) for x in complete_dirs + complete_zips]}); attach exactly one so the build under test is unambiguous")
    if complete_dirs:
        root = complete_dirs[0]
        prov["source"] = "attached-directory"
    elif complete_zips:
        chosen_zip = complete_zips[0]
        prov["bundle_zip"] = str(chosen_zip)
        prov["bundle_zip_sha256"] = sha256_file(chosen_zip)
        if expected_bundle_zip_sha256 and prov["bundle_zip_sha256"] != expected_bundle_zip_sha256.lower():
            raise BundleError(f"bundle ZIP SHA256 {prov['bundle_zip_sha256']} != expected {expected_bundle_zip_sha256}")
        wd = pathlib.Path(work_dir)
        if wd.exists():
            shutil.rmtree(wd)
        root = extract_zip(chosen_zip, wd)
        prov["source"] = "attached-zip"
    elif hf_fallback_enabled():
        if not fallback:
            raise BundleError("STOKASTISK_ALLOW_HF_FALLBACK=1 but this notebook has no pinned fallback release configured")
        wd = pathlib.Path(work_dir); wd.parent.mkdir(parents=True, exist_ok=True)
        zp = download_pinned(fallback["url"], wd.parent / "fallback-release.zip", expected_sha256=fallback["sha256"], expected_size=fallback["size"])
        prov.update(bundle_zip=str(zp), bundle_zip_sha256=sha256_file(zp), source="HUGGING-FACE-FALLBACK (opt-in)")
        if wd.exists():
            shutil.rmtree(wd)
        root = extract_zip(zp, wd)
    else:
        seen = sorted(str(q.relative_to(input_root)) for q in pathlib.Path(input_root).glob("*/*") if pathlib.Path(input_root).is_dir())[:20] if pathlib.Path(input_root).is_dir() else ["(input root missing)"]
        why = "; ".join(rejected) or f"nothing recognisable is attached; top of {input_root}: {seen}"
        loose = f" Loose wheel(s) found: {[x.name for x in c['wheels']]} - a wheel alone is not a release." if c["wheels"] else ""
        raise BundleError(f"no complete Stokastisk release bundle is attached under {input_root} ({why}).{loose} Attach {RELEASE_ROOT_NAME}-kaggle-input.zip as a notebook input. "
                          "No network fallback is used (set STOKASTISK_ALLOW_HF_FALLBACK=1 to opt in to the pinned Hugging Face copy).")
    prov.update(verify_release(root, expected_wheel_name=expected_wheel_name, expected_wheel_sha256=expected_wheel_sha256, expected_sums_sha256=expected_sums_sha256))
    prov["finished"] = time.strftime("%FT%T%z")
    print(f"RELEASE  : {RELEASE_ROOT_NAME}   source: {prov['source']}")
    print(f"WHEEL    : {prov['wheel']}   sha256 {prov['wheel_sha256']}   {prov['wheel_size']} bytes")
    if prov.get("bundle_zip_sha256"):
        print(f"BUNDLE ZIP sha256 {prov['bundle_zip_sha256']}")
    print(f"VERIFIED : {prov['files_verified']} files against SHA256SUMS; manifest sha256 {prov['manifest_sha256']}")
    return prov
