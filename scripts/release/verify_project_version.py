#!/usr/bin/env python3.13
"""Verify that every Stokastisk PROJECT-version declaration equals the permanently locked version. See VERSION_POLICY.md.

usage: verify_project_version.py [--release-dir DIR] [--zip ZIP] [--wheel WHL] [--latest-json FILE] [--model-card FILE] [--installed] [--tags]
Exit status 0 only if every inspected declaration says exactly LOCKED_VERSION. Dependency versions (vLLM, CUDA, torch, triton, python) are NOT project versions.
This script never writes anything and never derives or increments a version. Changing LOCKED_VERSION requires an explicit instruction from the project owner."""
import argparse, hashlib, io, json, pathlib, re, subprocess, sys, tempfile, zipfile

LOCKED_VERSION = "0.1.0"
RELEASE_NAME = f"stokastisk-{LOCKED_VERSION}-kaggle-t4-sm75"
ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = []


def check(name, ok, detail=""):
    RES.append((name, bool(ok), detail))


def tracked():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True).stdout.decode().split("\0")
    return [pathlib.Path(p) for p in out if p]


# Patterns that DECLARE the project version (never match dependency versions).
RELEASE_NAME_RE = re.compile(r"stokastisk-(\d+\.\d+\.\d+)(?:[-.a-z0-9]*?)(?=-kaggle-t4|-py3|\.tar|\.zip|\b)", re.I)
RELEASE_DIR_RE = re.compile(r"stokastisk-(\d+\.\d+\.\d+)-kaggle-t4-sm75(-rc\d+)?")
HEADLINE_RE = re.compile(r"^#+\s*Stokastisk\s+v?(\d+\.\d+\.\d+)", re.I | re.M)
PIN_RE = re.compile(r"\bstokastisk\s*==\s*(\d+\.\d+\.\d+)", re.I)
TEXT = {".py", ".md", ".toml", ".json", ".yml", ".yaml", ".sh", ".txt", ".ipynb", ".cfg", ".cpp"}


def source_checks():
    v = (ROOT / "VERSION").read_bytes()
    check("VERSION file is exactly '0.1.0\\n'", v == f"{LOCKED_VERSION}\n".encode(), repr(v))
    py = (ROOT / "pyproject.toml").read_text()
    static = re.findall(r'^version\s*=\s*"([^"]+)"', py, re.M)
    check("pyproject.toml has no static version other than the locked one", all(s == LOCKED_VERSION for s in static), static)
    check("pyproject.toml derives the version from VERSION", 'dynamic = ["version"]' in py and 'version = {file = "VERSION"}' in py)
    m = re.search(r'__version__\s*=\s*"([^"]+)"', (ROOT / "sdk/stokastisk/_version.py").read_text())
    check("sdk/stokastisk/_version.py __version__", m and m.group(1) == LOCKED_VERSION, m and m.group(1))
    bad = []
    for p in tracked():
        if p.suffix not in TEXT or not (ROOT / p).is_file() or p.parts[0] in ("tests",) and p.name == "test_version_policy.py":
            continue
        if p.name == "verify_project_version.py":
            continue
        t = (ROOT / p).read_text(errors="ignore")
        for rx in (RELEASE_DIR_RE, HEADLINE_RE, PIN_RE):
            for m in rx.finditer(t):
                if m.group(1) != LOCKED_VERSION or (rx is RELEASE_DIR_RE and m.group(2)):
                    bad.append(f"{p}: {m.group(0)}")
    check("no tracked file declares a different Stokastisk release/headline/pin", not bad, bad[:10])
    nbbad = []
    for p in sorted((ROOT / "notebooks").glob("*.ipynb")):
        for m in re.finditer(r'RELEASE\s*=\s*\\?"([^"\\]+)', p.read_text()):
            if m.group(1) != RELEASE_NAME:
                nbbad.append(f"{p.name}: {m.group(1)}")
    check(f"notebook RELEASE constants are {RELEASE_NAME}", not nbbad, nbbad)


def wheel_checks(path):
    path = pathlib.Path(path); z = zipfile.ZipFile(path)
    meta = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
    m = re.search(r"^Version:\s*(\S+)", z.read(meta).decode(), re.M)
    check(f"wheel METADATA Version ({path.name})", m and m.group(1) == LOCKED_VERSION, m and m.group(1))
    check(f"wheel filename ({path.name})", path.name.startswith(f"stokastisk-{LOCKED_VERSION}-"), path.name)
    vm = [n for n in z.namelist() if n.endswith("stokastisk/_version.py")]
    if vm:
        mm = re.search(r'__version__\s*=\s*"([^"]+)"', z.read(vm[0]).decode())
        check("wheel _version.py", mm and mm.group(1) == LOCKED_VERSION, mm and mm.group(1))


def dir_checks(d):
    d = pathlib.Path(d)
    check("release directory name", d.name == RELEASE_NAME, d.name)
    vf = d / "VERSION"
    check("release VERSION file", vf.is_file() and vf.read_text().strip() == LOCKED_VERSION)
    rm = json.loads((d / "manifests/RELEASE-MANIFEST.json").read_text())
    check("release manifest name/version", rm.get("name") == RELEASE_NAME and rm.get("version") == LOCKED_VERSION, {k: rm.get(k) for k in ("name", "version")})
    for w in sorted((d / "wheels").glob("stokastisk-*.whl")):
        wheel_checks(w)
    inst = (d / "scripts/kaggle/02_install.py").read_text()
    check("installer carries the locked version", f'LOCK = "{LOCKED_VERSION}"' in inst)
    bad = []
    for p in d.rglob("*"):
        if p.is_file() and p.suffix in TEXT:
            for m in RELEASE_DIR_RE.finditer(p.read_text(errors="ignore")):
                if m.group(1) != LOCKED_VERSION or m.group(2):
                    bad.append(f"{p.relative_to(d)}: {m.group(0)}")
    check("no file in the release names another Stokastisk release", not bad, bad[:10])


def zip_checks(zp):
    zp = pathlib.Path(zp); z = zipfile.ZipFile(zp)
    check("zip filename", zp.name == f"{RELEASE_NAME}-kaggle-input.zip", zp.name)
    names = z.namelist()
    check("zip has a single top-level directory with the release name", all(n.startswith(RELEASE_NAME + "/") for n in names))
    for n in names:
        if re.search(r"/wheels/stokastisk-[^/]+\.whl$", n):
            with tempfile.TemporaryDirectory() as td:
                f = pathlib.Path(td) / pathlib.Path(n).name; f.write_bytes(z.read(n)); wheel_checks(f)
    mf = f"{RELEASE_NAME}/manifests/RELEASE-MANIFEST.json"
    if mf in names:
        rm = json.loads(z.read(mf)); check("zip release manifest version", rm.get("version") == LOCKED_VERSION and rm.get("name") == RELEASE_NAME, rm.get("version"))


def latest_checks(p):
    j = json.loads(pathlib.Path(p).read_text())
    check("latest.json project/version", j.get("project") == "stokastisk" and j.get("version") == LOCKED_VERSION, {k: j.get(k) for k in ("project", "version")})
    check("latest.json release", j.get("release") == RELEASE_NAME, j.get("release"))
    s = json.dumps(j)
    check("latest.json mentions no other Stokastisk release", all(m.group(1) == LOCKED_VERSION and not m.group(2) for m in RELEASE_DIR_RE.finditer(s)))


def card_checks(p):
    t = pathlib.Path(p).read_text()
    hs = HEADLINE_RE.findall(t)
    check("model card headline version", hs and all(h == LOCKED_VERSION for h in hs), hs)
    check("model card release name", RELEASE_NAME in t)
    check("model card names no other Stokastisk release", all(m.group(1) == LOCKED_VERSION and not m.group(2) for m in RELEASE_DIR_RE.finditer(t)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--release-dir"); ap.add_argument("--zip"); ap.add_argument("--wheel"); ap.add_argument("--latest-json")
    ap.add_argument("--model-card"); ap.add_argument("--installed", action="store_true"); ap.add_argument("--tags", action="store_true")
    a = ap.parse_args()
    source_checks()
    if a.release_dir: dir_checks(a.release_dir)
    if a.zip: zip_checks(a.zip)
    if a.wheel: wheel_checks(a.wheel)
    if a.latest_json: latest_checks(a.latest_json)
    if a.model_card: card_checks(a.model_card)
    if a.installed:
        import importlib.metadata as md
        try: v = md.version("stokastisk")
        except md.PackageNotFoundError: v = None
        check("installed stokastisk distribution", v == LOCKED_VERSION, v)
    if a.tags:
        tags = subprocess.run(["git", "tag", "--list"], cwd=ROOT, capture_output=True, text=True).stdout.split()
        check("only tag v" + LOCKED_VERSION + " may exist as a project tag", all(t == f"v{LOCKED_VERSION}" for t in tags if re.fullmatch(r"v?\d+\.\d+.*", t)), tags)
    import os
    ref = os.environ.get("GITHUB_REF_NAME", "")
    if os.environ.get("GITHUB_REF_TYPE") == "tag":
        check("CI tag is v" + LOCKED_VERSION, ref == f"v{LOCKED_VERSION}", ref)
    for n, ok, d in RES:
        print(("PASS " if ok else "FAIL ") + n + ("" if ok or d == "" else f"  -> {d}"))
    bad = [n for n, ok, _ in RES if not ok]
    print(f"\n{len(RES) - len(bad)}/{len(RES)} checks passed; locked project version = {LOCKED_VERSION}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
