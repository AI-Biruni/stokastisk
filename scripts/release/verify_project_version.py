#!/usr/bin/env python3.13
"""Verify that every Stokastisk PROJECT-version declaration equals the locked version (VERSION_POLICY.md). Never writes or derives a version.
usage: verify_project_version.py [--wheel WHEEL] [--tag TAG]       exit 0 only if everything inspected says exactly 0.1.0
Checked: VERSION, pyproject.toml (dynamic version read from VERSION), sdk/stokastisk/_version.py, the wheel's METADATA/dist-info name, a git tag name (`v0.1.0`).
Dependency versions (vLLM 0.25.1, CUDA 13.0, torch, Triton, Python) and the wheel BUILD tag are not project versions."""
import pathlib, re, sys, tomllib, zipfile

LOCKED_VERSION = "0.1.0"
ROOT = pathlib.Path(__file__).resolve().parents[2]


def main(argv):
    bad = []

    def chk(name, ok, detail=""):
        print(("PASS " if ok else "FAIL ") + name + ("" if ok else f" -- {detail}"))
        if not ok:
            bad.append(name)

    chk("VERSION file is exactly 0.1.0", (ROOT / "VERSION").read_bytes() == f"{LOCKED_VERSION}\n".encode(), (ROOT / "VERSION").read_bytes())
    py = tomllib.loads((ROOT / "pyproject.toml").read_text())
    chk("pyproject reads the version from VERSION", "version" in py["project"].get("dynamic", []) and py["tool"]["setuptools"]["dynamic"]["version"] == {"file": "VERSION"}, py["project"].get("dynamic"))
    chk("_version.py is 0.1.0", re.search(r'__version__\s*=\s*"([^"]+)"', (ROOT / "sdk/stokastisk/_version.py").read_text()).group(1) == LOCKED_VERSION)
    if "--wheel" in argv:
        z = zipfile.ZipFile(argv[argv.index("--wheel") + 1])
        meta = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA") and ".data/" not in n)
        chk("wheel METADATA says Version: 0.1.0", re.search(r"^Version: (.*)$", z.read(meta).decode(), re.M).group(1) == LOCKED_VERSION)
        chk("wheel dist-info is stokastisk-0.1.0", meta.startswith(f"stokastisk-{LOCKED_VERSION}.dist-info/"), meta)
    if "--tag" in argv:
        t = argv[argv.index("--tag") + 1]
        chk("tag is v0.1.0", t == f"v{LOCKED_VERSION}", t)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
