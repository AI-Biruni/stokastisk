#!/usr/bin/env python3.13
"""Build the Stokastisk 0.1.0 *integrated, Flash-free* wheel (build tag 3 by default): SDK + bundled vLLM 0.25.1 tree for Tesla T4 / SM75 / TRITON_ATTN.
usage: build_integrated_wheel.py OUT_DIR [--build-tag 3] [--report-dir DIR]

 * Project version stays exactly 0.1.0 (VERSION file). The PEP 427 build tag distinguishes this wheel from build 1 (815f9783...) and build 2 (7e4b7009...); both stay untouched.
 * `stokastisk/**` comes from a fresh setuptools build of sdk/.
 * `vllm/**` comes from the byte-pinned vLLM wheel (cc3bcafb...) THROUGH scripts/release/t4_flash_free.py: every FlashAttention/FlashInfer/FlashMLA module is removed, imports of
   them are redirected to one explicit placeholder module, the backend registries and AttentionConfig refuse them, and an import guard is installed. The transform's report is
   written next to the wheel. No `.so` is touched.
 * Dependencies (METADATA): vLLM's own requirements with the exact pins of requirements/{core,structured-output,serving}.roots, torch==2.11.0+cu130, triton==3.6.0; the
   `models` extra carries requirements/models.roots. xgrammar is 0.2.3 (metadata-compatible with transformers 5.x and apache-tvm-ffi 0.1.9); nothing is patched.
 * Reproducible: sorted members, fixed timestamps (SOURCE_DATE_EPOCH if set, else 2026-11-13), no build-host paths. Remaining nondeterminism is recorded in the build manifest
   (the SDK sub-build is done by setuptools; only its Python sources are used, so its timestamps do not enter the wheel)."""
import base64, csv, hashlib, io, json, os, pathlib, re, subprocess, sys, tempfile, time, zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/release")); sys.path.insert(0, str(ROOT / "sdk"))
import t4_flash_free  # noqa: E402
from stokastisk import prohibited  # noqa: E402

LOCKED = "0.1.0"
assert (ROOT / "VERSION").read_text() == LOCKED + "\n", "VERSION is not the locked project version"
args = sys.argv[1:]
tag = "3"; report_dir = None
if "--build-tag" in args:
    i = args.index("--build-tag"); tag = args[i + 1]; del args[i:i + 2]
if "--report-dir" in args:
    i = args.index("--report-dir"); report_dir = pathlib.Path(args[i + 1]); del args[i:i + 2]
assert tag.isdigit() and int(tag) >= 3 and len(args) == 1, "usage: build_integrated_wheel.py OUT_DIR [--build-tag N>=3] [--report-dir DIR]"
OUT = pathlib.Path(args[0]); OUT.mkdir(parents=True, exist_ok=True)
VLLM_WHL = ROOT / "inputs/vllm-0.25.1-cp38-abi3-linux_x86_64.whl"
PIN = "cc3bcafb6516791990ca302073ae153a08465d3df1e460d666e3ec09b4e98f5e"
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
assert sha(VLLM_WHL.read_bytes()) == PIN, "vLLM wheel is not the byte-pinned Kaggle-verified wheel"
target = OUT / f"stokastisk-{LOCKED}-{tag}-cp313-cp313-linux_x86_64.whl"
assert not target.exists(), f"{target} exists (never overwrite a built artifact)"
EPOCH = int(os.environ.get("SOURCE_DATE_EPOCH", "1794528000"))     # 2026-11-13T00:00:00Z
DATE = time.gmtime(EPOCH)[:6]


def roots(name):
    out = []
    for ln in (ROOT / "requirements" / f"{name}.roots").read_text().splitlines():
        ln = ln.split("#")[0].strip()
        if ln:
            out.append(ln)
    return out


def canon(s):
    return re.sub(r"[-_.]+", "-", s).lower()


core = {canon(re.split(r"[=<>!~\[]", p)[0]): p for p in roots("core") + roots("structured-output") + roots("serving")}


def locked(profile):
    """Every package of a hash-pinned lock as an exact `name==version` requirement (the lock's full closure, not only its roots)."""
    return [f"{m.group(1)}=={m.group(2)}" for m in re.finditer(r"^([A-Za-z0-9_.\-]+)==(\S+) \\$", (ROOT / "requirements" / f"{profile}.lock").read_text(), re.M)]


# Plain `pip install` runs the resolver. If transitive new packages (rich-toolkit, supervisor ...) were left floating, a newer release could make pip UPGRADE a protected Kaggle
# package to satisfy it. Pinning the whole lock closure exactly makes the set pip downloads deterministic; everything else is already satisfied by the Kaggle image and kept.
hard_extra = sorted({r for prof in ("core", "structured-output", "serving") for r in locked(prof)}, key=str.lower)
assert {canon(r.split("==")[0]) for r in core} <= {canon(r.split("==")[0]) for r in hard_extra}, "a lock root is missing from the pinned closure"
EXTRA_REQ = ["torch==2.11.0+cu130", "triton==3.6.0"]

with tempfile.TemporaryDirectory() as td:
    r = subprocess.run([sys.executable, "-m", "build", "--wheel", "--no-isolation", "-o", td, str(ROOT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    sdkw = zipfile.ZipFile(next(pathlib.Path(td).glob("stokastisk-0.1.0-*.whl")))
    vz = zipfile.ZipFile(VLLM_WHL)
    DI = f"stokastisk-{LOCKED}.dist-info"; VDI = "vllm-0.25.1.dist-info"
    files: dict[str, bytes] = {}
    for n in sdkw.namelist():
        if n.startswith("stokastisk/") and not n.endswith("/") and "__pycache__" not in n:
            files[n] = sdkw.read(n)
    need = {"stokastisk/compat.py", "stokastisk/serving.py", "stokastisk/engine.py", "stokastisk/prohibited.py", "stokastisk/dependencies.py", "stokastisk/structured.py"}
    assert need <= set(files), need - set(files)
    # the shipped policy module is the repository's, byte for byte
    assert files["stokastisk/prohibited.py"] == (ROOT / "sdk/stokastisk/prohibited.py").read_bytes()
    vtree = {n: vz.read(n) for n in vz.namelist() if n.startswith("vllm/") and not n.endswith("/") and "__pycache__" not in n}
    vtree, treport = t4_flash_free.transform(vtree)
    files.update(vtree)
    files[f"{DI}/licenses/LICENSE"] = sdkw.read(f"{DI}/licenses/LICENSE")
    data = f"stokastisk-{LOCKED}.data/platlib/"
    for n in ("METADATA", "entry_points.txt", "top_level.txt", "licenses/LICENSE"):
        files[data + f"{VDI}/{n}"] = vz.read(f"{VDI}/{n}")

    meta = sdkw.read(f"{DI}/METADATA").decode()
    head, body = meta.split("\n\n", 1)
    vm = vz.read(f"{VDI}/METADATA").decode().split("\n\n", 1)[0]
    reqs, models_extra = [], []
    for line in vm.splitlines():
        if line.startswith("Requires-Dist:"):
            spec = line[len("Requires-Dist:"):].strip()
            name = canon(re.split(r"[ ;=<>!~\[(]", spec)[0])
            if "extra ==" in spec:
                continue                                         # vLLM's own extras are replaced by the explicit profiles below
            if name in core:
                continue                                         # replaced by the exact lock root below
            if name == "transformers":
                spec = "transformers>=5.5.3,<6"
            reqs.append("Requires-Dist: " + spec)
    reqs += ["Requires-Dist: " + r for r in hard_extra] + ["Requires-Dist: " + r for r in EXTRA_REQ]
    for r in roots("models"):
        models_extra.append(f'Requires-Dist: {r} ; extra == "models"')
    head = "\n".join(l for l in head.splitlines() if not l.startswith(("Requires-Dist:", "Dynamic:", "Provides-Extra:")))
    files[f"{DI}/METADATA"] = (head + "\n" + "\n".join(reqs + ["Provides-Extra: models"] + models_extra) + "\n\n" + body).encode()
    files[f"{DI}/top_level.txt"] = b"stokastisk\nvllm\n"
    files[f"{DI}/entry_points.txt"] = vz.read(f"{VDI}/entry_points.txt")
    files[f"{DI}/WHEEL"] = (f"Wheel-Version: 1.0\nGenerator: stokastisk-build_integrated_wheel\nRoot-Is-Purelib: false\nTag: cp313-cp313-linux_x86_64\nBuild: {tag}\n").encode()

    # ---- the finished file set must satisfy the Flash policy before it is ever written
    scan = prohibited.scan_members(list(files), files.__getitem__)
    assert scan["clean"] and not scan["pycache_dirs"], {k: scan[k] for k in ("path_hits", "native_hits", "binding_hits", "pycache_dirs")}

    rec = io.StringIO(); w = csv.writer(rec, lineterminator="\n")
    for n in sorted(files):
        w.writerow([n, "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(files[n]).digest()).rstrip(b"=").decode(), len(files[n])])
    w.writerow([f"{DI}/RECORD", "", ""])
    files[f"{DI}/RECORD"] = rec.getvalue().encode()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zo:
        for n in sorted(files):
            zi = zipfile.ZipInfo(n, DATE); zi.compress_type = zipfile.ZIP_DEFLATED; zi.external_attr = 0o644 << 16
            zo.writestr(zi, files[n])
wsha = sha(target.read_bytes())
if report_dir:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "flash-free-transform-report.json").write_text(json.dumps(treport, indent=1, sort_keys=True))
    (report_dir / "wheel-flash-scan.json").write_text(json.dumps({k: v for k, v in scan.items() if k != "references"} | {"reference_files": sorted(scan["references"])}, indent=1, sort_keys=True))
print(json.dumps({"wheel": target.name, "sha256": wsha, "size": target.stat().st_size, "members": len(files), "transform": treport["summary"]}))
