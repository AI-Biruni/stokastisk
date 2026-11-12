#!/usr/bin/env python3.13
"""Build the Stokastisk 0.1.0 *integrated SDK candidate* wheel (one installable wheel = SDK + bundled vLLM 0.25.1 T4/Triton tree + CUDA objects).
usage: build_sdk_candidate_wheel.py OUT_DIR [--build-tag 1]

 * Project version stays exactly 0.1.0 (read from VERSION; refuses anything else). The PEP 427 build tag (default 1) is what distinguishes this file from
   the historical, Kaggle-verified `stokastisk-0.1.0-py3-none-any.whl`; neither that wheel nor the vLLM wheel is touched (both are only READ and hash-checked).
 * Result: stokastisk-0.1.0-1-cp313-cp313-linux_x86_64.whl
 * stokastisk/** comes from a fresh setuptools build of sdk/ (it therefore contains compat.py, serving.py, LLM.chat ...).
 * vllm/** is copied byte-for-byte from the byte-pinned Kaggle-verified vLLM wheel (cc3bcafb...). No file of it is edited.
 * vLLM resolves its own version through importlib.metadata.version("vllm") (vllm/platforms/__init__.py, vllm/triton_utils/importing.py), so a wheel that
   shipped the vllm/ tree without a `vllm` distribution would break platform/Triton detection. The wheel therefore also installs `vllm-0.25.1.dist-info/`
   (via the wheel's .data/platlib scheme path) so the interpreter sees vllm==0.25.1 as installed.
 * Dependencies: vLLM's own Requires-Dist (with the Kaggle-resolved core pins) + torch==2.11.0+cu130 + triton==3.6.0. `vllm==0.25.1` is NOT a dependency.
Nothing here executes a GPU, vLLM or Kaggle."""
import base64, csv, hashlib, io, pathlib, re, subprocess, sys, tempfile, zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
LOCKED = "0.1.0"
assert (ROOT / "VERSION").read_text() == LOCKED + "\n", "VERSION is not the locked project version"
args = sys.argv[1:]
tag = "1"
if "--build-tag" in args:
    i = args.index("--build-tag"); tag = args[i + 1]; del args[i:i + 2]
assert tag.isdigit() and args, "usage: build_sdk_candidate_wheel.py OUT_DIR [--build-tag N]"
OUT = pathlib.Path(args[0]); OUT.mkdir(parents=True, exist_ok=True)
HIST = ROOT / "dist-rc/stokastisk-0.1.0-kaggle-t4-sm75/wheels"
VLLM_WHL = HIST / "vllm-0.25.1-cp38-abi3-linux_x86_64.whl"
PIN = "cc3bcafb6516791990ca302073ae153a08465d3df1e460d666e3ec09b4e98f5e"
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
assert sha(VLLM_WHL.read_bytes()) == PIN, "vLLM wheel is not the byte-pinned Kaggle-verified wheel"
target = OUT / f"stokastisk-{LOCKED}-{tag}-cp313-cp313-linux_x86_64.whl"
assert not target.exists(), f"{target} exists (never overwrite a built artifact)"
assert target.name != "stokastisk-0.1.0-py3-none-any.whl"

# core pins resolved on Kaggle (kept in the release dependency manifest)
import json
dm = json.loads((ROOT / "dist-rc/stokastisk-0.1.0-kaggle-t4-sm75/manifests/DEPENDENCY-MANIFEST.json").read_text())
core = {re.split(r"[=<>!~]", p)[0].lower().replace("_", "-"): p for p in dm["core_pins"]}
EXTRA_REQ = ["torch==2.11.0+cu130", "triton==3.6.0"]

with tempfile.TemporaryDirectory() as td:
    r = subprocess.run([sys.executable, "-m", "build", "--wheel", "--no-isolation", "-o", td, str(ROOT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    sdkw = zipfile.ZipFile(next(pathlib.Path(td).glob("stokastisk-0.1.0-*.whl")))
    vz = zipfile.ZipFile(VLLM_WHL)
    DI = f"stokastisk-{LOCKED}.dist-info"; VDI = "vllm-0.25.1.dist-info"
    files: dict[str, bytes] = {}
    for n in sdkw.namelist():
        if n.startswith("stokastisk/") and not n.endswith("/"): files[n] = sdkw.read(n)
    assert {"stokastisk/compat.py", "stokastisk/serving.py", "stokastisk/engine.py"} <= set(files)
    assert b"def chat" in files["stokastisk/engine.py"] and b"check_dtype" in files["stokastisk/engine.py"]
    for n in vz.namelist():
        if n.startswith("vllm/") and not n.endswith("/"): files[n] = vz.read(n)
    files[f"{DI}/licenses/LICENSE"] = sdkw.read(f"{DI}/licenses/LICENSE")

    # --- vllm distribution metadata, installed next to the package through the wheel .data/platlib scheme path
    data = f"stokastisk-{LOCKED}.data/platlib/"
    for n in ("METADATA", "entry_points.txt", "top_level.txt", "licenses/LICENSE"):
        files[data + f"{VDI}/{n}"] = vz.read(f"{VDI}/{n}")

    # --- stokastisk METADATA: setuptools body, dependency block replaced
    meta = sdkw.read(f"{DI}/METADATA").decode()
    head, body = meta.split("\n\n", 1)
    vm = zipfile.ZipFile(VLLM_WHL).read(f"{VDI}/METADATA").decode().split("\n\n", 1)[0]
    reqs, extras = [], []
    for line in vm.splitlines():
        if line.startswith("Provides-Extra:"): extras.append(line)
        elif line.startswith("Requires-Dist:"):
            spec = line[len("Requires-Dist:"):].strip()
            name = re.split(r"[ ;=<>!~\[(]", spec)[0].lower().replace("_", "-")
            if "extra ==" not in spec and name in core: spec = core[name]
            if name == "transformers": spec = "transformers>=5.5.3,<6"
            reqs.append("Requires-Dist: " + spec)
    reqs += ["Requires-Dist: " + r for r in EXTRA_REQ]
    head = "\n".join(l for l in head.splitlines() if not l.startswith(("Requires-Dist:", "Dynamic:")))
    head = head.replace("Summary: ", "Summary: ", 1)
    pcs = "\n".join(reqs + extras)
    files[f"{DI}/METADATA"] = (head + "\n" + pcs + "\n\n" + body).encode()
    files[f"{DI}/top_level.txt"] = b"stokastisk\nvllm\n"
    ep = vz.read(f"{VDI}/entry_points.txt")
    files[f"{DI}/entry_points.txt"] = ep
    files[f"{DI}/WHEEL"] = (f"Wheel-Version: 1.0\nGenerator: stokastisk-build_sdk_candidate_wheel\nRoot-Is-Purelib: false\n"
                            f"Tag: cp313-cp313-linux_x86_64\nBuild: {tag}\n").encode()
    rec = io.StringIO(); w = csv.writer(rec, lineterminator="\n")
    for n in sorted(files):
        w.writerow([n, "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(files[n]).digest()).rstrip(b"=").decode(), len(files[n])])
    w.writerow([f"{DI}/RECORD", "", ""])
    files[f"{DI}/RECORD"] = rec.getvalue().encode()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zo:
        for n in sorted(files):
            zi = zipfile.ZipInfo(n, (2026, 11, 12, 0, 0, 0)); zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            zo.writestr(zi, files[n])
print(target, sha(target.read_bytes()), target.stat().st_size)
