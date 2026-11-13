#!/usr/bin/env python3.13
"""Assemble the Stokastisk 0.1.0 BUILD 3 Kaggle input bundle from the CURRENT source tree. Never uploads anything; refuses to overwrite anything.
usage: assemble_build3_release.py [--build-tag 3]

 1. builds the integrated, Flash-free wheel (scripts/release/build_integrated_wheel.py) into dist-rc/integrated-0.1.0-build3/
 2. verifies it independently (scripts/audit/verify_build3_wheel.py) and audits its native objects (scripts/audit/native_audit.py)
 3. writes dist-rc/stokastisk-0.1.0-build3-kaggle-t4-sm75/ : wheel, installer + stage scripts, hash-pinned locks, manifests, docs, notebook 15
 4. builds the reproducible ZIP + .sha256, plus a STANDALONE notebook that also carries the ZIP's SHA256 (the ZIP cannot contain its own hash)
 5. re-verifies from a fresh extraction: SHA256SUMS, the notebook's own discovery code against a simulated /kaggle/input, a pip --target install of the wheel and the strict
    Flash scan of the installed tree (ordinary pip, no virtualenv, no container, no GPU)
Historical artifacts (build 1, build 2, the legacy two-wheel release) are only READ."""
import hashlib, json, pathlib, shutil, subprocess, sys, tempfile, time, zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/kaggle")); sys.path.insert(0, str(ROOT / "sdk")); sys.path.insert(0, str(ROOT / "scripts/release"))
args = sys.argv[1:]; tag = args[args.index("--build-tag") + 1] if "--build-tag" in args else "3"
assert (ROOT / "VERSION").read_text() == "0.1.0\n"
NAME = "stokastisk-0.1.0-build3-kaggle-t4-sm75"
WORK = ROOT / "dist-rc" / f"integrated-0.1.0-build{tag}"; B = ROOT / "dist-rc" / NAME
assert not WORK.exists() and not B.exists(), "never overwrite a built release"
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()  # noqa: E731
run = lambda *a, **k: subprocess.run(a, capture_output=True, text=True, cwd=ROOT, **k)  # noqa: E731
git = lambda *a: run("git", *a).stdout.strip()  # noqa: E731
head, dirty = git("rev-parse", "HEAD"), git("status", "--porcelain", "--untracked-files=no").splitlines()
LEG = ROOT / "dist-rc/stokastisk-0.1.0-kaggle-t4-sm75/wheels"
VLLM_PIN = LEG / "vllm-0.25.1-cp38-abi3-linux_x86_64.whl"
hist = {"vllm-0.25.1-cp38-abi3-linux_x86_64.whl": sha(VLLM_PIN), "stokastisk-0.1.0-py3-none-any.whl": sha(LEG / "stokastisk-0.1.0-py3-none-any.whl"),
        "build1": sha(ROOT / "dist-rc/sdk-candidate-0.1.0-build1/stokastisk-0.1.0-1-cp313-cp313-linux_x86_64.whl"), "build2": sha(ROOT / "dist-rc/integrated-0.1.0-build2/stokastisk-0.1.0-2-cp313-cp313-linux_x86_64.whl")}
assert hist["vllm-0.25.1-cp38-abi3-linux_x86_64.whl"] == "cc3bcafb6516791990ca302073ae153a08465d3df1e460d666e3ec09b4e98f5e", "historical vLLM wheel altered"
assert hist["build1"] == "815f978346cfc6d85b7b47f9b96ce6805b5630ea2e0d3c0070c45f538c11cf6b" and hist["build2"] == "7e4b700929ec7f3489570e93252ef5a49da2d1682c111c5feefb55efead84369", "historical builds altered"

r = run(sys.executable, "scripts/release/build_integrated_wheel.py", str(WORK), "--build-tag", tag, "--report-dir", str(WORK / "reports"))
assert r.returncode == 0, r.stderr[-2000:]
whl = next(WORK.glob("*.whl")); print(r.stdout.strip().splitlines()[-1])
assert whl.name not in {"stokastisk-0.1.0-1-cp313-cp313-linux_x86_64.whl", "stokastisk-0.1.0-2-cp313-cp313-linux_x86_64.whl"} and sha(whl) not in hist.values()
r = run(sys.executable, "scripts/audit/verify_build3_wheel.py", str(whl), str(VLLM_PIN), str(WORK / "static-verification.json"))
assert r.returncode == 0, "independent wheel verification failed:\n" + r.stdout[-2500:]
r = run(sys.executable, "scripts/audit/native_audit.py", str(whl), str(WORK / "native-audit.json")); assert r.returncode == 0, r.stderr[-800:]

# ------------------------------------------------------------------------------------------------------------------------------------------------ bundle
for d in ("wheels", "scripts/kaggle", "requirements/closure", "manifests", "evidence", "notebooks", "docs"):
    (B / d).mkdir(parents=True)
shutil.copy2(whl, B / "wheels" / whl.name)
KS = ["02_install_integrated.py", "_common.py", "_inventory.py", "bundle.py", "prohibited.py", "structured_output_cpu_check.py", "candidate_15_common.py", "candidate_15_static.py",
      "candidate_15_engine.py", "candidate_15_server.py", "candidate_15_summary.py", "candidate_15_integration.py"]
for f in KS:
    shutil.copy2(ROOT / "scripts/kaggle" / f, B / "scripts/kaggle" / f)
for p in ("core", "structured-output", "serving", "models"):
    for ext in (".lock", ".roots"):
        shutil.copy2(ROOT / "requirements" / f"{p}{ext}", B / "requirements" / f"{p}{ext}")
    shutil.copy2(ROOT / "requirements/closure" / f"{p}.closure.json", B / "requirements/closure" / f"{p}.closure.json")
shutil.copy2(ROOT / "requirements/kaggle-baseline-2026-11-13.json", B / "requirements/kaggle-baseline-2026-11-13.json")
(B / "requirements/SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(B / 'requirements').as_posix()}\n" for p in sorted((B / "requirements").rglob("*")) if p.is_file() and p.name != "SHA256SUMS"))
for f in ("LICENSE", "VERSION", "README.md", "INSTALL.md", "CHANGELOG.md"):
    shutil.copy2(ROOT / f, B / f)
for f in ("DEPENDENCY_COMPATIBILITY.md", "KAGGLE_BUILD3_VERIFICATION.md"):
    shutil.copy2(ROOT / "docs" / f, B / "docs" / f)
for f in ("static-verification.json", "native-audit.json"):
    shutil.copy2(WORK / f, B / "evidence" / f)
for f in ("flash-free-transform-report.json", "wheel-flash-scan.json"):
    shutil.copy2(WORK / "reports" / f, B / "evidence" / f)
import make_notebook15_build3 as mk
wsha = sha(whl)
mk.nbformat.write(mk.build(wsha, None), B / "notebooks/15_stokastisk_candidate_sdk_integration.ipynb")

created = time.strftime("%FT%T%z")
manifest = {"name": NAME, "version": "0.1.0", "kind": "integrated-wheel", "created": created, "source_commit": head, "source_tree_dirty_tracked_files": dirty,
            "integrated_wheel": {"filename": whl.name, "sha256": wsha, "size": whl.stat().st_size, "build_tag": tag, "tag": "cp313-cp313-linux_x86_64"},
            "status": "BUILT, INDEPENDENTLY VERIFIED, CPU-TESTED; KAGGLE VERIFICATION PENDING (notebook 15, build 3)", "kaggle_verified": False,
            "locks": {"python": "3.13", "vllm": "0.25.1", "cuda": "13.0", "torch": "2.11.0+cu130", "triton": "3.6.0", "gpu": "2x Tesla T4 SM75", "attention": "TRITON_ATTN", "structured_output_backend": "xgrammar"},
            "vllm_tree": "vllm-0.25.1-cp38-abi3-linux_x86_64.whl (cc3bcafb...) through scripts/release/t4_flash_free.py; native objects byte-identical; 28 Flash-family files removed (evidence/flash-free-transform-report.json)",
            "dependency_profiles": {"installed_by_default": ["core", "structured-output", "serving"], "optional": ["models"], "unsupported": ["observability (opentelemetry exporters conflict with Kaggle protobuf 7.x)"]},
            "supersedes": {"build1": hist["build1"], "build2": hist["build2"], "note": "builds 1 and 2 stay published and untouched; build 2 failed Kaggle notebook 15 (evidence/kaggle/build-2-failed-notebook-15/)"},
            "legacy_release": {"name": "stokastisk-0.1.0-kaggle-t4-sm75", "status": "KAGGLE_T4_VERIFIED", "wheels": {k: hist[k] for k in ("vllm-0.25.1-cp38-abi3-linux_x86_64.whl", "stokastisk-0.1.0-py3-none-any.whl")}}}
(B / "manifests/RELEASE-MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
build = {"timestamp": created, "source_revision": head, "source_tree_dirty_tracked_files": dirty, "build_command": f"python3.13 scripts/release/build_integrated_wheel.py {WORK.name} --build-tag {tag}", "python": sys.version.split()[0],
         "setuptools_build": "python -m build --wheel --no-isolation", "wheel": whl.name, "wheel_size": whl.stat().st_size, "wheel_sha256": wsha, "vllm_wheel_pinned_sha256": hist["vllm-0.25.1-cp38-abi3-linux_x86_64.whl"],
         "nvcc_native_rebuilt": False, "pip": run(sys.executable, "-m", "pip", "--version").stdout.strip(),
         "reproducibility": "members sorted, fixed timestamps (SOURCE_DATE_EPOCH or 2026-11-13), no host paths; two consecutive builds of the same tree gave byte-identical wheels. Nondeterministic inputs: none known; "
                            "the wheel depends on the byte-pinned vLLM wheel, requirements/*.roots and the SDK sources at source_revision."}
(B / "manifests/BUILD-MANIFEST.json").write_text(json.dumps(build, indent=1) + "\n")
(B / "manifests/DEPENDENCY-MANIFEST.json").write_text(json.dumps({"locks": {p: f"requirements/{p}.lock" for p in ("core", "structured-output", "serving", "models")}, "baseline_inventory": "requirements/kaggle-baseline-2026-11-13.json",
    "protected_exact": {"torch": "2.11.0+cu130", "triton": "3.6.0"}, "transformers": ">=5.5.3,<6", "pip_check_policy": "no new conflict tolerated", "xgrammar": "0.2.3 (metadata-compatible; not patched)",
    "closure": {p: len(json.loads((ROOT / f"requirements/closure/{p}.closure.json").read_text())["new_packages"]) for p in ("core", "structured-output", "serving", "models")}}, indent=1) + "\n")
# SHA256SUMS must list every file except itself (the discovery code rejects anything unlisted) - rewrite deterministically
files = sorted(p for p in B.rglob("*") if p.is_file() and p != B / "manifests/SHA256SUMS")
(B / "manifests/SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(B).as_posix()}\n" for p in files))

# ------------------------------------------------------------------------------------------------------------------------------------------------ reproducible ZIP + standalone notebook
zp = WORK / f"{NAME}-kaggle-input.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for p in sorted(B.rglob("*")):
        if p.is_file():
            zi = zipfile.ZipInfo(f"{NAME}/{p.relative_to(B).as_posix()}", (2026, 11, 13, 0, 0, 0)); zi.compress_type = zipfile.ZIP_DEFLATED; zi.external_attr = 0o644 << 16
            z.writestr(zi, p.read_bytes())
zsha = sha(zp)
(WORK / (zp.name + ".sha256")).write_text(f"{zsha}  {zp.name}\n")
standalone = WORK / "15_stokastisk_candidate_sdk_integration.ipynb"
mk.nbformat.write(mk.build(wsha, zsha), standalone)

# ------------------------------------------------------------------------------------------------------------------------------------------------ re-verification from the finished artifacts
import bundle
with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    fake_input = td / "kaggle" / "input" / "stokastisk-build3"; fake_input.mkdir(parents=True); shutil.copy2(zp, fake_input / zp.name)
    prov = bundle.locate_and_verify(input_root=td / "kaggle" / "input", work_dir=td / "work" / "release", expected_wheel_name=whl.name, expected_wheel_sha256=wsha, expected_bundle_zip_sha256=zsha)
    assert prov["wheel_sha256"] == wsha and prov["bundle_zip_sha256"] == zsha
    inst = td / "inst"
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--target", str(inst), "--no-deps", "--no-index", "--disable-pip-version-check", str(whl)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    from stokastisk import prohibited
    scan = prohibited.scan_tree(inst / "vllm"); assert scan["clean"], scan["path_hits"][:5]
    for f in KS:
        assert sha(pathlib.Path(prov["release_root"]) / "scripts/kaggle" / f) == sha(ROOT / "scripts/kaggle" / f), f
print(json.dumps({"wheel": whl.name, "wheel_sha256": wsha, "wheel_size": whl.stat().st_size, "zip": zp.name, "zip_sha256": zsha, "zip_size": zp.stat().st_size, "standalone_notebook": standalone.name,
                  "source_commit": head, "dirty_tracked": dirty, "installed_tree_scan": {k: scan[k] for k in ("files", "py", "pyc", "native", "path_hits", "clean")}}, indent=1))
