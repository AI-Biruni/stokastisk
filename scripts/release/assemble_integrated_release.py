#!/usr/bin/env python3.13
"""Assemble the canonical integrated release `stokastisk-0.1.0-integrated-kaggle-t4-sm75` from the CURRENT source tree. Never uploads anything.
usage: assemble_integrated_release.py [--build-tag 2]
 1. builds the integrated wheel (scripts/release/build_sdk_candidate_wheel.py) into dist-rc/integrated-0.1.0-buildN/
 2. statically verifies it (scripts/audit/verify_sdk_candidate.py) and audits its native objects (scripts/audit/native_audit.py)
 3. writes dist-rc/stokastisk-0.1.0-integrated-kaggle-t4-sm75/ (wheel, installer + stage scripts, hashed pins, notebook 15, docs, manifests, SHA256SUMS)
 4. builds the reproducible Kaggle/Hub ZIP + .sha256 into dist-rc/integrated-0.1.0-buildN/
Historical artifacts (legacy wheels, ZIP, build-tag-1 candidate) are only READ. Refuses to overwrite anything."""
import hashlib, json, pathlib, shutil, subprocess, sys, time, zipfile
ROOT = pathlib.Path(__file__).resolve().parents[2]
args = sys.argv[1:]; tag = args[args.index("--build-tag") + 1] if "--build-tag" in args else "2"
assert (ROOT / "VERSION").read_text() == "0.1.0\n"
NAME = "stokastisk-0.1.0-integrated-kaggle-t4-sm75"; WORK = ROOT / "dist-rc" / f"integrated-0.1.0-build{tag}"; B = ROOT / "dist-rc" / NAME
assert not WORK.exists() and not B.exists(), "never overwrite a built release"
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()  # noqa: E731
run = lambda *a: subprocess.run(a, capture_output=True, text=True, cwd=ROOT)  # noqa: E731
git = lambda *a: run("git", *a).stdout.strip()  # noqa: E731
head, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--untracked-files=no"))
LEG = ROOT / "dist-rc/stokastisk-0.1.0-kaggle-t4-sm75/wheels"
hist = {n: sha(LEG / n) for n in ("stokastisk-0.1.0-py3-none-any.whl", "vllm-0.25.1-cp38-abi3-linux_x86_64.whl")}
assert hist == {"stokastisk-0.1.0-py3-none-any.whl": "29edaa550f49a1cd343a8f043ddd2b5c98c9154ab1da0172fc3fc8725aa3ee49",
                "vllm-0.25.1-cp38-abi3-linux_x86_64.whl": "cc3bcafb6516791990ca302073ae153a08465d3df1e460d666e3ec09b4e98f5e"}, "historical wheels altered"
cmd = [sys.executable, "scripts/release/build_sdk_candidate_wheel.py", str(WORK), "--build-tag", tag]
r = run(*cmd); assert r.returncode == 0, r.stderr[-1500:]
whl = next(WORK.glob("*.whl")); print(r.stdout.strip().splitlines()[-1])
r = run(sys.executable, "scripts/audit/verify_sdk_candidate.py", str(whl), str(LEG / "vllm-0.25.1-cp38-abi3-linux_x86_64.whl"), str(LEG / "stokastisk-0.1.0-py3-none-any.whl"), str(WORK / "static-verification.json"))
assert r.returncode == 0, "static verification failed:\n" + r.stdout[-1500:]
r = run(sys.executable, "scripts/audit/native_audit.py", str(whl), str(WORK / "native-audit.json")); assert r.returncode == 0, r.stderr[-800:]
# ---- bundle
for d in ("wheels", "scripts/kaggle", "requirements", "manifests", "evidence", "notebooks", "docs", "LICENSES"): (B / d).mkdir(parents=True)
shutil.copy2(whl, B / "wheels" / whl.name)
for f in ("_common.py", "02_install_integrated.py", "candidate_15_integration.py"): shutil.copy2(ROOT / "scripts/kaggle" / f, B / "scripts/kaggle" / f)
for f in ("kaggle-t4-core.txt", "kaggle-t4-core.hashes.txt", "kaggle-t4-constraints.txt", "kaggle-t4-prohibited.txt"): shutil.copy2(ROOT / "requirements" / f, B / "requirements" / f)
shutil.copy2(ROOT / "notebooks/kaggle-t4-sm75/integrated/15_stokastisk_candidate_sdk_integration.ipynb", B / "notebooks")
for f in ("INSTALL.md", "RELEASE.md", "LICENSE", "VERSION", "VERSION_POLICY.md", "CHANGELOG.md"): shutil.copy2(ROOT / f, B / f)
shutil.copy2(ROOT / "README.md", B / "README.md")
for f in ("INTEGRATED_RELEASE.md", "DEPENDENCY_COMPATIBILITY.md", "KAGGLE_T4_VERIFICATION.md", "SOURCE_DIFF_VS_KAGGLE_EXECUTED.md", "VLLM_T4_COVERAGE.md", "ARCHITECTURE.md", "DEPENDENCIES.md"):
    if (ROOT / "docs" / f).is_file(): shutil.copy2(ROOT / "docs" / f, B / "docs" / f)
shutil.copy2(WORK / "static-verification.json", B / "evidence/static-verification.json"); shutil.copy2(WORK / "native-audit.json", B / "evidence/native-audit.json")
dm = json.loads((ROOT / "dist-rc/stokastisk-0.1.0-kaggle-t4-sm75/manifests/DEPENDENCY-MANIFEST.json").read_text())
manifest = {"name": NAME, "version": "0.1.0", "kind": "integrated-wheel", "created": time.strftime("%FT%T%z"),
            "integrated_wheel": {"filename": whl.name, "sha256": sha(whl), "size": whl.stat().st_size, "build_tag": tag, "tag": "cp313-cp313-linux_x86_64"},
            "status": "BUILT, STATICALLY INSPECTED, CPU-TESTED; KAGGLE VERIFICATION PENDING (notebook 15)", "kaggle_verified": False,
            "locks": {"python": "3.13", "vllm": "0.25.1", "cuda": "13.0", "torch": "2.11.0+cu130", "triton": "3.6.0", "gpu": "2x Tesla T4 SM75", "attention": "TRITON_ATTN"},
            "vllm_tree": "every file byte-identical to vllm-0.25.1-cp38-abi3-linux_x86_64.whl cc3bcafb6516791990ca302073ae153a08465d3df1e460d666e3ec09b4e98f5e",
            "legacy_release": {"name": "stokastisk-0.1.0-kaggle-t4-sm75", "status": "KAGGLE_T4_VERIFIED", "wheels": hist, "installer": "02_install.py (exactly two wheels; refuses this bundle)"},
            "superseded_candidate": "sdk-candidate/stokastisk-0.1.0-build1 (build tag 1, sha256 815f978346cfc6d85b7b47f9b96ce6805b5630ea2e0d3c0070c45f538c11cf6b)"}
(B / "manifests/RELEASE-MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
build = {"timestamp": manifest["created"], "source_revision": head, "source_tree_dirty_tracked_files": dirty, "build_command": " ".join(cmd), "python": sys.version.split()[0],
         "setuptools_build": "python -m build --wheel --no-isolation", "wheel": whl.name, "wheel_size": whl.stat().st_size, "wheel_sha256": sha(whl),
         "vllm_wheel_pinned_sha256": hist["vllm-0.25.1-cp38-abi3-linux_x86_64.whl"], "nvcc_native_rebuilt": False, "pip": subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True, text=True).stdout.strip()}
(B / "manifests/BUILD-MANIFEST.json").write_text(json.dumps(build, indent=1) + "\n"); shutil.copy2(B / "manifests/BUILD-MANIFEST.json", B / "evidence/BUILD-MANIFEST.json")
(B / "manifests/DEPENDENCY-MANIFEST.json").write_text(json.dumps({"core_pins": dm["core_pins"], "hash_file": "requirements/kaggle-t4-core.hashes.txt", "protected_exact": {"torch": "2.11.0+cu130", "triton": "3.6.0"},
    "transformers": ">=5.5.3,<6", "known_pip_check_conflict": "xgrammar 0.2.5.post1 has requirement transformers<5,>=4.38.0 (docs/DEPENDENCY_COMPATIBILITY.md)", "prohibited": sorted(json.loads(json.dumps(dm["profiles"]))["kaggle-t4-prohibited.txt"])}, indent=1) + "\n")
lines = [f"{sha(p)}  {p.relative_to(B).as_posix()}" for p in sorted(B.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"]
(B / "manifests/SHA256SUMS").write_text("\n".join(lines) + "\n")
# ---- reproducible ZIP
zp = WORK / f"{NAME}-kaggle-input.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for p in sorted(B.rglob("*")):
        if p.is_file():
            zi = zipfile.ZipInfo(f"{NAME}/{p.relative_to(B).as_posix()}", (2026, 11, 12, 0, 0, 0)); zi.compress_type = zipfile.ZIP_DEFLATED; zi.external_attr = 0o644 << 16
            z.writestr(zi, p.read_bytes())
(WORK / (zp.name + ".sha256")).write_text(f"{sha(zp)}  {zp.name}\n")
# re-verify from a fresh extraction
import tempfile
with tempfile.TemporaryDirectory() as td:
    zipfile.ZipFile(zp).extractall(td); bad = [l for l in (pathlib.Path(td) / NAME / "manifests/SHA256SUMS").read_text().splitlines() if sha(pathlib.Path(td) / NAME / l.split(None, 1)[1]) != l.split()[0]]
    assert not bad, bad
print(json.dumps({"wheel": whl.name, "wheel_sha256": sha(whl), "wheel_size": whl.stat().st_size, "zip": zp.name, "zip_sha256": sha(zp), "zip_size": zp.stat().st_size, "head": head, "dirty": dirty}, indent=1))
