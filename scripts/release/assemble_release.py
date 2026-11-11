#!/usr/bin/env python3.13
"""Assemble THE Stokastisk release `stokastisk-0.1.0-kaggle-t4-sm75` from a private baseline directory (source of the vLLM wheel and native-binary lineage).
usage: assemble_release.py BASE_NAME        BASE_NAME = directory under dist-rc/ that holds the byte-pinned vLLM wheel (never modified; hashes asserted before and after)
 * PROJECT VERSION LOCK: refuses to run unless VERSION, sdk/stokastisk/_version.py and the built wheel all say exactly 0.1.0 (VERSION_POLICY.md). Nothing here derives or increments a version.
 * the vLLM wheel (all native code) is carried over BYTE-IDENTICAL; the SDK wheel is rebuilt from sdk/ and diffed against it
 * CPU tests must all pass with zero skips; native audit re-run on the (unchanged) vLLM wheel; hygiene scans; manifests; SHA256SUMS
 * the ZIP is built by make_kaggle_input_zip.py (reproducible) and re-verified from a fresh extraction, then scripts/release/verify_project_version.py runs on the bundle + ZIP
 * status is ALWAYS "KAGGLE_GPU_VERIFICATION_PENDING" here: only an owner-run Kaggle execution can change that."""
import hashlib, json, os, pathlib, re, shutil, subprocess, sys, tempfile, time, zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
D = ROOT / "dist-rc"; BASE = sys.argv[1]
LOCKED = "0.1.0"; NEW = f"stokastisk-{LOCKED}-kaggle-t4-sm75"
assert (ROOT / "VERSION").read_text() == LOCKED + "\n", "VERSION is not the locked project version"
assert re.search(r'__version__ = "([^"]+)"', (ROOT / "sdk/stokastisk/_version.py").read_text()).group(1) == LOCKED, "_version.py is not the locked project version"
assert not os.environ.get("SETUPTOOLS_SCM_PRETEND_VERSION") and not os.environ.get("PBR_VERSION"), "version override environment variables are forbidden"
S, B = D / BASE, D / NEW
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()  # noqa: E731
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=ROOT).stdout.strip()  # noqa: E731
assert S.is_dir() and not B.exists(), "base must exist and the new release must not (never overwrite a release)"
BASE_PIN = {"wheels/vllm-0.25.1-cp38-abi3-linux_x86_64.whl": "cc3bcafb6516791990ca302073ae153a08465d3df1e460d666e3ec09b4e98f5e"}   # the vLLM 0.25.1 SM75/CUDA13 wheel, byte-pinned
for rel, h in BASE_PIN.items(): assert sha(S / rel) == h, f"base vLLM wheel altered: {rel}"
base_snapshot = {str(p.relative_to(S)): sha(p) for p in S.rglob("*") if p.is_file()}

shutil.copytree(S, B)
subprocess.run(["chmod", "-R", "u+w", str(B)], check=True)   # the base may be a read-only history copy; the new release must be writable
ign = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
def overlay(rel, dst=None):
    src, dst = ROOT / rel, B / (dst or rel)
    if dst.exists(): shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
    shutil.copytree(src, dst, ignore=ign) if src.is_dir() else (dst.parent.mkdir(parents=True, exist_ok=True), shutil.copy2(src, dst))
for rel in ("scripts/kaggle", "scripts/release", "scripts/audit", "tests", "patches", "native", "requirements", "docs"): overlay(rel)
for rel in ("scripts/fetch_release.py", "scripts/make_notebooks.py", "scripts/build_vllm_sm75.sh", "README.md", "INSTALL.md", "BUILD.md", "RELEASE.md", "VERSION", "VERSION_POLICY.md", "LICENSE"): overlay(rel)
shutil.rmtree(B / "docs", ignore_errors=True); shutil.copytree(ROOT / "docs", B / "docs", ignore=shutil.ignore_patterns("HUB_README.md"))   # Hub card is not part of the bundle
for old in (B / "scripts/make_notebooks.py",):
    if old.exists(): old.unlink()
shutil.rmtree(B / "notebooks"); shutil.copytree(ROOT / "notebooks" / "kaggle-t4-sm75" / "notebooks", B / "notebooks", ignore=ign)
nbs = sorted(p.name for p in (B / "notebooks").glob("*.ipynb")); assert [n[:2] for n in nbs] == [f"{i:02d}" for i in range(1, 15)], nbs
for f in ("evidence/test-results.json", "evidence/sdk-test-results.json", "evidence/notebook-validation.json", "evidence/installer-test.json", "evidence/release-integrity.json",
          "evidence/source-diff-report.json", "evidence/wheel-vs-source.json"):
    (B / f).unlink(missing_ok=True)

# ---- wheels: vLLM byte-identical, SDK rebuilt and diffed against sdk/
sys.path.insert(0, str(ROOT / "tests")); from test_wheel_matches_source import wheel_drift  # noqa: E402
vname = "vllm-0.25.1-cp38-abi3-linux_x86_64.whl"
for w in (B / "wheels").glob("stokastisk-*.whl"): w.unlink()
with tempfile.TemporaryDirectory() as td:
    r = subprocess.run([sys.executable, "-m", "build", "--wheel", "--no-isolation", "-o", td, str(ROOT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    nw = next(pathlib.Path(td).glob("stokastisk-*.whl")); drift = wheel_drift(nw); assert not drift, drift
    shutil.copy2(nw, B / "wheels" / nw.name)
sname = nw.name; assert sname.startswith(f"stokastisk-{LOCKED}-"), sname
wheels = {w.name: sha(w) for w in sorted((B / "wheels").glob("*.whl"))}
assert set(wheels) == {vname, sname}, wheels
assert wheels[vname] == BASE_PIN["wheels/" + vname] == sha(S / "wheels" / vname), "vLLM wheel must be byte-identical to the pinned base"
ns = {w.name: [n for n in zipfile.ZipFile(w).namelist() if re.search(r"velocit\w*|stoch\w*tisk", n, re.I)] for w in (B / "wheels").glob("*.whl")}
assert not any(ns.values()), ns
flash = {w.name: [n for n in zipfile.ZipFile(w).namelist() if re.search(r"flashinfer|vllm_flash_attn/_vllm_fa|flashmla|flash_attn_2_cuda", n, re.I) and not n.endswith(".py")] for w in (B / "wheels").glob("*.whl")}
assert not any(flash.values()), flash
sdk_wheel_members = zipfile.ZipFile(B / "wheels" / sname).namelist()
assert not [n for n in sdk_wheel_members if n.endswith((".so", ".cpp", ".cu"))], "SDK wheel must be pure Python"
(B / "evidence").mkdir(exist_ok=True)
(B / "evidence/wheel-vs-source.json").write_text(json.dumps({"stokastisk_wheel": sname, "sha256": wheels[sname], "drift_vs_sdk": drift, "pure_python": True, "members": len(sdk_wheel_members)}, indent=1))
(B / "evidence/namespace-audit.json").write_text(json.dumps({"canonical_package": "stokastisk", "wheel_members_with_obsolete_or_misspelled_names": ns}, indent=1))
(B / "evidence/flashattention-flashinfer-audit.json").write_text(json.dumps({"native_flash_members_in_wheels": flash, "note": "inert upstream *.py adapter modules are allowed; native flash code is not"}, indent=1))

# ---- native audit of the (unchanged) vLLM wheel
na = subprocess.run([sys.executable, str(ROOT / "scripts/audit/native_audit.py"), str(B / "wheels" / vname), str(B / "evidence/native-binary-audit.json")], capture_output=True, text=True)
assert na.returncode == 0, na.stderr[-800:]
nsum = json.loads((B / "evidence/native-binary-audit.json").read_text())["summary"]
assert nsum["all_x86_64"] and not nsum["any_needs_cuda12_soname"] and not nsum["any_cuda12_toolkit_string"] and not nsum["any_older_than_sm75_cubin"] and not nsum["any_flash_markers"], nsum
shutil.copy2(B / "evidence/native-binary-audit.json", B / "manifests/NATIVE-BINARY-AUDIT.json")

# ---- the experimental C++ transport must still compile for sm_75 and link only libcudart.so.13
tmp = pathlib.Path(tempfile.mkdtemp()); so = tmp / "libstokastisk_xfer.so"
nv = subprocess.run(["nvcc", "-O2", "-std=c++17", "-shared", "-Xcompiler", "-fPIC", "-gencode", "arch=compute_75,code=sm_75", str(ROOT / "native/host_staged_xfer.cpp"), "-o", str(so), "-lcudart"], capture_output=True, text=True)
native_ev = {"compiled": nv.returncode == 0, "stderr_tail": nv.stderr[-300:], "executed_on_two_gpus": False,
             "note": "compile + link check only; execution on 2 GPUs is notebook 10 (NOT_RUN)"}
if nv.returncode == 0:
    ldd = subprocess.run(["ldd", str(so)], capture_output=True, text=True).stdout
    native_ev["links"] = sorted(set(re.findall(r"(libcudart\.so\.\d+|libcublas\S*|libnccl\S*|libnvshmem\S*)", ldd)))
    assert native_ev["links"] == ["libcudart.so.13"], native_ev
    native_ev["sha256_of_local_build"] = sha(so)
(B / "evidence/experimental-transport-build.json").write_text(json.dumps(native_ev, indent=1)); shutil.rmtree(tmp, ignore_errors=True)
assert native_ev["compiled"], native_ev

# ---- CPU tests (zero skips, zero failures)
env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
tests = {}
for t in sorted(p.name for p in (ROOT / "tests").glob("test_*.py")):
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(ROOT / "tests" / t)], capture_output=True, text=True, env=env, cwd=ROOT)
    m = re.search(r"(\d+) passed", r.stdout); f = re.search(r"(\d+) failed", r.stdout); sk = re.search(r"(\d+) skipped", r.stdout)
    tests[t] = {"returncode": r.returncode, "passed": int(m.group(1)) if m else 0, "failed": int(f.group(1)) if f else 0, "skipped": int(sk.group(1)) if sk else 0}
assert all(v["returncode"] == 0 and v["failed"] == 0 for v in tests.values()), tests
assert all(v["skipped"] == 0 for v in tests.values()), f"skipped tests would be NOT_RUN, not PASS: {tests}"

# ---- notebook / script static validation
import ast
nbv = []
for f in sorted((B / "notebooks").glob("*.ipynb")):
    j = json.loads(f.read_text()); code = [c for c in j["cells"] if c["cell_type"] == "code"]
    for c in code: ast.parse("".join(c["source"]))
    assert all(c["outputs"] == [] and c["execution_count"] is None for c in code), f.name
    nbv.append({"notebook": f.name, "code_cells": len(code), "parse": True, "executed": False})
now = time.strftime("%FT%T%z")
(B / "evidence/notebook-validation.json").write_text(json.dumps({"timestamp": now, "kind": "static (JSON + syntax + unexecuted). NOT executed on Kaggle.", "notebooks": nbv, "execution_status": "NOT_RUN (pending owner on Kaggle)"}, indent=1))
(B / "evidence/test-results.json").write_text(json.dumps({"timestamp": now, "kind": "CPU-only tests with mocks / fake workers; NOT GPU or Kaggle evidence", "tests": tests,
        "totals": {k: sum(v[k] for v in tests.values()) for k in ("passed", "failed", "skipped")}, "git_head": git("rev-parse", "HEAD"), "git_branch": git("branch", "--show-current"),
        "git_dirty": bool(git("status", "--porcelain"))}, indent=1))

# ---- manifests
rm0 = json.loads((S / "manifests/RELEASE-MANIFEST.json").read_text())
status = ("KAGGLE_GPU_VERIFICATION_PENDING — CPU-tested only (mocks, fake workers); this SDK wheel has not been executed on Kaggle. Notebooks 01-12 were executed with the earlier SDK wheel "
          "a6b8596b (see manifests/PRIOR-KAGGLE-EVIDENCE.json); they say nothing about this wheel. Notebooks 13-14 are new and NOT_RUN. "
          "The vLLM wheel and every native CUDA binary are byte-identical to the earlier baseline build.")
rm = {"name": NEW, "project": "stokastisk", "version": LOCKED, "timestamp": now, "status": status, "validation_status": "KAGGLE_GPU_VERIFICATION_PENDING", "supersedes": None, "prior_wheel_executed_on_kaggle": {"stokastisk-0.1.0-py3-none-any.whl": "a6b8596bb70e31d82a3f00ae15bbffa271876c718da9200943d7f42bfe5b8585"},
      "kaggle_acceptance": {n[:2]: "NOT_RUN" for n in nbs},
      "platform": {"python": "3.13", "cuda_toolkit": "13.0", "vllm": "0.25.1", "torch": "2.11.0+cu130", "triton": "3.6.0", "gpu": "2 x NVIDIA Tesla T4 (SM75)", "attention_backend": "TRITON_ATTN"},
      "wheels": {n: {"size": (B / "wheels" / n).stat().st_size, "sha256": h} for n, h in wheels.items()},
      "vllm_wheel_identical_to_base": True, "native_binaries_changed": False, "native_audit": nsum,
      "sdk_fix": "replica routing: an unobserved worker is weighted at the mean observed speed (previously 1.0 tokens/s), which sent every request to the first-warmed GPU in notebook-09 scaling runs",
      "new_features": ["topology discovery", "execution strategies (single_gpu, tensor_parallel, pipeline_parallel[experimental], replica_parallel[experimental], auto[recommendation-only])",
                       "ReplicaPool + token-aware scheduler (routing fix)", "recommend_parallelism", "telemetry", "cuda helpers", "experimental host-staged transport (Python + optional C++)", "benchmarks", "notebooks 07-14 (13: TP/PP fulfilment, 14: replica balance)"],
      "git": {"head": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"), "dirty": bool(git("status", "--porcelain"))}}
(B / "manifests/RELEASE-MANIFEST.json").write_text(json.dumps(rm, indent=1)); (B / "evidence/RELEASE-MANIFEST.json").write_text(json.dumps(rm, indent=1))
bm = {"timestamp": now, "kind": "assemble_release.py", "project_version": LOCKED, "tool_versions": {"python": sys.version.split()[0], "nvcc": subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]},
      "native_rebuilt": False, "sdk_wheel_rebuilt": True, "inputs": dict(BASE_PIN)}
(B / "manifests/BUILD-MANIFEST.json").write_text(json.dumps(bm, indent=1)); (B / "evidence/BUILD-MANIFEST.json").write_text(json.dumps(bm, indent=1))
dm = json.loads((S / "manifests/DEPENDENCY-MANIFEST.json").read_text()); dm["note"] = "no new required dependency; pynvml, cuda-python, opentelemetry, Nsight, nvcc are optional and detected at runtime"
(B / "manifests/DEPENDENCY-MANIFEST.json").write_text(json.dumps(dm, indent=1))
(B / "manifests/CUDA-ARCH-MANIFEST.json").write_text(json.dumps({"target": "sm_75 only", "ptx": False, "cuda_toolkit": "13.0 only", "native_objects_unchanged_from": "pinned baseline vLLM wheel", "audit": nsum,
        "experimental_transport": {"device_code": False, "links": native_ev.get("links")}}, indent=1))
(B / "manifests/PRIOR-KAGGLE-EVIDENCE.json").write_text(json.dumps({"note": "Kaggle T4 x2 execution of notebooks 01-12 on 2026-11-08..10 used the SDK wheel below, NOT the wheel in this release", "sdk_wheel_sha256_executed": "a6b8596bb70e31d82a3f00ae15bbffa271876c718da9200943d7f42bfe5b8585", "vllm_wheel_sha256_executed": wheels[vname], "vllm_wheel_unchanged": True, "results": {"01": "PASS", "02": "PASS", "03": "PASS", "04": "PASS", "05": "WARNING", "06": "PASS", "07": "PASS", "08": "PASS", "09": "PASS (validation); scaling defect", "10": "WARNING", "11": "PASS", "12": "PASS (shallow)"}, "details": "reports/STOKASTISK_010_KAGGLE_T4_EXECUTION_AUDIT.md"}, indent=1))
assert (B / "manifests/WHEEL-LINEAGE.json").exists()
if (B / "cuda/README.md").exists(): pass

# ---- hygiene
bad_names = [p for p in B.rglob("*") if p.is_file() and (p.name in {"tokens.env", ".env", "id_rsa", "id_ed25519", "os.ps"} or p.suffix in {".pyc", ".pyo"})]
assert not bad_names, bad_names
SECRET = re.compile(("gh" + "p_[A-Za-z0-9]{20}|github" + "_pat_[A-Za-z0-9_]{20}|hf" + "_[A-Za-z0-9]{30}|AKIA[0-9A-Z]{16}").encode())
secrets = [str(p.relative_to(B)) for p in B.rglob("*") if p.is_file() and p.suffix in (".py", ".md", ".json", ".txt", ".ipynb", ".sh", ".patch", ".cpp", ".toml") and SECRET.search(p.read_bytes())]
assert not secrets, secrets
assert not list(B.rglob("*executed*")) and not any("notebooks-executed" in str(p) for p in B.rglob("*"))
for p in B.rglob("*.ipynb"): assert not any(c.get("outputs") for c in json.loads(p.read_text())["cells"] if c["cell_type"] == "code"), p
req_text = "\n".join(p.read_text() for p in (B / "requirements").glob("kaggle-t4-core.txt"))
assert not re.search(r"flashinfer|flash-attn|nvidia-nccl-cu12|cuda-python|pynvml|nvidia-ml-py", req_text, re.I), "core pins must not include optional/prohibited packages"

# ---- checksums, ZIP
(B / "manifests/SHA256SUMS").unlink(missing_ok=True)
files = sorted(p for p in B.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
(B / "manifests/SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(B)}\n" for p in files if p.name != "SHA256SUMS"))
out = ROOT / "stokastisk-kaggle-input"
r = subprocess.run([sys.executable, str(ROOT / "scripts/release/make_kaggle_input_zip.py"), NEW, str(out)], capture_output=True, text=True); assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
zp = out / f"{NEW}-kaggle-input.zip"
with tempfile.TemporaryDirectory() as td:
    z = zipfile.ZipFile(zp); names = z.namelist(); assert z.testzip() is None
    assert len(names) == len(set(names)), "duplicate ZIP paths"
    z.extractall(td); root = pathlib.Path(td) / NEW
    bad = [l for l in (root / "manifests/SHA256SUMS").read_text().splitlines() if not (root / l.split(None, 1)[1].strip()).is_file() or sha(root / l.split(None, 1)[1].strip()) != l.split()[0]]
    assert not bad, bad[:5]
    assert all(n.startswith(NEW + "/") for n in names), "ZIP must have the single top-level release directory"
    assert not any(re.search(r"tokens\.env|\.env$|/\.git/|__pycache__|os\.ps|notebooks-executed|flashinfer_python|vllm_flash_attn-", n) for n in names)
    zip_wheels = sorted(n for n in names if n.endswith(".whl")); assert len(zip_wheels) == 2
    wheel_ok = all(sha(root / "wheels" / pathlib.Path(n).name) == wheels[pathlib.Path(n).name] for n in zip_wheels)
integ = {"zip": zp.name, "size": zp.stat().st_size, "sha256": sha(zp), "entries": len(names), "testzip": "ok", "duplicate_paths": 0, "hash_mismatches": 0, "wheels_match_manifest": wheel_ok,
         "secret_hits": 0, "executed_notebooks": 0, "base_unmodified": True}
assert wheel_ok
# ---- the base must be untouched
now_snapshot = {str(p.relative_to(S)): sha(p) for p in S.rglob("*") if p.is_file()}
assert now_snapshot == base_snapshot, "BASE RELEASE WAS MODIFIED"
(out / f"{NEW}-kaggle-input.zip.sha256").write_text(f"{sha(zp)}  {zp.name}\n")
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports/stokastisk-010-release-integrity.json").write_text(json.dumps({"integrity": integ, "wheels": wheels, "tests": tests, "native_audit": nsum, "experimental_transport_build": native_ev}, indent=1))
vr = subprocess.run([sys.executable, str(ROOT / "scripts/release/verify_project_version.py"), "--release-dir", str(B), "--zip", str(zp)], capture_output=True, text=True)
assert vr.returncode == 0, "PROJECT VERSION LOCK VIOLATION:\n" + vr.stdout[-2500:]
integ["project_version_verified"] = LOCKED
print(json.dumps(integ, indent=1)); print("wheels:", json.dumps(wheels, indent=1)); print(vr.stdout.splitlines()[-1])
