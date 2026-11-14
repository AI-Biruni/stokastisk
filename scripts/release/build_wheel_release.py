#!/usr/bin/env python3.13
"""Build, audit and describe one integrated Stokastisk wheel (build tag N >= 3) WITHOUT any GPU, Kaggle or network access.
usage: build_wheel_release.py --build-tag N --out DIR --reference-wheel OLD.whl [--allow-sdk stokastisk/x.py ...] [--docstring-only stokastisk/x.py ...]
                              [--ast-identical stokastisk/x.py:name,name ...] [--reproduce] [--cpu-tests "<command>=<result>"]

DIR must not exist and nothing is ever overwritten. Steps: build (scripts/release/build_integrated_wheel.py, from the byte-pinned vLLM wheel in inputs/), independent verification
(scripts/audit/verify_integrated_wheel.py), member-by-member delta against the reference wheel (scripts/audit/wheel_delta.py), optional second build to prove reproducibility, and
``release-manifest.json``. The manifest keeps the evidence classes apart: it records what was RUN on Kaggle (inherited, with where it came from), that this build was NOT run on Kaggle,
the proven equivalence of the unchanged implementation, and the CPU verification. It never sets a Kaggle-verified flag for a build that was not executed."""
import hashlib, json, pathlib, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
PY = sys.executable
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, **kw)
    return r


def opt(argv, flag, many=False):
    vals = []
    while flag in argv:
        i = argv.index(flag); vals.append(argv[i + 1]); del argv[i:i + 2]
    return vals if many else (vals[0] if vals else None)


def main(argv):
    tag, out, ref = opt(argv, "--build-tag"), opt(argv, "--out"), opt(argv, "--reference-wheel")
    allow, docs, asts = opt(argv, "--allow-sdk", True), opt(argv, "--docstring-only", True), opt(argv, "--ast-identical", True)
    cpu = opt(argv, "--cpu-tests", True)
    reproduce = "--reproduce" in argv
    assert tag and tag.isdigit() and int(tag) >= 3 and out and ref, __doc__
    out = pathlib.Path(out)
    assert not out.exists(), f"{out} exists: nothing is overwritten"
    assert (ROOT / "VERSION").read_text() == "0.1.0\n"
    (out / "reports").mkdir(parents=True)
    vllm = ROOT / "inputs/vllm-0.25.1-cp38-abi3-linux_x86_64.whl"
    r = run([PY, "scripts/release/build_integrated_wheel.py", str(out), "--build-tag", tag, "--report-dir", str(out / "reports")])
    assert r.returncode == 0, r.stdout[-800:] + r.stderr[-1500:]
    built = json.loads(r.stdout.strip().splitlines()[-1]); wheel = out / built["wheel"]
    v = run([PY, "scripts/audit/verify_integrated_wheel.py", str(wheel), str(vllm), str(out / "reports/verify.json")])
    verify = json.loads((out / "reports/verify.json").read_text())
    cmd = [PY, "scripts/audit/wheel_delta.py", ref, str(wheel), str(out / "reports/delta-vs-reference.json")]
    for a in allow: cmd += ["--allow-sdk", a]
    for a in docs: cmd += ["--docstring-only", a]
    for a in asts: cmd += ["--ast-identical", a]
    d = run(cmd); delta = json.loads((out / "reports/delta-vs-reference.json").read_text())
    repro = None
    if reproduce:
        with tempfile.TemporaryDirectory() as td:
            r2 = run([PY, "scripts/release/build_integrated_wheel.py", td, "--build-tag", tag]); w2 = next(pathlib.Path(td).glob("*.whl"))
            repro = {"second_build_sha256": sha(w2.read_bytes()), "identical": sha(w2.read_bytes()) == built["sha256"]}
    head = run(["git", "rev-parse", "HEAD"]).stdout.strip(); dirty = run(["git", "status", "--porcelain", "--untracked-files=no"]).stdout.strip().splitlines()
    lock = {p.name: sha(p.read_bytes()) for p in sorted((ROOT / "requirements").glob("*.lock")) + sorted((ROOT / "requirements").glob("*.roots"))}
    ev = json.loads((ROOT / "evidence/kaggle/evidence-manifest.json").read_text())
    equivalent = delta["status"] == "PASS"
    man = {"schema": 1, "project_version": "0.1.0", "build_tag": tag,
           "wheel": {"filename": wheel.name, "size": wheel.stat().st_size, "sha256": built["sha256"], "python_tag": "cp313-cp313-linux_x86_64", "requires_python": ">=3.13,<3.14"},
           "base_vllm_wheel": {"filename": vllm.name, "size": vllm.stat().st_size, "sha256": sha(vllm.read_bytes())},
           "source": {"git_commit": head, "tracked_files_modified_at_build": dirty},
           "target": {"vllm": "0.25.1", "cuda_toolkit": "13.0", "torch": "2.11.0+cu130", "triton": "3.6.0", "gpu": "Tesla T4 x 2 (SM75)", "dtype": "float16", "attention": "TRITON_ATTN"},
           "dependency_locks_sha256": lock,
           "audits": {"verify_integrated_wheel": {"status": verify["status"], "checks": len(verify["checks"]), "report": "reports/verify.json"},
                      "record_verified": any(c["check"].startswith("RECORD") and c["status"] == "PASS" for c in verify["checks"]),
                      "flash_removal_audit": [c["status"] for c in verify["checks"] if "Flash policy" in c["check"]],
                      "native_binary_comparison_with_pinned_vllm": [c["status"] for c in verify["checks"] if "native object" in c["check"]],
                      "delta_vs_reference": {"status": delta["status"], "reference": delta["old_wheel"], "summary": delta["summary"], "report": "reports/delta-vs-reference.json"},
                      "reproducible_build": repro},
           "sdk_file_delta_against_reference": {"changed": delta["summary"]["by_category_changed"]["sdk"], "allowed": sorted(allow), "docstring_only": sorted(docs), "ast_analysis": delta["ast_analysis"]},
           "verification": {
               "build3_kaggle_acceptance": {"executed_on_kaggle": True, "on_wheel": ev["notebook_15_acceptance"]["acceptance_wheel_sha256"], "stages": {k: f'{v["passed"]}/{v["checks"]}' for k, v in ev["notebook_15_acceptance"]["stages"].items()},
                                            "raw_evidence": "evidence/kaggle/evidence-manifest.json"},
               "build3_end_user_notebooks": {"executed_on_kaggle": True, "notebooks": ["16", "18", "19"], "raw_evidence": "evidence/kaggle/evidence-manifest.json"},
               "this_build_executed_on_kaggle": False,
               "this_build_kaggle_status": "NOT_RUN (the owner prohibited further T4 verification); nothing below is a claim that this build ran on a GPU",
               "inference_code_equivalent_to_build3": {"proven_by_delta_audit": equivalent,
                                                      "meaning": "all vllm/** members and native objects are byte-identical, every dependency pin is unchanged, and the only SDK edits are the allowed files (docstring-only where listed; the runtime-imported compat names are AST-identical). "
                                                                 "Build-3 Kaggle evidence is therefore inherited for the unchanged implementation paths." if equivalent else "NOT PROVEN - see the delta report"},
               "cpu_verification": {"wheel_audit": verify["status"], "delta_audit": delta["status"], "reproducible_build": None if repro is None else repro["identical"], "tests": cpu}},
           "publication": {"huggingface_wheel_verified": None, "pip_index_verified": None, "note": "filled in only after the remote files were downloaded again and compared"}}
    (out / "release-manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    ok = verify["status"] == "PASS" and delta["status"] == "PASS" and (repro is None or repro["identical"])
    print(json.dumps({"wheel": wheel.name, "sha256": built["sha256"], "size": wheel.stat().st_size, "verify": verify["status"], "delta": delta["status"], "reproducible": None if repro is None else repro["identical"]}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
