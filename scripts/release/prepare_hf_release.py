#!/usr/bin/env python3.13
"""Stage the Hugging Face repository tree for the locked release and write latest.json. Never uploads anything.
usage: prepare_hf_release.py [STAGE_DIR]          (default: dist-rc/hf-stage; must not exist)
Refuses unless the assembled release and ZIP exist and scripts/release/verify_project_version.py passes on the staged tree (project version 0.1.0 only)."""
import hashlib, json, pathlib, shutil, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[2]
LOCKED = "0.1.0"; NAME = f"stokastisk-{LOCKED}-kaggle-t4-sm75"
assert (ROOT / "VERSION").read_text() == LOCKED + "\n", "VERSION is not the locked project version"
REL = ROOT / "dist-rc" / NAME; ZIP = ROOT / "stokastisk-kaggle-input" / f"{NAME}-kaggle-input.zip"
STAGE = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dist-rc" / "hf-stage"
assert REL.is_dir() and ZIP.is_file(), "run scripts/release/assemble_release.py first"
assert not STAGE.exists(), f"{STAGE} exists (never overwrite a stage)"
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()  # noqa: E731

dst = STAGE / "releases" / NAME
(dst / "wheels").mkdir(parents=True); (dst / "manifests").mkdir()
shutil.copy2(ZIP, dst / ZIP.name); (dst / (ZIP.name + ".sha256")).write_text(f"{sha(ZIP)}  {ZIP.name}\n")
for w in sorted((REL / "wheels").glob("*.whl")): shutil.copy2(w, dst / "wheels" / w.name)
for m in ("RELEASE-MANIFEST.json", "BUILD-MANIFEST.json", "DEPENDENCY-MANIFEST.json", "CUDA-ARCH-MANIFEST.json", "SHA256SUMS"): shutil.copy2(REL / "manifests" / m, dst / "manifests" / m)
(dst / "VERSION").write_text(LOCKED + "\n")
shutil.copy2(ROOT / "INSTALL.md", dst / "INSTALL.md")                                              # installation instructions
(dst / "validation").mkdir(); shutil.copy2(REL / "evidence/test-results.json", dst / "validation/test-results.json")   # validation metadata (CPU tests)
shutil.copy2(REL / "evidence/notebook-validation.json", dst / "validation/notebook-validation.json")
shutil.copy2(ROOT / "docs/HUB_README.md", STAGE / "README.md")
(STAGE / "scripts").mkdir(); shutil.copy2(ROOT / "scripts/fetch_release.py", STAGE / "scripts/fetch_release.py")
rm = json.loads((REL / "manifests/RELEASE-MANIFEST.json").read_text())
kv = json.loads((ROOT / "docs/kaggle-t4-verification.json").read_text())
assert kv["sdk_wheel_sha256_released"] == sha(next((REL / "wheels").glob("stokastisk-*.whl"))), "verification record names a different SDK wheel than the release"
base = f"releases/{NAME}"
latest = {"project": "stokastisk", "version": LOCKED, "release": NAME, "release_dir": base, "updated": time.strftime("%Y-%m-%d"),
          "validation_status": kv["validation_status"], "status": kv["performance_conclusion"], "verification": kv,
          "zip": {"path": f"{base}/{ZIP.name}", "sha256": sha(ZIP), "size": ZIP.stat().st_size},
          "wheels": {w.name: {"path": f"{base}/wheels/{w.name}", "sha256": sha(w), "size": w.stat().st_size} for w in sorted((dst / "wheels").glob("*.whl"))},
          "manifests": {m.name: f"{base}/manifests/{m.name}" for m in sorted((dst / "manifests").iterdir())},
          "installer": "scripts/fetch_release.py",
          "target": {"python": "3.13", "torch": "2.11.0+cu130", "cuda": "13.0", "triton": "3.6.0", "vllm": "0.25.1", "gpus": "2x Tesla T4 SM75", "attention": "TRITON_ATTN"},
          "kaggle_acceptance": kv["kaggle_acceptance"]}
(STAGE / "latest.json").write_text(json.dumps(latest, indent=1) + "\n")
r = subprocess.run([sys.executable, str(ROOT / "scripts/release/verify_project_version.py"), "--zip", str(ZIP), "--latest-json", str(STAGE / "latest.json"),
                    "--model-card", str(STAGE / "README.md"), "--wheel", str(next((dst / "wheels").glob("stokastisk-*.whl")))], capture_output=True, text=True)
if r.returncode:
    shutil.rmtree(STAGE); sys.exit("PROJECT VERSION LOCK VIOLATION:\n" + r.stdout[-2500:])
print(r.stdout.splitlines()[-1]); print("staged:", STAGE)
