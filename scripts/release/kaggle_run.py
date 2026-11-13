#!/usr/bin/env python3.13
"""Maintainer tool: publish the build-3 bundle as a Kaggle dataset version, run notebook 15 on Kaggle (2 x Tesla T4, Internet ON) and download its outputs.
usage: kaggle_run.py BUNDLE_ZIP NOTEBOOK.ipynb OUT_DIR [--timeout SECONDS]
Needs an authenticated Kaggle CLI (~/.kaggle). The dataset and kernel ids are fixed below; both stay private."""
import json, pathlib, shutil, subprocess, sys, tempfile, time

USER = "waqasm86"
DATASET = f"{USER}/stokastisk-0-1-0-build3-t4-sm75-kaggle-input"
KERNEL = f"{USER}/15-stokastisk-build3-t4-verification"       # separate from the owner's hand-run notebook so the two never overwrite each other


def kg(*a, check=True):
    r = subprocess.run(["kaggle", *a], capture_output=True, text=True)
    print("$ kaggle", *a, "\n", (r.stdout + r.stderr).strip()[-1500:])
    if check and r.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(a)} failed")
    return r


def main(argv):
    zp, nb, out = map(pathlib.Path, argv[:3]); timeout = int(argv[argv.index("--timeout") + 1]) if "--timeout" in argv else 3600
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        d = pathlib.Path(td) / "ds"; d.mkdir(); shutil.copy2(zp, d / zp.name)
        (d / "dataset-metadata.json").write_text(json.dumps({"title": "stokastisk-0.1.0-build3-t4-sm75-kaggle-input", "id": DATASET, "licenses": [{"name": "Apache 2.0"}]}))
        kg("datasets", "version", "-p", str(d), "-m", f"build3 bundle {time.strftime('%F %T')} {zp.name}", "--dir-mode", "zip")
        t_push = time.time() - 120
        for _ in range(90):                                               # `status` can still report the PREVIOUS version as ready: wait until the new SHA256SUMS is listed
            s = kg("datasets", "status", DATASET, check=False).stdout.strip()
            f = kg("datasets", "files", DATASET, "--page-size", "200", check=False).stdout
            line = next((l for l in f.splitlines() if l.startswith(f"{zp.stem.replace('-kaggle-input', '')}/manifests/SHA256SUMS")), "")
            try:
                created = time.mktime(time.strptime(line.split()[-2] + " " + line.split()[-1][:8], "%Y-%m-%d %H:%M:%S")) - time.timezone
            except (ValueError, IndexError):
                created = 0
            if s.endswith("ready") and created >= t_push:
                break
            time.sleep(10)
        k = pathlib.Path(td) / "k"; k.mkdir(); shutil.copy2(nb, k / nb.name)
        (k / "kernel-metadata.json").write_text(json.dumps({"id": KERNEL, "title": "15 stokastisk build3 t4 verification", "code_file": nb.name, "language": "python", "kernel_type": "notebook",
                                                            "is_private": "true", "enable_gpu": "true", "enable_internet": "true", "dataset_sources": [DATASET], "competition_sources": [], "kernel_sources": []}))
        kg("kernels", "push", "-p", str(k), "--accelerator", "NvidiaTeslaT4", "-t", str(timeout))
    t0 = time.time()
    while time.time() - t0 < timeout + 600:
        time.sleep(30)
        s = kg("kernels", "status", KERNEL, check=False).stdout
        if any(w in s.lower() for w in ("complete", "error", "cancel")):
            break
    kg("kernels", "output", KERNEL, "-p", str(out))
    print("outputs in", out)


if __name__ == "__main__":
    main(sys.argv[1:])
