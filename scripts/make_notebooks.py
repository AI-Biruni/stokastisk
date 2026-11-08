#!/usr/bin/env python3.13
"""Generate the six UNEXECUTED Kaggle notebooks (07-12) into notebooks/. Cells only call the bundled scripts; nothing runs here.
Notebooks 01-06 (the baseline acceptance) are historical and are not regenerated."""
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"
RELEASE = "stokastisk-0.1.0-kaggle-t4-sm75"
_FETCH = (ROOT / "scripts/fetch_release.py").read_text().split('if __name__ == "__main__":')[0].split('"""', 2)[2].strip()

FIND = """import glob, hashlib, json, os, pathlib, shutil, sys, time, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor
""" + _FETCH + """

# Only the bundle directory named RELEASE whose RELEASE-MANIFEST.json carries the same name is accepted (never an older rc).
RELEASE = "@@RELEASE@@"
cands = [pathlib.Path(p).parent.parent for p in glob.glob("/kaggle/input/**/manifests/SHA256SUMS", recursive=True)]
if os.environ.get("STOKASTISK_BUNDLE"): cands = [pathlib.Path(os.environ["STOKASTISK_BUNDLE"])]
match = [c for c in cands if c.name == RELEASE]
if not match:
    print(f"{RELEASE} is not attached (attached: {sorted(str(c) for c in cands)}); downloading from Hugging Face (needs Internet ON and a published release) ...")
    match = [fetch_release(RELEASE)]
assert len({(c / "manifests" / "SHA256SUMS").read_bytes() for c in match}) == 1, f"several DIFFERENT copies of {RELEASE}: {match}"
BUNDLE = match[0]
_m = json.loads((BUNDLE / "manifests" / "RELEASE-MANIFEST.json").read_text())
assert _m.get("name") == RELEASE, f"manifest name {_m.get('name')!r} != expected {RELEASE!r}"
SCRIPTS = BUNDLE / "scripts" / "kaggle"
os.environ["STOKASTISK_OUT"] = "/kaggle/working/stokastisk-acceptance"
print("release:", RELEASE, "| bundle:", BUNDLE, "| manifest sha256:", hashlib.sha256((BUNDLE / "manifests" / "SHA256SUMS").read_bytes()).hexdigest())""".replace("@@RELEASE@@", RELEASE)

INSTALL = '''import subprocess, sys, os
r = subprocess.run([sys.executable, str(SCRIPTS / "02_install.py"), str(BUNDLE)],
                   env=dict(os.environ, STOKASTISK_OUT="/kaggle/working/stokastisk-acceptance/prereq-install"))
assert r.returncode == 0, "prerequisite install failed (see output above); do not continue"'''

PROVENANCE = '''import hashlib
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
for w in sorted((BUNDLE / "wheels").glob("*.whl")): print(f"{sha(w)}  {w.name}")
expected = {l.split(None, 1)[1].strip(): l.split()[0] for l in (BUNDLE / "manifests" / "SHA256SUMS").read_text().splitlines()}
for rel in [f"scripts/kaggle/{s.name}" for s in SCRIPTS.glob("*.py")] + [f"wheels/{w.name}" for w in (BUNDLE / "wheels").glob("*.whl")]:
    assert expected.get(rel) == sha(BUNDLE / rel), f"{rel} does not match manifests/SHA256SUMS"
print("all executed scripts and wheels match manifests/SHA256SUMS")'''

VERIFY = '''import importlib, importlib.metadata as md, zipfile
for w in sorted((BUNDLE / "wheels").glob("*.whl")):
    pkg = w.name.split("-")[0]; mod = importlib.import_module(pkg); root = pathlib.Path(mod.__file__).resolve().parent
    z = zipfile.ZipFile(w); bad = []
    for n in z.namelist():
        if n.startswith(pkg + "/") and n.endswith((".py", ".so")):
            f = root.parent / n
            if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest() != hashlib.sha256(z.read(n)).hexdigest(): bad.append(n)
    print(f"{pkg} {md.version(pkg)}: {'identical to ' + w.name if not bad else 'MISMATCH ' + str(bad)}")
    assert not bad, f"installed {pkg} differs from the bundle wheel {w.name}"'''

RUNPY = '''import subprocess, sys, os
def run_stage(script, *args, timeout=1800, log=None):
    """Run a bundled stage script with a finite timeout; returns its exit code (124 = timeout). Failures are judged in the verdict cell."""
    env = dict(os.environ, PYTHONPATH=str(SCRIPTS) + ":" + os.environ.get("PYTHONPATH", ""), VLLM_LOGGING_LEVEL="INFO")
    out = open("/kaggle/working/stokastisk-acceptance/" + (log or script + ".log"), "w") if log is not False else None
    try:
        r = subprocess.run([sys.executable, str(SCRIPTS / script), *args], cwd=SCRIPTS, env=env, timeout=timeout, stdout=out, stderr=subprocess.STDOUT if out else None)
        rc = r.returncode
    except subprocess.TimeoutExpired:
        rc = 124
    print(f"{script} {' '.join(args)} exit code: {rc}")
    return rc
os.makedirs("/kaggle/working/stokastisk-acceptance", exist_ok=True)'''

ZIP = '''import shutil
print(shutil.make_archive("/kaggle/working/{nn}-stokastisk-acceptance", "zip", "/kaggle/working", "stokastisk-acceptance"))'''
HEAD = ("**UNEXECUTED — Kaggle verification pending.** Stokastisk 0.1.0 features are verified only when *you* run this notebook on Kaggle "
        "(Accelerator **GPU T4 x2**, Internet ON, Python 3.13). Statuses: PASS / FAIL / ERROR / WARNING / SKIPPED / NOT_RUN. "
        "A feature that is unsupported is reported as such; it is never turned into a PASS.")


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.strip().splitlines(True)}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip().splitlines(True)}
def setup(): return [md("## 1. Locate the release bundle"), code(FIND), md("## 2. Install into this fresh session (`--no-deps`, protected packages untouched)"), code(INSTALL),
                     md("## 3. Provenance: scripts and wheels must match `manifests/SHA256SUMS`"), code(PROVENANCE), md("## 4. Installed packages must be byte-identical to the bundle"), code(VERIFY),
                     md("## 5. Stages"), code(RUNPY)]
def pack(nn): return [md(f"## Package evidence → `/kaggle/working/{nn}-stokastisk-acceptance.zip`"), code(ZIP.replace("{nn}", nn))]
def verdict(required, info=""):
    return [md("## Verdict (fails the notebook if a REQUIRED stage failed)" + (f"\n\n{info}" if info else "")),
            code(f"required = {required!r}\nbad = {{k: v for k, v in RC.items() if k in required and v != 0}}\nprint('all stage exit codes:', RC)\nassert not bad, f'FAILED required stage(s): {{bad}}'")]


NB = {
 "07_stokastisk_gpu_topology_validation": ("07 — GPU topology validation",
   "Discovers the real topology: GPUs, UUIDs, compute capability, VRAM, PCIe, CUDA peer access in both directions, link class, NCCL availability, telemetry, Nsight availability. Differences from the baseline-observed environment (PHB, no P2P, no NVLink) are WARNINGs.",
   lambda: setup() + [code('RC = {"07_topology": run_stage("dual_07_topology.py", timeout=300)}')] + pack("07") + verdict(["07_topology"])),
 "08_stokastisk_pipeline_parallel_validation": ("08 — Pipeline parallel (PP=2) validation",
   "Tries PP=2 on the bundled vLLM wheel (Qwen2.5-0.5B), then benchmarks TP=1, TP=2 and PP=2 on one workload (Qwen2.5-1.5B). If PP=2 is unsupported the notebook FAILS with the real reason — it never fabricates a PASS.",
   lambda: setup() + [code('RC = {}\nRC["08_pp_validation"] = run_stage("dual_08_pp_validation.py", timeout=1200)'),
     md("### PP=2 vs TP=1 vs TP=2 (same model, prompts and sampling; every repeat recorded)"),
     code('M = "Qwen/Qwen2.5-1.5B-Instruct"\nfor mode in ("tp1", "tp2", "pp2"):\n    RC["bench_" + mode] = run_stage("dual_bench.py", mode, M, "standard", timeout=1800, log=f"bench_{mode}.log")')]
     + pack("08") + verdict(["08_pp_validation", "bench_tp1"], "PP=2 results are required to be *reported*; TP=2 / PP=2 benchmark failures are shown but do not hide the PP validation result above.")),
 "09_stokastisk_dual_gpu_replica_inference": ("09 — Independent dual-GPU replica inference",
   "Two TP=1 vLLM engines, one per T4: distinct UUIDs/PIDs, independent memory, 16 concurrent requests, clean shutdown. Then a scaling benchmark (1/8/16/32/64 requests) against TP=1 and TP=2.",
   lambda: setup() + [code('RC = {}\nRC["09_replica_validation"] = run_stage("dual_09_replica_validation.py", timeout=1500)'),
     code('M = "Qwen/Qwen2.5-1.5B-Instruct"\nfor mode in ("tp1", "replicas2", "tp2"):\n    RC["bench_" + mode] = run_stage("dual_bench.py", mode, M, "scaling", timeout=2400, log=f"bench_{mode}_scaling.log")')]
     + pack("09") + verdict(["09_replica_validation", "bench_tp1", "bench_replicas2"])),
 "10_stokastisk_cuda_communication_benchmark": ("10 — CUDA communication benchmark (EXPERIMENTAL)",
   "Host-staged double-buffered transfer (PyTorch pinned memory, and the C++ CUDA-Runtime build if nvcc works) versus NCCL send/recv, 1 KiB – 128 MiB. A slower transport stays experimental.",
   lambda: setup() + [code('RC = {"10_comm": run_stage("dual_10_comm.py", timeout=1800)}')] + pack("10") + verdict(["10_comm"])),
 "11_stokastisk_gpu_load_balancing_benchmark": ("11 — GPU load-balancing benchmark",
   "Runs TP=1, TP=2, PP=2 and 2 replicas on the same workloads, then reports throughput, TTFT, inter-token latency, per-worker compute/idle time and imbalance %. GPU utilisation alone is not treated as proof of balance.",
   lambda: setup() + [code('RC = {}\nM = "Qwen/Qwen2.5-1.5B-Instruct"\nfor mode in ("tp1", "replicas2", "tp2", "pp2"):\n    RC["bench_" + mode] = run_stage("dual_bench.py", mode, M, "standard", timeout=1800, log=f"bench_{mode}.log")\nRC["11_balance"] = run_stage("dual_11_balance.py", timeout=300)')]
     + pack("11") + verdict(["bench_tp1", "bench_replicas2", "11_balance"])),
 "12_stokastisk_consolidated_acceptance": ("12 — Consolidated acceptance",
   "One session: topology, recommender, unchanged baseline API (TP=1), replicas (all MANDATORY) plus PP=2 and host-staged transport (EXPERIMENTAL: reported with their real status, never counted as successes).",
   lambda: setup() + [code('RC = {"12_acceptance": run_stage("dual_12_acceptance.py", timeout=5400, log="12_acceptance.log")}'), code('print(open("/kaggle/working/stokastisk-acceptance/12_acceptance.log").read()[-3000:])')]
     + pack("12") + verdict(["12_acceptance"])),
}

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, (title, body, cells) in NB.items():
        nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}},
              "cells": [md(f"# {title}\n\n{HEAD}\n\n{body}")] + cells()}
        (OUT / f"{name}.ipynb").write_text(json.dumps(nb, indent=1))
    print("wrote", len(NB), "notebooks to", OUT)
