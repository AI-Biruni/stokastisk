"""Stage 12 (0.4.0): consolidated acceptance. Runs the sub-stages as subprocesses (each in its own fresh interpreter, bounded time),
then classifies every feature. MANDATORY features must PASS; EXPERIMENTAL ones are reported with their real status and never
silently counted as successes. usage: v040_12_acceptance.py"""
import json, pathlib, subprocess, sys, time
from _common import OUT, check, finish, gpu_used_mib, run_guarded, save, versions

HERE = pathlib.Path(__file__).resolve().parent
MANDATORY = {"07_topology": "GPU topology discovery", "13_api_compat": "0.3.0 public API still works (TP=1 generation)", "09_replica_validation": "independent dual-GPU replicas",
             "14_recommender": "parallelism recommendation (rule-based)"}
EXPERIMENTAL = {"08_pp_validation": "pipeline parallelism (PP=2)", "10_comm": "host-staged CUDA transport vs NCCL"}


def run_stage(script, args=(), timeout=1500):
    t = time.time()
    try:
        r = subprocess.run([sys.executable, str(HERE / script), *args], capture_output=True, text=True, timeout=timeout, cwd=str(HERE))
        (OUT / f"{script}.log").write_text(r.stdout + r.stderr); return r.returncode, time.time() - t
    except subprocess.TimeoutExpired as e:
        (OUT / f"{script}.log").write_text(f"TIMEOUT after {timeout}s\n{(e.stdout or b'')[-2000:]}"); return 124, time.time() - t


def main() -> int:
    R = []
    plan = [("v040_07_topology.py", "07_topology", ()), ("v040_14_recommender.py", "14_recommender", ()), ("v040_13_api_compat.py", "13_api_compat", ()),
            ("v040_09_replica_validation.py", "09_replica_validation", ()), ("v040_08_pp_validation.py", "08_pp_validation", ()), ("v040_10_comm.py", "10_comm", ())]
    stages, timing = {}, {}
    for script, stage, args in plan:
        rc, secs = run_stage(script, args); timing[stage] = round(secs, 1)
        f = OUT / f"{stage}.json"
        stages[stage] = json.loads(f.read_text())["status"] if f.exists() else ("TIMEOUT" if rc == 124 else "ERROR")
        print(f"{stage}: {stages[stage]} ({secs:.0f}s)")
        mem = gpu_used_mib() or []
        if mem and any(x > 800 for x in mem):
            time.sleep(10)
    for s, d in MANDATORY.items():
        check(R, f"MANDATORY: {d}", stages.get(s) == "PASS", stages.get(s))
    classification = {}
    for s, d in EXPERIMENTAL.items():
        st = stages.get(s, "NOT_RUN")
        classification[s] = {"feature": d, "status": st, "class": "EXPERIMENTAL-VERIFIED" if st == "PASS" else "EXPERIMENTAL-UNSUPPORTED/UNPROVEN"}
        R.append({"check": f"EXPERIMENTAL (reported, not required): {d}", "status": "PASS" if st == "PASS" else "WARNING", "detail": st})
    finish("12_acceptance", R, {"versions": versions(), "stage_status": stages, "stage_seconds": timing, "experimental_classification": classification,
                                "note": "WARNING on an experimental feature means NOT verified; it is never a success."})
    return 0


if __name__ == "__main__":
    run_guarded("12_acceptance", main)
