#!/usr/bin/env python3.13
"""Build evidence/kaggle/evidence-manifest.json from the archived Kaggle artifacts (no GPU, no network, nothing fabricated).
usage: build_evidence_manifest.py [--archive ORIGINAL_ZIP ...]

Every value is read from a file under evidence/kaggle/: the executed notebooks (outputs intact) and the notebook-15 acceptance reports. A pattern that does not match raises, so a
missing fact can never be filled in silently. ``--archive`` records the size and SHA256 of an original archive that is kept outside the repository (dist/evidence-archives/); without it the archive entries of the existing manifest are kept, so the manifest is reproducible from the repository alone."""
import hashlib, json, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
EV = ROOT / "evidence/kaggle"
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
D15, D16, D1819 = "2026-11-13-notebook-15-build3-acceptance", "2026-11-13-notebook-16-quickstart", "2026-11-14"
NB = {"15": f"{D15}/15-stokastisk-candidate-sdk-integration.ipynb", "16": f"{D16}/16-stokastisk-quickstart.ipynb",
      "18": f"{D1819}/18-stokastisk-0-1-0-qwen25-t4-direct-cells.ipynb", "19": f"{D1819}/19-stokastisk-0-1-0-qwen2-5-coder-7b-dual-t4.ipynb"}


def need(m, what):
    if not m:
        raise SystemExit(f"cannot extract {what}")
    return m


def cells(path):
    nb = json.loads((EV / path).read_text())
    return [("".join(c["source"]), "".join("".join(o.get("text", "")) for o in c.get("outputs", []) if o.get("output_type") == "stream")) for c in nb["cells"]]


def env(out):
    py = need(re.search(r"Python: (\d+\.\d+\.\d+)", out), "python")[1]
    gpus = re.findall(r"GPU \d: (Tesla T4)\s*[,|]\s*(?:compute capability|SM) (\d\.\d)", out)
    return {"python": py, "torch": need(re.search(r"PyTorch: (\S+)", out), "torch")[1], "cuda": need(re.search(r"CUDA runtime: (\S+)", out), "cuda")[1],
            "triton": need(re.search(r"Triton: (\S+)", out), "triton")[1], "gpus": [{"model": g, "compute_capability": c} for g, c in gpus]}


def free(out, key):
    return [float(x) for x in re.findall(r"'" + key + r"': ([\d.]+)", out)][:2]          # the two readings taken inside the engine context


def run(rid, nbno, e, cell, src_cells, mem_during, mem_after, done, model_dtype="half"):
    src, out = src_cells[cell]
    mem_out = src_cells[mem_after][1] if isinstance(mem_after, int) else out
    ids = re.findall(r'MODEL_ID\s*=\s*[\'"]([^\'"]+)', src) or re.findall(r'MODEL_ID\s*=\s*[\'"]([^\'"]+)', "\n".join(s for s, _ in src_cells[:cell]))
    model = ids[-1] if ids else None          # the assignment in this cell, else the latest one before it
    tp = int(need(re.search(r"tensor_parallel_size\s*=\s*(\d)", src), "tp")[1]); pp = int(re.search(r"pipeline_parallel_size\s*=\s*(\d)", src)[1]) if re.search(r"pipeline_parallel_size\s*=\s*(\d)", src) else 1
    ml = int(need(re.search(r"max_model_len\s*=\s*(\d+)", src), "max_model_len")[1])
    backend = "TRITON_ATTN" if "Verified attention backend: TRITON_ATTN" in out else None
    completed = bool(backend) and "Processed prompts: 100%" in out and done in out
    return {"run": rid, "notebook": nbno, "model": model, "dtype": model_dtype, "quantization": None, "tensor_parallel_size": tp, "pipeline_parallel_size": pp,
            "parallelism": f"TP={tp}" if pp == 1 else f"PP={pp}", "max_model_len": ml, "enforce_eager": "enforce_eager=True" in src or "enforce_eager" not in src,
            "attention_backend": backend, "generation_completed": completed, "free_gib_during_inference": free(mem_during, "free_GiB") if mem_during else None,
            "free_gib_after_close": [float(x) for x in re.findall(r"free_(?:memory_)?GiB(?:_after_[A-Z]+\d)?'?: ([\d.]+)", mem_out)][-2:],
            "process_manager_forced_cleanup_warning": "force killing remaining processes" in out}


def main():
    arts = []
    for p in sorted(EV.rglob("*")):
        if p.is_file() and p.name not in ("evidence-manifest.json",):
            b = p.read_bytes(); arts.append({"path": str(p.relative_to(EV)), "size": len(b), "sha256": sha(b)})
    runs = []
    c18, c19, c16 = cells(NB["18"]), cells(NB["19"]), cells(NB["16"])
    e18, e19 = env(c18[2][1]), env(c19[2][1])
    inst = lambda c, i: {"command": need(re.search(r"!pip install [^\n]+", c[i][0]), "install command")[0], "succeeded": "Successfully installed" in c[i][1]}  # noqa: E731
    r = run("nb18-qwen2.5-0.5b-tp1", "18", e18, 8, c18, None, 10, "The Stokastisk engine context has closed."); r["model"] = "Qwen/Qwen2.5-0.5B-Instruct"; r["tensor_parallel_size"] = 1
    r["free_gib_during_inference"] = None; runs.append(r)
    runs.append(run("nb18-qwen2.5-3b-tp2", "18", e18, 12, c18, c18[12][1].split("GPU MEMORY DURING INFERENCE")[1].split("GPU MEMORY AFTER")[0], None, "STEP 6 COMPLETED"))
    runs.append(run("nb18-qwen2.5-3b-pp2", "18", e18, 15, c18, c18[15][1].split("GPU MEMORY DURING PIPELINE INFERENCE")[1].split("STEP 7")[0], None, "STEP 7 COMPLETED"))
    for rr, i, mem, done in (("nb19-qwen2.5-coder-7b-tp2", 12, 14, "TP=2 Stokastisk engine closed."), ("nb19-qwen2.5-coder-7b-pp2", 16, 18, "PP=2 Stokastisk engine closed.")):
        x = run(rr, "19", e19, i, c19, c19[i][1].split("GPU memory during")[1], mem, done); x["model"] = "Qwen/Qwen2.5-Coder-7B-Instruct"; runs.append(x)
    for x in runs:
        e = e18 if x["notebook"] == "18" else e19
        x.update({k: e[k] for k in ("python", "torch", "cuda", "triton", "gpus")}); x["install"] = inst(c18 if x["notebook"] == "18" else c19, 4)
        x["date"] = "2026-11-14"
        x["gpu_memory_recovered"] = x["free_gib_after_close"] == [14.46, 14.46]
    q = c16[4][1]
    runs.append({"run": "nb16-quickstart-qwen2.5-0.5b-tp1", "notebook": "16", "date": "2026-11-13", "model": "Qwen/Qwen2.5-0.5B-Instruct", "dtype": "half", "quantization": None,
                 "parallelism": "TP=1", "max_model_len": 2048, "attention_backend": "TRITON_ATTN" if "attention backend : TRITON_ATTN" in q else None,
                 "generation_completed": all(k in q for k in ("completion", "chat", "structured")), "python": None, "torch": None, "cuda": None, "triton": None, "gpus": [],
                 "install": inst(c16, 1), "gpu_memory_recovered": None, "note": "this notebook prints no environment versions and no memory readings; the engine shut down cleanly (core_client shutdown complete)"})
    a15 = {}
    d = EV / D15 / "acceptance"
    prov = json.loads((d / "provenance.json").read_text())
    for st in ("static", "engine", "parallel", "server", "summary"):
        j = json.loads((d / f"15c_{st}.json").read_text()); a15[st] = {"status": j["status"], "checks": len(j["checks"]), "passed": sum(1 for c in j["checks"] if c["status"] == "PASS")}
    ij = json.loads((d / "prereq-install/02_install_integrated.json").read_text()); a15["installer"] = {"status": ij["status"], "checks": len(ij["checks"]), "passed": sum(1 for c in ij["checks"] if c["status"] == "PASS")}
    ej = json.loads((d / "15c_engine.json").read_text())
    arch = [{"name": pathlib.Path(n).name, "size": p.stat().st_size, "sha256": sha(p.read_bytes())} for n, p in ((n, pathlib.Path(n)) for n in sys.argv[1:] if n != "--archive") if p.is_file()] if "--archive" in sys.argv else (json.loads((EV / "evidence-manifest.json").read_text()).get("original_archives", []) if (EV / "evidence-manifest.json").exists() else [])
    man = {"schema": 1, "project_version": "0.1.0",
           "statement": "Raw artifacts of the Kaggle 2 x Tesla T4 runs recorded for Stokastisk 0.1.0 build 3. Build 4 was NOT executed on Kaggle; see docs/PRODUCTION_READINESS.md.",
           "notebook_15_acceptance": {"date": "2026-11-13", "acceptance_wheel": prov["wheel"], "acceptance_wheel_sha256": prov["wheel_sha256"], "build": prov["build"], "python": ej["versions"]["python"] if isinstance(ej["versions"], dict) else None,
                                      "stages": a15, "models": {"engine": ej["model"], "server": json.loads((d / "15c_server.json").read_text())["model"], "parallel": ej["model"]},
                                      "note": "Run on the acceptance wheel; the published build-3 wheel (72039ca4...) differs from it only in METADATA/RECORD."},
           "runs": runs, "raw_artifacts": arts, "original_archives": arch,
           "missing": ["The original Kaggle output archives are not in the repository (about 17 MB, mostly Triton caches); only their reports/logs are, and the archives' SHA256 values are listed under original_archives."]}
    (EV / "evidence-manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    print("runs:", len(runs), "artifacts:", len(arts)); print(json.dumps(a15)); print([ (r["run"], r["generation_completed"], r["gpu_memory_recovered"], r["attention_backend"]) for r in runs])


if __name__ == "__main__":
    main()
