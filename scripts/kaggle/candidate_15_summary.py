"""Stage `summary`: apply the cross-stage rules. A missing stage report is NOT_RUN (never a pass); only PASS passes; all stages must describe the same wheel and runtime."""
import json, pathlib
from _common import OUT, check, finish, error
from candidate_15_common import *   # noqa: F401,F403

STAGES = ("static", "engine", "server")


def summary() -> int:
    R = []; recs = {}; modes = {}
    for m in STAGES:
        p = OUT / f"15c_{m}.json"
        if not p.is_file():
            R.append({"check": f"{m} report exists", "status": "NOT_RUN", "detail": f"{p.name} is missing - the stage did not run or crashed before writing evidence"}); modes[m] = "NOT_RUN"; print(f"NOT_RUN {m}"); continue
        try: recs[m] = json.loads(p.read_text())
        except ValueError as e:
            error(R, f"{m} report is readable JSON", e); modes[m] = "ERROR"; continue
        st = recs[m].get("status"); modes[m] = st
        R.append({"check": f"{m} status is PASS", "status": "PASS" if st == "PASS" else (st if st in ("FAIL", "ERROR", "WARNING", "SKIPPED", "NOT_RUN") else "FAIL"), "detail": st})
        print(f"{R[-1]['status']} {m} status is PASS -- {st}")
    inst = OUT / "prereq-install" / "02_install_integrated.json"
    if inst.is_file():
        ij = json.loads(inst.read_text()); st = ij.get("status")
        R.append({"check": "installer status is PASS", "status": "PASS" if st == "PASS" else (st if st in ("FAIL", "ERROR", "WARNING", "SKIPPED", "NOT_RUN") else "FAIL"), "detail": st}); modes["installer"] = st
    else:
        R.append({"check": "installer report exists", "status": "NOT_RUN", "detail": "prereq-install/02_install_integrated.json missing"}); modes["installer"] = "NOT_RUN"
    provs = {m: r.get("provenance", {}) for m, r in recs.items()}
    if inst.is_file(): provs["installer"] = ij.get("provenance", {}) | {"wheel_sha256": ij.get("wheel_sha256")}
    bundle = json.loads((OUT / "provenance.json").read_text()) if (OUT / "provenance.json").is_file() else {}
    shas = {m: p.get("wheel_sha256") for m, p in provs.items()}
    check(R, "all stages and the installer report the SAME wheel SHA256, equal to the attached bundle's", len({*shas.values()}) == 1 and None not in shas.values() and set(shas.values()) == {bundle.get("wheel_sha256")}, {**shas, "bundle": bundle.get("wheel_sha256")})
    check(R, f"all stages verified build {BUILD} ({WHEEL})", all(p.get("wheel") == WHEEL for p in provs.values()) and bundle.get("wheel") == WHEEL, {m: p.get("wheel") for m, p in provs.items()})
    ver = {m: (r.get("versions") or {}) for m, r in recs.items()}
    keyv = {m: (v.get("python"), v.get("torch"), v.get("triton"), v.get("vllm"), v.get("stokastisk"), v.get("transformers")) for m, v in ver.items() if v}
    check(R, "all stages ran on the same Python / torch / Triton / vLLM / Stokastisk / transformers", len(set(keyv.values())) == 1 and len(keyv) >= 2, keyv)
    check(R, "the release came from the attached input (no remote fallback)", str(bundle.get("source", "")).startswith("attached"), bundle.get("source"))
    # kernel-level backend evidence from the captured stage logs
    for m, need in (("engine", True), ("server", True)):
        lp = OUT / f"15c_{m}.log"
        if lp.is_file():
            t = lp.read_text(errors="replace")
            check(R, f"{m} log: the worker selected AttentionBackendEnum.TRITON_ATTN", "Using AttentionBackendEnum.TRITON_ATTN backend" in t, "found" if "Using AttentionBackendEnum.TRITON_ATTN backend" in t else "line not found")
            check(R, f"{m} log: no Flash backend was selected", not any(k in t for k in ("Using AttentionBackendEnum.FLASH", "Using FlashAttention", "Using FlashInfer")), "none")
        else:
            R.append({"check": f"{m} log exists", "status": "NOT_RUN", "detail": f"{lp.name} missing"})
    overall = "PASS" if all(r["status"] == "PASS" for r in R) else ("FAIL" if any(r["status"] in ("FAIL", "ERROR") for r in R) else "NOT_RUN" if any(r["status"] == "NOT_RUN" for r in R) else "WARNING")
    finish("15c_summary", R, {"modes": modes, "wheel": WHEEL, "bundle": bundle, "overall": overall}, script_file=__file__); return 0
