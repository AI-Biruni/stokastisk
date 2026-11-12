#!/usr/bin/env python3.13
"""Static validation of the fourteen Kaggle notebooks. No GPU, no network, no notebook execution.

usage: validate_notebooks.py [--notebooks DIR] [--json OUT]
Checks (each notebook): valid nbformat 4 JSON; cell structure; no stored outputs/execution counts (the author has NOT run them);
every code cell parses (IPython '!'/'%' lines are handled); names are defined by an earlier cell/import/builtin before use;
RELEASE constant equals the locked release; no older Stokastisk release, no credential-looking strings, no unpinned
'pip install' of torch/triton/vllm/nccl; accelerator/prerequisite/status wording present; the set is exactly 01..14.
Exit 0 only when every check passes. This validates STRUCTURE ONLY; it is not Kaggle validation."""
import argparse, ast, builtins, json, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
LOCKED = (ROOT / "VERSION").read_text().strip()
RELEASE = f"stokastisk-{LOCKED}-kaggle-t4-sm75"
EXPECTED = ["01_stokastisk_environment_audit", "02_stokastisk_installation", "03_stokastisk_single_gpu_validation",
            "04_stokastisk_dual_gpu_nccl_validation", "05_stokastisk_vllm_inference", "06_stokastisk_tensor_parallel_benchmark",
            "07_stokastisk_gpu_topology_validation", "08_stokastisk_pipeline_parallel_validation",
            "09_stokastisk_dual_gpu_replica_inference", "10_stokastisk_cuda_communication_benchmark",
            "11_stokastisk_gpu_load_balancing_benchmark", "12_stokastisk_consolidated_acceptance",
            "13_stokastisk_tp_pp_fulfilment", "14_stokastisk_replica_balance"]
SECRET_RE = re.compile(r"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|KAGGLE_KEY\s*=\s*['\"][0-9a-f]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY")
OLD_REL_RE = re.compile(r"stokastisk-(\d+\.\d+\.\d+)(-rc\d+)?-kaggle-t4")
PIP_BAD_RE = re.compile(r"pip\s+install[^\n]*\b(torch|triton|vllm|nvidia-nccl|flash[-_]?attn|flashinfer)\b", re.I)
PEND = ("PENDING", "UNEXECUTED", "NOT EXECUTED", "not executed", "KAGGLE-EXECUTED")   # a notebook either states it is pending or states which Kaggle run it records
INTEGRATED = ["15_stokastisk_candidate_sdk_integration"]
RELEASE_INT = f"stokastisk-{LOCKED}-integrated-kaggle-t4-sm75"


def code_of(cell):
    lines = []
    for l in "".join(cell.get("source", [])).splitlines():
        lines.append("pass" if l.lstrip().startswith(("!", "%")) else l)
    return "\n".join(lines)


def defined_names(tree):
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)): names.add(n.name)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)): names.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names: names.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.arg): names.add(n.arg)
        elif isinstance(n, ast.ExceptHandler) and n.name: names.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)): names.update(n.names)
    return names


def check_notebook(path, release=None):
    release = release or RELEASE
    errs, warns = [], []
    try:
        nb = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return [f"invalid JSON: {e}"], warns
    if nb.get("nbformat") != 4: errs.append(f"nbformat {nb.get('nbformat')} != 4")
    cells = nb.get("cells")
    if not isinstance(cells, list) or not cells: return errs + ["no cells"], warns
    if cells[0].get("cell_type") != "markdown": errs.append("first cell must be a markdown cell stating prerequisites/status")
    head = "".join(cells[0].get("source", []))
    if not any(w in head for w in PEND): errs.append("first cell must state the Kaggle verification is pending/unexecuted")
    if not re.search(r"T4", head): errs.append("first cell must name the required accelerator (T4)")
    defined = set(dir(builtins)) | {"get_ipython", "display", "__file__", "__name__"}
    release_seen = False
    for i, c in enumerate(cells):
        if c.get("cell_type") not in ("markdown", "code", "raw"): errs.append(f"cell {i}: unknown type")
        if c.get("cell_type") != "code": continue
        if c.get("outputs"): errs.append(f"cell {i}: has stored outputs (notebook must be unexecuted)")
        if c.get("execution_count") not in (None, 0): errs.append(f"cell {i}: execution_count set")
        src = "".join(c.get("source", []))
        if SECRET_RE.search(src): errs.append(f"cell {i}: credential-looking string")
        for m in OLD_REL_RE.finditer(src):
            if m.group(1) != LOCKED or m.group(2): errs.append(f"cell {i}: references another Stokastisk release {m.group(0)}")
        if PIP_BAD_RE.search(src): errs.append(f"cell {i}: pip-installs a protected package (torch/triton/vllm/nccl/flash)")
        for m in re.finditer(r'RELEASE\s*=\s*"([^"]+)"', src):
            release_seen = True
            if m.group(1) != release: errs.append(f"cell {i}: RELEASE={m.group(1)!r}, expected {release!r}")
        try:
            tree = ast.parse(code_of(c))
        except SyntaxError as e:
            errs.append(f"cell {i}: syntax error line {e.lineno}: {e.msg}"); continue
        # names used at module level of this cell must be defined earlier or within this cell (function bodies see later globals)
        local = defined_names(tree)
        for n in ast.walk(tree):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in defined and n.id not in local:
                errs.append(f"cell {i}: name {n.id!r} is used but never defined in this or an earlier cell")
        defined |= local
    if not release_seen and not path.name.startswith("01_"): warns.append("no RELEASE constant (cannot pin the release)")
    return sorted(set(errs)), warns


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--notebooks", default=str(ROOT / "notebooks" / "kaggle-t4-sm75" / "notebooks")); ap.add_argument("--json")
    a = ap.parse_args(argv); d = pathlib.Path(a.notebooks)
    found = sorted(p.stem for p in d.glob("*.ipynb")); result = {"locked_version": LOCKED, "release": RELEASE, "kaggle_execution": "NOT_RUN", "notebooks": {}}
    set_errs = []
    if found != EXPECTED: set_errs.append(f"notebook set is {found}, expected {EXPECTED}")
    ok = not set_errs
    for stem in found:
        errs, warns = check_notebook(d / f"{stem}.ipynb")
        result["notebooks"][stem] = {"static_validation": "PASS" if not errs else "FAIL", "kaggle_status": "NOT_RUN", "errors": errs, "warnings": warns}
        ok &= not errs
        print(("PASS " if not errs else "FAIL ") + stem + "".join(f"\n    - {e}" for e in errs) + "".join(f"\n    ! {w}" for w in warns))
    idir = d.parent / "integrated"
    if d == ROOT / "notebooks" / "kaggle-t4-sm75" / "notebooks":
        ifound = sorted(p.stem for p in idir.glob("*.ipynb"))
        if ifound != INTEGRATED: set_errs.append(f"integrated notebook set is {ifound}, expected {INTEGRATED}"); ok = False
        for stem in ifound:
            errs, warns = check_notebook(idir / f"{stem}.ipynb", RELEASE_INT)
            result["notebooks"][stem] = {"static_validation": "PASS" if not errs else "FAIL", "kaggle_status": "NOT_RUN", "errors": errs, "warnings": warns}
            ok &= not errs
            print(("PASS " if not errs else "FAIL ") + stem + "".join(f"\n    - {e}" for e in errs) + "".join(f"\n    ! {w}" for w in warns))
    for e in set_errs: print("FAIL set:", e)
    result["set_errors"] = set_errs; result["overall"] = "STATIC_PASS" if ok else "STATIC_FAIL"
    if a.json: pathlib.Path(a.json).write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nstatic validation: {result['overall']} (Kaggle execution: NOT_RUN for all notebooks)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
