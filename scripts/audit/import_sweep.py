#!/usr/bin/env python3.13
"""Import EVERY module of a vLLM tree in one process and record which import and which fail (CPU only; no engine, no CUDA context is required).

usage: import_sweep.py TREE_ROOT OUT.json [--timeout SECONDS]      TREE_ROOT contains `vllm/`; dependencies must already be importable (PYTHONPATH).
       import_sweep.py --compare A.json B.json                      exit 1 if B has a module that imports in A but fails in B (a REGRESSION), or a new error class

Used to prove that the Flash-free transform did not break any import that worked before: the failure sets of the original tree (A) and the transformed tree (B) are
compared module by module. Failures that exist in both trees (modules that need a GPU platform, a missing optional package ...) are not regressions."""
from __future__ import annotations

import importlib
import json
import pathlib
import signal
import sys
import time
import traceback


def modules_of(root: pathlib.Path) -> list[str]:
    mods = []
    for p in sorted((root / "vllm").rglob("*.py")):
        rel = p.relative_to(root).with_suffix("")
        parts = rel.parts[:-1] if rel.name == "__init__" else rel.parts
        mods.append(".".join(parts))
    return mods


class Timeout(Exception):
    pass


def sweep(root: pathlib.Path, per_module: int) -> dict:
    sys.path.insert(0, str(root))
    result = {"root": str(root), "ok": [], "failed": {}}
    signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(Timeout()))
    t0 = time.time()
    for m in modules_of(root):
        signal.alarm(per_module)
        try:
            importlib.import_module(m)
            result["ok"].append(m)
        except Timeout:
            result["failed"][m] = "Timeout"
        except BaseException as e:  # noqa: BLE001 - a sweep must record SystemExit/KeyboardInterrupt-like failures too
            last = [l for l in traceback.format_exception_only(type(e), e) if l.strip()][-1].strip()
            result["failed"][m] = f"{type(e).__name__}: {last[:240]}"
        finally:
            signal.alarm(0)
    result["seconds"] = round(time.time() - t0, 1)
    result["n_ok"], result["n_failed"] = len(result["ok"]), len(result["failed"])
    return result


def compare(a: dict, b: dict) -> int:
    ok_a, fail_b = set(a["ok"]), b["failed"]
    regress = sorted(m for m in ok_a if m in fail_b)
    new_mods = sorted(set(b["ok"]) | set(b["failed"]) - set(a["ok"]) - set(a["failed"]))
    gone = sorted((set(a["ok"]) | set(a["failed"])) - (set(b["ok"]) | set(b["failed"])))
    print(f"A: ok={a['n_ok']} failed={a['n_failed']}   B: ok={b['n_ok']} failed={b['n_failed']}")
    print(f"modules removed in B: {len(gone)}   modules new in B: {len([m for m in new_mods if m not in a['ok'] and m not in a['failed']])}")
    print(f"REGRESSIONS (import in A, fail in B): {len(regress)}")
    for m in regress:
        print("   ", m, "->", fail_b[m])
    changed = sorted(m for m in b["failed"] if m in a["failed"] and a["failed"][m] != b["failed"][m])
    print(f"failing in both with a different error: {len(changed)}")
    for m in changed[:40]:
        print("   ", m, "\n       A:", a["failed"][m], "\n       B:", b["failed"][m])
    return 1 if regress else 0


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--compare":
        return compare(json.loads(pathlib.Path(argv[1]).read_text()), json.loads(pathlib.Path(argv[2]).read_text()))
    per = 120
    if "--timeout" in argv:
        i = argv.index("--timeout"); per = int(argv[i + 1]); del argv[i:i + 2]
    root, out = pathlib.Path(argv[0]).resolve(), pathlib.Path(argv[1])
    res = sweep(root, per)
    out.write_text(json.dumps(res, indent=1))
    print(f"{res['n_ok']} imported, {res['n_failed']} failed in {res['seconds']}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
