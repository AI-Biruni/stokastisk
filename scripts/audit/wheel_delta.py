#!/usr/bin/env python3.13
"""Member-by-member delta between two integrated Stokastisk wheels (e.g. the published build 3 and a new build).
usage: wheel_delta.py OLD_WHEEL NEW_WHEEL OUT.json [--allow-sdk stokastisk/compat.py ...] [--docstring-only stokastisk/serving.py ...] [--ast-identical stokastisk/compat.py:check_dtype,check_feature ...] [--literals-only stokastisk/serving.py ...]

Classifies every member (vllm/**, native binaries, stokastisk/**, dist-info, data) and reports identical / changed / added / removed with sha256 values. The result is PASS only when:
all vllm/** members and all native objects are byte-identical, no Flash-family path or binary exists in the new wheel, METADATA differs only in its description and the build/WHEEL tag
(the Requires-Dist set, i.e. every dependency pin, is identical), and the only changed stokastisk/** files are those named with --allow-sdk; every --docstring-only file differs from the old one in docstrings only (identical AST once docstrings are
removed - so no executable line changed); every --literals-only file differs from the old one in string constants only (identical AST once every string constant is blanked - so no code,
name or control flow changed; the changed literals are listed in the report); and every name listed with --ast-identical (the functions/constants the runtime imports) has an identical AST in both versions. Exit 1 otherwise."""
import ast, hashlib, json, pathlib, re, sys, zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sdk"))
from stokastisk import prohibited  # noqa: E402

sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
DI = "stokastisk-0.1.0.dist-info"


def requires(meta: str) -> list[str]:
    return sorted(l for l in meta.split("\n\n", 1)[0].splitlines() if l.startswith(("Requires-Dist:", "Provides-Extra:", "Requires-Python:")))


def header_without_description(meta: str) -> list[str]:
    return [l for l in meta.split("\n\n", 1)[0].splitlines() if not l.startswith(("Requires-Dist:", "Provides-Extra:"))]


def _strip_docstrings(tree):
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant) \
                and isinstance(n.body[0].value.value, str):
            n.body = n.body[1:] or [ast.Pass()]
    return tree


def _top_level(tree):
    out = {}
    for n in tree.body:
        names = [n.name] if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) else [t.id for t in getattr(n, "targets", []) if isinstance(t, ast.Name)] + ([n.target.id] if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) else [])
        for name in names:
            out[name] = ast.dump(n)
    return out


def _blank_strings(tree):
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            n.value = ""
    return tree


def _strings(tree):
    return sorted({n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)})


def delta(old_path, new_path, allow_sdk, docstring_only=(), ast_identical=(), literals_only=()):
    a, b = zipfile.ZipFile(old_path), zipfile.ZipFile(new_path)
    an, bn = set(a.namelist()), set(b.namelist())
    changed = sorted(n for n in an & bn if a.read(n) != b.read(n))
    added, removed = sorted(bn - an), sorted(an - bn)
    cat = lambda n: "vllm" if n.startswith("vllm/") else "sdk" if n.startswith("stokastisk/") else "dist-info" if ".dist-info/" in n else "other"  # noqa: E731
    natives = sorted(n for n in an | bn if n.startswith("vllm/") and prohibited.is_binary(n))
    row = lambda n: {"member": n, "old_sha256": sha(a.read(n)) if n in an else None, "new_sha256": sha(b.read(n)) if n in bn else None}  # noqa: E731
    summary = {"members_old": len(an), "members_new": len(bn), "identical": len(an & bn) - len(changed), "changed": len(changed), "added": len(added), "removed": len(removed),
               "by_category_changed": {c: [n for n in changed if cat(n) == c] for c in ("vllm", "sdk", "dist-info", "other")},
               "vllm_members": sum(1 for n in bn if cat(n) == "vllm"), "vllm_members_identical": sum(1 for n in bn if cat(n) == "vllm" and n in an and a.read(n) == b.read(n)),
               "native_binaries": len(natives), "native_binaries_identical": sum(1 for n in natives if n in an and n in bn and a.read(n) == b.read(n))}
    checks = []

    def chk(name, ok, detail=""):
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": str(detail)[:500]}); print(("PASS " if ok else "FAIL ") + name + ("" if ok else f" -- {str(detail)[:200]}"))

    chk("no member added or removed", not added and not removed, (added + removed)[:5])
    chk("every vllm/** member is byte-identical", not summary["by_category_changed"]["vllm"] and summary["vllm_members"] == summary["vllm_members_identical"], summary["by_category_changed"]["vllm"][:5])
    chk("every native object (.so/.cubin/.ptx/...) is byte-identical", summary["native_binaries"] == summary["native_binaries_identical"] and summary["native_binaries"] > 0, summary["native_binaries"])
    chk("the bundled vllm-0.25.1 dist-info under .data/ is byte-identical", not [n for n in changed if ".data/" in n], [n for n in changed if ".data/" in n])
    unexpected = [n for n in summary["by_category_changed"]["sdk"] if n not in set(allow_sdk)]
    chk("only the explicitly allowed stokastisk/** files changed", not unexpected, unexpected)
    chk("every allowed SDK file really changed (the allow-list is not padding)", all(n in changed for n in allow_sdk), [n for n in allow_sdk if n not in changed])
    chk("only METADATA, WHEEL and RECORD changed in the dist-info", set(summary["by_category_changed"]["dist-info"]) <= {f"{DI}/METADATA", f"{DI}/WHEEL", f"{DI}/RECORD"}, summary["by_category_changed"]["dist-info"])
    chk("no other member category changed", not summary["by_category_changed"]["other"], summary["by_category_changed"]["other"])
    ma, mb = a.read(f"{DI}/METADATA").decode(), b.read(f"{DI}/METADATA").decode()
    chk("every dependency pin is identical (Requires-Dist / Provides-Extra / Requires-Python)", requires(ma) == requires(mb), sorted(set(requires(ma)) ^ set(requires(mb)))[:6])
    chk("METADATA headers (name, version, license, ...) are identical apart from the description", header_without_description(ma) == header_without_description(mb), sorted(set(header_without_description(ma)) ^ set(header_without_description(mb)))[:4])
    wa, wb = a.read(f"{DI}/WHEEL").decode(), b.read(f"{DI}/WHEEL").decode()
    strip = lambda w: [l for l in w.splitlines() if not l.startswith("Build:")]  # noqa: E731
    chk("WHEEL differs only in its Build tag", strip(wa) == strip(wb) and wa != wb, (wa, wb))
    ast_report = {}
    for f in docstring_only:
        same = ast.dump(_strip_docstrings(ast.parse(a.read(f)))) == ast.dump(_strip_docstrings(ast.parse(b.read(f))))
        ast_report[f] = {"docstring_only_change": same}
        chk(f"{f}: the change is confined to docstrings (identical AST once docstrings are removed)", same and a.read(f) != b.read(f))
    for f in literals_only:
        ta, tb = ast.parse(a.read(f)), ast.parse(b.read(f))
        same = ast.dump(_blank_strings(ast.parse(a.read(f)))) == ast.dump(_blank_strings(ast.parse(b.read(f))))
        sa, sb = set(_strings(ta)), set(_strings(tb))
        ast_report.setdefault(f, {})["string_literals_only_change"] = same
        ast_report[f]["string_literals_removed"] = sorted(sa - sb); ast_report[f]["string_literals_added"] = sorted(sb - sa)
        chk(f"{f}: the change is confined to string constants (identical AST once every string is blanked)", same and a.read(f) != b.read(f))
    for spec in ast_identical:
        f, names = spec.split(":"); oa, ob = _top_level(_strip_docstrings(ast.parse(a.read(f)))), _top_level(_strip_docstrings(ast.parse(b.read(f))))
        for n in names.split(","):
            chk(f"{f}: {n} is AST-identical in both builds (the runtime imports it)", n in oa and n in ob and oa[n] == ob[n], n)
        ast_report.setdefault(f, {})["top_level_names_changed"] = sorted(n for n in set(oa) | set(ob) if oa.get(n) != ob.get(n))
        ast_report[f]["runtime_names_identical"] = names.split(",")
    sc = prohibited.scan_members(sorted(bn), b.read)
    chk("no Flash-family path, native marker or binding in the new wheel", sc["clean"] and not sc["pycache_dirs"], {k: sc[k] for k in ("path_hits", "native_hits", "binding_hits")})
    status = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"
    return {"status": status, "old_wheel": {"name": pathlib.Path(old_path).name, "sha256": sha(pathlib.Path(old_path).read_bytes()), "size": pathlib.Path(old_path).stat().st_size},
            "new_wheel": {"name": pathlib.Path(new_path).name, "sha256": sha(pathlib.Path(new_path).read_bytes()), "size": pathlib.Path(new_path).stat().st_size},
            "allowed_sdk_changes": sorted(allow_sdk), "ast_analysis": ast_report, "summary": summary, "changed_members": [row(n) for n in changed], "checks": checks}


def main(argv):
    opts = {"--allow-sdk": [], "--docstring-only": [], "--ast-identical": [], "--literals-only": []}
    for k in opts:
        while k in argv:
            i = argv.index(k); opts[k].append(argv[i + 1]); del argv[i:i + 2]
    rep = delta(argv[0], argv[1], opts["--allow-sdk"], opts["--docstring-only"], opts["--ast-identical"], opts["--literals-only"])
    pathlib.Path(argv[2]).write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    print("OVERALL:", rep["status"])
    return 0 if rep["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
