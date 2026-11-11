#!/usr/bin/env python3.13
"""Static inventory and T4/SM75 classification of every Python module in the shipped vLLM 0.25.1 wheel (plus the modules the stage archive has and the wheel dropped).

usage: vllm_inventory.py WHEEL_EXTRACT_DIR STAGE_DIR OUT_JSON
  WHEEL_EXTRACT_DIR  directory holding the extracted `vllm/` package of the Kaggle-executed wheel
  STAGE_DIR          the extracted stage.zip `stage/` directory (raw vLLM tree; used only to list removed modules)

This is SOURCE INSPECTION ONLY. Nothing is imported, no GPU is touched and no test is run. A class is never "SUPPORTED" because a module merely
imports: SUPPORTED requires direct Kaggle execution evidence (see KAGGLE_EVIDENCE). Classes:
SUPPORTED, SUPPORTED_WITH_CONDITIONS, NEEDS_ADAPTATION, NOT_SUPPORTED_SM75, EXCLUDED_FLASH_DEPENDENCY, NOT_IMPLEMENTED, UNKNOWN_REQUIRES_EVIDENCE."""
import ast, collections, json, pathlib, re, sys

FLASH = ("flashinfer", "flash_attn", "vllm_flash_attn", "flashmla", "flash_mla")
# Files whose Triton kernels were compiled and launched on Kaggle 2xT4 (found as .cubin names in the executed acceptance ZIPs, mapped to source by `def <kernel>`).
KAGGLE_KERNEL_FILES = {
    "v1/attention/ops/triton_unified_attention.py", "v1/attention/ops/triton_reshape_and_cache_flash.py", "v1/sample/ops/topk_topp_triton.py",
    "v1/worker/gpu/structured_outputs.py", "v1/worker/gpu/buffer_utils.py", "v1/worker/gpu/sample/bad_words.py", "v1/worker/gpu/sample/logit_bias.py",
    "v1/worker/gpu/sample/penalties.py", "v1/worker/gpu/input_batch.py", "v1/worker/gpu/block_table.py", "v1/worker/gpu/sample/gumbel.py",
    "v1/worker/gpu/sample/min_p.py", "v1/worker/gpu/sample/prompt_logprob.py", "v1/worker/gpu/sample/logprob.py"}
# Modules on the path the executed notebooks exercise (vllm.LLM offline generation of Qwen2.5 at TP=1, TP=2, PP=2 with TRITON_ATTN; mp executor; patched sampler/communicator).
KAGGLE_PATH_FILES = {
    "entrypoints/llm.py": "notebooks 05/06/13 construct vllm.LLM and generate",
    "v1/attention/backends/triton_attn.py": "TRITON_ATTN selected and confirmed in notebooks 05, 13",
    "model_executor/models/qwen2.py": "Qwen2.5-0.5B/1.5B-Instruct generated in notebooks 05, 06, 08, 09, 11, 13, 14",
    "distributed/device_communicators/cuda_communicator.py": "TP=2 NCCL path (notebooks 05, 06, 13); patched 0003",
    "v1/sample/ops/topk_topp_sampler.py": "sampling on the default path; patched 0003",
}
# Name patterns that need hardware newer than SM75 (static; names only, confirmed per-file by cap-gate scan where available).
SM80_PLUS = re.compile(r"(deep_?gemm|deep_?ep|cutlass_(?:moe|w4a8|fp4)|nvfp4|mxfp4|mxfp8|fp4|trtllm|blackwell|hopper|sm90|sm100|sm120|cutedsl|marlin|machete|nixl|mnnvl|pplx|nvlink_(?:one|two)_sided|fbgemm|experts_int8|fp_quant|fa3|fa4)", re.I)
NON_CUDA = re.compile(r"(^|/)(rocm|xpu|hpu|tpu|neuron|spyre)|_rocm|rocm_|_xpu\b|xpu_|cpu_attn|/cpu/|aiter", re.I)
PLATFORM_DIRS = ("platforms/rocm.py", "platforms/xpu.py", "platforms/cpu.py", "platforms/tpu.py")
CAP = re.compile(r"(?:has_device_capability|is_device_capability(?:_family)?|get_min_capability|has_compute_capability)\s*\(\s*(?:cls\s*\)\s*:?\s*return\s*)?\(?\s*(\d+)")
RETCAP = re.compile(r"def get_min_capability\(cls\)[^\n]*\n(?:\s+(?:\"\"\"[\s\S]*?\"\"\"|#[^\n]*)\n)*\s+return\s+(\d+)")


def is_flash_lib(mod: str) -> bool:
    """True for an import of an EXTERNAL flash library or of the vendored vllm_flash_attn / flashmla packages. Imports of inert helper modules that merely
    probe lazily (vllm.utils.flashinfer, fa_utils, ...) are NOT flash dependencies; absent intra-vllm modules are reported separately as broken imports."""
    root = mod.split(".")[0]
    return root in ("flashinfer", "flash_attn", "flash_mla", "flashmla") or mod.startswith(("vllm.vllm_flash_attn", "vllm.third_party.flashmla"))


def modname(rel: str) -> str:
    m = rel[:-3].replace("/", ".")
    return "vllm" + ("" if m == "__init__" else "." + m).replace(".__init__", "")


def analyse(path: pathlib.Path, rel: str):
    src = path.read_text(errors="replace")
    d = {"loc": src.count("\n") + 1, "flash_refs": [], "flash_top_level": False, "flash_guarded": False, "uncond_imports": [], "triton_jit": 0, "top_imports": [], "all_imports": [], "cap_gates": [], "main_guard": False, "purpose": ""}
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        d["syntax_error"] = str(e); return d
    doc = ast.get_docstring(tree) or ""
    d["purpose"] = doc.strip().splitlines()[0][:110] if doc.strip() else ""
    pkg = modname(rel).split(".")
    if not rel.endswith("__init__.py"): pkg = pkg[:-1]

    def resolve(node):
        out = []
        if isinstance(node, ast.Import):
            out = [(a.name, []) for a in node.names]
        else:
            base = node.module or ""
            if node.level:
                up = pkg[: len(pkg) - (node.level - 1)]
                base = ".".join(up + ([base] if base else []))
            out = [(base, [a.name for a in node.names])]
        return out

    def is_type_checking(n):
        return isinstance(n, ast.If) and ((isinstance(n.test, ast.Name) and n.test.id == "TYPE_CHECKING") or (isinstance(n.test, ast.Attribute) and n.test.attr == "TYPE_CHECKING"))

    def walk(nodes, top, uncond):
        for n in nodes:
            if is_type_checking(n): continue
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                for mod, names in resolve(n):
                    d["all_imports"].append((mod, names))
                    if top: d["top_imports"].append((mod, names))
                    if top and uncond: d["uncond_imports"].append((mod, names))
                    if is_flash_lib(mod):
                        d["flash_refs"].append(mod + ("" if not names else " :: " + ",".join(names)))
                        if top and uncond: d["flash_top_level"] = True
                        elif top: d["flash_guarded"] = True
            for fld in ("body", "orelse", "finalbody", "handlers"):
                sub = getattr(n, fld, None)
                if isinstance(sub, list):
                    is_blk = isinstance(n, (ast.If, ast.Try, ast.With, ast.ExceptHandler))
                    walk([x for x in sub if isinstance(x, ast.AST)], top and is_blk, uncond and False)
    walk(tree.body, True, True)
    d["triton_jit"] = len(re.findall(r"@triton\.jit|@triton\.autotune|@torch\.library\.triton_op", src))
    d["cap_gates"] = sorted({int(x) for x in CAP.findall(src)} | {int(x) for x in RETCAP.findall(src)})
    d["main_guard"] = bool(re.search(r"if __name__\s*==\s*[\"']__main__[\"']", src))
    return d


def main(wheel_dir, stage_dir, out_json):
    W, S = pathlib.Path(wheel_dir) / "vllm", pathlib.Path(stage_dir) / "vllm"
    wheel_py = sorted(str(p.relative_to(W)) for p in W.rglob("*.py"))
    stage_py = sorted(str(p.relative_to(S)) for p in S.rglob("*.py"))
    removed = sorted(set(stage_py) - set(wheel_py))
    mods = {modname(r): r for r in wheel_py}
    info = {r: analyse(W / r, r) for r in wheel_py}
    for r in removed: info[r] = analyse(S / r, r); info[r]["removed_from_wheel"] = True
    removed_mods = {modname(r) for r in removed}

    def is_module(name): return name in mods
    broken = collections.defaultdict(list)
    graph = {}
    for r in wheel_py:
        deps = set()
        for mod, names in info[r]["top_imports"]:
            if not mod.startswith("vllm"): continue
            unc = (mod, names) in info[r]["uncond_imports"]
            hits = []
            if is_module(mod): hits.append(mod)
            for n in names:
                if is_module(mod + "." + n): hits.append(mod + "." + n)
            if unc and (mod in removed_mods or any(mod + "." + n in removed_mods for n in names)):
                broken[r].append(mod if mod in removed_mods else [mod + "." + n for n in names if mod + "." + n in removed_mods][0])
            deps.update(hits)
        graph[modname(r)] = deps
    # reachable closure from the executed entry points
    roots = ["vllm.entrypoints.llm", "vllm.v1.engine.core", "vllm.v1.engine.llm_engine", "vllm.v1.worker.gpu_worker", "vllm.v1.executor.multiproc_executor",
             "vllm.model_executor.models.qwen2", "vllm.v1.attention.backends.triton_attn", "vllm.distributed.parallel_state"]
    reach, stack = set(), [x for x in roots if x in graph]
    while stack:
        m = stack.pop()
        if m in reach: continue
        reach.add(m); stack.extend(graph.get(m, ()))
    rows = []
    for r in sorted(info):
        d = info[r]; m = modname(r); cls = None; why = ""; ev = "static"
        top = r.split("/")[0] if "/" in r else "(root)"
        if d.get("removed_from_wheel"):
            refs = ",".join(sorted(set(d["flash_refs"]))[:3]) or "none found at module level"
            if any(k in r.lower() for k in ("flash", "fa_")) or d["flash_top_level"]:
                cls, why, ev = "EXCLUDED_FLASH_DEPENDENCY", "removed from the T4 wheel; " + ("unconditional" if d["flash_top_level"] else "flash-named") + " flash import(s): " + refs, "wheel-vs-stage diff + AST"
            else:
                cls, why, ev = "NEEDS_ADAPTATION", ("removed from the T4 wheel by the module-level flash-import rule; its flash import(s) sit inside a module-level try/if (" + refs + "), so the module could be restored with a patch (needs a new vLLM wheel)"), "wheel-vs-stage diff + AST"
        elif r in KAGGLE_KERNEL_FILES:
            cls, why, ev = "SUPPORTED", "Triton kernel(s) from this file were compiled and launched on Kaggle 2xT4 (cubins in the executed acceptance ZIPs)", "kaggle-kernel"
        elif r in KAGGLE_PATH_FILES:
            cls, why, ev = "SUPPORTED", KAGGLE_PATH_FILES[r], "kaggle-path"
        elif r in broken and not d["flash_refs"]:
            cls, why = "NEEDS_ADAPTATION", "top-level import of a module that is absent from the wheel: " + ", ".join(sorted(set(broken[r])))
        elif any(k in r.lower() for k in ("flashinfer", "flashmla", "flash_attn", "vllm_flash_attn", "flash_mla")):
            cls, why = "EXCLUDED_FLASH_DEPENDENCY", "flash-named module kept as an inert adapter; no Flash native code is shipped"
        elif r in PLATFORM_DIRS or NON_CUDA.search(r):
            cls, why = "NOT_SUPPORTED_SM75", "targets a non-CUDA platform (ROCm/XPU/CPU/TPU/HPU), not an NVIDIA T4"
        elif d["flash_top_level"]:
            cls, why = "NEEDS_ADAPTATION", "unguarded top-level flash import: " + ",".join(sorted(set(d["flash_refs"]))[:3])
        elif SM80_PLUS.search(r):
            cls, why = "NOT_SUPPORTED_SM75", "name indicates a feature that needs SM80+/SM89+/SM90+/SM100+ hardware or libraries (static, by file name)"
        elif d["cap_gates"] and min(d["cap_gates"]) >= 80 and d["triton_jit"] == 0 and not any(g < 80 for g in d["cap_gates"]):
            cls, why = "NOT_SUPPORTED_SM75", f"capability gate(s) {d['cap_gates']} with no SM75 alternative in this file (static)"
        elif r.startswith(("model_executor/models/", "models/")) and not r.endswith(("registry.py", "__init__.py", "utils.py", "interfaces.py", "interfaces_base.py", "adapters.py")):
            cls, why = "UNKNOWN_REQUIRES_EVIDENCE", "model implementation; only Qwen2 was run on Kaggle; fp16 on SM75 and Triton attention compatibility not demonstrated for this architecture"
        elif m in reach:
            cls, why = "SUPPORTED_WITH_CONDITIONS", "statically reachable (top-level imports) from the executed LLM / engine / worker / qwen2 / triton_attn entry points; this module itself is not individually proven executed" + (f"; capability gates {d['cap_gates']}" if d["cap_gates"] else ""); ev = "static-import-closure"
        else:
            cls, why = "UNKNOWN_REQUIRES_EVIDENCE", "not reached from the executed entry points by top-level imports; no Kaggle or CPU evidence" + (f"; capability gates {d['cap_gates']}" if d["cap_gates"] else "")
        rows.append({"path": "vllm/" + r, "module": m, "area": top, "class": cls, "evidence": ev, "reason": why, "loc": d["loc"], "triton_jit": d["triton_jit"], "cap_gates": d["cap_gates"],
                     "flash_refs": sorted(set(d["flash_refs"]))[:4], "main_guard": d["main_guard"], "purpose": d["purpose"]})
    counts = collections.Counter(x["class"] for x in rows)
    json.dump({"wheel_py": len(wheel_py), "stage_py": len(stage_py), "removed": removed, "reachable_from_executed_roots": len(reach), "counts": dict(counts), "rows": rows,
               "broken_top_level_imports": {k: sorted(set(v)) for k, v in broken.items()}}, open(out_json, "w"), indent=1)
    print(json.dumps({"wheel_py": len(wheel_py), "stage_py": len(stage_py), "removed": len(removed), "reachable": len(reach), "counts": dict(counts)}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
