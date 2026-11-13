"""Stage `static`: the installed build 3 is the one the user attached, the runtime is untouched, the package contents obey the Flash policy, the dependency closure is complete,
and every CPU-checkable behaviour (config, compat, serving command, backend registry, import guard, structured-output grammar stack) is correct on the REAL Kaggle runtime."""
import hashlib, importlib, importlib.metadata as md, json, os, pathlib, subprocess, sys, traceback
from _common import OUT, check, error, finish, versions, warn
from candidate_15_common import *   # noqa: F401,F403
import _inventory as inv


def static() -> int:
    R = []; extra = {}
    v = versions(); extra["versions"] = v
    check(R, "Python 3.13", sys.version_info[:2] == (3, 13), v.get("python"))
    check(R, "PyTorch 2.11.0+cu130", v.get("torch") == "2.11.0+cu130", v.get("torch"))
    check(R, "Triton 3.6.0", v.get("triton") == "3.6.0", v.get("triton"))
    check(R, "CUDA 13.0 (torch build)", v.get("torch_cuda") == "13.0", v.get("torch_cuda"))
    gp = v.get("gpus") or []
    check(R, "two Tesla T4 GPUs, SM75", len(gp) == 2 and all(tuple(g["capability"]) == (7, 5) and "T4" in g["name"] for g in gp), gp)
    # ---------------------------------------------------------------- provenance: the attached build is the installed build
    prov = json.loads((OUT / "provenance.json").read_text()) if (OUT / "provenance.json").is_file() else None
    check(R, "bundle provenance record exists (notebook cell 1)", bool(prov), "provenance.json" if prov else "missing")
    inst = json.loads(INSTALL_JSON.read_text()) if INSTALL_JSON.is_file() else None
    check(R, "installer report exists and its status is PASS", bool(inst) and inst.get("status") == "PASS", inst and inst.get("status"))
    if prov and inst:
        check(R, "installed wheel SHA256 == attached bundle's wheel SHA256", inst.get("wheel_sha256") == prov.get("wheel_sha256"), (inst.get("wheel_sha256"), prov.get("wheel_sha256")))
        check(R, f"wheel is {WHEEL} (build {BUILD})", prov.get("wheel") == WHEEL and inst.get("wheel") == WHEEL, (prov.get("wheel"), inst.get("wheel")))
        check(R, "the bundle came from the attached input (no remote fallback)", str(prov.get("source", "")).startswith("attached"), prov.get("source"))
        d = inst.get("diff", {})
        check(R, "installer: protected Kaggle packages unchanged and only planned packages added", d.get("ok") is True, {k: d.get(k) for k in ("changed", "removed", "unexpected_added", "expected_but_absent")})
        extra["install"] = {k: inst.get(k) for k in ("wheel_sha256", "build", "plan", "diff", "pip_check_new", "installer_sha256", "downloaded_bytes")}
    # ---------------------------------------------------------------- inventory: ONE implementation, live vs installer snapshots
    live = inv.snapshot()
    (OUT / "15c_package_inventory.txt").write_text("\n".join(f"{n}=={ver}" for n, ver in live["packages"].items()) + "\n")
    extra["inventory"] = {"method": live["method"], "count": live["count"], "duplicates": live["duplicates"]}
    aft = OUT / "prereq-install" / "02_packages_after.json"; bef = OUT / "prereq-install" / "02_packages_before.json"
    if aft.is_file() and bef.is_file():
        after_s, before_s = json.loads(aft.read_text()), json.loads(bef.read_text())
        same = inv.same_inventory(after_s, live["packages"])
        check(R, "live inventory == the installer's AFTER inventory (same method, PEP 503 names)", same["ok"], {k: same[k] for k in ("changed", "removed", "added")} if not same["ok"] else f"{live['count']} packages identical")
        ref = inv.diff(before_s, live["packages"], expected_new=list((inst or {}).get("plan", {}).get("install", {})), own=inv_own())
        check(R, "live inventory vs BEFORE: no pre-installed package changed or removed (incl. dotted names such as jaraco.*, lazr.*, ruamel.yaml)", not ref["changed"] and not ref["removed"], {"changed": ref["changed"], "removed": ref["removed"]} if (ref["changed"] or ref["removed"]) else "identical")
        dotted = sorted(n for n in before_s if n.replace("-", "") != n and n in {"jaraco-classes", "jaraco-context", "jaraco-functools", "lazr-restfulclient", "lazr-uri", "path-py", "ruamel-yaml", "keyrings-google-artifactregistry-auth"})
        check(R, "the 8 distributions build 2 misreported as removed are present", all(n in live["packages"] for n in dotted), {n: live["packages"].get(n) for n in dotted})
    else:
        error(R, "installer inventories exist", "02_packages_before/after.json missing")
    # ---------------------------------------------------------------- imports, API
    try:
        import stokastisk, vllm
        sv, vv = md.version("stokastisk"), md.version("vllm")
        check(R, "Stokastisk version 0.1.0 (module + distribution)", stokastisk.__version__ == sv == "0.1.0", (stokastisk.__version__, sv))
        check(R, "vLLM version 0.25.1 (module + distribution)", vllm.__version__ == vv == "0.25.1", (vllm.__version__, vv))
        check(R, "stokastisk imported from site-packages (not a source tree)", "site-packages" in stokastisk.__file__ or "dist-packages" in stokastisk.__file__, stokastisk.__file__)
        check(R, "a single vLLM package tree", len(list(vllm.__path__)) == 1, list(vllm.__path__))
    except BaseException:  # noqa: BLE001
        error(R, "import stokastisk / vllm", traceback.format_exc()[-800:]); finish("15c_static", R, extra, script_file=__file__); return 1
    names = [n for n in stokastisk.__all__ if n != "SamplingParams"]
    missing = [n for n in names if not hasattr(stokastisk, n)]
    check(R, "every name in stokastisk.__all__ resolves", not missing, missing or f"{len(names)} names")
    for m in ("stokastisk.compat", "stokastisk.serving", "stokastisk.engine", "stokastisk.structured", "stokastisk.dependencies", "stokastisk.prohibited", "stokastisk.parallel",
              "stokastisk.parallel.replica", "stokastisk.topology", "stokastisk.telemetry", "stokastisk.communication", "stokastisk.cuda", "stokastisk.limits", "stokastisk.diagnostics",
              "stokastisk.benchmarks", "stokastisk.nccl"):
        try: importlib.import_module(m); check(R, f"import {m}", True)
        except BaseException: error(R, f"import {m}", traceback.format_exc()[-500:])
    check(R, "LLM.chat / complete / generate_structured are defined", all(callable(getattr(stokastisk.LLM, n, None)) for n in ("chat", "complete", "generate_structured")))
    # ---------------------------------------------------------------- config / compat on the real runtime
    import torch
    cap = torch.cuda.get_device_capability(0)
    check(R, "bfloat16 is rejected on SM75 (not silently accepted)", raises(lambda: stokastisk.check_dtype("bfloat16", capability=cap), stokastisk.StokastiskCompatibilityError))
    check(R, "float16 ('half') is accepted", stokastisk.check_dtype("half") == "half")
    check(R, "LLM(dtype='bfloat16') fails before any engine starts", raises(lambda: stokastisk.LLM(MODEL, config=stokastisk.StokastiskConfig(), dtype="bfloat16"), stokastisk.StokastiskCompatibilityError))
    check(R, "LLM(attention_backend='FLASH_ATTN') fails before any engine starts", raises(lambda: stokastisk.LLM(MODEL, attention_backend="FLASH_ATTN"), stokastisk.StokastiskCompatibilityError))
    check(R, "LLM(structured_outputs_config backend 'auto') fails before any engine starts", raises(lambda: stokastisk.LLM(MODEL, structured_outputs_config={"backend": "auto"}), stokastisk.ConfigurationError))
    check(R, "no engine process was started by the pre-flight checks (GPU memory idle)", all(x < IDLE_MIB for x in (gpu_used_mib() or [10**9])), gpu_used_mib())
    for bad, why in ((dict(tensor_parallel_size=True), "bool as int"), (dict(tensor_parallel_size=3), "TP=3"), (dict(tensor_parallel_size=2, pipeline_parallel_size=2), "TP2 x PP2 on two GPUs"),
                     (dict(structured_output_backend="auto"), "hidden fallback"), (dict(gpu_memory_utilization=True), "bool as number")):
        check(R, f"invalid configuration rejected: {why}", raises(lambda b=bad: stokastisk.StokastiskConfig(**b), stokastisk.ConfigurationError))
    for feat in ("flash_attention", "flashinfer", "flashmla", "direct_upstream_vllm"):
        check(R, f"check_feature({feat!r}) is reported unsupported", raises(lambda f=feat: stokastisk.check_feature(f), stokastisk.StokastiskCompatibilityError))
    fm = stokastisk.feature_matrix()
    check(R, "feature matrix is non-empty and every entry has status/evidence", bool(fm) and all({"status", "evidence"} <= set(r) for r in fm.values()), len(fm))
    try: stokastisk.check_model(MODEL); check(R, f"check_model({MODEL}) does not raise", True)
    except BaseException: error(R, f"check_model({MODEL})", traceback.format_exc()[-500:])
    # ---------------------------------------------------------------- serving command + validation
    from stokastisk import serving
    cmd = stokastisk.build_serve_command(MODEL, stokastisk.StokastiskConfig(max_model_len=2048), port=8000)
    g = lambda flag: cmd[cmd.index(flag) + 1]
    check(R, "serve command pins TRITON_ATTN, fp16 and the xgrammar structured-output backend", g("--attention-backend") == "TRITON_ATTN" and g("--dtype") == "half" and json.loads(g("--structured-outputs-config")) == {"backend": "xgrammar"}, cmd[4:])
    for extra_args in (["--attention-backend", "FLASH_ATTN"], ["--structured-outputs-config", '{"backend":"auto"}']):
        check(R, f"serve refuses the override {extra_args[0]}", raises(lambda a=extra_args: stokastisk.build_serve_command(MODEL, stokastisk.StokastiskConfig(), extra_args=a), stokastisk.StokastiskCompatibilityError))
    for bad in (dict(host="bad host"), dict(port=0), dict(port=70000), dict(port=True)):
        check(R, f"serve refuses {bad}", raises(lambda b=bad: stokastisk.build_serve_command(MODEL, stokastisk.StokastiskConfig(), **b), stokastisk.ConfigurationError))
    check(R, "serve refuses a model id that looks like an option", raises(lambda: stokastisk.build_serve_command("--help", stokastisk.StokastiskConfig()), stokastisk.ConfigurationError))
    try:
        from vllm.utils.argparse_utils import FlexibleArgumentParser
        from vllm.entrypoints.openai.cli_args import make_arg_parser
        ns = make_arg_parser(FlexibleArgumentParser()).parse_args(cmd[cmd.index("serve") + 1:])
        check(R, "vLLM's real argument parser accepts the exact serve command", str(getattr(ns, "attention_backend", "")).endswith("TRITON_ATTN") and ns.structured_outputs_config.backend == "xgrammar", getattr(ns, "attention_backend", None))
    except BaseException: error(R, "vLLM argument parser accepts the serve command", traceback.format_exc()[-700:])
    miss = serving.check_server_requirements(); extra["server_requirements_missing"] = miss
    check(R, "every package `vllm serve` needs is importable (baseline + serving + structured-output locks)", not miss, miss or f"{len(serving.SERVER_REQUIREMENTS)} import names")
    # ---------------------------------------------------------------- backend registry + Flash policy
    from vllm.v1.attention.backends.registry import AttentionBackendEnum as A
    from vllm.t4_backend_policy import ProhibitedBackendError
    check(R, "TRITON_ATTN backend class resolves", A.TRITON_ATTN.get_class().__name__ == "TritonAttentionBackend")
    for n in ("FLASH_ATTN", "FLASHINFER", "FLASHMLA", "FLASH_ATTN_MLA"):
        check(R, f"AttentionBackendEnum.{n}.get_class() raises ProhibitedBackendError", raises(getattr(A, n).get_class, ProhibitedBackendError))
    try:
        from vllm.platforms.cuda import CudaPlatformBase
        from vllm.platforms.interface import DeviceCapability
        from vllm.v1.attention.selector import AttentionSelectorConfig
        valid, invalid = CudaPlatformBase.get_valid_backends(DeviceCapability(7, 5), AttentionSelectorConfig(head_size=64, dtype=torch.float16, kv_cache_dtype="auto", block_size=16))
        names_ = [c.backend.name for c in valid]
        check(R, "CUDA platform auto-selection on SM7.5 puts TRITON_ATTN first and never a Flash backend", bool(names_) and names_[0] == "TRITON_ATTN" and not any("FLASH" in n for n in names_), names_)
        extra["sm75_backend_selection"] = {"valid": names_, "invalid": {k.name: str(v[1])[:120] for k, v in invalid.items()}}
    except BaseException: error(R, "CUDA platform backend selection on SM7.5", traceback.format_exc()[-700:])
    from vllm.config.attention import AttentionConfig
    check(R, "AttentionConfig(backend='FLASH_ATTN') is rejected at construction", raises(lambda: AttentionConfig(backend="FLASH_ATTN"), Exception))
    guard = [m for m in stokastisk.prohibited.BLOCKED_IMPORT_NAMES if sys.modules.get(m, "x") is not None]
    check(R, "Flash import guard: import flashinfer / flash_attn / flash_mla fails explicitly", not guard and raises(lambda: importlib.import_module("flashinfer"), ImportError), guard or "blocked")
    vroot = pathlib.Path(vllm.__file__).parent
    scan = stokastisk.prohibited.scan_tree(vroot); extra["installed_tree_scan"] = {k: v for k, v in scan.items() if k not in ("references", "allowed_flash_named")}; extra["flash_named_but_allowed"] = scan["allowed_flash_named"]
    check(R, "installed vllm tree: 0 Flash-family files (source, bytecode, native), 0 native markers, 0 bindings", scan["clean"], scan["path_hits"][:8] or scan["native_hits"] or scan["binding_hits"] or f"{scan['files']} files: {scan['py']} .py, {scan['pyc']} .pyc, {scan['native']} native")
    pr = stokastisk.prohibited_report(); check(R, "prohibited_report is clean", pr["clean"], pr)
    check(R, "no Flash library mapped in this process", not flat(flash_libs_tree([os.getpid()])), flat(flash_libs_tree([os.getpid()])) or "none")
    # ---------------------------------------------------------------- dependency state
    pc = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True); lines = pc.stdout.strip().splitlines()
    extra["pip_check"] = {"returncode": pc.returncode, "lines": lines}
    before_lines = (inst or {}).get("pip_check_before", [])
    new = [l for l in lines if l not in set(before_lines)]
    check(R, "pip check: no conflict introduced by the installation (none tolerated)", not new, new or f"{len(lines)} pre-existing Kaggle lines unchanged")
    check(R, "pip check: no line mentions xgrammar, stokastisk, vllm, lm-format-enforcer or model-hosting-container-standards", not [l for l in lines if l.split(" ")[0].lower() in ("xgrammar", "stokastisk", "vllm", "lm-format-enforcer", "model-hosting-container-standards")], "clean")
    check(R, "xgrammar 0.2.3 + apache-tvm-ffi 0.1.9 + transformers 5.x (metadata-compatible; no patched distribution)", md.version("xgrammar") == "0.2.3" and md.version("apache-tvm-ffi") == "0.1.9" and md.version("transformers").startswith("5."), (md.version("xgrammar"), md.version("apache-tvm-ffi"), md.version("transformers")))
    from stokastisk import dependencies
    check(R, "dependency lock completeness: every import name of core + structured-output + serving is importable", not dependencies.missing_for(*dependencies.PROFILES), dependencies.missing_for(*dependencies.PROFILES) or "complete")
    # ---------------------------------------------------------------- structured-output stack on CPU with the installed packages
    r = subprocess.run([sys.executable, str(pathlib.Path(__file__).with_name("structured_output_cpu_check.py")), MODEL, "--json", str(OUT / "15c_structured_cpu.json")], capture_output=True, text=True)
    try: sj = json.loads((OUT / "15c_structured_cpu.json").read_text())
    except (OSError, ValueError): sj = None
    check(R, "vLLM XgrammarBackend + xgrammar + transformers: constrained decode yields schema-valid JSON, rejects invalid tokens (CPU; NOT the GPU proof)", r.returncode == 0 and bool(sj) and sj["status"] == "PASS", (sj or {}).get("status") or r.stdout[-400:] + r.stderr[-300:])
    extra["structured_cpu"] = sj
    finish("15c_static", R, extra, script_file=__file__); return 0


def inv_own():
    return ("stokastisk", "vllm")
