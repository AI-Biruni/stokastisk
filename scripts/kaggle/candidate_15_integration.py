"""Stage 15 (candidate): verify the INTEGRATED Stokastisk 0.1.0 wheel (SDK + bundled vLLM 0.25.1) on Kaggle 2 x Tesla T4.
usage: candidate_15_integration.py MODE        MODE: static | engine | server | summary
Each mode is a separate process (a fresh CUDA context and clean memory accounting). Modes write 15c_<MODE>.json; `summary` applies the cross-mode rules.
Honesty rules: a feature that fails is FAIL/ERROR with the traceback; nothing unsupported is turned into PASS; this script never installs anything."""
import hashlib, importlib, importlib.metadata as md, json, os, pathlib, shutil, subprocess, sys, time, traceback
from _common import OUT, check, finish, gpu_used_mib, loaded_libs, process_tree, run_guarded, save, sha256, tree_loaded_libs, versions, wait_children_gone, warn, skipped

MODEL = os.environ.get("STOKASTISK_CANDIDATE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
IDLE_MIB = 500
FLASH_KEYS = ("vllm_flash_attn", "_vllm_fa", "flashinfer", "flashmla")
WHEEL = "stokastisk-0.1.0-2-cp313-cp313-linux_x86_64.whl"
WHEEL_SHA256 = os.environ.get("STOKASTISK_EXPECTED_WHEEL_SHA256", "")      # the notebook passes the manifest value; empty = not enforced here (the installer enforces it)
INSTALL_JSON = OUT / "prereq-install" / "02_install_integrated.json"


def flash_in(libmap):
    return sorted({x for v in libmap.values() if isinstance(v, list) for x in v if any(k in x.lower() for k in FLASH_KEYS)})


def static() -> int:
    R = []; extra = {}
    v = versions(); extra["versions"] = v
    check(R, "Python 3.13", sys.version_info[:2] == (3, 13), v.get("python"))
    check(R, "PyTorch 2.11.0+cu130", v.get("torch") == "2.11.0+cu130", v.get("torch"))
    check(R, "Triton 3.6.0", v.get("triton") == "3.6.0", v.get("triton"))
    check(R, "CUDA 13.0 (torch build)", v.get("torch_cuda") == "13.0", v.get("torch_cuda"))
    gp = v.get("gpus") or []
    check(R, "two Tesla T4 GPUs, SM75", len(gp) == 2 and all(tuple(g["capability"]) == (7, 5) and "T4" in g["name"] for g in gp), gp)
    # installation evidence (written by 02_install_integrated.py in this same session)
    inst = json.loads(INSTALL_JSON.read_text()) if INSTALL_JSON.is_file() else None
    check(R, "integrated installer evidence exists and its status is PASS", bool(inst) and inst.get("status") == "PASS", inst and inst.get("status"))
    if inst:
        extra["install"] = {k: inst.get(k) for k in ("wheel_sha256", "release", "legacy_two_wheel_detected", "already_installed", "diff", "pip_check_new")}
        check(R, "installed wheel SHA256 is recorded", bool(inst.get("wheel_sha256")), inst.get("wheel_sha256"))
        if WHEEL_SHA256: check(R, "installed wheel SHA256 equals the expected published value", inst.get("wheel_sha256") == WHEEL_SHA256, inst.get("wheel_sha256"))
        d = inst.get("diff", {}); check(R, "protected Kaggle packages were not modified (installer before/after diff)", d.get("ok") is True, d.get("preinstalled_changed_or_removed") or "identical")
        for c in inst.get("checks", []):
            if c["status"] == "WARNING": warn(R, "installer: " + c["check"], c["detail"])
    # live protected-package check independent of the installer
    if (OUT / "prereq-install" / "02_packages_before.json").is_file():
        before = json.loads((OUT / "prereq-install" / "02_packages_before.json").read_text())
        now = {d.metadata["Name"].lower().replace("_", "-"): d.version for d in md.distributions() if d.metadata["Name"]}
        changed = {n: (ver, now.get(n)) for n, ver in before.items() if now.get(n) != ver and n not in ("vllm", "stokastisk")}
        check(R, "no preinstalled package changed since the pre-install snapshot (live comparison)", not changed, changed or "identical")
    try:
        import stokastisk, vllm
        sv, vv = md.version("stokastisk"), md.version("vllm")
        check(R, "Stokastisk version 0.1.0 (module + distribution)", stokastisk.__version__ == sv == "0.1.0", (stokastisk.__version__, sv))
        check(R, "vLLM version 0.25.1 (module + distribution)", vllm.__version__ == vv == "0.25.1", (vllm.__version__, vv))
        check(R, "stokastisk imported from site-packages (not a source tree)", "site-packages" in stokastisk.__file__ or "dist-packages" in stokastisk.__file__, stokastisk.__file__)
        check(R, "a single vLLM package tree", len(list(vllm.__path__)) == 1, list(vllm.__path__))
    except BaseException:  # noqa: BLE001
        check(R, "import stokastisk / vllm", False, traceback.format_exc()[-800:]); finish("15c_static", R, extra); return 1
    # public API presence
    names = ["LLM", "StokastiskConfig", "check_model", "check_dtype", "check_feature", "check_quantization", "feature_matrix", "ServerProcess", "build_serve_command",
             "discover_topology", "recommend_parallelism", "StokastiskCompatibilityError"]
    missing = [n for n in names if not hasattr(stokastisk, n)]
    check(R, "public API exports present", not missing, missing or f"{len(names)} names")
    for m in ("stokastisk.compat", "stokastisk.serving", "stokastisk.parallel.replica", "stokastisk.topology", "stokastisk.telemetry", "stokastisk.communication", "stokastisk.cuda"):
        try: importlib.import_module(m); check(R, f"import {m}", True)
        except BaseException: check(R, f"import {m}", False, traceback.format_exc()[-500:])
    check(R, "LLM.chat is defined", callable(getattr(stokastisk.LLM, "chat", None)))
    # compat: bfloat16 is detected as unsupported on the REAL T4, never attempted
    import torch
    cap = torch.cuda.get_device_capability(0)
    try: stokastisk.check_dtype("bfloat16", capability=cap); check(R, "bfloat16 is rejected on SM75 (not silently accepted)", False, "accepted")
    except stokastisk.StokastiskCompatibilityError as e: check(R, "bfloat16 is rejected on SM75 (not silently accepted)", True, str(e)[:160])
    check(R, "float16 ('half') is accepted", stokastisk.check_dtype("half") == "half")
    try: stokastisk.LLM(MODEL, config=stokastisk.StokastiskConfig(), dtype="bfloat16"); check(R, "LLM(dtype='bfloat16') fails before any engine starts", False, "constructed")
    except stokastisk.StokastiskCompatibilityError: check(R, "LLM(dtype='bfloat16') fails before any engine starts", True)
    check(R, "no engine process was started by the bfloat16 check", gpu_used_mib() is not None and all(x < IDLE_MIB for x in gpu_used_mib()), gpu_used_mib())
    for feat in ("flash_attention", "flashinfer"):
        try: stokastisk.check_feature(feat); check(R, f"check_feature({feat!r}) is reported unsupported", False, "accepted")
        except stokastisk.StokastiskCompatibilityError: check(R, f"check_feature({feat!r}) is reported unsupported", True)
        except KeyError: check(R, f"check_feature({feat!r}) is reported unsupported", False, "unknown feature name")
    fm = stokastisk.feature_matrix(); extra["feature_matrix_size"] = len(fm)
    check(R, "feature matrix is non-empty and every entry has status/evidence", bool(fm) and all({"status", "evidence"} <= set(r) for r in fm.values()), len(fm))
    try:
        r = stokastisk.check_model(MODEL); extra["check_model"] = r if isinstance(r, (dict, list, str)) else str(r); check(R, f"check_model({MODEL}) does not raise", True)
    except BaseException: check(R, f"check_model({MODEL}) does not raise", False, traceback.format_exc()[-500:])
    # serving: pure command construction
    cmd = stokastisk.build_serve_command(MODEL, stokastisk.StokastiskConfig(max_model_len=2048), port=8000)
    check(R, "serve command pins --attention-backend TRITON_ATTN and --dtype half", cmd[cmd.index("--attention-backend") + 1] == "TRITON_ATTN" and cmd[cmd.index("--dtype") + 1] == "half", cmd[4:])
    try: stokastisk.build_serve_command(MODEL, stokastisk.StokastiskConfig(), extra_args=["--attention-backend", "FLASH_ATTN"]); check(R, "serve refuses an attention-backend override", False, "accepted")
    except stokastisk.StokastiskCompatibilityError: check(R, "serve refuses an attention-backend override", True)
    # bundled vLLM: backend class + no Flash binaries on disk
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    check(R, "TRITON_ATTN backend class resolves", AttentionBackendEnum.TRITON_ATTN.get_class().__name__ == "TritonAttentionBackend")
    vroot = pathlib.Path(vllm.__file__).parent
    badfiles = sorted(str(p.relative_to(vroot)) for p in vroot.rglob("*") if p.is_file() and not p.name.endswith(".py") and any(k in p.name.lower() for k in FLASH_KEYS + ("libflash",)))
    check(R, "no FlashAttention/FlashInfer/FlashMLA binary files in the installed vllm tree", not badfiles, badfiles or "none")
    prohib = stokastisk.prohibited_report(); check(R, "prohibited_report is clean", prohib["clean"], prohib)
    check(R, "no Flash library mapped in this process", not flash_in({str(os.getpid()): loaded_libs(os.getpid())}))
    # honest pip check
    pc = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True); lines = pc.stdout.strip().splitlines()
    extra["pip_check"] = {"returncode": pc.returncode, "lines": lines}
    own = [l for l in lines if l.startswith(("stokastisk ", "vllm "))]
    check(R, "pip check: no conflict reported FOR stokastisk or vllm", not own, own or "none")
    xg = [l for l in lines if l.startswith("xgrammar ")]
    if xg: warn(R, "pip check is NOT clean: xgrammar metadata conflict (documented, see docs/DEPENDENCY_COMPATIBILITY.md)", xg)
    if inst: check(R, "installer found no NEW pip conflict beyond the documented one", not inst["pip_check_new"]["other"] and not inst["pip_check_new"]["own"], inst["pip_check_new"])
    # xgrammar functional check on CPU (structured-output grammar engine under the installed transformers 5.x)
    try:
        import xgrammar as xgr, transformers, torch as _t
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL); info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=151936)
        schema = json.dumps({"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]})
        cg = xgr.GrammarCompiler(info, max_threads=2).compile_json_schema(schema); m = xgr.GrammarMatcher(cg); bm = xgr.allocate_token_bitmask(1, info.vocab_size)
        ok = True
        for t in tok.encode('{"name": "Ann", "age": 7}', add_special_tokens=False):
            m.fill_next_token_bitmask(bm, 0); ok = ok and bool(bm.any()) and m.accept_token(t)
        m2 = xgr.GrammarMatcher(cg); rej = not m2.accept_token(tok.encode("hello", add_special_tokens=False)[0])
        check(R, f"xgrammar {md.version('xgrammar')} + transformers {transformers.__version__}: JSON-schema grammar accepts valid and rejects invalid tokens", ok and rej, {"valid_accepted": ok, "invalid_rejected": rej})
    except BaseException: check(R, "xgrammar functional check", False, traceback.format_exc()[-800:])
    # provenance + package inventory
    inv = sorted(f"{d.metadata['Name']}=={d.version}" for d in md.distributions() if d.metadata["Name"])
    (OUT / "15c_package_inventory.txt").write_text("\n".join(inv) + "\n")
    prov = {"script_sha256": sha256(__file__), "common_sha256": sha256(os.path.join(os.path.dirname(__file__), "_common.py")), "package_inventory_count": len(inv),
            "installed_stokastisk_files_sha256": hashlib.sha256("".join(sorted(f"{p}{sha256(p)}" for p in map(str, pathlib.Path(stokastisk.__file__).parent.rglob("*.py")))).encode()).hexdigest()}
    save("15c_provenance.json", prov); extra["provenance"] = prov
    finish("15c_static", R, extra); return 0


def engine() -> int:
    cache = OUT / "triton-cache-15c-engine"; shutil.rmtree(cache, ignore_errors=True); cache.mkdir(parents=True); os.environ["TRITON_CACHE_DIR"] = str(cache)
    import stokastisk
    from vllm import SamplingParams
    R = []; stage = "15c_engine"
    check(R, "parent process has not initialised CUDA before the engine is created", not __import__("torch").cuda.is_initialized())
    before = gpu_used_mib(); check(R, "both GPUs idle before the engine starts", before is not None and all(x < IDLE_MIB for x in before), before)
    cfg = stokastisk.StokastiskConfig(max_model_len=2048)
    t0 = time.time()
    try: llm = stokastisk.LLM(MODEL, config=cfg)
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc(); print(tb, file=sys.stderr)
        R.append({"check": "Qwen2.5-0.5B-Instruct FP16 engine initialisation", "status": "ERROR", "detail": f"{type(e).__name__}: {str(e)[:500]}"}); finish(stage, R, {"traceback": tb}); return 1
    extra = {"versions": versions(), "model": MODEL, "load_seconds": time.time() - t0}
    try:
        check(R, "dtype is float16", str(llm.engine.llm_engine.vllm_config.model_config.dtype).endswith("float16"), llm.engine.llm_engine.vllm_config.model_config.dtype)
        check(R, "attention backend == TRITON_ATTN (live VllmConfig)", llm.selected_attention_backend() == "TRITON_ATTN" and llm.verify_attention_backend() == "TRITON_ATTN", llm.selected_attention_backend())
        sp = SamplingParams(max_tokens=32, temperature=0.0)
        g1 = [o.outputs[0].text for o in llm.generate(["The capital of France is", "Two plus two equals"], sp)]; g2 = [o.outputs[0].text for o in llm.generate(["The capital of France is", "Two plus two equals"], sp)]
        check(R, "LLM.generate returns non-empty text for every prompt", all(t.strip() for t in g1), [t[:40] for t in g1])
        check(R, "greedy decoding is deterministic", g1 == g2)
        check(R, "LLM.generate([]) returns [] without touching the engine", llm.generate([]) == [])
        ch = llm.chat([{"role": "user", "content": "Reply with one short sentence about the sea."}], SamplingParams(max_tokens=32, temperature=0.0))
        txt = ch[0].outputs[0].text; check(R, "LLM.chat returns non-empty text", bool(txt.strip()), txt[:80]); extra["chat_text"] = txt
        check(R, "LLM.complete returns text", bool(llm.complete("Hello,", max_tokens=8).strip()))
        for bad, exc in ((lambda: llm.generate([1]), TypeError), (lambda: llm.chat([]), ValueError)):
            try: bad(); check(R, "invalid input is rejected", False, "accepted")
            except exc: check(R, "invalid input is rejected with a clear exception", True)
        # structured output through the bundled xgrammar backend
        try:
            from vllm.sampling_params import StructuredOutputsParams
            schema = {"type": "object", "properties": {"city": {"type": "string"}, "population": {"type": "integer"}}, "required": ["city", "population"]}
            so = llm.generate(["Give a JSON object describing Paris."], SamplingParams(max_tokens=64, temperature=0.0, structured_outputs=StructuredOutputsParams(json=schema)))[0].outputs[0].text
            parsed = json.loads(so); check(R, "structured output (xgrammar, JSON schema) yields schema-valid JSON", isinstance(parsed.get("city"), str) and isinstance(parsed.get("population"), int), so[:120])
        except BaseException: check(R, "structured output (xgrammar, JSON schema) yields schema-valid JSON", False, traceback.format_exc()[-700:])
        peak = gpu_used_mib() or []; check(R, "exactly one GPU holds the model (TP=1)", len(peak) == 2 and sum(x > IDLE_MIB for x in peak) == 1, peak)
        arts = sorted({p.name for p in cache.rglob("*") if p.is_file() and "kernel_unified_attention" in p.name}); check(R, "Triton cache holds compiled kernel_unified_attention", bool(arts), arts[:3] or "none")
        libs = tree_loaded_libs(); bad = flash_in(libs); check(R, "no Flash library mapped in any process of the engine tree", not bad, bad or f"{len(libs)} processes inspected")
        extra.update(texts=g1, gpu_used_MiB={"before": before, "peak": peak}, loaded_libs=libs)
    finally:
        llm.close()
    check(R, "LLM.closed is True and generate() after close raises EngineError", llm.closed and _raises(lambda: llm.generate(["x"]), stokastisk.EngineError))
    time.sleep(2); live, ignored, waited = wait_children_gone(timeout=30); after = gpu_used_mib() or []
    check(R, "no live engine process after close", not live, live); check(R, "GPU memory released after close (<500 MiB each)", all(x < IDLE_MIB for x in after), after)
    finish(stage, R, {**extra, "gpu_after_close": after, "shutdown": {"live": live, "ignored": ignored, "waited_s": waited}}); return 0


def _raises(fn, exc):
    try: fn()
    except exc: return True
    except BaseException: return False
    return False


def server() -> int:
    cache = OUT / "triton-cache-15c-server"; shutil.rmtree(cache, ignore_errors=True); cache.mkdir(parents=True); os.environ["TRITON_CACHE_DIR"] = str(cache)
    import stokastisk
    from stokastisk import serving
    R = []; stage = "15c_server"; extra = {"model": MODEL}
    before = gpu_used_mib(); check(R, "both GPUs idle before the server starts", before is not None and all(x < IDLE_MIB for x in before), before)
    miss = serving.check_server_requirements(); extra["server_requirements_missing"] = miss
    if miss:
        R.append({"check": "OpenAI server requirements importable", "status": "FAIL", "detail": f"missing: {miss} (Stokastisk does not install packages; this feature is NOT verified)"}); finish(stage, R, extra); return 1
    log = str(OUT / "15c_server.log"); srv = serving.ServerProcess(MODEL, stokastisk.StokastiskConfig(max_model_len=2048), port=8011, log_path=log)
    try:
        srv.start(); srv.wait_ready(timeout=900)
        check(R, "server became healthy (GET /health 200)", True)
        base = srv.base_url
        check(R, "GET /v1/models lists the served model", MODEL in serving.list_models(base), serving.list_models(base))
        txt = serving.chat(base, MODEL, [{"role": "user", "content": "Say hello in five words."}], max_tokens=24); check(R, "POST /v1/chat/completions returns non-empty text", bool(txt.strip()), txt[:80]); extra["chat_text"] = txt
        chunks = list(serving.stream_chat(base, MODEL, [{"role": "user", "content": "Count from one to five."}], max_tokens=32))
        check(R, "streaming (SSE) yields several non-empty deltas", len(chunks) >= 2 and bool("".join(chunks).strip()), {"n": len(chunks), "text": "".join(chunks)[:80]})
        met = serving.fetch_metrics(base); check(R, "/metrics exposes vllm: samples", any(k.startswith("vllm:") for k in met), len(met))
        peak = gpu_used_mib() or []; check(R, "exactly one GPU holds the model while serving", len(peak) == 2 and sum(x > IDLE_MIB for x in peak) == 1, peak)
        pid = srv._proc.pid; libs = {str(p): loaded_libs(p) for p in [pid, *process_tree(pid)]}; bad = flash_in(libs)
        check(R, "no Flash library mapped in the server process tree", not bad, bad or f"{len(libs)} processes inspected"); extra.update(gpu_used_MiB={"before": before, "peak": peak}, loaded_libs=libs)
        arts = sorted({p.name for p in cache.rglob("*") if p.is_file() and "kernel_unified_attention" in p.name}); check(R, "Triton cache holds compiled kernel_unified_attention (server)", bool(arts), arts[:3] or "none")
        try: log_text = open(log, errors="replace").read(); check(R, "server log shows TRITON_ATTN and no Flash backend selection", "TRITON_ATTN" in log_text.upper() and "FLASH_ATTN" not in log_text.upper().replace("NO FLASH", ""), "TRITON_ATTN" in log_text.upper())
        except OSError as e: check(R, "server log readable", False, repr(e))
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc(); print(tb, file=sys.stderr); R.append({"check": "server start / requests", "status": "ERROR", "detail": f"{type(e).__name__}: {str(e)[:400]}"}); extra["traceback"] = tb
    finally:
        rc = srv.stop(); extra["server_exit_code"] = rc
    time.sleep(3); live, ignored, waited = wait_children_gone(timeout=30); after = gpu_used_mib() or []
    check(R, "server process stopped (exit code 0 or signal-terminated)", rc is not None, rc)
    check(R, "no live engine process after stop", not live, live); check(R, "GPU memory released after stop (<500 MiB each)", all(x < IDLE_MIB for x in after), after)
    finish(stage, R, {**extra, "gpu_after_close": after}); return 0


def summary() -> int:
    R = []; recs = {}
    for m in ("static", "engine", "server"):
        p = OUT / f"15c_{m}.json"; ok = p.is_file(); check(R, f"{m} evidence exists (missing = NOT_RUN, never a pass)", ok, p.name)
        if ok: recs[m] = json.loads(p.read_text()); check(R, f"{m} status PASS", recs[m].get("status") == "PASS", recs[m].get("status"))
    if len(recs) == 3:
        check(R, "all modes ran on the same installed versions", len({json.dumps(r.get("versions", {}).get("stokastisk"), sort_keys=True) for r in recs.values() if r.get("versions")}) <= 1)
    finish("15c_summary", R, {"modes": {m: r.get("status") for m, r in recs.items()}, "wheel": WHEEL}); return 0


if __name__ == "__main__":
    import multiprocessing; multiprocessing.freeze_support()
    modes = {"static": static, "engine": engine, "server": server, "summary": summary}
    if len(sys.argv) < 2 or sys.argv[1] not in modes: raise SystemExit(__doc__)
    run_guarded(f"15c_{sys.argv[1]}", modes[sys.argv[1]])
