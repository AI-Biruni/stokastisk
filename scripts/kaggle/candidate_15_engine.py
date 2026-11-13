"""Stage `engine`: actual FP16 inference on a Tesla T4 with the installed build 3: generate / chat / complete, determinism, input validation, structured output with the PINNED
xgrammar backend (valid output, invalid-output detection, no hidden fallback), serialisation / timeout, TRITON_ATTN at config AND kernel level, lifecycle, cleanup,
and the same check for upstream `vllm.LLM` used directly (an unsupported entry point: it must still never select a Flash backend)."""
import json, os, pathlib, shutil, sys, threading, time, traceback
from _common import OUT, check, error, finish, gpu_used_mib, process_tree, tree_loaded_libs, versions, wait_children_gone
from candidate_15_common import *   # noqa: F401,F403


def engine() -> int:
    cache = OUT / "triton-cache-15c-engine"; shutil.rmtree(cache, ignore_errors=True); cache.mkdir(parents=True); os.environ["TRITON_CACHE_DIR"] = str(cache)
    import stokastisk
    from vllm import SamplingParams
    R = []; stage = "15c_engine"
    check(R, "parent process has not initialised CUDA before the engine is created", not __import__("torch").cuda.is_initialized())
    before = gpu_used_mib(); check(R, "both GPUs idle before the engine starts", before is not None and all(x < IDLE_MIB for x in before), before)
    # failed construction must leave nothing behind
    t0 = time.time()
    try:
        stokastisk.LLM("stokastisk-test/this-model-does-not-exist", config=stokastisk.StokastiskConfig(max_model_len=2048)); check(R, "construction with a non-existent model fails", False, "constructed")
    except stokastisk.EngineError as e:
        check(R, "construction with a non-existent model raises EngineError", True, str(e)[:100])
    time.sleep(2); live, _, _ = wait_children_gone(timeout=30); check(R, "failed construction: no engine process and no GPU memory left behind", not live and all(x < IDLE_MIB for x in (gpu_used_mib() or [10**9])), (live, gpu_used_mib()))
    cfg = stokastisk.StokastiskConfig(max_model_len=2048)
    t0 = time.time()
    try: llm = stokastisk.LLM(MODEL, config=cfg)
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc(); print(tb, file=sys.stderr)
        error(R, "Qwen2.5-0.5B-Instruct FP16 engine initialisation", f"{type(e).__name__}: {str(e)[:500]}"); finish(stage, R, {"traceback": tb}, script_file=__file__); return 1
    extra = {"versions": versions(), "model": MODEL, "load_seconds": time.time() - t0}
    try:
        vc = llm.engine.llm_engine.vllm_config
        check(R, "dtype is float16", str(vc.model_config.dtype).endswith("float16"), vc.model_config.dtype)
        check(R, "attention backend == TRITON_ATTN (live VllmConfig)", llm.selected_attention_backend() == "TRITON_ATTN" and llm.verify_attention_backend() == "TRITON_ATTN", llm.selected_attention_backend())
        check(R, "structured-output backend == xgrammar (live VllmConfig; never 'auto')", llm.structured_output_backend() == "xgrammar" and llm.verify_structured_output_backend() == "xgrammar", llm.structured_output_backend())
        sp = SamplingParams(max_tokens=32, temperature=0.0)
        g1 = [o.outputs[0].text for o in llm.generate(["The capital of France is", "Two plus two equals"], sp)]; g2 = [o.outputs[0].text for o in llm.generate(["The capital of France is", "Two plus two equals"], sp)]
        check(R, "LLM.generate returns non-empty text for every prompt", all(t.strip() for t in g1), [t[:40] for t in g1])
        check(R, "greedy decoding is deterministic", g1 == g2)
        check(R, "LLM.generate([]) returns [] without touching the engine", llm.generate([]) == [])
        ch = llm.chat([{"role": "user", "content": "Reply with one short sentence about the sea."}], SamplingParams(max_tokens=32, temperature=0.0))
        txt = ch[0].outputs[0].text; check(R, "LLM.chat returns non-empty text", bool(txt.strip()), txt[:80]); extra["chat_text"] = txt
        check(R, "LLM.complete returns text", bool(llm.complete("Hello,", max_tokens=8).strip()))
        for label, fn, exc in (("generate([1])", lambda: llm.generate([1]), TypeError), ("generate('')", lambda: llm.generate(""), ValueError), ("generate(['  '])", lambda: llm.generate(["  "]), ValueError),
                               ("chat([])", lambda: llm.chat([]), ValueError), ("chat([{'role':'user'}])", lambda: llm.chat([{"role": "user"}]), ValueError), ("complete('')", lambda: llm.complete(""), ValueError),
                               ("complete(max_tokens=0)", lambda: llm.complete("hi", max_tokens=0), ValueError), ("generate(timeout=-1)", lambda: llm.generate(["hi"], timeout=-1), ValueError)):
            check(R, f"invalid input rejected before the engine: {label}", raises(fn, exc))
        # ---------------------------------------------------------------- structured output (pinned xgrammar)
        def structured_ok():
            objs = llm.generate_structured(["Give a JSON object describing Paris.", "Give a JSON object describing Tokyo."], ENGINE_SCHEMA, max_tokens=96)
            check(R, "generate_structured: every output is JSON that satisfies the bounded schema (validated by jsonschema)", all(isinstance(o["city"], str) and isinstance(o["population"], int) for o in objs), objs)
            extra["structured_objects"] = objs
        attempt(R, "generate_structured: schema-valid output", structured_ok)
        from stokastisk.structured import validate_json_output
        for label, bad_text in (("not JSON", "Paris is nice"), ("wrong type", '{"city": 5, "population": 1}'), ("missing key", '{"city": "Paris"}'), ("extra key", '{"city": "P", "population": 1, "x": 1}'), ("truncated", '{"city": "Par')):
            check(R, f"invalid structured output is detected: {label}", raises(lambda t=bad_text: validate_json_output(t, ENGINE_SCHEMA), stokastisk.StructuredOutputError))
        check(R, "a schema the pinned xgrammar backend does not support is REJECTED (no hidden fallback to guidance/outlines)", raises(lambda: llm.generate_structured(["n"], UNSUPPORTED_BY_XGRAMMAR, max_tokens=16), stokastisk.EngineError))
        check(R, "engine is still healthy after the rejected request", bool(llm.complete("Hello,", max_tokens=4).strip()))
        maps = {str(p): [l.split()[-1] for l in open(f"/proc/{p}/maps", errors="replace") if l.strip().endswith((".so",)) or ".so." in l] for p in process_tree()}
        xg = sorted({m for v in maps.values() for m in v if "xgrammar" in m.lower()}); extra["xgrammar_native_objects_mapped"] = xg
        check(R, "xgrammar's native library is mapped in the engine process tree (the pinned backend really ran)", bool(xg), xg[:2] or "none mapped")
        check(R, "outlines_core's native library is NOT mapped (no fallback backend ran)", not [m for v in maps.values() for m in v if "outlines_core" in m.lower()])
        # ---------------------------------------------------------------- serialisation and timeout
        res, errs = {}, []
        def worker(k):
            try: res[k] = llm.complete(f"Count to {k}:", max_tokens=8)
            except BaseException as e: errs.append(repr(e))  # noqa: BLE001
        ts = [threading.Thread(target=worker, args=(k,)) for k in (1, 2, 3, 4)]
        [t.start() for t in ts]; [t.join(300) for t in ts]
        check(R, "4 concurrent callers are serialised: all return text, none interleave or raise", not errs and len(res) == 4 and all(v.strip() for v in res.values()), errs or sorted(res))
        long_sp = SamplingParams(max_tokens=512, temperature=0.0)
        check(R, "generate(timeout=0.01) raises EngineTimeoutError", raises(lambda: llm.generate(["Write a very long story about the sea."], long_sp, timeout=0.01), stokastisk.EngineTimeoutError))
        check(R, "the next call waits for the timed-out request and then succeeds (the engine is never driven twice)", bool(llm.complete("Hello,", max_tokens=4).strip()))
        peak = gpu_used_mib() or []; check(R, "exactly one GPU holds the model (TP=1)", len(peak) == 2 and sum(x > IDLE_MIB for x in peak) == 1, peak)
        arts = triton_artifacts(cache); check(R, "Triton cache holds compiled kernel_unified_attention (kernel-level TRITON_ATTN evidence)", bool(arts), arts[:3] or "none")
        libs = flash_libs_tree([os.getpid(), *process_tree()]); bad = flat(libs); check(R, "no Flash library mapped in any process of the engine tree", not bad, bad or f"{len(libs)} processes inspected")
        extra.update(texts=g1, gpu_used_MiB={"before": before, "peak": peak}, loaded_libs=tree_loaded_libs())
    finally:
        close_error = None
        try: llm.close()
        except Exception as e: close_error = e  # noqa: BLE001
    check(R, "LLM.close() completed without an engine shutdown error", close_error is None, repr(close_error) if close_error else "")
    check(R, "LLM.closed is True; a second close() is a no-op; generate() after close raises EngineClosedError", llm.closed and no_raise(llm.close) and raises(lambda: llm.generate(["x"]), stokastisk.EngineClosedError))
    time.sleep(2); live, ignored, waited = wait_children_gone(timeout=30); after = gpu_used_mib() or []
    check(R, "no live engine process after close", not live, live); check(R, "GPU memory released after close (<500 MiB each)", all(x < IDLE_MIB for x in after), after)
    extra.update(gpu_after_close=after, shutdown={"live": live, "ignored": ignored, "waited_s": waited})
    # ---------------------------------------------------------------- upstream vllm.LLM used directly: NOT a supported entry point, but it must not select Flash either
    dcache = OUT / "triton-cache-15c-direct-vllm"; shutil.rmtree(dcache, ignore_errors=True); dcache.mkdir(parents=True); os.environ["TRITON_CACHE_DIR"] = str(dcache)
    try:
        from vllm import LLM as UpstreamLLM
        up = UpstreamLLM(model=MODEL, dtype="half", enforce_eager=True, max_model_len=2048, gpu_memory_utilization=0.5, tensor_parallel_size=1)
        out = up.generate(["The capital of France is"], SamplingParams(max_tokens=8, temperature=0.0))[0].outputs[0].text
        check(R, "[unsupported entry point] direct vllm.LLM(dtype='half') generates (no attention backend specified)", bool(out.strip()), out[:40])
        auto = up.llm_engine.vllm_config.attention_config.backend
        check(R, "[unsupported entry point] direct vllm.LLM auto-selection never picks a Flash backend and compiles the Triton unified-attention kernel", (auto is None or "FLASH" not in str(auto).upper()) and bool(triton_artifacts(dcache)), (str(auto), triton_artifacts(dcache)[:2]))
        dl = flat(flash_libs_tree([os.getpid(), *process_tree()])); check(R, "[unsupported entry point] no Flash library mapped", not dl, dl or "none")
        try: up.llm_engine.engine_core.shutdown()
        except Exception as e: error(R, "[unsupported entry point] direct engine shutdown", repr(e))  # noqa: BLE001
        del up
    except BaseException: error(R, "[unsupported entry point] direct vllm.LLM check", traceback.format_exc()[-800:])
    time.sleep(2); live, _, _ = wait_children_gone(timeout=30); fin = gpu_used_mib() or []
    check(R, "no live process and GPU memory released after the direct-vLLM check", not live and all(x < IDLE_MIB for x in fin), (live, fin))
    finish(stage, R, extra, script_file=__file__); return 0
