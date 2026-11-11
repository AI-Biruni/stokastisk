"""Stage 15 : tensor- and pipeline-parallel fulfilment on 2 x Tesla T4 (SM75).
usage: dual_15_tp_pp_fulfilment.py MODE [MODEL]      MODE: tp1 | tp2 | pp2 | summary
One engine per process (the same isolation the benchmarks use). Each mode writes 15_fulfil_<MODE>.json; `summary` reads the three
and applies the cross-mode requirements. A vLLM failure is reported as ERROR with the traceback and is NEVER converted to PASS."""
import json, os, shutil, sys, time, traceback
from _common import OUT, check, finish, gpu_used_mib, process_tree, run_guarded, save, tree_loaded_libs, versions, wait_children_gone, warn

MODES = {"tp1": (1, 1, "single_gpu"), "tp2": (2, 1, "tensor_parallel"), "pp2": (1, 2, "pipeline_parallel")}
PROMPTS = ["The capital of France is", "The capital of Japan is"]
IDLE_MIB = 500          # a GPU holding less than this is not running a model shard


def run_mode(mode: str, model: str) -> int:
    tp, pp, strategy = MODES[mode]
    cache = OUT / f"triton-cache-fulfil-{mode}"; shutil.rmtree(cache, ignore_errors=True); cache.mkdir(parents=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache)
    import stokastisk
    from vllm import SamplingParams
    R = []; stage = f"15_fulfil_{mode}"
    stokastisk.validate_environment(require_gpus=2)
    before = gpu_used_mib()
    check(R, "both GPUs idle before the engine starts", before is not None and all(x < IDLE_MIB for x in before), before)
    cfg = stokastisk.StokastiskConfig(tensor_parallel_size=tp, pipeline_parallel_size=pp, execution_strategy=strategy, max_model_len=4096)
    check(R, f"config: TP={tp} PP={pp} strategy {strategy}", (cfg.tensor_parallel_size, cfg.pipeline_parallel_size, cfg.strategy) == (tp, pp, strategy),
          (cfg.tensor_parallel_size, cfg.pipeline_parallel_size, cfg.strategy))
    t0 = time.time()
    try:
        llm = stokastisk.LLM(model, config=cfg)
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc(); print(tb, file=sys.stderr)
        R.append({"check": f"{mode} engine initialisation on vLLM 0.25.1", "status": "ERROR", "detail": f"{type(e).__name__}: {str(e)[:600]}"})
        finish(stage, R, {"versions": versions(), "traceback": tb, "limitation": f"{mode} failed to initialise; it is NOT supported by this build until resolved"})
        return 1
    extra = {"mode": mode, "model": model, "load_seconds": time.time() - t0}
    try:
        check(R, "attention backend == TRITON_ATTN", llm.selected_attention_backend() == "TRITON_ATTN", llm.selected_attention_backend())
        pc = llm.engine.llm_engine.vllm_config.parallel_config
        check(R, f"vLLM parallel_config reports TP={tp} PP={pp}", (pc.tensor_parallel_size, pc.pipeline_parallel_size) == (tp, pp), (pc.tensor_parallel_size, pc.pipeline_parallel_size))
        sp = SamplingParams(max_tokens=48, temperature=0.0)
        first = [o.outputs[0].text for o in llm.generate(PROMPTS, sp)]
        again = [o.outputs[0].text for o in llm.generate(PROMPTS, sp)]
        check(R, "every prompt returned non-empty text", all(t.strip() for t in first), [t[:40] for t in first])
        check(R, "deterministic at temperature 0 (two identical runs)", first == again)
        check(R, "France prompt mentions Paris", "paris" in first[0].lower(), first[0][:60])
        peak = gpu_used_mib() or []
        busy = [x > IDLE_MIB for x in peak]
        if mode == "tp1":
            check(R, "exactly one GPU holds the model (the other is idle)", len(peak) == 2 and sum(busy) == 1, peak)
        else:
            check(R, "BOTH GPUs hold a model shard while generating", len(peak) == 2 and all(busy), peak)
        arts = sorted({p.name for p in cache.rglob("*") if p.is_file() and "kernel_unified_attention" in p.name})
        check(R, "Triton cache holds compiled kernel_unified_attention (kernel ran)", bool(arts), arts[:3] or "none")
        flat = sorted({x for v in tree_loaded_libs().values() if isinstance(v, list) for x in v})
        bad = [x for x in flat if any(k in x.lower() for k in ("vllm_flash_attn", "_vllm_fa", "flashinfer", "flashmla"))]
        check(R, "no Flash library mapped in any process", not bad, bad or f"{len(flat)} libs inspected")
        extra.update(texts=first, gpu_used_MiB={"before": before, "peak": peak}, engine_processes=len([p for p in (process_tree() or []) if p]))
    finally:
        llm.close()
    time.sleep(2); live, ignored, waited = wait_children_gone(timeout=30)
    after = gpu_used_mib() or []
    check(R, "no live engine process after close", not live, live)
    check(R, "GPU memory released after close (<500 MiB each)", all(x < IDLE_MIB for x in after), after)
    finish(stage, R, {"versions": versions(), **extra, "gpu_after_close": after, "shutdown": {"live": live, "ignored": ignored, "waited_s": waited}})
    return 0


def summary() -> int:
    R = []; recs = {}
    for m in MODES:
        p = OUT / f"15_fulfil_{m}.json"
        ok = p.is_file()
        check(R, f"{m} stage evidence exists (a missing stage is NOT_RUN, never a pass)", ok, p.name)
        if ok:
            recs[m] = json.loads(p.read_text()); check(R, f"{m} stage status PASS", recs[m].get("status") == "PASS", recs[m].get("status"))
    if len(recs) == 3:
        ref = recs["tp1"].get("texts") or []
        for m in ("tp2", "pp2"):
            t = recs[m].get("texts") or []
            same = t == ref
            if same:
                check(R, f"{m} greedy output equals tp1", True)
            else:   # fp16 reductions in a different order can legitimately change a late token; the first-prompt prefix must still agree
                pref = bool(t) and bool(ref) and t[0][:20] == ref[0][:20]
                check(R, f"{m} output agrees with tp1 on the first 20 characters of the first prompt", pref, {"tp1": ref[:1], m: t[:1]})
                warn(R, f"{m} output is not byte-identical to tp1 (fp16 numerics across parallel layouts)", "informational")
        versions_seen = {json.dumps(r.get("versions", {}), sort_keys=True) for r in recs.values()}
        check(R, "all three modes ran on the same installed versions", len(versions_seen) == 1)
    finish("15_fulfil_summary", R, {"modes": {m: recs[m].get("status") for m in recs}})
    return 0


if __name__ == "__main__":
    import multiprocessing; multiprocessing.freeze_support()
    if len(sys.argv) < 2 or sys.argv[1] not in (*MODES, "summary"):
        raise SystemExit(__doc__)
    if sys.argv[1] == "summary":
        run_guarded("15_fulfil_summary", summary)
    else:
        mdl = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-1.5B-Instruct"
        run_guarded(f"15_fulfil_{sys.argv[1]}", lambda: run_mode(sys.argv[1], mdl))
