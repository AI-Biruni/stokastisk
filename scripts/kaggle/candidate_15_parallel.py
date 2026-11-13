"""Stage `parallel`: the multi-GPU paths of the installed build on two Tesla T4: tensor parallel (TP=2), pipeline parallel (PP=2, experimental) and independent replicas (ReplicaPool),
each with real generation, both GPUs engaged, TRITON_ATTN, no Flash library, orderly shutdown and GPU memory released. Topology and the recommender run last (they initialise CUDA in this process).
Honest expectations: for a model that fits one T4 these paths are SLOWER than TP=1 (PCIe host bridge, no NVLink); this stage proves correctness and cleanup, not speed."""
import os, shutil, sys, time, traceback
from _common import OUT, check, error, finish, gpu_used_mib, process_tree, wait_children_gone
from candidate_15_common import *   # noqa: F401,F403

PROMPTS = ["The capital of France is", "Two plus two equals", "The sea is", "Once upon a time"]


def settle(R, label):
    time.sleep(2); live, _, _ = wait_children_gone(timeout=40); after = gpu_used_mib() or []
    check(R, f"{label}: no live engine process and GPU memory released (<{IDLE_MIB} MiB each)", not live and all(x < IDLE_MIB for x in after), (live, after))


def engine_case(R, label, cfg, expect_gpus):
    import stokastisk
    from vllm import SamplingParams
    llm = None
    try:
        llm = stokastisk.LLM(MODEL, config=cfg)
        check(R, f"{label}: backend TRITON_ATTN (live config)", llm.verify_attention_backend() == "TRITON_ATTN")
        out = [o.outputs[0].text for o in llm.generate(PROMPTS, SamplingParams(max_tokens=24, temperature=0.0))]
        check(R, f"{label}: generate returns non-empty text for every prompt", len(out) == len(PROMPTS) and all(t.strip() for t in out), [t[:30] for t in out])
        used = gpu_used_mib() or []
        check(R, f"{label}: {expect_gpus} GPU(s) hold model memory", sum(1 for x in used if x > IDLE_MIB) == expect_gpus, used)
        bad = flat(flash_libs_tree([os.getpid(), *process_tree()])); check(R, f"{label}: no Flash library mapped in the process tree", not bad, bad or "none")
    except BaseException:  # noqa: BLE001
        error(R, f"{label}: engine run", traceback.format_exc()[-900:])
    finally:
        err = None
        try:
            if llm is not None: llm.close()
        except Exception as e: err = e  # noqa: BLE001
        check(R, f"{label}: close() completed without an engine shutdown error", err is None, repr(err) if err else "")
    settle(R, label)


def parallel() -> int:
    cache = OUT / "triton-cache-15c-parallel"; shutil.rmtree(cache, ignore_errors=True); cache.mkdir(parents=True); os.environ["TRITON_CACHE_DIR"] = str(cache)
    import stokastisk
    R = []; extra = {}
    before = gpu_used_mib(); check(R, "both GPUs idle before the stage", before is not None and all(x < IDLE_MIB for x in before), before)
    # ---- tensor parallel
    engine_case(R, "TP=2", stokastisk.StokastiskConfig(tensor_parallel_size=2, max_model_len=2048), expect_gpus=2)
    # ---- pipeline parallel (experimental)
    engine_case(R, "PP=2 (experimental)", stokastisk.StokastiskConfig(pipeline_parallel_size=2, execution_strategy="pipeline_parallel", max_model_len=2048), expect_gpus=2)
    # ---- replicas
    pool = None
    try:
        pool = stokastisk.ReplicaPool(MODEL, devices=[0, 1], config=stokastisk.StokastiskConfig(max_model_len=2048), ready_timeout=900)
        info = pool.info; extra["replica_info"] = {str(k): v for k, v in info.items()}
        check(R, "replicas: two workers on two distinct GPUs (distinct UUIDs) and distinct processes", len({v.get("uuid") for v in info.values()}) == 2 and len({v["pid"] for v in info.values()}) == 2, [(v["device"], v.get("uuid")) for v in info.values()])
        check(R, "replicas: every worker reports TRITON_ATTN", all(v.get("backend") == "TRITON_ATTN" for v in info.values()), [v.get("backend") for v in info.values()])
        prompts = [f"{p} {i}" for i, p in enumerate(PROMPTS * 4)]
        out = pool.generate(prompts, max_tokens=16)
        check(R, "replicas: one non-empty result per prompt, in order", len(out) == len(prompts) and all(t.strip() for t in out), len(out))
        assign = pool.last_stats.get("assignment", {}); extra["replica_assignment"] = assign
        check(R, "replicas: both GPUs received work", len(assign) == 2 and all(v > 0 for v in assign.values()), assign)
        used = gpu_used_mib() or []; check(R, "replicas: both GPUs hold a copy of the model", sum(1 for x in used if x > IDLE_MIB) == 2, used)
        bad = flat(flash_libs_tree([os.getpid(), *process_tree()])); check(R, "replicas: no Flash library mapped in the process tree", not bad, bad or "none")
    except BaseException:  # noqa: BLE001
        error(R, "replicas: pool run", traceback.format_exc()[-900:])
    finally:
        err = None
        try:
            if pool is not None: extra["replica_close_report"] = pool.close()
        except Exception as e: err = e  # noqa: BLE001
        check(R, "replicas: close() completed with every worker exiting 0", err is None and all(v == "exit 0" for v in (pool.close_report if pool else {"x": "exit 0"}).values()), repr(err) if err else (pool.close_report if pool else ""))
    settle(R, "replicas")
    # ---- topology and recommender (initialises CUDA in this process: last)
    try:
        t = stokastisk.discover_topology()
        check(R, "topology: two Tesla T4 devices, SM75", t.gpu_count == 2 and all(tuple(d.compute_capability) == (7, 5) for d in t.devices if getattr(d, "compute_capability", None)), [d.name for d in t.devices])
        rec = stokastisk.recommend_parallelism("Qwen/Qwen2.5-0.5B-Instruct", t)
        check(R, "recommender: a model that fits one T4 is recommended single_gpu (TP=1)", rec.strategy == "single_gpu", (rec.strategy, rec.reason[:100]))
        extra["topology"] = {"link_class": t.link_class, "peer_access_all_pairs": t.peer_access.all_pairs_available, "nvlink": t.nvlink_available, "recommended": t.recommended_parallelism}
    except BaseException:  # noqa: BLE001
        error(R, "topology / recommender", traceback.format_exc()[-900:])
    finish("15c_parallel", R, extra, script_file=__file__); return 0
