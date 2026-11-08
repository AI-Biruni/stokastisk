#!/usr/bin/env python3.13
"""CPU-only tests: queue, routing, scheduler, recommender, memory estimation, and the REAL ReplicaPool process lifecycle with a fake (GPU-less) worker."""
import os, pathlib, sys, unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sdk")); sys.path.insert(0, str(ROOT / "tests"))
import stokastisk  # noqa: E402
from stokastisk.parallel import ModelSpec, Request, RequestQueue, Scheduler, WorkerLoad, choose_worker, recommend_parallelism  # noqa: E402
from stokastisk.topology.models import DeviceInfo, PeerAccess, Topology  # noqa: E402

QWEN15 = ModelSpec("Qwen2.5-1.5B", 28, 1536, 12, 2, 8960, 151936, tie_word_embeddings=True, architecture="Qwen2ForCausalLM")
QWEN05 = ModelSpec("Qwen2.5-0.5B", 24, 896, 14, 2, 4864, 151936, tie_word_embeddings=True, architecture="Qwen2ForCausalLM")
BIG20 = ModelSpec("20B", 48, 6144, 48, 8, 16384, 128000)
LLAMA7 = ModelSpec("7B", 32, 4096, 32, 32, 11008, 32000, architecture="LlamaForCausalLM")
BIG13 = ModelSpec("13B", 40, 5120, 40, 40, 13824, 32000)


def topo(n=2, p2p=False, nvlink=False, mib=14912.0, ram=25.0):
    t = Topology(devices=[DeviceInfo(index=i, name="Tesla T4", compute_capability=(7, 5), total_vram_mib=mib) for i in range(n)], system_ram_available_gib=ram)
    t.peer_access = PeerAccess({a: {b: p2p for b in range(n) if b != a} for a in range(n)}); t.nvlink_available = nvlink
    t.link_class = {"0-1": "NVLINK" if nvlink else "PHB"} if n > 1 else {}
    return t


class QueueRouting(unittest.TestCase):
    def test_queue_fifo_and_tokens(self):
        q = RequestQueue(maxsize=2); a, b = Request(1, "x" * 40, 10), Request(2, "y", 5)
        q.put(a); q.put(b); self.assertEqual(len(q), 2); self.assertEqual(q.pending_tokens, a.cost + b.cost)
        with self.assertRaises(OverflowError): q.put(Request(3, "z"))
        self.assertIs(q.get(), a); self.assertIs(q.get(), b); self.assertIsNone(q.get())
        self.assertEqual(a.input_tokens, 10); self.assertEqual(a.cost, 20)

    def test_least_loaded_and_tiebreak(self):
        w = [WorkerLoad(0, 0), WorkerLoad(1, 1)]
        self.assertEqual(choose_worker(w, 10), 0)                       # tie -> lowest id
        w[0].pending_tokens = 100; self.assertEqual(choose_worker(w, 10), 1)

    def test_speed_weighting(self):
        w = [WorkerLoad(0, 0, pending_tokens=100, speed=4.0), WorkerLoad(1, 1, pending_tokens=50, speed=1.0)]
        self.assertEqual(choose_worker(w, 10), 0)                       # 110/4 < 60/1

    def test_excludes_dead_notready_and_full(self):
        w = [WorkerLoad(0, 0, alive=False), WorkerLoad(1, 1, free_vram_mib=10.0), WorkerLoad(2, 2, ready=False)]
        self.assertIsNone(choose_worker(w, 1)); w.append(WorkerLoad(3, 3)); self.assertEqual(choose_worker(w, 1), 3)


class SchedulerTests(unittest.TestCase):
    def reqs(self, costs): return [Request(i, "p", max_tokens=c, input_tokens=1) for i, c in enumerate(costs)]

    def test_deterministic(self):
        runs = []
        for _ in range(3):
            s = Scheduler([WorkerLoad(0, 0), WorkerLoad(1, 1)]); plan = s.assign_batch(self.reqs([50, 10, 40, 30, 20, 10]))
            runs.append({k: [r.id for r in v] for k, v in plan.items()})
        self.assertEqual(runs[0], runs[1]); self.assertEqual(runs[1], runs[2])

    def test_lpt_balances_tokens(self):
        s = Scheduler([WorkerLoad(0, 0), WorkerLoad(1, 1)]); s.assign_batch(self.reqs([90, 80, 10, 10, 10, 10, 10, 10]))
        a, b = s.workers[0].pending_tokens, s.workers[1].pending_tokens
        self.assertLessEqual(abs(a - b), 12)

    def test_roundrobin_would_be_worse_for_skewed_load(self):
        costs = [100, 1, 100, 1, 100, 1]                                 # round robin gives worker0 all the 100s
        s = Scheduler([WorkerLoad(0, 0), WorkerLoad(1, 1)]); s.assign_batch(self.reqs(costs))
        rr = [sum(c + 1 for c in costs[0::2]), sum(c + 1 for c in costs[1::2])]
        self.assertLess(abs(s.workers[0].pending_tokens - s.workers[1].pending_tokens), abs(rr[0] - rr[1]))

    def test_complete_updates_load_and_speed(self):
        s = Scheduler([WorkerLoad(0, 0)]); r = Request(1, "p", 10, input_tokens=5); s.assign(r)
        self.assertEqual(s.workers[0].pending_tokens, 15); s.complete(1, seconds=1.5)
        self.assertEqual(s.workers[0].pending_tokens, 0); self.assertAlmostEqual(s.workers[0].speed, 10.0)

    def test_dead_worker_requeues_within_retry_budget(self):
        s = Scheduler([WorkerLoad(0, 0), WorkerLoad(1, 1)], max_attempts=2); r = Request(1, "p", 10); s.assign(r)
        lost = s.mark_dead(0); self.assertEqual([x.id for x in lost], [1]); self.assertEqual(s.assign(lost[0]), 1)
        s2 = Scheduler([WorkerLoad(0, 0)], max_attempts=1); r2 = Request(2, "p"); s2.assign(r2); self.assertEqual(s2.mark_dead(0), [])
        self.assertIsNone(s2.assign(Request(3, "p"))); s2.revive(0); self.assertEqual(s2.assign(Request(4, "p")), 0)


class MemoryAndRecommender(unittest.TestCase):
    def test_param_estimate_close_to_published(self):
        self.assertAlmostEqual(QWEN15.params / 1e9, 1.54, delta=0.05)    # Qwen2.5-1.5B: 1.54B
        self.assertAlmostEqual(QWEN05.params / 1e9, 0.49, delta=0.03)    # Qwen2.5-0.5B: 0.49B
        self.assertAlmostEqual(LLAMA7.params / 1e9, 6.74, delta=0.1); self.assertAlmostEqual(BIG13.params / 1e9, 13.0, delta=0.5)

    def test_from_hf_config(self):
        s = ModelSpec.from_hf_config("x", {"num_hidden_layers": 2, "hidden_size": 64, "num_attention_heads": 4, "vocab_size": 100, "architectures": ["LlamaForCausalLM"]})
        self.assertEqual(s.num_kv_heads, 4)
        with self.assertRaises(stokastisk.ConfigurationError): ModelSpec.from_hf_config("x", {"hidden_size": 1})

    def test_fits_one_gpu_latency_is_tp1(self):
        r = recommend_parallelism(QWEN15, topo(), "latency")
        self.assertEqual((r.strategy, r.tensor_parallel_size), ("single_gpu", 1)); self.assertIn("fits on one GPU", r.reason)
        self.assertTrue(any(a["strategy"] == "pipeline_parallel" and a.get("experimental") for a in r.alternatives))
        self.assertEqual(r.to_config().strategy, "single_gpu")

    def test_fits_one_gpu_throughput_is_replicas_experimental(self):
        r = recommend_parallelism(QWEN15, topo(), "throughput")
        self.assertEqual(r.strategy, "replica_parallel"); self.assertTrue(r.experimental); self.assertTrue(any("notebook 09" in w for w in r.warnings))

    def test_too_big_for_one_needs_two(self):
        r = recommend_parallelism(LLAMA7, topo(), "latency", max_model_len=2048)
        self.assertEqual((r.strategy, r.tensor_parallel_size), ("tensor_parallel", 2))
        self.assertTrue(any("no NVLink" in w for w in r.warnings)); self.assertEqual(r.to_config().tensor_parallel_size, 2)

    def test_too_big_for_two(self):
        for m in (BIG20, BIG13):          # 13B fp16 (~26 GB) does not fit 2 x ~12 GiB usable either
            self.assertEqual(recommend_parallelism(m, topo(), "latency").strategy, "insufficient_memory", m.name)
        r = recommend_parallelism(BIG20, topo(), "latency")
        with self.assertRaises(stokastisk.ConfigurationError): r.to_config()

    def test_single_gpu_and_no_gpu(self):
        self.assertEqual(recommend_parallelism(QWEN15, topo(1), "throughput").strategy, "single_gpu")
        self.assertEqual(recommend_parallelism(QWEN15, topo(0), "latency").strategy, "no_gpu")

    def test_unsupported_architecture(self):
        m = ModelSpec("ds", 10, 64, 4, 4, 128, 100, architecture="DeepseekV3ForCausalLM")
        self.assertEqual(recommend_parallelism(m, topo()).strategy, "unsupported_architecture")

    def test_does_not_mutate_environment(self):
        before = dict(os.environ); recommend_parallelism(QWEN15, topo(), "throughput"); self.assertEqual(before, dict(os.environ))

    def test_bad_workload(self):
        with self.assertRaises(stokastisk.ConfigurationError): recommend_parallelism(QWEN15, topo(), "fast")


class ReplicaLifecycle(unittest.TestCase):
    """Real multiprocessing (spawn) with a fake worker: device assignment, isolation checks, failure, retry, restart, shutdown."""
    def pool(self, **env):
        import stokastisk.parallel.replica as rp
        from _fake_replica_worker import worker_main as fake
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(rp, "worker_main", fake).start()
        for k, v in env.items(): mock.patch.dict(os.environ, {k: v}).start()
        return rp.ReplicaPool("fake-model", devices=[0, 1], ready_timeout=60)

    def test_two_workers_distinct_devices_and_ordering(self):
        p = self.pool()
        try:
            self.assertEqual(sorted(v["cuda_visible_devices"] for v in p.info.values()), ["0", "1"])
            self.assertEqual(len({v["uuid"] for v in p.info.values()}), 2); self.assertEqual(len({v["pid"] for v in p.info.values()}), 2)
            prompts = [f"p{i}" for i in range(10)]; out = p.generate(prompts, max_tokens=8)
            self.assertEqual([o.split(":", 1)[1] for o in out], prompts)                  # order preserved
            self.assertEqual(sum(p.last_stats["assignment"].values()), 10); self.assertTrue(all(v > 0 for v in p.last_stats["assignment"].values()))
        finally:
            rep = p.close()
        self.assertEqual(set(rep.values()), {"exit 0"})
        with self.assertRaises(stokastisk.ReplicaError): p.generate(["x"])

    def test_same_gpu_is_refused(self):
        with self.assertRaisesRegex(stokastisk.ReplicaError, "share a GPU"): self.pool(FAKE_SAME_UUID="1")

    def test_start_failure_reported(self):
        with self.assertRaisesRegex(stokastisk.ReplicaError, "failed to start"): self.pool(FAKE_FAIL_START="1")

    def test_duplicate_devices_rejected(self):
        from stokastisk.parallel.replica import ReplicaPool
        with self.assertRaises(stokastisk.UnsupportedParallelismError): ReplicaPool("m", devices=[0, 0])
        with self.assertRaises(stokastisk.UnsupportedParallelismError): ReplicaPool("m", devices=[0, 1], config=stokastisk.StokastiskConfig(tensor_parallel_size=2))

    def test_worker_death_is_retried_on_the_other_worker(self):
        p = self.pool(FAKE_DIE_ON_FIRST="0")
        try:
            out = p.generate([f"q{i}" for i in range(6)], max_tokens=4)
            self.assertEqual(len(out), 6)
            self.assertFalse(p._procs[0].is_alive()); self.assertTrue(p.sched.workers[1].alive); self.assertFalse(p.sched.workers[0].alive)
            self.assertTrue(all(o.startswith("w1:") for o in out))                            # everything was served by the survivor
        finally:
            p.close()


if __name__ == "__main__":
    unittest.main()
