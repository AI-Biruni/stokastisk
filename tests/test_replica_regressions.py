"""ReplicaPool regressions with FAKE workers (same message protocol, no GPU). CPU mocks do NOT prove GPU failure recovery - that stays 'verification pending' until Kaggle."""
import multiprocessing as mp, os, pathlib, sys, threading, time, unittest
from unittest import mock
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "sdk")); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stokastisk
from stokastisk import ReplicaError


class Base(unittest.TestCase):
    def pool(self, **env):
        import stokastisk.parallel.replica as rp
        from _fake_replica_worker import worker_main as fake
        self.addCleanup(mock.patch.stopall); mock.patch.object(rp, "worker_main", fake).start()
        for k, v in env.items(): mock.patch.dict(os.environ, {k: v}).start()
        return rp.ReplicaPool("fake-model", devices=[0, 1], ready_timeout=60)

    def orphans(self): return [c for c in mp.active_children() if c.name.startswith("stokastisk-replica")]


class Correlation(Base):
    def test_batch_ids_are_unique_for_the_life_of_the_pool(self):
        p = self.pool()
        try:
            seen = []
            orig = p._in[0].put
            for _ in range(3): p.generate(["a", "b", "c", "d"])
            self.assertGreaterEqual(p._next_batch, 3)
            b0 = p._next_batch; p.generate(["x"]); self.assertGreater(p._next_batch, b0)
        finally: p.close()

    def test_a_late_result_of_an_earlier_call_is_never_taken_for_a_batch_of_the_next_call(self):
        p = self.pool()
        try:
            p.generate(["warm", "up"]); stale_id = p._next_batch          # an id the pool already issued (a late answer of a call that timed out can only carry such an id)
            p._out.put(("result", 0, stale_id, {999999: ("STALE", 1)}, {"seconds": 0.0, "memory_total_mib": None, "memory_used_mib": None}))    # as if a timed-out call's worker answered late
            out = p.generate([f"q{i}" for i in range(6)])
            self.assertEqual(len(out), 6); self.assertTrue(all(o.startswith("w") and ":q" in o for o in out)); self.assertNotIn("STALE", " ".join(out))
        finally: p.close()

    def test_request_to_result_correlation_under_concurrency(self):
        p = self.pool(); res = {}
        try:
            ts = [threading.Thread(target=lambda k=k: res.__setitem__(k, p.generate([f"{k}{i}" for i in range(12)], max_tokens=1 + (ord(k) % 3)))) for k in "abcdef"]
            [t.start() for t in ts]; [t.join(120) for t in ts]
            for k in "abcdef": self.assertEqual([o.split(":", 1)[1] for o in res[k]], [f"{k}{i}" for i in range(12)])
        finally: p.close()


class Validation(Base):
    def test_arguments(self):
        p = self.pool()
        try:
            for kw in (dict(temperature=-1), dict(temperature=True), dict(temperature="0"), dict(timeout=0), dict(timeout=True)):
                with self.subTest(kw=kw), self.assertRaises(ValueError): p.generate(["x"], **kw)
            with self.assertRaises(ValueError): p.generate(["ok", "  "])
        finally: p.close()


class Failures(Base):
    def test_worker_start_failure_leaves_no_orphans(self):
        with self.assertRaises(ReplicaError): self.pool(FAKE_FAIL_START="1")
        time.sleep(1.5); self.assertEqual(self.orphans(), [])

    def test_worker_crash_is_retried_and_the_pool_can_restart_it(self):
        p = self.pool(FAKE_DIE_ON_FIRST="0")
        try:
            out = p.generate([f"q{i}" for i in range(6)]); self.assertEqual(len(out), 6); self.assertTrue(all(o.startswith("w1:") for o in out))
            mock.patch.dict(os.environ, {"FAKE_DIE_ON_FIRST": ""}).start()
            self.assertEqual(p.restart_dead(timeout=60), [0]); out = p.generate([f"r{i}" for i in range(8)]); self.assertEqual({o.split(":")[0] for o in out}, {"w0", "w1"})     # both serve again
        finally: p.close()

    def test_all_workers_dead_is_an_error_not_a_hang(self):
        p = self.pool()
        try:
            for pr in p._procs.values(): pr.terminate(); pr.join(10)
            with self.assertRaises(ReplicaError): p.generate(["x", "y"], timeout=30)
        finally:
            try: p.close()
            except ReplicaError: pass

    def test_hung_worker_times_out(self):
        p = self.pool(FAKE_HANG_ON_GENERATE="1") if False else self.pool()
        try:
            for pr in p._procs.values(): os.kill(pr.pid, 19)                       # SIGSTOP both workers: alive but silent
            t0 = time.time()
            with self.assertRaises(ReplicaError) as c: p.generate(["x"], timeout=2)
            self.assertIn("timed out", str(c.exception)); self.assertLess(time.time() - t0, 20)
            for pr in p._procs.values(): os.kill(pr.pid, 18)
        finally:
            try: p.close(timeout=20)
            except ReplicaError: pass

    def test_close_failure_is_reported_after_complete_cleanup_and_leaves_no_orphans(self):
        p = self.pool(FAKE_CLOSE_ERROR="0", FAKE_HANG_ON_SHUTDOWN="1")
        with self.assertRaises(ReplicaError) as c: p.close(timeout=3)
        msg = str(c.exception); self.assertIn("replica 1", msg); self.assertIn("terminated", msg); self.assertIn("replica 0", msg); self.assertIn("LLM.close failed", msg)
        time.sleep(1); self.assertEqual(self.orphans(), []); self.assertEqual(p.close(), p.close_report)

    def test_queues_are_closed_and_use_after_close_is_an_error(self):
        p = self.pool(); p.generate(["a"]); p.close()
        with self.assertRaises(ReplicaError): p.generate(["a"])
        with self.assertRaises(ReplicaError): p.restart_dead()
        for q in (*p._in.values(), p._out):
            with self.assertRaises((ValueError, OSError)): q.put(1)

    def test_repeated_creation_and_shutdown_leak_nothing(self):
        for _ in range(3):
            p = self.pool(); p.generate(["a", "b"]); p.close()
        time.sleep(1); self.assertEqual(self.orphans(), [])

    def test_context_manager_keeps_the_body_exception(self):
        with self.assertRaises(KeyError):
            with self.pool(FAKE_HANG_ON_SHUTDOWN="1"): raise KeyError("body")
        time.sleep(0.5)


if __name__ == "__main__":
    unittest.main()
