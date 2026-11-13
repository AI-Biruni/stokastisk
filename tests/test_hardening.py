"""Edge cases for the production-readiness audit: configuration validation, LLM facade lifecycle (stub vLLM, no GPU), compat checks,
ReplicaPool input validation / failure / repeated lifecycle (fake spawn workers). CPU only; none of this is GPU evidence."""
import os, sys, types, unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "sdk"))
import stokastisk
from stokastisk import StokastiskConfig, EngineError, ConfigurationError, UnsupportedParallelismError, StokastiskCompatibilityError, AttentionBackendError


def stub_vllm(backend="TRITON_ATTN", fail=False):
    mod = types.ModuleType("vllm")
    class Eng:
        def __init__(self, **kw):
            if fail: raise RuntimeError("model not found")
            self.kw = kw; self.shutdown_calls = 0
            outer = self
            class Core:
                def shutdown(self): outer.shutdown_calls += 1
            self.llm_engine = types.SimpleNamespace(engine_core=Core(), vllm_config=types.SimpleNamespace(attention_config=types.SimpleNamespace(backend=types.SimpleNamespace(name=backend))))
        def generate(self, prompts, sp=None, **kw): return [f"out:{p}" for p in prompts]
        def chat(self, messages, sp=None, **kw): return ["chat-out"]
    def LLM(model, **kw): return Eng(model=model, **kw)
    mod.LLM = LLM; mod.SamplingParams = lambda **k: k
    return mod


class ConfigValidation(unittest.TestCase):
    def test_rejects(self):
        for kw, exc in [({"tensor_parallel_size": 3}, ConfigurationError), ({"tensor_parallel_size": True}, ConfigurationError), ({"tensor_parallel_size": "1"}, ConfigurationError),
                        ({"max_model_len": 0}, ConfigurationError), ({"max_model_len": True}, ConfigurationError), ({"gpu_memory_utilization": 1.5}, ConfigurationError),
                        ({"gpu_memory_utilization": "0.9"}, ConfigurationError), ({"dtype": "bfloat16"}, ConfigurationError), ({"kv_cache_dtype": ""}, ConfigurationError),
                        ({"attention_backend": "FLASH_ATTN"}, ConfigurationError), ({"attention_backend": "FLASHINFER"}, ConfigurationError), ({"attention_backend": "XFORMERS"}, ConfigurationError),
                        ({"tensor_parallel_size": 2, "pipeline_parallel_size": 2}, UnsupportedParallelismError),
                        ({"execution_strategy": "tensor_parallel", "tensor_parallel_size": 1}, UnsupportedParallelismError), ({"execution_strategy": "nonsense"}, UnsupportedParallelismError),
                        ({"execution_strategy": "replica_parallel", "replica_devices": (0, 0)}, UnsupportedParallelismError), ({"replica_devices": (0, 1)}, UnsupportedParallelismError)]:
            with self.subTest(kw=kw), self.assertRaises(exc): StokastiskConfig(**kw)

    def test_bad_env(self):
        with mock.patch.dict(os.environ, {"STOKASTISK_TENSOR_PARALLEL_SIZE": "two"}), self.assertRaises(ConfigurationError): StokastiskConfig.from_env()
        with mock.patch.dict(os.environ, {"STOKASTISK_ENFORCE_EAGER": "maybe"}), self.assertRaises(ConfigurationError): StokastiskConfig.from_env()

    def test_defaults_are_the_verified_ones(self):
        c = StokastiskConfig(); kw = c.engine_kwargs()
        self.assertEqual((c.attention_backend, c.dtype, kw["attention_config"]), ("TRITON_ATTN", "half", {"backend": "TRITON_ATTN"})); self.assertNotIn("pipeline_parallel_size", kw)

    def test_import_does_not_start_cuda(self):
        import subprocess
        r = subprocess.run([sys.executable, "-c", "import sys, stokastisk; assert 'vllm' not in sys.modules; import torch; assert not torch.cuda.is_initialized()"], capture_output=True, text=True,
                           env={**os.environ, "PYTHONPATH": os.path.join(HERE, "..", "sdk")})
        self.assertEqual(r.returncode, 0, r.stderr[-600:])


class LLMLifecycle(unittest.TestCase):
    def llm(self, **kw):
        self.addCleanup(mock.patch.stopall)
        mock.patch.dict(sys.modules, {"vllm": stub_vllm(**kw)}).start()
        return stokastisk.LLM("m", config=StokastiskConfig())

    def test_generate_chat_close(self):
        l = self.llm()
        self.assertEqual(l.generate("a"), ["out:a"]); self.assertEqual(l.generate([]), []); self.assertEqual(l.chat([{"role": "user", "content": "hi"}]), ["chat-out"])
        eng = l.engine; l.close(); l.close()
        self.assertTrue(l.closed); self.assertEqual(eng.shutdown_calls, 1)                       # idempotent
        for call in (lambda: l.generate(["x"]), lambda: l.chat([{"role": "user", "content": "x"}])):
            with self.assertRaises(EngineError): call()

    def test_invalid_inputs(self):
        l = self.llm()
        for bad in ([1, 2], 5, None, ["ok", None]):
            with self.subTest(bad=bad), self.assertRaises(TypeError): l.generate(bad)
        for bad in ([], "hi", [{"role": "user"}], [[]], None):
            with self.subTest(bad=bad), self.assertRaises(ValueError): l.chat(bad)
        with self.assertRaises(ValueError): l.complete("")
        with self.assertRaises(ValueError): l.complete("x", max_tokens=0)

    def test_context_manager(self):
        with self.llm() as l: self.assertFalse(l.closed)
        self.assertTrue(l.closed)

    def test_model_validation_and_engine_init_failure(self):
        mock.patch.dict(sys.modules, {"vllm": stub_vllm(fail=True)}).start(); self.addCleanup(mock.patch.stopall)
        with self.assertRaises(ConfigurationError): stokastisk.LLM("  ")
        with self.assertRaises(ConfigurationError): stokastisk.LLM(None)
        with self.assertRaisesRegex(EngineError, "model not found"): stokastisk.LLM("m", config=StokastiskConfig())

    def test_wrong_backend_fails_loudly_and_closes_engine(self):
        self.addCleanup(mock.patch.stopall); v = stub_vllm(backend="FLASH_ATTN"); mock.patch.dict(sys.modules, {"vllm": v}).start()
        created = []
        orig = v.LLM; v.LLM = lambda model, **kw: created.append(orig(model, **kw)) or created[-1]
        with self.assertRaises(AttentionBackendError): stokastisk.LLM("m", config=StokastiskConfig())
        self.assertEqual(created[0].shutdown_calls, 1)

    def test_bf16_rejected_before_engine_start(self):
        self.addCleanup(mock.patch.stopall); mock.patch.dict(sys.modules, {"vllm": stub_vllm()}).start()
        with self.assertRaises(StokastiskCompatibilityError): stokastisk.LLM("m", config=StokastiskConfig(), dtype="bfloat16")


class CompatChecks(unittest.TestCase):
    def test_capability_gating(self):
        self.assertEqual(stokastisk.check_dtype("half"), "half") if isinstance(stokastisk.check_dtype("half"), str) else None
        with self.assertRaises(StokastiskCompatibilityError): stokastisk.check_dtype("bfloat16")
        with self.assertRaises(StokastiskCompatibilityError): stokastisk.check_feature("flash_attention")
        stokastisk.check_dtype("bfloat16", capability=(8, 0))                      # supported on Ampere: the gate is the capability, not the name

    def test_unknown_feature_is_not_silently_supported(self):
        with self.assertRaises(Exception): stokastisk.check_feature("definitely_not_a_feature")


class ReplicaEdgeCases(unittest.TestCase):
    def pool(self, **env):
        import stokastisk.parallel.replica as rp
        from _fake_replica_worker import worker_main as fake
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(rp, "worker_main", fake).start()
        for k, v in env.items(): mock.patch.dict(os.environ, {k: v}).start()
        return rp.ReplicaPool("fake-model", devices=[0, 1], ready_timeout=60)

    def test_input_validation_and_empty(self):
        p = self.pool()
        try:
            self.assertEqual(p.generate([]), [])
            for bad in ("a string", [1], None, ["ok", 3]):
                with self.subTest(bad=bad), self.assertRaises(TypeError): p.generate(bad)
            for mt in (0, -1, True, 2.5):
                with self.subTest(mt=mt), self.assertRaises(ValueError): p.generate(["x"], max_tokens=mt)
        finally: p.close()

    def test_repeated_initialisation_and_idempotent_close(self):
        for _ in range(2):
            p = self.pool(); self.assertEqual(len(p.generate(["a", "b"])), 2)
            r1 = p.close(); r2 = p.close(); self.assertEqual(r1, r2); self.assertEqual(set(r1.values()), {"exit 0"})

    def test_concurrent_generate_calls_are_serialised(self):
        import threading
        p = self.pool(); res = {}
        try:
            def run(k): res[k] = p.generate([f"{k}{i}" for i in range(8)])
            ts = [threading.Thread(target=run, args=(k,)) for k in "abc"]; [t.start() for t in ts]; [t.join(60) for t in ts]
            for k in "abc": self.assertEqual([o.split(":", 1)[1] for o in res[k]], [f"{k}{i}" for i in range(8)])
        finally: p.close()

    def test_worker_that_ignores_shutdown_is_terminated(self):
        p = self.pool(FAKE_HANG_ON_SHUTDOWN="1")
        with self.assertRaises(stokastisk.ReplicaError) as c: p.close(timeout=3)          # cleanup completes, then the failure is REPORTED (never swallowed)
        self.assertIn("replica 1", str(c.exception)); self.assertIn("terminated", str(c.exception))
        self.assertEqual(p.close_report[1], "terminated"); self.assertEqual(p.close_report[0], "exit 0")
        self.assertEqual(p.close(), p.close_report)                                         # idempotent: the second call is quiet

    def test_start_failure_leaves_no_processes(self):
        import multiprocessing as mp
        with self.assertRaises(stokastisk.ReplicaError): self.pool(FAKE_FAIL_START="0")
        import time; time.sleep(1)
        self.assertEqual([c for c in mp.active_children() if c.name.startswith("stokastisk-replica")], [])

    def test_missing_gpu_count_and_bad_devices(self):
        from stokastisk.parallel.replica import ReplicaPool
        for devs in ([], [0, 0], [0, 1, 2]):
            with self.subTest(devs=devs), self.assertRaises(UnsupportedParallelismError): ReplicaPool("m", devices=devs)


if __name__ == "__main__":
    unittest.main()
