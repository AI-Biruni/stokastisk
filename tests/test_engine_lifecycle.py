"""stokastisk.LLM lifecycle, validation, serialisation, timeout, structured output - with a fake vllm engine (CPU). Says nothing about GPU behaviour; Kaggle notebook 15 does."""
import sys, threading, time, types, unittest
import torch  # noqa: F401  imported once, BEFORE sys.modules is patched, so the patch never removes torch submodules
from unittest import mock
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "sdk"))
import stokastisk
from stokastisk import engine as E
from stokastisk.errors import (AttentionBackendError, ConfigurationError, EngineCloseError, EngineClosedError, EngineError, EngineTimeoutError, StokastiskCompatibilityError, StructuredOutputError)


class FakeCore:
    def __init__(self, fail=False): self.fail, self.shutdowns = fail, 0
    def shutdown(self):
        self.shutdowns += 1
        if self.fail: raise RuntimeError("core refused to stop")


def out(text, finish_reason="stop"): return types.SimpleNamespace(outputs=[types.SimpleNamespace(text=text, token_ids=[1, 2], finish_reason=finish_reason)])


class FakeVLLM:
    instances = []
    attention = "TRITON_ATTN"; structured = "xgrammar"; init_error = None; core_fail = False; delay = 0.0; texts = None

    def __init__(self, model, **kw):
        if FakeVLLM.init_error: raise FakeVLLM.init_error
        self.model, self.kw, self.calls, self.active, self.max_active = model, kw, 0, 0, 0
        cfg = types.SimpleNamespace(attention_config=types.SimpleNamespace(backend=types.SimpleNamespace(name=FakeVLLM.attention)), structured_outputs_config=types.SimpleNamespace(backend=FakeVLLM.structured))
        self.llm_engine = types.SimpleNamespace(vllm_config=cfg, engine_core=FakeCore(FakeVLLM.core_fail)); self._lock = threading.Lock(); FakeVLLM.instances.append(self)

    def _run(self, prompts):
        with self._lock: self.active += 1; self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(FakeVLLM.delay); self.calls += 1
            return [out((FakeVLLM.texts or ["ok:{}"])[min(i, len(FakeVLLM.texts or [0]) - 1)].format(p) if FakeVLLM.texts else f"ok:{p}") for i, p in enumerate(prompts)]
        finally:
            with self._lock: self.active -= 1

    def generate(self, prompts, sp=None, **kw): return self._run(prompts)
    def chat(self, messages, sp=None, **kw): return self._run(["chat"])


def install_fake():
    vllm = types.ModuleType("vllm"); vllm.LLM = FakeVLLM
    vllm.SamplingParams = lambda **kw: types.SimpleNamespace(**kw)
    sp = types.ModuleType("vllm.sampling_params"); sp.StructuredOutputsParams = lambda **kw: types.SimpleNamespace(**kw)
    return mock.patch.dict(sys.modules, {"vllm": vllm, "vllm.sampling_params": sp})


class Base(unittest.TestCase):
    def setUp(self):
        FakeVLLM.instances.clear(); FakeVLLM.attention, FakeVLLM.structured, FakeVLLM.init_error, FakeVLLM.core_fail, FakeVLLM.delay, FakeVLLM.texts = "TRITON_ATTN", "xgrammar", None, False, 0.0, None
        p = install_fake(); p.start(); self.addCleanup(p.stop)
        self.cfg = stokastisk.StokastiskConfig(max_model_len=2048)

    def llm(self, **kw): return stokastisk.LLM("m", config=self.cfg, **kw)


class Construction(Base):
    def test_valid_and_repeated_construction(self):
        a, b = self.llm(), self.llm()
        self.assertFalse(a.closed or b.closed); self.assertEqual(a.verify_attention_backend(), "TRITON_ATTN"); self.assertEqual(a.verify_structured_output_backend(), "xgrammar")
        self.assertEqual(a.engine.kw["attention_config"], {"backend": "TRITON_ATTN"}); self.assertEqual(a.engine.kw["structured_outputs_config"], {"backend": "xgrammar"}); self.assertEqual(a.engine.kw["dtype"], "half")
        a.close(); b.close()

    def test_model_loading_failure_is_an_engine_error_and_leaves_nothing(self):
        FakeVLLM.init_error = OSError("404 model not found")
        with self.assertRaises(EngineError) as c: self.llm()
        self.assertIn("404", str(c.exception)); self.assertIsInstance(c.exception.__cause__, OSError)

    def test_wrong_backend_closes_the_engine_before_raising(self):
        FakeVLLM.attention = "FLASH_ATTN"
        with self.assertRaises(AttentionBackendError): self.llm()
        self.assertEqual(FakeVLLM.instances[-1].llm_engine.engine_core.shutdowns, 1)

    def test_policy_kwargs_cannot_change_the_enforced_backends(self):
        for kw in (dict(attention_backend="FLASH_ATTN"), dict(attention_backend="flashinfer"), dict(attention_config={"backend": "FLASHINFER"})):
            with self.subTest(kw=kw), self.assertRaises(StokastiskCompatibilityError): self.llm(**kw)
        for kw in (dict(structured_outputs_config={"backend": "auto"}), dict(structured_outputs_config={"backend": "guidance"})):
            with self.subTest(kw=kw), self.assertRaises(ConfigurationError): self.llm(**kw)
        self.llm(attention_backend="triton_attn", structured_outputs_config={"backend": "xgrammar"}).close()          # repeating the enforced value is fine
        self.assertEqual(len(FakeVLLM.instances), 1)                                                                  # nothing above reached the engine

    def test_dtype_and_model_validation_before_any_engine(self):
        with self.assertRaises(StokastiskCompatibilityError): self.llm(dtype="bfloat16")
        for m in ("", "  ", None, 5):
            with self.subTest(m=m), self.assertRaises(ConfigurationError): stokastisk.LLM(m, config=self.cfg)
        self.assertEqual(FakeVLLM.instances, [])


class Inference(Base):
    def test_generate_chat_complete(self):
        l = self.llm()
        self.assertEqual([o.outputs[0].text for o in l.generate(["a", "b"])], ["ok:a", "ok:b"]); self.assertEqual(l.generate("a")[0].outputs[0].text, "ok:a")
        self.assertEqual(l.generate([]), []); self.assertEqual(l.engine.calls, 2)
        self.assertEqual(l.chat([{"role": "user", "content": "hi"}])[0].outputs[0].text, "ok:chat"); l.close()

    def test_invalid_input_never_reaches_the_engine(self):
        l = self.llm()
        for fn, exc in ((lambda: l.generate([1]), TypeError), (lambda: l.generate(None), TypeError), (lambda: l.generate(""), ValueError), (lambda: l.generate(["a", " "]), ValueError),
                        (lambda: l.chat([]), ValueError), (lambda: l.chat("hi"), ValueError), (lambda: l.chat([{"role": "user"}]), ValueError), (lambda: l.chat([[]]), ValueError),
                        (lambda: l.generate(["a"], timeout=0), ValueError), (lambda: l.generate(["a"], timeout=True), ValueError), (lambda: l.generate(["a"], timeout="5"), ValueError),
                        (lambda: l.complete(""), ValueError), (lambda: l.complete("x", max_tokens=0), ValueError), (lambda: l.complete("x", max_tokens=True), ValueError), (lambda: l.complete("x", temperature=-1), ValueError)):
            with self.subTest(), self.assertRaises(exc): fn()
        self.assertEqual(l.engine.calls, 0)

    def test_inference_failure_is_wrapped_with_the_cause(self):
        l = self.llm()
        with mock.patch.object(l.engine, "generate", side_effect=RuntimeError("CUDA out of memory")):
            with self.assertRaises(EngineError) as c: l.generate(["a"])
        self.assertIn("CUDA out of memory", str(c.exception)); self.assertIsInstance(c.exception.__cause__, RuntimeError)

    def test_concurrent_callers_are_serialised(self):
        FakeVLLM.delay = 0.05; l = self.llm(); res, errs = {}, []
        def w(k):
            try: res[k] = l.generate([f"p{k}"])[0].outputs[0].text
            except BaseException as e: errs.append(e)  # noqa: BLE001
        ts = [threading.Thread(target=w, args=(k,)) for k in range(8)]; [t.start() for t in ts]; [t.join(30) for t in ts]
        self.assertEqual(errs, []); self.assertEqual(res, {k: f"ok:p{k}" for k in range(8)}); self.assertEqual(l.engine.max_active, 1)          # request-to-result correlation, never two calls inside the engine

    def test_timeout_then_the_next_call_waits_for_the_slow_one(self):
        FakeVLLM.delay = 0.4; l = self.llm(); t0 = time.time()
        with self.assertRaises(EngineTimeoutError): l.generate(["slow"], timeout=0.05)
        self.assertLess(time.time() - t0, 0.3)
        self.assertEqual(l.generate(["next"])[0].outputs[0].text, "ok:next"); self.assertEqual(l.engine.max_active, 1)
        FakeVLLM.delay = 0.3
        with self.assertRaises(EngineTimeoutError): l.generate(["a"], timeout=0.05)
        with self.assertRaises(EngineTimeoutError) as c: l.generate(["b"], timeout=0.05)                     # busy: the earlier request still holds the engine
        self.assertIn("busy", str(c.exception)); time.sleep(0.8); l.close()

    def test_timeout_path_returns_results_and_propagates_errors(self):
        l = self.llm(); self.assertEqual(l.generate(["a"], timeout=5)[0].outputs[0].text, "ok:a")
        with mock.patch.object(l.engine, "generate", side_effect=RuntimeError("boom")):
            with self.assertRaises(EngineError): l.generate(["a"], timeout=5)
        self.assertEqual(l.generate(["a"], timeout=5)[0].outputs[0].text, "ok:a")                            # the lock was released by the helper thread


class Close(Base):
    def test_repeated_close_and_use_after_close(self):
        l = self.llm(); l.close(); l.close(); self.assertTrue(l.closed); self.assertEqual(l.engine, None)
        for fn in (lambda: l.generate(["a"]), lambda: l.generate([]), lambda: l.chat([{"role": "user", "content": "x"}]), lambda: l.complete("x"), lambda: l.generate_structured(["x"], {"type": "object"})):
            with self.subTest(), self.assertRaises(EngineClosedError): fn()
        self.assertEqual(FakeVLLM.instances[0].llm_engine.engine_core.shutdowns, 1)

    def test_close_error_is_raised_after_cleanup_and_never_swallowed(self):
        FakeVLLM.core_fail = True; l = self.llm()
        with self.assertRaises(EngineCloseError) as c: l.close()
        self.assertIn("core refused", str(c.exception)); self.assertIsInstance(c.exception.__cause__, RuntimeError); self.assertTrue(l.closed)
        l.close()                                                                                             # already closed: quiet

    def test_context_manager_does_not_mask_the_body_exception(self):
        FakeVLLM.core_fail = True
        with self.assertRaises(KeyError):
            with self.llm(): raise KeyError("body failure")
        with self.assertRaises(EngineCloseError):
            with self.llm(): pass

    def test_close_after_failed_initialisation(self):
        FakeVLLM.init_error = RuntimeError("x")
        l = stokastisk.LLM.__new__(stokastisk.LLM)
        with self.assertRaises(EngineError): stokastisk.LLM.__init__(l, "m", config=self.cfg)
        self.assertTrue(l.closed); l.close()                                                                  # closing a half-built object is a no-op

    def test_close_while_a_call_is_queued(self):
        FakeVLLM.delay = 0.3; l = self.llm(); box = []
        t = threading.Thread(target=lambda: box.append(self._try(lambda: l.generate(["a"])))); t.start(); time.sleep(0.05)
        t2 = threading.Thread(target=lambda: box.append(self._try(lambda: l.generate(["b"])))); t2.start(); time.sleep(0.05); l.close(); t.join(5); t2.join(5)
        self.assertTrue(any(isinstance(x, EngineClosedError) for x in box), box)                              # the queued request finds a closed engine, it does not run on a dead one

    @staticmethod
    def _try(fn):
        try: return fn()
        except BaseException as e: return e  # noqa: BLE001


class Structured(Base):
    S = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"], "additionalProperties": False}

    def test_valid_output_is_parsed(self):
        FakeVLLM.texts = ['{{"city": "Paris"}}']; l = self.llm()
        self.assertEqual(l.generate_structured(["x"], self.S), [{"city": "Paris"}])

    def test_a_truncated_output_is_explained_not_just_rejected(self):
        """Kaggle: an unbounded integer ran to max_tokens ('"population": 2000000000...'); the error must say why and what to do."""
        l = self.llm()
        with mock.patch.object(l.engine, "generate", return_value=[out('{"city": "Paris", "population": 20000000000000', "length")]):
            with self.assertRaises(StructuredOutputError) as c: l.generate_structured(["x"], self.S, max_tokens=64)
        for needle in ("max_tokens=64", "finish_reason='length'", "maximum", "maxLength"): self.assertIn(needle, str(c.exception))

    def test_invalid_outputs_are_detected(self):
        for bad in ("Paris", '{"city": 5}', "{}", '{"city": "P", "x": 1}', '{"city": "Pa'):
            FakeVLLM.texts = [bad.replace("{", "{{").replace("}", "}}")]
            with self.subTest(bad=bad), self.assertRaises(StructuredOutputError): self.llm().generate_structured(["x"], self.S)

    def test_engine_with_another_backend_is_refused(self):
        FakeVLLM.structured = "auto"; l = self.llm()
        with self.assertRaises(StructuredOutputError): l.generate_structured(["x"], self.S)

    def test_bad_schema_and_arguments(self):
        l = self.llm()
        for s in (None, {}, "not json", {"type": "nonsense"}): 
            with self.subTest(s=s), self.assertRaises(StructuredOutputError): l.generate_structured(["x"], s)
        with self.assertRaises(ValueError): l.generate_structured(["x"], self.S, max_tokens=0)


if __name__ == "__main__":
    unittest.main()
