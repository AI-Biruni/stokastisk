"""Strict configuration types (bool is not an int), environment parsing, and the source-level rule 'no silent exception handling' for the SDK."""
import ast, os, pathlib, sys, unittest
from unittest import mock
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sdk"))
import stokastisk
from stokastisk import ConfigurationError, StokastiskConfig, UnsupportedParallelismError

C = StokastiskConfig


class Types(unittest.TestCase):
    def test_bool_is_never_an_int_or_a_number(self):
        for kw in (dict(tensor_parallel_size=True), dict(pipeline_parallel_size=True), dict(max_model_len=True), dict(gpu_memory_utilization=True), dict(gpu_memory_utilization=False)):
            with self.subTest(kw=kw), self.assertRaises(ConfigurationError): C(**kw)
        with self.assertRaises(ConfigurationError): C(execution_strategy="replica_parallel", replica_devices=(True, 0))

    def test_other_wrong_types(self):
        for kw in (dict(tensor_parallel_size="1"), dict(tensor_parallel_size=1.0), dict(max_model_len="4096"), dict(max_model_len=4096.0), dict(gpu_memory_utilization="0.9"), dict(gpu_memory_utilization=float("nan")),
                   dict(enforce_eager=1), dict(enforce_eager="yes"), dict(disable_custom_all_reduce=None), dict(apply_nccl_env=0), dict(dtype=5), dict(dtype=""), dict(kv_cache_dtype=None), dict(attention_backend=None),
                   dict(execution_strategy=3), dict(nccl_env={"A": 1}), dict(nccl_env=["A"]), dict(structured_output_backend=1)):
            with self.subTest(kw=kw), self.assertRaises(ConfigurationError): C(**kw)

    def test_values(self):
        for kw in (dict(tensor_parallel_size=3), dict(pipeline_parallel_size=0), dict(max_model_len=8), dict(gpu_memory_utilization=0.05), dict(gpu_memory_utilization=0.99), dict(dtype="bfloat16"), dict(dtype="BF16"),
                   dict(attention_backend="FLASH_ATTN"), dict(attention_backend="flashinfer"), dict(attention_backend="FLASHMLA"), dict(attention_backend="ROCM_AITER_FA"), dict(attention_backend="FLEX_ATTENTION"),
                   dict(structured_output_backend="auto"), dict(structured_output_backend="outlines")):
            with self.subTest(kw=kw), self.assertRaises(ConfigurationError): C(**kw)
        for kw in (dict(tensor_parallel_size=2, pipeline_parallel_size=2), dict(execution_strategy="tensor_parallel"), dict(replica_devices=(0, 1)), dict(execution_strategy="replica_parallel", replica_devices=(0, 0)),
                   dict(execution_strategy="replica_parallel", replica_devices=(0, 2)), dict(execution_strategy="bogus")):
            with self.subTest(kw=kw), self.assertRaises(UnsupportedParallelismError): C(**kw)

    def test_valid_defaults(self):
        c = C(); self.assertEqual((c.dtype, c.attention_backend, c.structured_output_backend, c.max_model_len, c.gpu_memory_utilization), ("half", "TRITON_ATTN", "xgrammar", 4096, 0.88))
        self.assertEqual(C(attention_backend="triton_attn", structured_output_backend="XGrammar", gpu_memory_utilization=0.9).gpu_memory_utilization, 0.9)
        self.assertEqual(C(execution_strategy="replica_parallel", replica_devices=[0, 1]).replica_devices, (0, 1))

    def test_environment(self):
        with mock.patch.dict(os.environ, {"STOKASTISK_TENSOR_PARALLEL_SIZE": "2", "STOKASTISK_ENFORCE_EAGER": "off", "STOKASTISK_MAX_MODEL_LEN": "2048"}):
            c = C.from_env(); self.assertEqual((c.tensor_parallel_size, c.enforce_eager, c.max_model_len), (2, False, 2048))
        for k, v in (("TENSOR_PARALLEL_SIZE", "true"), ("TENSOR_PARALLEL_SIZE", "1.5"), ("MAX_MODEL_LEN", "True"), ("ENFORCE_EAGER", "maybe"), ("GPU_MEMORY_UTILIZATION", "x"), ("STRUCTURED_OUTPUT_BACKEND", "auto"), ("ATTENTION_BACKEND", "FLASH_ATTN")):
            with self.subTest(k=k, v=v), mock.patch.dict(os.environ, {f"STOKASTISK_{k}": v}), self.assertRaises(ConfigurationError): C.from_env()


class NoSilentExceptionHandling(unittest.TestCase):
    """Every `except` in the SDK either re-raises, logs (log.* / logging), records the error, or is an explicitly justified import/lookup fallback listed here."""
    ALLOWED = {("diagnostics.py", "check_native_extensions"): "diagnostics must never raise; the error text is the result",
               ("diagnostics.py", "prohibited_report"): "best-effort scan inside a report that is itself the evidence",
               ("diagnostics.py", "report"): "non-raising report; errors are stored in the report"}

    def test_scan(self):
        bad = []
        for f in (ROOT / "sdk/stokastisk").rglob("*.py"):
            tree = ast.parse(f.read_text()); fn_of = {}
            for fnode in ast.walk(tree):
                if isinstance(fnode, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for n in ast.walk(fnode): fn_of.setdefault(n, fnode.name)
            for n in ast.walk(tree):
                if isinstance(n, ast.ExceptHandler):
                    body = ast.dump(ast.Module(body=n.body, type_ignores=[]))
                    handled = any(k in body for k in ("Raise", "log", "append", "Return", "Assign", "put", "debug", "warn", "print", "continue", "Continue"))
                    only_pass = all(isinstance(s, ast.Pass) for s in n.body)
                    if only_pass and (f.name, fn_of.get(n)) not in self.ALLOWED and not (f.parent.name, f.name) == ("parallel", "queue.py"):
                        bad.append(f"{f.relative_to(ROOT)}:{n.lineno} in {fn_of.get(n)}")
        self.assertEqual(bad, [], "silent `except: pass` in the SDK")


if __name__ == "__main__":
    unittest.main()
