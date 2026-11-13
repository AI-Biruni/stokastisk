"""The Flash-free transform of the pinned vLLM tree, and the runtime policy module it installs. Needs the byte-pinned vLLM wheel (private local baseline): NOT_RUN (skipped) if absent."""
import ast, hashlib, json, os, pathlib, subprocess, sys, tempfile, textwrap, unittest, zipfile
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sdk")); sys.path.insert(0, str(ROOT / "scripts/release"))
PIN = ROOT / "inputs/vllm-0.25.1-cp38-abi3-linux_x86_64.whl"
PIN_SHA = "cc3bcafb6516791990ca302073ae153a08465d3df1e460d666e3ec09b4e98f5e"
import t4_flash_free as t
from stokastisk import prohibited


def tree():
    z = zipfile.ZipFile(PIN)
    return {n: z.read(n) for n in z.namelist() if n.startswith("vllm/") and not n.endswith("/")}


@unittest.skipUnless(PIN.is_file(), "pinned vLLM wheel not available locally (NOT_RUN)")
class Transform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert hashlib.sha256(PIN.read_bytes()).hexdigest() == PIN_SHA
        cls.src = tree(); cls.out, cls.rep = t.transform(dict(cls.src))

    def test_all_prohibited_files_are_gone_and_nothing_else_native_changed(self):
        self.assertEqual(len(self.rep["removed"]), 28); self.assertEqual([p for p in self.out if prohibited.is_prohibited_path(p)], [])
        for n in self.src:
            if prohibited.is_binary(n): self.assertEqual(self.out[n], self.src[n], n)

    def test_the_source_wheel_really_contained_them(self):
        self.assertGreaterEqual(len(prohibited.scan_members(list(self.src), self.src.__getitem__)["path_hits"]), 28)
        self.assertTrue(prohibited.scan_members(list(self.out), self.out.__getitem__)["clean"])

    def test_deterministic(self):
        again, rep = t.transform(dict(self.src)); self.assertEqual(again, self.out); self.assertEqual(rep["stub_entries"], self.rep["stub_entries"])

    def test_no_unguarded_import_of_a_removed_module_remains(self):
        removed = {t.module_name(r["path"]) for r in self.rep["removed"] if r["path"].endswith(".py")}
        for path, data in self.out.items():
            if not path.endswith(".py"): continue
            tr = ast.parse(data.decode()); par = {c: n for n in ast.walk(tr) for c in ast.iter_child_nodes(n)}
            for n in ast.walk(tr):
                mods = [n.module] if isinstance(n, ast.ImportFrom) and n.module and n.level == 0 else [a.name for a in n.names] if isinstance(n, ast.Import) else []
                for m in mods:
                    if m in removed and path != t.STUB_PATH: self.assertTrue(t.guarded(n, par), f"{path}:{n.lineno} imports {m}")

    def test_every_rewritten_file_still_parses_and_stub_names_exist(self):
        for path, data in self.out.items():
            if path.endswith(".py"): ast.parse(data.decode())
        stub = ast.parse(self.out[t.STUB_PATH].decode()); defined = {tg.id for n in stub.body if isinstance(n, ast.Assign) for tg in n.targets if isinstance(tg, ast.Name)}
        for path, data in self.out.items():
            if not path.endswith(".py") or path == t.STUB_PATH: continue
            for n in ast.walk(ast.parse(data.decode())):
                if isinstance(n, ast.ImportFrom) and n.module == t.STUB_MODULE:
                    for a in n.names: self.assertIn(a.name, defined, f"{path}: {a.name}")

    def test_registries_config_and_selector_are_patched(self):
        self.assertIn("raise_if_prohibited_backend", self.out["vllm/v1/attention/backends/registry.py"].decode())
        self.assertIn("raise_if_prohibited_backend", self.out["vllm/v1/attention/backends/mla/prefill/registry.py"].decode())
        self.assertIn("validate_backend_not_prohibited", self.out["vllm/config/attention.py"].decode())
        self.assertIn("import vllm.t4_backend_policy", self.out["vllm/__init__.py"].decode())
        self.assertIn('f"{type(e).__name__}: {e}"', self.out["vllm/platforms/cuda.py"].decode())

    def test_default_path_warmup_placeholders_are_noops_not_errors(self):
        """Kaggle run 2: kernel_warmup() calls these for EVERY model; a raising placeholder made the engine fail to start."""
        tr = ast.parse(self.out["vllm/model_executor/warmup/kernel_warmup.py"].decode()); fn = next(n for n in tr.body if isinstance(n, ast.FunctionDef) and n.name == "kernel_warmup")
        called = {s.value.func.id for s in fn.body if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and isinstance(s.value.func, ast.Name)}     # unconditional top-level calls
        stubs = {n for n in called if n in self.rep["stub_entries"]}
        self.assertIn("flashinfer_sparse_mla_decode_autotune_warmup", stubs); self.assertIn("deepseek_v4_sparse_mla_attention_warmup", stubs)
        for n in stubs: self.assertEqual(self.rep["stub_entries"][n], "none", f"{n} is called unconditionally by kernel_warmup() and must be a no-op")
        stub_src = self.out[t.STUB_PATH].decode(); self.assertIn("none_function('flashinfer_sparse_mla_decode_autotune_warmup')", stub_src)

    def test_anchor_mismatch_fails_the_build(self):
        broken = dict(self.src); broken["vllm/config/attention.py"] = self.src["vllm/config/attention.py"].replace(b'@field_validator("mla_prefill_backend", mode="before")', b"# gone")
        with self.assertRaises(AssertionError): t.transform(broken)

    def test_triton_attention_modules_are_untouched(self):
        for n in ("vllm/v1/attention/backends/triton_attn.py", "vllm/v1/attention/ops/triton_unified_attention.py", "vllm/v1/attention/ops/triton_reshape_and_cache_flash.py"):
            if n in self.src: self.assertEqual(self.out[n], self.src[n], n)


class PolicyModule(unittest.TestCase):
    """vllm/t4_backend_policy.py executed standalone in a SUBPROCESS (it installs an import guard into sys.modules)."""

    def run_policy(self, body, pre=""):
        src = (t.POLICY_SOURCE % {"blocked": tuple(prohibited.BLOCKED_IMPORT_NAMES)})
        with tempfile.TemporaryDirectory() as d:
            (pathlib.Path(d) / "t4_backend_policy.py").write_text(src)
            code = pre + "\nimport sys; sys.path.insert(0, %r)\n" % d + textwrap.dedent(body)
            return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))

    def test_guard_blocks_imports_and_find_spec(self):
        r = self.run_policy("""
            import t4_backend_policy as p, importlib, importlib.util
            for m in ('flashinfer', 'flash_attn', 'flash_mla', 'vllm_flash_attn'):
                try: importlib.import_module(m); print('IMPORTED', m)
                except ModuleNotFoundError: pass
                assert importlib.util.find_spec(m) is None, m
            print('GUARD-OK')""")
        self.assertIn("GUARD-OK", r.stdout, r.stderr)

    def test_a_package_imported_before_vllm_is_refused(self):
        r = self.run_policy("import t4_backend_policy", pre="import sys, types; sys.modules['flashinfer'] = types.ModuleType('flashinfer')")
        self.assertNotEqual(r.returncode, 0); self.assertIn("ProhibitedBackendError", r.stderr)

    def test_errors_predicates_and_placeholders(self):
        r = self.run_policy("""
            import t4_backend_policy as p
            assert issubclass(p.ProhibitedBackendError, ImportError)
            try: p.raise_if_prohibited_backend('FLASH_ATTN'); raise SystemExit('no raise')
            except p.ProhibitedBackendError as e: assert 'TRITON_ATTN' in str(e)
            for n in ('FLASHINFER', 'FLASHMLA_SPARSE', 'FLASH_ATTN_MLA', 'ROCM_AITER_FA'): assert p.is_prohibited_backend(n), n
            for n in ('TRITON_ATTN', 'FLEX_ATTENTION', 'CPU_ATTN', 'TURBOQUANT', 'NO_ATTENTION'): assert not p.is_prohibited_backend(n), n
            p.raise_if_prohibited_backend('CUSTOM', 'my.module.Impl')
            try: p.raise_if_prohibited_backend('CUSTOM', 'vllm.v1.attention.backends.flash_attn.X'); raise SystemExit('no raise 2')
            except p.ProhibitedBackendError: pass
            assert p.false_predicate('has_flashinfer')() is False and p.none_function('get_flash_attn_version')() is None
            K = p.prohibited_class('FlashInferFoo'); assert K.is_supported()[0] is False and K.can_implement(None)[0] is False and K.get_min_capability() > 100
            try: K(); raise SystemExit('instantiated')
            except p.ProhibitedBackendError: pass
            try: p.prohibited_function('flash_attn_varlen_func')(); raise SystemExit('called')
            except p.ProhibitedBackendError: pass
            try: p.check_attention_backend('FLASHINFER'); raise SystemExit('config accepted')
            except ValueError: pass
            p.check_attention_backend('TRITON_ATTN'); p.check_attention_backend(None)
            print('POLICY-OK')""")
        self.assertIn("POLICY-OK", r.stdout, r.stderr + r.stdout)


if __name__ == "__main__":
    unittest.main()
