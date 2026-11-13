"""Dependency-lock completeness and the root causes found in the failed build-2 Kaggle run. CPU only; the closure files are the audit trail of scripts/release/lock_closure.py."""
import ast, importlib.util, json, pathlib, re, sys, unittest, zipfile
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sdk")); sys.path.insert(0, str(ROOT / "scripts/kaggle")); sys.path.insert(0, str(ROOT / "scripts/audit")); sys.path.insert(0, str(ROOT / "scripts/release"))
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name as cn
from stokastisk import dependencies as dep
import import_closure as ic

R = ROOT / "requirements"
PROFILES = ("core", "structured-output", "serving", "models")
BASE = json.loads((R / "kaggle-baseline-2026-11-13.json").read_text())
PIN = ROOT / "dist-rc/stokastisk-0.1.0-kaggle-t4-sm75/wheels/vllm-0.25.1-cp38-abi3-linux_x86_64.whl"


def lock_pins(p):
    return {cn(m.group(1)): m.group(2) for m in re.finditer(r"^([A-Za-z0-9_.\-]+)==(\S+) \\$", (R / f"{p}.lock").read_text(), re.M)}


class Locks(unittest.TestCase):
    def test_every_profile_is_resolved_without_conflict_against_the_kaggle_baseline(self):
        for p in PROFILES:
            c = json.loads((R / "closure" / f"{p}.closure.json").read_text())
            self.assertEqual((c["conflicts"], c["unresolvable"]), ([], []), p)

    def test_locks_contain_only_packages_absent_from_kaggle_and_pin_every_root(self):
        for p in PROFILES:
            pins = lock_pins(p)
            self.assertFalse(set(pins) & set(BASE), (p, sorted(set(pins) & set(BASE))))      # the protected baseline is never reinstalled
            for line in (R / f"{p}.roots").read_text().splitlines():
                line = line.split("#")[0].strip()
                if line:
                    r = Requirement(line); self.assertEqual(str(r.specifier)[:2], "==", line)
                    self.assertEqual(pins.get(cn(r.name)), str(r.specifier)[2:], line)

    def test_every_lock_entry_has_hashes(self):
        for p in PROFILES:
            for blk in re.split(r"\n(?=\S)", (R / f"{p}.lock").read_text()):
                if blk.startswith("#") or not blk.strip(): continue
                self.assertRegex(blk, r"--hash=sha256:[0-9a-f]{64}", blk[:40])

    def test_protected_packages_are_never_in_a_lock(self):
        for p in PROFILES:
            for n in lock_pins(p):
                self.assertFalse(n.startswith(("torch", "triton", "nvidia-")) or n in {"numpy", "transformers", "tokenizers", "pydantic", "pydantic-core", "fastapi", "starlette", "protobuf"}, (p, n))

    def test_wheel_requirements_equal_the_lock_roots(self):
        s = (ROOT / "scripts/release/build_integrated_wheel.py").read_text()
        for prof in ("core", "structured-output", "serving", "models"): self.assertIn(f'roots("{prof}")', s)


class XgrammarConflict(unittest.TestCase):
    """build 2: xgrammar 0.2.5.post1 requires transformers<5 but Kaggle has transformers 5.x and vLLM 0.25.1 needs >=5.5.3."""

    def test_the_locked_xgrammar_is_metadata_compatible_with_transformers_5_and_tvm_ffi_0_1_9(self):
        c = json.loads((R / "closure/core.closure.json").read_text())
        self.assertEqual(c["new_packages"]["xgrammar"]["version"], "0.2.3"); self.assertEqual(c["new_packages"]["apache-tvm-ffi"]["version"], "0.1.9")
        self.assertIn("transformers", c["satisfied_by_baseline"]); self.assertTrue(c["satisfied_by_baseline"]["transformers"]["baseline"].startswith("5."))
        self.assertNotIn("<5", c["satisfied_by_baseline"]["transformers"]["spec"])

    def test_the_build2_pin_is_reproduced_as_a_conflict_by_the_resolver(self):
        from lock_closure import Closure
        from packaging.requirements import Requirement as Rq
        import tempfile
        cache = pathlib.Path(tempfile.mkdtemp())
        meta = {"requires_dist": ["apache-tvm-ffi>=0.1.9", "torch>=1.10.0", "transformers<5,>=4.38.0", "numpy"]}
        c = Closure({**BASE, "apache-tvm-ffi": "0.1.9"}, cache, offline=True)       # tvm-ffi 0.1.9 as the lock installs it; everything else is the real Kaggle baseline
        c.walk = lambda *a, **k: None
        c.chosen = {}
        # emulate xgrammar 0.2.5.post1's requirements against the Kaggle baseline
        for rq in meta["requires_dist"]: c.add(Rq(rq), "xgrammar==0.2.5.post1")
        self.assertTrue(any(x["package"] == "transformers" for x in c.conflicts), c.conflicts)


class StructuredOutputRootCause(unittest.TestCase):
    """build 2: 'No module named lmformatenforcer' - vLLM imports backend_lm_format_enforcer for EVERY structured request."""

    @unittest.skipUnless(PIN.is_file(), "pinned vLLM wheel not available (NOT_RUN)")
    def test_lmformatenforcer_is_imported_at_import_time_by_every_structured_output_validation(self):
        z = zipfile.ZipFile(PIN)
        src = z.read("vllm/v1/structured_output/backend_lm_format_enforcer.py").decode()
        self.assertNotIn("from __future__ import annotations", src)                      # => the dataclass annotation below is evaluated at import
        self.assertIn("token_enforcer: lmformatenforcer.TokenEnforcer", src)
        sp = z.read("vllm/sampling_params.py").decode()
        self.assertIn("from vllm.v1.structured_output.backend_lm_format_enforcer import (", sp)
        fn = sp[sp.index("def _validate_structured_outputs"):]
        self.assertLess(fn.index("backend_lm_format_enforcer"), fn.index('if backend.startswith("xgrammar")'))      # imported BEFORE the backend is even looked at
        self.assertIn("lm-format-enforcer", lock_pins("structured-output")); self.assertIn("lmformatenforcer", dep.LOCKED["structured-output"])

    def test_registry_names_match_locks(self):
        for prof in ("core", "structured-output", "serving"):
            self.assertTrue(dep.LOCKED[prof]); self.assertLessEqual(len(dep.LOCKED[prof]), len(lock_pins(prof)) + 2)
        self.assertEqual(set(dep.PROFILES), {"core", "structured-output", "serving"})
        for need in ("model_hosting_container_standards", "prometheus_fastapi_instrumentator", "sse_starlette", "fastapi_cli", "pydantic_settings"): self.assertIn(need, dep.LOCKED["serving"])

    def test_missing_dependency_detection(self):
        self.assertEqual(dep.missing(("json", "no_such_module_xyz")), ["no_such_module_xyz"])
        from stokastisk import serving
        self.assertEqual(serving.check_server_requirements(("os", "no_such_module_xyz")), ["no_such_module_xyz"])
        self.assertIn("model_hosting_container_standards", serving.SERVER_REQUIREMENTS)       # the module build 2 died on
        self.assertIn("lmformatenforcer", serving.SERVER_REQUIREMENTS)


@unittest.skipUnless(PIN.is_file(), "pinned vLLM wheel not available (NOT_RUN)")
class ImportClosure(unittest.TestCase):
    """Every third-party module imported at module level on the vllm.LLM / vllm serve / structured-output paths is provided by the Kaggle baseline or by a lock."""

    @classmethod
    def setUpClass(cls):
        import t4_flash_free, tempfile
        z = zipfile.ZipFile(PIN); files = {n: z.read(n) for n in z.namelist() if n.startswith("vllm/") and not n.endswith("/")}
        out, _ = t4_flash_free.transform(files)
        cls.root = pathlib.Path(tempfile.mkdtemp())
        for n, d in out.items():
            p = cls.root / n; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(d)
        for n in ("sdk/stokastisk",):
            for f in (ROOT / n).rglob("*.py"):
                p = cls.root / "stokastisk" / f.relative_to(ROOT / n); p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(f.read_bytes())
        cls.mods = ic.load_tree(cls.root)
        provided = {cn(k) for k in BASE} | {cn(k) for p in PROFILES[:3] for k in lock_pins(p)}
        cls.provided = provided

    def third_party(self, *entries):
        _, third = ic.closure(self.mods, list(entries))
        eager = {k: v for k, v in third.items() if v.get("eager")}
        return {k: ic.IMPORT_TO_DIST.get(k, k.replace("_", "-")) for k in eager}

    def assertProvided(self, entries):
        optional = {"modelscope": "imported only when VLLM_USE_MODELSCOPE=1 (vllm/transformers_utils/config.py)"}
        missing = {k: d for k, d in self.third_party(*entries).items() if cn(d) not in self.provided and k not in ("torch", "triton") and k not in optional}
        self.assertEqual(missing, {}, f"module-level imports not provided by the Kaggle baseline or a lock: {missing}")

    def test_offline_llm_path(self):
        self.assertProvided(["vllm", "vllm.sampling_params", "vllm.entrypoints.llm", "vllm.v1.structured_output", "stokastisk", "stokastisk.engine"])

    def test_serve_path(self):
        self.assertProvided(["vllm.entrypoints.cli.main", "vllm.entrypoints.openai.api_server", "stokastisk.serving"])

    def test_the_serve_path_really_needs_model_hosting_container_standards(self):
        self.assertIn("model_hosting_container_standards", self.third_party("vllm.entrypoints.openai.api_server"))


if __name__ == "__main__":
    unittest.main()
