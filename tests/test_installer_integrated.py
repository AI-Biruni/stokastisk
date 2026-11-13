"""Pure decision functions of scripts/kaggle/02_install_integrated.py (build 3). No pip install, no network, no GPU."""
import importlib.util, io, os, pathlib, sys, tempfile, unittest, zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("STOKASTISK_OUT", tempfile.mkdtemp())
sys.path.insert(0, str(ROOT / "scripts" / "kaggle"))
spec = importlib.util.spec_from_file_location("inst", ROOT / "scripts/kaggle/02_install_integrated.py"); inst = importlib.util.module_from_spec(spec); spec.loader.exec_module(inst)

LOCK_TEXT = "# c\nfoo-bar==1.0 \\\n    --hash=sha256:" + "a" * 64 + " \\\n    --hash=sha256:" + "b" * 64 + "\nBaz_Qux==2 \\\n    --hash=sha256:" + "c" * 64 + "\n"


class Locks(unittest.TestCase):
    def test_parse(self):
        l = inst.parse_lock(LOCK_TEXT); self.assertEqual(set(l), {"foo-bar", "baz-qux"}); self.assertEqual(len(l["foo-bar"]["hashes"]), 2); self.assertEqual(l["baz-qux"]["version"], "2")

    def test_entry_without_hash_is_rejected(self):
        with self.assertRaises(ValueError): inst.parse_lock("foo==1\n")
        with self.assertRaises(ValueError): inst.parse_lock("not a requirement\n")

    def test_every_shipped_lock_parses_and_is_fully_hashed(self):
        for prof in ("core", "structured-output", "serving", "models"):
            l = inst.parse_lock((ROOT / "requirements" / f"{prof}.lock").read_text()); self.assertTrue(l, prof)
            for n, e in l.items(): self.assertTrue(e["hashes"] and all(len(h) == 64 for h in e["hashes"]), (prof, n))


class Plan(unittest.TestCase):
    L = {"core": inst.parse_lock(LOCK_TEXT)}

    def test_absent_is_installed_equal_is_kept_different_is_a_conflict(self):
        self.assertEqual(inst.make_plan({}, self.L)["install"], {"foo-bar": "1.0", "baz-qux": "2"})
        p = inst.make_plan({"foo-bar": "1.0"}, self.L); self.assertEqual((p["install"], p["keep"]), ({"baz-qux": "2"}, {"foo-bar": "1.0"}))
        p = inst.make_plan({"foo-bar": "0.9"}, self.L); self.assertEqual(len(p["conflicts"]), 1); self.assertIn("protected", p["conflicts"][0])

    def test_two_locks_pinning_one_package_differently_conflict(self):
        L = {"a": inst.parse_lock("x==1 \\\n    --hash=sha256:" + "a" * 64 + "\n"), "b": inst.parse_lock("x==2 \\\n    --hash=sha256:" + "b" * 64 + "\n")}
        self.assertTrue(inst.make_plan({}, L)["conflicts"])


class Baseline(unittest.TestCase):
    OK = {"torch": "2.11.0+cu130", "triton": "3.6.0", "transformers": "5.20.2"}

    def test_baseline(self):
        self.assertEqual(inst.check_baseline(self.OK), [])
        for bad in ({"torch": "2.12.0"}, {"triton": "3.7.0"}, {"transformers": "4.57.6"}, {"transformers": "5.5.2"}, {"transformers": "6.0.0"}, {"flashinfer-python": "1"}, {"flash_attn": "2"}):
            self.assertTrue(inst.check_baseline({**self.OK, **bad}), bad)


class PipCheck(unittest.TestCase):
    PRE = ["gradio 6 has requirement pillow<13, but you have pillow 13."]

    def test_no_new_conflict_is_tolerated_in_build3(self):
        self.assertEqual(inst.classify_pip_check(self.PRE, self.PRE)["new"], [])
        xg = "xgrammar 0.2.5.post1 has requirement transformers<5,>=4.38.0, but you have transformers 5.20.2."
        c = inst.classify_pip_check(self.PRE, self.PRE + [xg]); self.assertEqual(c["other"], [xg])              # the build-2 'documented' conflict is now just a failure
        c = inst.classify_pip_check(self.PRE, self.PRE + ["vllm 0.25.1 has requirement foo>=2, but you have foo 1."]); self.assertEqual(len(c["own"]), 1)
        c = inst.classify_pip_check(self.PRE, self.PRE + ["stokastisk 0.1.0 requires lm-format-enforcer==0.11.3, which is not installed."]); self.assertEqual(len(c["own"]), 1)


class WheelInspection(unittest.TestCase):
    def wheel(self, files):
        p = pathlib.Path(tempfile.mkdtemp()) / "w.whl"
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("stokastisk-0.1.0.dist-info/METADATA", "Name: stokastisk\n"); z.writestr("stokastisk-0.1.0.dist-info/WHEEL", "Tag: cp313-cp313-linux_x86_64\nBuild: 3\n")
            z.writestr("stokastisk-0.1.0.data/platlib/vllm-0.25.1.dist-info/METADATA", "x")
            for n, d in files.items(): z.writestr(n, d)
        return p

    def test_clean_wheel(self):
        w = inst.inspect_wheel(self.wheel({"stokastisk/__init__.py": "", "vllm/__init__.py": "", "vllm/ops/triton_reshape_and_cache_flash.py": "# kernel"}))
        self.assertTrue(w["scan"]["clean"]); self.assertEqual((w["tag"], w["build"], w["top_level"]), ("cp313-cp313-linux_x86_64", "3", ["stokastisk", "vllm"]))

    def test_every_flash_form_is_refused(self):
        for name, data in (("vllm/v1/attention/backends/flash_attn.py", ""), ("vllm/utils/flashinfer.py", ""), ("vllm/x/__pycache__/flashinfer.cpython-313.pyc", b"\0"),
                           ("vllm/libflash_fwd.so", b"\x7fELF"), ("vllm/y/fa_utils.py", ""), ("vllm/vllm_flash_attn/__init__.py", ""), ("vllm/z.so", b"\x7fELF flash_fwd_kernel"), ("vllm/w.cubin", b"flashinfer")):
            w = inst.inspect_wheel(self.wheel({"stokastisk/__init__.py": "", name: data})); self.assertFalse(w["scan"]["clean"], name)

    def test_documentation_text_is_not_a_false_positive(self):
        w = inst.inspect_wheel(self.wheel({"stokastisk/__init__.py": "", "vllm/doc.py": '"""FlashAttention and FlashInfer are not supported; flash_attn is mentioned in prose only."""\nX = 1\n',
                                           "vllm/models/longcat_flash.py": "", "vllm/dflash.py": ""}))
        self.assertTrue(w["scan"]["clean"], w["scan"])


class Sources(unittest.TestCase):
    def test_installer_is_hash_pinned_offline_and_resolver_free(self):
        t = (ROOT / "scripts/kaggle/02_install_integrated.py").read_text()
        for s in ("--require-hashes", "--no-deps", "--no-index", "--only-binary=:all:", "--find-links"): self.assertIn(s, t)
        self.assertNotIn('"--upgrade"', t); self.assertNotIn("--force-reinstall\", \"torch", t)


if __name__ == "__main__":
    unittest.main()
