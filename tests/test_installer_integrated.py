"""Pure decision functions of scripts/kaggle/02_install_integrated.py (no pip, no GPU)."""
import importlib.util, io, pathlib, sys, unittest, zipfile, tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "kaggle"))
spec = importlib.util.spec_from_file_location("inst", ROOT / "scripts/kaggle/02_install_integrated.py"); inst = importlib.util.module_from_spec(spec); spec.loader.exec_module(inst)

KNOWN = "xgrammar 0.2.5.post1 has requirement transformers<5,>=4.38.0, but you have transformers 5.20.2."


class Decisions(unittest.TestCase):
    def test_hash_blocks(self):
        t = "# c\nfoo-bar==1.0 \\\n    --hash=sha256:aa \\\n    --hash=sha256:bb\nbaz==2 \\\n    --hash=sha256:cc\n"
        b = inst.parse_hash_blocks(t); self.assertEqual(set(b), {"foo-bar", "baz"}); self.assertIn("sha256:bb", b["foo-bar"])

    def test_pip_check_classification(self):
        pre = ["gradio 6 has requirement pillow<13, but you have pillow 13."]
        c = inst.classify_pip_check(pre, pre + [KNOWN]); self.assertEqual((c["known"], c["own"], c["other"]), ([KNOWN], [], []))
        c = inst.classify_pip_check(pre, pre + [KNOWN, "vllm 0.25.1 has requirement foo>=2, but you have foo 1."]); self.assertEqual(len(c["own"]), 1)
        c = inst.classify_pip_check(pre, pre + ["msgspec 0.22 has requirement bar, which is not installed."]); self.assertEqual(len(c["other"]), 1)
        c = inst.classify_pip_check(pre, pre + ["xgrammar 0.2.5.post1 has requirement transformers<4, but you have transformers 5.1."]); self.assertEqual(len(c["other"]), 1)   # only the exact documented line is tolerated
        self.assertEqual(inst.classify_pip_check(pre, pre), {"new": [], "known": [], "own": [], "other": []})

    def test_baseline(self):
        ok = {"torch": "2.11.0+cu130", "triton": "3.6.0", "transformers": "5.20.2"}
        self.assertEqual(inst.check_baseline(ok), [])
        self.assertTrue(inst.check_baseline({**ok, "torch": "2.12.0"})); self.assertTrue(inst.check_baseline({**ok, "triton": "3.7.0"}))
        self.assertTrue(inst.check_baseline({**ok, "transformers": "4.57.6"})); self.assertTrue(inst.check_baseline({**ok, "transformers": "6.0.0"}))
        self.assertTrue(inst.check_baseline({**ok, "flashinfer-python": "1"}))

    def test_diff_protects_everything_but_own_distributions(self):
        before = {"torch": "2.11.0+cu130", "numpy": "1.26.4", "vllm": "0.25.1", "stokastisk": "0.1.0"}
        self.assertTrue(inst.diff_snapshots(before, {**before}, [])["ok"])
        self.assertFalse(inst.diff_snapshots(before, {**before, "torch": "2.12.0"}, [])["ok"])
        self.assertFalse(inst.diff_snapshots(before, {k: v for k, v in before.items() if k != "numpy"}, [])["ok"])
        self.assertFalse(inst.diff_snapshots(before, {**before, "flashinfer-python": "1"}, ["xgrammar"])["ok"])
        self.assertTrue(inst.diff_snapshots(before, {**before, "xgrammar": "0.2.5.post1"}, ["xgrammar"])["ok"])

    def test_plan(self):
        from packaging.requirements import Requirement as R
        p = inst.make_plan({"numpy": "2.1.3", "transformers": "5.20.2", "torch": "2.11.0+cu130"}, {"xgrammar": "0.2.5.post1"},
                           [R("numpy"), R("transformers>=5.5.3,<6"), R("torch==2.11.0+cu130"), R("xgrammar==0.2.5.post1")])
        self.assertEqual((p["conflicts"], p["install"]), ([], {"xgrammar": "0.2.5.post1"}))
        self.assertTrue(inst.make_plan({"transformers": "4.57.6"}, {}, [R("transformers>=5.5.3,<6")])["conflicts"])
        self.assertTrue(inst.make_plan({}, {}, [R("msgspec==0.22.0")])["conflicts"])                    # missing and unpinned: abort, never resolve
        self.assertTrue(inst.make_plan({}, {"flashinfer-python": "1"}, [])["conflicts"])

    def test_inspect_wheel(self):
        def whl(extra):
            f = tempfile.NamedTemporaryFile(suffix=".whl", delete=False); z = zipfile.ZipFile(f, "w")
            z.writestr("stokastisk/__init__.py", ""); z.writestr("vllm/__init__.py", ""); z.writestr("stokastisk-0.1.0.dist-info/METADATA", "Name: stokastisk\n")
            z.writestr("stokastisk-0.1.0.dist-info/WHEEL", "Tag: cp313-cp313-linux_x86_64\nBuild: 1\n"); z.writestr("stokastisk-0.1.0.data/platlib/vllm-0.25.1.dist-info/METADATA", "x")
            for n in extra: z.writestr(n, "")
            z.close(); return f.name
        i = inst.inspect_wheel(whl([])); self.assertEqual((i["top_level"], i["prohibited"], i["tag"], i["build"], i["vllm_dist_info_bundled"]), (["stokastisk", "vllm"], [], "cp313-cp313-linux_x86_64", "1", True))
        self.assertTrue(inst.inspect_wheel(whl(["vllm/_vllm_fa2_C.abi3.so"]))["prohibited"]); self.assertTrue(inst.inspect_wheel(whl(["vllm/vllm_flash_attn/__init__.py"]))["prohibited"])
        self.assertEqual(inst.inspect_wheel(whl(["flashinfer/x.py"]))["top_level"], ["flashinfer", "stokastisk", "vllm"])


if __name__ == "__main__":
    unittest.main()
