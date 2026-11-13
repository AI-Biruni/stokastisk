"""Inventory comparison (regression for the build-2 'eight missing packages': two different name normalisers on one comparison)."""
import pathlib, sys, unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "kaggle"))
import _inventory as inv

DOTTED = {"jaraco.classes": "3.4.0", "jaraco.context": "6.1.2", "jaraco.functools": "4.6.0", "keyrings.google-artifactregistry-auth": "1.1.2", "lazr.restfulclient": "0.14.6",
          "lazr.uri": "1.0.6", "path.py": "12.5.0", "ruamel.yaml": "0.19.1"}


class Inventory(unittest.TestCase):
    def test_canon_is_pep503(self):
        for raw, want in (("jaraco.classes", "jaraco-classes"), ("ruamel.yaml", "ruamel-yaml"), ("Pillow", "pillow"), ("typing_extensions", "typing-extensions"), ("a._-b", "a-b")):
            self.assertEqual(inv.canon(raw), want)

    def test_build2_bug_reproduced_and_fixed(self):
        installer_style = {inv.canon(k): v for k, v in DOTTED.items()}                      # what the installer stored
        buggy_live = {k.lower().replace("_", "-"): v for k, v in DOTTED.items()}            # what the static stage compared with: dots survive
        missing = {n for n in installer_style if n not in buggy_live}
        self.assertEqual(len(missing), 8, "the old normaliser makes all eight dotted names look removed")
        live = {inv.canon(k): v for k, v in DOTTED.items()}                                 # one implementation, both sides
        self.assertTrue(inv.same_inventory(installer_style, live)["ok"])

    def test_diff_rules(self):
        before = {"torch": "2.11.0+cu130", "numpy": "1.26.4", "vllm": "0.25.1"}
        self.assertTrue(inv.diff(before, dict(before), expected_new=[])["ok"])
        self.assertEqual(inv.diff(before, {**before, "torch": "2.12.0"})["changed"], {"torch": ["2.11.0+cu130", "2.12.0"]})
        self.assertEqual(inv.diff(before, {"torch": before["torch"], "vllm": "0.25.1"})["removed"], ["numpy"])
        d = inv.diff(before, {**before, "flashinfer-python": "1"}, expected_new=["xgrammar"])
        self.assertEqual(d["unexpected_added"], ["flashinfer-python"]); self.assertEqual(d["expected_but_absent"], ["xgrammar"]); self.assertFalse(d["ok"])
        self.assertTrue(inv.diff(before, {**before, "xgrammar": "0.2.3"}, expected_new=["XGrammar"])["ok"])     # names are canonicalised on both sides
        self.assertTrue(inv.diff(before, {**before, "vllm": "9"}, own=("vllm",))["ok"])

    def test_snapshot_uses_one_method_and_reports_duplicates(self):
        s = inv.snapshot()
        self.assertEqual(s["method"], "python -m pip list --format=json"); self.assertEqual(s["count"], len(s["packages"]))
        self.assertTrue(all(k == inv.canon(k) for k in s["packages"]))
        self.assertIsInstance(s["duplicates"], dict)


if __name__ == "__main__":
    unittest.main()
