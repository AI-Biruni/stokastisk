"""Static checks of the integrated-release assets (CPU only; says nothing about Kaggle behaviour)."""
import json, pathlib, re, unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks/kaggle-t4-sm75"


class IntegratedAssets(unittest.TestCase):
    def test_scripts_compile(self):
        for f in ("scripts/kaggle/02_install_integrated.py", "scripts/kaggle/candidate_15_integration.py", "scripts/release/build_sdk_candidate_wheel.py",
                  "scripts/release/assemble_integrated_release.py", "scripts/release/gen_pin_hashes.py", "scripts/audit/verify_sdk_candidate.py", "scripts/audit/xgrammar_cpu_check.py"):
            compile((ROOT / f).read_text(), f, "exec")

    def test_notebook15_is_unexecuted_and_calls_existing_scripts(self):
        nb = json.loads((NB / "integrated/15_stokastisk_candidate_sdk_integration.ipynb").read_text())
        head = "".join(nb["cells"][0]["source"]); self.assertIn("UNEXECUTED", head); self.assertIn("PENDING", head)
        code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        for s in re.findall(r'"([\w]+\.py)"', code):
            self.assertTrue((ROOT / "scripts/kaggle" / s).is_file(), s)
        self.assertTrue(all(not c.get("outputs") and c.get("execution_count") is None for c in nb["cells"] if c["cell_type"] == "code"))
        for mode in ("static", "engine", "server", "summary"): self.assertIn(f'"{mode}"', code)

    def test_executed_notebooks_13_14_no_longer_claim_to_be_unexecuted(self):
        for n in ("13_stokastisk_tp_pp_fulfilment", "14_stokastisk_replica_balance"):
            head = "".join(json.loads((NB / f"notebooks/{n}.ipynb").read_text())["cells"][0]["source"])
            self.assertIn("KAGGLE-EXECUTED", head); self.assertNotIn("UNEXECUTED", head); self.assertNotIn("verification pending", head)

    def test_installer_refuses_wrong_build_and_old_installer_refuses_one_wheel(self):
        old = (ROOT / "scripts/kaggle/02_install.py").read_text(); new = (ROOT / "scripts/kaggle/02_install_integrated.py").read_text()
        self.assertIn('"exactly two wheels"', old); self.assertIn("exactly one wheel", new)
        self.assertIn("--require-hashes", new); self.assertIn("--no-deps", new); self.assertIn("--no-index", new)
        self.assertNotRegex(new, r"pip\(\s*\"install\"[^\n]*(?<!--no-deps )torch")

    def test_pins_all_have_hashes(self):
        pins = [l.split("==")[0].strip().replace("_", "-") for l in (ROOT / "requirements/kaggle-t4-core.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
        h = (ROOT / "requirements/kaggle-t4-core.hashes.txt").read_text()
        for p in pins: self.assertIn(p + "==", h, f"{p} has no hash block")
        self.assertGreaterEqual(h.count("--hash=sha256:"), len(pins))

    def test_evidence_preserved(self):
        e = ROOT / "evidence/kaggle-legacy-two-wheel"
        for f in ("2026-11-12-notebook-13/15_fulfil_summary.json", "2026-11-12-notebook-14/16_replica_balance.json", "notebooks-executed/13-stokastisk-tp-pp-fulfilment.ipynb",
                  "notebooks-executed/14-stokastisk-replica-balance.ipynb", "SHA256SUMS-original-artifacts"):
            self.assertTrue((e / f).is_file(), f)
        self.assertEqual(json.loads((e / "2026-11-12-notebook-14/16_replica_balance.json").read_text())["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
