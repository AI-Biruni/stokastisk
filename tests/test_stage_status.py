"""Result vocabulary and stage status (PASS FAIL ERROR WARNING SKIPPED NOT_RUN): only PASS passes; a missing report is never a pass."""
import json, os, pathlib, subprocess, sys, tempfile, unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]
K = ROOT / "scripts" / "kaggle"
os.environ.setdefault("STOKASTISK_OUT", tempfile.mkdtemp())
sys.path.insert(0, str(K))
import _common as c


def st(*statuses):
    return c.stage_status([{"status": s} for s in statuses])


class StageStatus(unittest.TestCase):
    def test_table(self):
        self.assertEqual(c.STATUSES, ("PASS", "FAIL", "ERROR", "WARNING", "SKIPPED", "NOT_RUN"))
        self.assertEqual(st(), "NOT_RUN")                                    # no checks recorded
        self.assertEqual(st("PASS", "PASS"), "PASS")
        self.assertEqual(st("PASS", "WARNING"), "WARNING")                   # a warning is not a pass
        self.assertEqual(st("WARNING"), "WARNING")                           # a warning-only stage is not a pass
        self.assertEqual(st("PASS", "SKIPPED"), "WARNING")                   # a skipped check means incomplete evidence
        self.assertEqual(st("SKIPPED"), "SKIPPED")
        self.assertEqual(st("PASS", "FAIL", "WARNING"), "FAIL")
        self.assertEqual(st("PASS", "ERROR", "FAIL"), "ERROR")
        self.assertEqual(st("PASS", "NOT_RUN"), "NOT_RUN")
        self.assertEqual(st("PASS", "INSUFFICIENT_EVIDENCE"), "FAIL")        # anything outside the vocabulary fails

    def test_finish_exit_codes(self):
        for statuses, rc in ((["PASS"], 0), (["PASS", "WARNING"], 1), (["FAIL"], 1), ([], 1)):
            with tempfile.TemporaryDirectory() as d:
                code = ("import sys; sys.path.insert(0, %r); import _common as c; R=[{'check':'x','status':s,'detail':''} for s in %r]; c.finish('t', R)" % (str(K), statuses))
                r = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, STOKASTISK_OUT=d, PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True)
                self.assertEqual(r.returncode, rc, (statuses, r.stderr[-300:]))
                doc = json.load(open(f"{d}/t.json")); self.assertIn("provenance", doc); self.assertEqual(doc["status"], st(*statuses))

    def test_summary_treats_a_missing_report_as_not_run_and_never_passes(self):
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([sys.executable, "candidate_15_integration.py", "summary"], cwd=K, env=dict(os.environ, STOKASTISK_OUT=d, PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True)
            self.assertEqual(r.returncode, 1)
            doc = json.load(open(f"{d}/15c_summary.json"))
            self.assertEqual(doc["modes"], {"static": "NOT_RUN", "engine": "NOT_RUN", "server": "NOT_RUN", "installer": "NOT_RUN"})
            self.assertNotEqual(doc["status"], "PASS"); self.assertNotEqual(doc["overall"], "PASS")

    def test_summary_rejects_stages_that_describe_different_wheels(self):
        with tempfile.TemporaryDirectory() as d:
            for m, sha in (("static", "a" * 64), ("engine", "b" * 64), ("server", "a" * 64)):
                json.dump({"stage": m, "status": "PASS", "checks": [], "versions": {"python": "3.13", "torch": "t", "triton": "3", "vllm": "0.25.1", "stokastisk": "0.1.0", "transformers": "5"},
                           "provenance": {"wheel": "stokastisk-0.1.0-3-cp313-cp313-linux_x86_64.whl", "wheel_sha256": sha}}, open(f"{d}/15c_{m}.json", "w"))
            json.dump({"wheel": "stokastisk-0.1.0-3-cp313-cp313-linux_x86_64.whl", "wheel_sha256": "a" * 64, "source": "attached-zip"}, open(f"{d}/provenance.json", "w"))
            r = subprocess.run([sys.executable, "candidate_15_integration.py", "summary"], cwd=K, env=dict(os.environ, STOKASTISK_OUT=d, PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True)
            self.assertEqual(r.returncode, 1)
            doc = json.load(open(f"{d}/15c_summary.json"))
            self.assertTrue(any("SAME wheel SHA256" in x["check"] and x["status"] == "FAIL" for x in doc["checks"]))


if __name__ == "__main__":
    unittest.main()
