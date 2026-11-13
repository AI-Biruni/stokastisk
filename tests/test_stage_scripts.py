"""Kaggle stage scripts: they parse, their structured-output schema is BOUNDED (Kaggle: an unbounded integer ran to max_tokens), and the summary's runtime check works when a stage ERRORs."""
import ast, json, os, pathlib, subprocess, sys, tempfile, unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]
K = ROOT / "scripts" / "kaggle"
os.environ.setdefault("STOKASTISK_OUT", tempfile.mkdtemp())
sys.path.insert(0, str(K)); sys.path.insert(0, str(ROOT / "sdk"))
WHEEL = "stokastisk-0.1.0-3-cp313-cp313-linux_x86_64.whl"
RT = {"python": "3.13.16", "torch": "2.11.0+cu130", "triton": "3.6.0", "vllm": "0.25.1", "stokastisk": "0.1.0", "transformers": "5.20.2", "xgrammar": "0.2.3"}


class Stages(unittest.TestCase):
    def test_every_script_parses(self):
        for f in K.glob("*.py"): ast.parse(f.read_text())

    def test_the_acceptance_schema_is_bounded(self):
        import candidate_15_common as c
        props = c.ENGINE_SCHEMA["properties"]
        self.assertIn("maximum", props["population"]); self.assertIn("minimum", props["population"]); self.assertIn("maxLength", props["city"])
        from stokastisk.structured import check_schema
        check_schema(c.ENGINE_SCHEMA)

    def test_attempt_turns_an_exception_into_an_error_check_and_continues(self):
        import candidate_15_common as c
        R = []; c.attempt(R, "boom", lambda: 1 / 0); self.assertEqual([r["status"] for r in R], ["ERROR"]); self.assertIn("ZeroDivisionError", R[0]["detail"])
        self.assertEqual(c.attempt(R, "fine", lambda: 7), 7)

    def test_runtime_versions_need_no_import_of_torch_or_cuda(self):
        import _common
        before = set(sys.modules)
        rv = _common.runtime_versions()
        self.assertEqual(rv["python"], sys.version.split()[0]); self.assertEqual(set(sys.modules) & {"torch", "vllm"}, set(sys.modules) & before & {"torch", "vllm"})

    def test_summary_compares_the_runtime_of_all_stages_even_when_one_errored(self):
        with tempfile.TemporaryDirectory() as d:
            for m, st in (("static", "PASS"), ("engine", "ERROR"), ("parallel", "PASS"), ("server", "FAIL")):
                json.dump({"stage": m, "status": st, "checks": [], "provenance": {"wheel": WHEEL, "wheel_sha256": "a" * 64, "runtime": RT}}, open(f"{d}/15c_{m}.json", "w"))
            pathlib.Path(f"{d}/prereq-install").mkdir()
            json.dump({"status": "PASS", "wheel_sha256": "a" * 64, "provenance": {"wheel": WHEEL, "wheel_sha256": "a" * 64, "runtime": RT}}, open(f"{d}/prereq-install/02_install_integrated.json", "w"))
            json.dump({"wheel": WHEEL, "wheel_sha256": "a" * 64, "source": "attached-directory"}, open(f"{d}/provenance.json", "w"))
            r = subprocess.run([sys.executable, "candidate_15_integration.py", "summary"], cwd=K, env=dict(os.environ, STOKASTISK_OUT=d, PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True)
            doc = json.load(open(f"{d}/15c_summary.json"))
            runtime = next(c for c in doc["checks"] if "same Python" in c["check"]); self.assertEqual(runtime["status"], "PASS", runtime)
            self.assertNotEqual(doc["status"], "PASS"); self.assertEqual(r.returncode, 1)
            drift = json.loads(json.dumps(RT)); drift["torch"] = "2.12.0"
            json.dump({"stage": "server", "status": "PASS", "checks": [], "provenance": {"wheel": WHEEL, "wheel_sha256": "a" * 64, "runtime": drift}}, open(f"{d}/15c_server.json", "w"))
            subprocess.run([sys.executable, "candidate_15_integration.py", "summary"], cwd=K, env=dict(os.environ, STOKASTISK_OUT=d, PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True)
            self.assertEqual(next(c for c in json.load(open(f"{d}/15c_summary.json"))["checks"] if "same Python" in c["check"])["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
