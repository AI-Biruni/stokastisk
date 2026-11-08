"""CPU tests for scripts/validation/validate_notebooks.py (static structure only; says nothing about Kaggle behaviour)."""
import copy, json, pathlib, shutil, sys, tempfile, unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/validation"))
import validate_notebooks as vn  # noqa: E402


class StaticNotebookValidation(unittest.TestCase):
    def mutated(self, fn, name="03_stokastisk_single_gpu_validation"):
        td = tempfile.TemporaryDirectory(); self.addCleanup(td.cleanup)
        for p in (ROOT / "notebooks").glob("*.ipynb"): shutil.copy(p, td.name)
        f = pathlib.Path(td.name) / f"{name}.ipynb"; nb = json.loads(f.read_text()); fn(nb)
        f.write_text(json.dumps(nb)); return pathlib.Path(td.name)

    def run_dir(self, d):
        return vn.main(["--notebooks", str(d)])

    def test_repository_notebooks_pass_and_none_is_marked_executed(self):
        self.assertEqual(vn.main([]), 0)
        for p in (ROOT / "notebooks").glob("*.ipynb"):
            nb = json.loads(p.read_text())
            self.assertTrue(all(not c.get("outputs") and c.get("execution_count") is None for c in nb["cells"] if c["cell_type"] == "code"), p.name)

    def test_exactly_twelve_expected_names(self):
        self.assertEqual(len(vn.EXPECTED), 12)
        self.assertEqual(sorted(p.stem for p in (ROOT / "notebooks").glob("*.ipynb")), vn.EXPECTED)

    def test_detects_wrong_release(self):
        def m(nb):
            for c in nb["cells"]:
                if c["cell_type"] == "code": c["source"] = [s.replace(vn.RELEASE, vn.RELEASE.replace(vn.LOCKED, "0.9.9")) for s in c["source"]]
        self.assertEqual(self.run_dir(self.mutated(m)), 1)

    def test_detects_syntax_error_undefined_name_secret_and_outputs(self):
        def add(src, **kw):
            def m(nb): nb["cells"].append({"cell_type": "code", "metadata": {}, "source": [src], "outputs": kw.get("outputs", []), "execution_count": kw.get("ec")})
            return m
        for fn in (add("def broken(:\n"), add("print(never_defined_anywhere)"), add("T = 'hf_" + "A" * 30 + "'"),
                   add("x = 1", outputs=[{"output_type": "stream"}]), add("x = 1", ec=3), add("!pip install torch==2.0")):
            self.assertEqual(self.run_dir(self.mutated(fn)), 1)

    def test_detects_malformed_json_and_missing_notebook(self):
        td = tempfile.TemporaryDirectory(); self.addCleanup(td.cleanup)
        for p in (ROOT / "notebooks").glob("*.ipynb"): shutil.copy(p, td.name)
        (pathlib.Path(td.name) / "05_stokastisk_vllm_inference.ipynb").write_text("{broken")
        self.assertEqual(self.run_dir(td.name), 1)
        (pathlib.Path(td.name) / "05_stokastisk_vllm_inference.ipynb").unlink()
        self.assertEqual(self.run_dir(td.name), 1)

    def test_detects_missing_pending_statement(self):
        self.assertEqual(self.run_dir(self.mutated(lambda nb: nb["cells"][0].update(source=["# Title only"]))), 1)

    def test_all_notebooks_embed_the_shipped_downloader(self):
        src = (ROOT / "scripts/fetch_release.py").read_text()
        want = src[src.index("def _download(url"):src.index("def _sha256(")].strip()
        for p in (ROOT / "notebooks").glob("*.ipynb"):
            code = "".join("".join(c["source"]) for c in json.loads(p.read_text())["cells"])
            self.assertIn(want, code, p.name)

    def test_report_marks_kaggle_not_run(self):
        out = pathlib.Path(tempfile.mkdtemp()) / "r.json"; vn.main(["--json", str(out)])
        r = json.loads(out.read_text())
        self.assertTrue(all(v["kaggle_status"] == "NOT_RUN" for v in r["notebooks"].values()))
        shutil.rmtree(out.parent)


if __name__ == "__main__":
    unittest.main()
