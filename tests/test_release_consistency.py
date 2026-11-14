"""One wheel identity everywhere: RELEASE.json is the source of truth; the install command, the file table, the pip index generator and the notebooks must agree with it. CPU-only."""
import json, pathlib, re, sys, unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/release"))
import make_pip_index as mpi  # noqa: E402

REL = json.loads((ROOT / "RELEASE.json").read_text())
CUR = REL["builds"][str(REL["current_build"])]
OLD = REL["builds"]["3"]
PIN = f'python -m pip install --index-url https://pypi.org/simple "stokastisk @ {CUR["url"]}#sha256={CUR["sha256"]}"'
DOCS = ["README.md", "INSTALL.md", "docs/HUB_README.md", "docs/guide/getting-started.md"]


class ReleaseConsistency(unittest.TestCase):
    def test_project_version_and_build_tags(self):
        self.assertEqual(REL["project_version"], "0.1.0"); self.assertEqual((ROOT / "VERSION").read_text(), "0.1.0\n")
        self.assertRegex(CUR["filename"], r"^stokastisk-0\.1\.0-4-cp313-cp313-linux_x86_64\.whl$"); self.assertRegex(OLD["filename"], r"^stokastisk-0\.1\.0-3-cp313")
        self.assertEqual(OLD["sha256"], "72039ca4044e48fdf5edca8051dcb48b42fe61a69e07715fa8803e029581fe12")        # build 3 is never replaced
        self.assertEqual(REL["base_vllm_wheel"]["sha256"], "cc3bcafb6516791990ca302073ae153a08465d3df1e460d666e3ec09b4e98f5e")
        self.assertTrue(CUR["url"].endswith("/" + CUR["filename"])); self.assertEqual(re.fullmatch(r"[0-9a-f]{64}", CUR["sha256"]) is not None, True)

    def test_pinned_command_everywhere(self):
        for d in DOCS:
            self.assertIn(PIN, (ROOT / d).read_text(), d)
        for nb in ("notebooks/18_stokastisk_qwen25_t4.ipynb", "notebooks/19_stokastisk_qwen25_coder_7b_dual_t4.ipynb"):
            self.assertIn("!" + PIN, "".join("".join(c["source"]) for c in json.loads((ROOT / nb).read_text())["cells"]), nb)

    def test_no_document_recommends_the_extra_index_as_the_install(self):
        for d in ("README.md", "INSTALL.md", "docs/HUB_README.md", "docs/guide/getting-started.md", *(f"notebooks/{n}" for n in ("18_stokastisk_qwen25_t4.ipynb", "19_stokastisk_qwen25_coder_7b_dual_t4.ipynb"))):
            t = (ROOT / d).read_text()
            for m in re.finditer(r"^(?:!)?(?:python -m )?pip install stokastisk --extra-index-url", t, re.M):
                self.fail(f"{d} uses --extra-index-url as its install command")

    def test_hash_and_size_tables_agree(self):
        for d in ("RELEASE.md", "docs/HUB_README.md", "docs/PRODUCTION_READINESS.md", "INSTALL.md"):
            t = (ROOT / d).read_text()
            self.assertIn(CUR["sha256"], t, d); self.assertIn(OLD["sha256"], t, d)
        for d in ("RELEASE.md", "docs/HUB_README.md", "docs/PRODUCTION_READINESS.md"):
            t = (ROOT / d).read_text()
            for w in (CUR, OLD):
                self.assertIn(f"{w['size']:,}", t, d)
        self.assertIn(REL["pip_index"], (ROOT / "scripts/release/make_pip_index.py").read_text())
        self.assertEqual(mpi.INDEX, REL["pip_index"]); self.assertEqual(mpi.HF + CUR["filename"], CUR["url"])

    def test_build4_is_never_called_kaggle_executed(self):
        self.assertFalse(CUR["executed_on_kaggle"]); self.assertTrue(OLD["executed_on_kaggle"])
        for d in ("README.md", "RELEASE.md", "docs/PRODUCTION_READINESS.md", "docs/HUB_README.md"):
            t = (ROOT / d).read_text()
            self.assertRegex(t, r"(?i)not (itself )?(separately )?(been )?(run|executed)( on Kaggle)?|not run", d)

    def test_wheel_description_has_no_hash(self):
        t = (ROOT / "README.wheel.md").read_text()
        self.assertNotIn(CUR["sha256"], t); self.assertNotIn(OLD["sha256"], t); self.assertIn('readme = "README.wheel.md"', (ROOT / "pyproject.toml").read_text())


if __name__ == "__main__":
    unittest.main()
