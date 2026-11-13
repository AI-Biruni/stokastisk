"""README/INSTALL examples: Python blocks parse, the locked version is stated, and no document points at removed material. CPU-only."""
import ast, pathlib, re, unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", ROOT / "INSTALL.md", ROOT / "RELEASE.md", ROOT / "docs/HUB_README.md"]


def python_blocks(text):
    return re.findall(r"```python\n(.*?)```", text, re.S)


class ReadmeExamples(unittest.TestCase):
    def test_blocks_parse(self):
        for d in DOCS:
            for b in python_blocks(d.read_text()):
                ast.parse(b)

    def test_documents_state_the_locked_version_and_the_unverified_gpu_status(self):
        for d in (ROOT / "README.md", ROOT / "RELEASE.md", ROOT / "docs/HUB_README.md"):
            t = d.read_text()
            self.assertIn("0.1.0", t); self.assertRegex(t, r"[Vv]erification pending|pending", d.name)

    def test_no_document_references_removed_files(self):
        gone = ("fetch_release.py", "02_install.py", "docs/VLLM_T4_COVERAGE.md", "docs/KAGGLE_T4_VERIFICATION.md", "STOKASTISK_0.1.0_FINAL_RELEASE_REPORT", "notebooks-executed-with-output", "dist-rc")
        for d in DOCS + list((ROOT / "docs").glob("*.md")) + [ROOT / "BUILD.md", ROOT / "CHANGELOG.md"]:
            for g in gone:
                if d.name == "CHANGELOG.md":
                    continue
                self.assertNotIn(g, d.read_text(), f"{d.name} mentions {g}")

    def test_every_relative_link_resolves(self):
        for d in DOCS + list((ROOT / "docs").glob("*.md")):
            for target in re.findall(r"\]\(([^)#\s]+)\)", d.read_text()):
                if target.startswith(("http", "mailto")):
                    continue
                base = ROOT if d.parent == ROOT else d.parent
                self.assertTrue((base / target).exists() or (ROOT / target).exists(), f"{d.name}: broken link {target}")


if __name__ == "__main__":
    unittest.main()
