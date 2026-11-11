"""README examples must be notebook-friendly (no main() wrappers) and the GitHub README / Hub model card must agree."""
import pathlib, re, unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", ROOT / "docs/HUB_README.md"]
V = (ROOT / "VERSION").read_text().strip()


def python_blocks(text):
    return re.findall(r"```python\n(.*?)```", text, re.S)


class ReadmeExamples(unittest.TestCase):
    def test_no_main_wrappers_in_examples(self):
        for d in DOCS:
            for b in python_blocks(d.read_text()):
                self.assertNotRegex(b, r"def main\s*\(", d.name)
                self.assertNotIn('__name__ == "__main__"', b, d.name)

    def test_blocks_parse(self):
        import ast
        for d in DOCS:
            for b in python_blocks(d.read_text()):
                ast.parse(b)

    def test_both_state_version_and_pending_status_and_release(self):
        for d in DOCS:
            t = d.read_text()
            self.assertIn(V, t); self.assertIn("KAGGLE_T4_VERIFIED", t)
            self.assertIn(f"stokastisk-{V}-kaggle-t4-sm75", t)
            self.assertIn("https://huggingface.co/waqasm86/stokastisk/resolve/main/scripts/fetch_release.py", t)

    def test_canonical_example_is_identical_in_both(self):
        needle = 'config = stokastisk.StokastiskConfig(tensor_parallel_size=1)'
        for d in DOCS:
            self.assertIn(needle, d.read_text())
            self.assertIn('llm = stokastisk.LLM("Qwen/Qwen2.5-0.5B-Instruct", config=config)', d.read_text())


if __name__ == "__main__":
    unittest.main()
