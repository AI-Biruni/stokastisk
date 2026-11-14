"""The static PEP 503 index behind the one-line install: format, hash fragment, python constraint, and that pip itself understands it (offline, via a file:// copy)."""
import hashlib, pathlib, subprocess, sys, tempfile, unittest, zipfile, html.parser
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/release"))
import make_pip_index as mpi


class Links(html.parser.HTMLParser):
    def __init__(self):
        super().__init__(); self.links = []
    def handle_starttag(self, tag, attrs):
        if tag == "a": self.links.append(dict(attrs))


class PipIndex(unittest.TestCase):
    def setUp(self):
        self.td = pathlib.Path(tempfile.mkdtemp()); self.w = self.td / "stokastisk-0.1.0-3-cp313-cp313-linux_x86_64.whl"
        with zipfile.ZipFile(self.w, "w") as z:                                  # a minimal but VALID wheel, so pip can really process it
            z.writestr("stokastisk-0.1.0.dist-info/METADATA", "Metadata-Version: 2.1\nName: stokastisk\nVersion: 0.1.0\n"); z.writestr("stokastisk-0.1.0.dist-info/WHEEL", "Wheel-Version: 1.0\nGenerator: t\nRoot-Is-Purelib: false\nTag: cp313-cp313-linux_x86_64\nBuild: 3\n"); z.writestr("stokastisk-0.1.0.dist-info/RECORD", "")
        self.sha = hashlib.sha256(self.w.read_bytes()).hexdigest(); self.out = self.td / "site"
        self.info = mpi.build(self.w, self.out)

    def test_layout_and_pep503_page(self):
        for f in (".nojekyll", "index.html", "simple/index.html", "simple/stokastisk/index.html"): self.assertTrue((self.out / f).exists(), f)
        p = Links(); p.feed((self.out / "simple/stokastisk/index.html").read_text()); (a,) = p.links
        self.assertEqual(a["href"], f"{mpi.HF}{self.w.name}#sha256={self.sha}"); self.assertEqual(a["data-requires-python"], ">=3.13,<3.14")
        self.assertIn('pypi:repository-version', (self.out / "simple/stokastisk/index.html").read_text())
        p = Links(); p.feed((self.out / "simple/index.html").read_text()); self.assertEqual(p.links[0]["href"], "stokastisk/")

    def test_the_landing_page_carries_the_pinned_command_and_the_security_note(self):
        page = (self.out / "index.html").read_text()
        self.assertIn(f'"stokastisk @ {self.info["url"]}#sha256={self.sha}"', page); self.assertIn("--index-url https://pypi.org/simple", page)
        self.assertIn("Security trade-off", page); self.assertIn(f"pip install stokastisk --extra-index-url {mpi.INDEX}", page)

    def test_wrong_wheel_names_are_refused(self):
        bad = self.td / "stokastisk-0.9.9-1-cp313-cp313-linux_x86_64.whl"; bad.write_bytes(b"x")
        with self.assertRaises(SystemExit): mpi.build(bad, self.td / "o2")

    def test_pip_resolves_and_verifies_the_hash_through_the_index(self):
        """pip against a file:// copy whose link points at a local file: right hash downloads, wrong hash is refused."""
        for ok, sha_suffix in ((True, self.sha), (False, "0" * 64)):
            site = self.td / f"s{ok}"; (site / "stokastisk").mkdir(parents=True)
            (site / "stokastisk" / "index.html").write_text(f'<!DOCTYPE html><html><body><a href="{self.w.as_uri()}#sha256={sha_suffix}" data-requires-python="&gt;=3.13,&lt;3.14">{self.w.name}</a></body></html>')
            r = subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "--no-cache-dir", "--disable-pip-version-check", "--index-url", site.as_uri() + "/", "-d", str(self.td / f"d{ok}"), "stokastisk==0.1.0"], capture_output=True, text=True)
            if ok:
                self.assertTrue(list((self.td / f"d{ok}").glob("*.whl")), r.stdout + r.stderr[-400:])
            else:
                self.assertNotEqual(r.returncode, 0); self.assertIn("HASHES", r.stderr.upper() + r.stdout.upper())


if __name__ == "__main__":
    unittest.main()
