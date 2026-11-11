"""Permanent project-version lock (VERSION_POLICY.md): the Stokastisk project version is exactly 0.1.0 everywhere. CPU-only."""
import importlib.util, json, pathlib, re, subprocess, sys, unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOCKED = "0.1.0"
OTHER = ".".join(["0", "9", "9"])   # a deliberately different version, built at runtime so no file literally declares it
spec = importlib.util.spec_from_file_location("verify_project_version", ROOT / "scripts/release/verify_project_version.py")
vpv = importlib.util.module_from_spec(spec); spec.loader.exec_module(vpv)


class VersionLock(unittest.TestCase):
    def test_locked_constant_is_0_1_0(self):
        self.assertEqual(vpv.LOCKED_VERSION, LOCKED)
        self.assertEqual(vpv.RELEASE_NAME, f"stokastisk-{LOCKED}-kaggle-t4-sm75")

    def test_version_file_is_exact(self):
        self.assertEqual((ROOT / "VERSION").read_bytes(), b"0.1.0\n")

    def test_policy_document_exists_and_states_the_lock(self):
        t = (ROOT / "VERSION_POLICY.md").read_text()
        for needle in (LOCKED, "explicit", "owner", "dependency"):
            self.assertIn(needle, t)

    def test_sdk_version_matches(self):
        sys.path.insert(0, str(ROOT / "sdk"))
        import stokastisk
        self.assertEqual(stokastisk.__version__, LOCKED)

    def test_pyproject_derives_version_from_VERSION_file(self):
        t = (ROOT / "pyproject.toml").read_text()
        self.assertIn('dynamic = ["version"]', t)
        self.assertIn('version = {file = "VERSION"}', t)
        self.assertIsNone(re.search(r'^version\s*=\s*"(?!0\.1\.0")', t, re.M))

    def test_notebooks_name_only_the_locked_release(self):
        for p in sorted((ROOT / "notebooks").rglob("*.ipynb")):
            self.assertIn(f"stokastisk-{LOCKED}-kaggle-t4-sm75", p.read_text(), p.name)
            for m in vpv.RELEASE_DIR_RE.finditer(p.read_text()):
                self.assertEqual((m.group(1), m.group(2)), (LOCKED, None), p.name)

    def test_installer_enforces_locked_version_and_force_reinstall(self):
        t = (ROOT / "scripts/kaggle/02_install.py").read_text()
        self.assertIn(f'LOCK = "{LOCKED}"', t)
        self.assertIn('"--force-reinstall", "--no-deps", "--no-index"', t)
        self.assertNotIn("--upgrade", t)

    def test_assembler_refuses_other_versions(self):
        t = (ROOT / "scripts/release/assemble_release.py").read_text()
        self.assertIn(f'LOCKED = "{LOCKED}"', t)
        self.assertIn("verify_project_version.py", t)

    def test_no_release_script_derives_or_bumps_a_version(self):
        for p in list((ROOT / "scripts").rglob("*.py")) + [ROOT / "pyproject.toml"]:
            if p.name == "verify_project_version.py":
                continue
            t = p.read_text()
            self.assertNotRegex(t, r"setuptools_scm|versioneer|bump2?version|semantic[_-]release|git describe", p.name)

    def test_whole_repo_verifier_passes(self):
        r = subprocess.run([sys.executable, str(ROOT / "scripts/release/verify_project_version.py")], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(r.returncode, 0, r.stdout[-2000:])

    def test_verifier_detects_drift(self):
        """Feed the verifier checks a fake latest.json / model card with a different version: it must fail."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            bad = pathlib.Path(td) / "latest.json"
            bad.write_text(json.dumps({"project": "stokastisk", "version": OTHER, "release": f"stokastisk-{OTHER}-kaggle-t4-sm75"}))
            card = pathlib.Path(td) / "README.md"; card.write_text(f"# Stokastisk {OTHER}\nstokastisk-{OTHER}-kaggle-t4-sm75\n")
            r = subprocess.run([sys.executable, str(ROOT / "scripts/release/verify_project_version.py"), "--latest-json", str(bad), "--model-card", str(card)], capture_output=True, text=True, cwd=ROOT)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("FAIL latest.json", r.stdout); self.assertIn("FAIL model card", r.stdout)


if __name__ == "__main__":
    unittest.main()
