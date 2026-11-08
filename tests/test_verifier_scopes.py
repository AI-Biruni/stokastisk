"""Tests for scripts/release/verify_project_version.py on synthetic project trees (no network, no GPU, no torch).

A historical reference must never fail the verifier; an incorrect ACTIVE declaration must. The 'other' version strings are
built at runtime so this file itself declares no foreign Stokastisk release."""
import json, pathlib, subprocess, sys, tempfile, unittest, zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
V = (ROOT / "VERSION").read_text().strip()
OTHER = ".".join(["0", "9", "9"])
REL = f"stokastisk-{V}-kaggle-t4-sm75"
OLD = f"stokastisk-{OTHER}-kaggle-t4-sm75"
PYPROJECT = '[project]\nname = "stokastisk"\ndynamic = ["version"]\n\n[tool.setuptools.dynamic]\nversion = {file = "VERSION"}\n'


def make_tree(td, **over):
    r = pathlib.Path(td)
    (r / "sdk/stokastisk").mkdir(parents=True); (r / "notebooks").mkdir()
    files = {"VERSION": f"{V}\n", "pyproject.toml": PYPROJECT, "sdk/stokastisk/_version.py": f'__version__ = "{V}"\n'}
    files.update(over)
    for k, v in files.items():
        if v is not None:
            (r / k).parent.mkdir(parents=True, exist_ok=True); (r / k).write_text(v)
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": [{"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": [f'RELEASE = "{REL}"']}]}
    if "notebooks/01.ipynb" not in over:
        (r / "notebooks/01.ipynb").write_text(json.dumps(nb))
    return r


def run(root, *args):
    p = subprocess.run([sys.executable, "-I", str(ROOT / "scripts/release/verify_project_version.py"), "--root", str(root), *args], capture_output=True, text=True)
    return p.returncode, p.stdout


class VerifierScopes(unittest.TestCase):
    def tree(self, **over):
        td = tempfile.TemporaryDirectory(); self.addCleanup(td.cleanup)
        return make_tree(td.name, **over)

    def test_correct_tree_passes(self):
        rc, out = run(self.tree()); self.assertEqual(rc, 0, out)

    def test_missing_version_file(self):
        rc, out = run(self.tree(VERSION=None)); self.assertNotEqual(rc, 0); self.assertIn("FAIL VERSION", out)

    def test_wrong_version_file(self):
        rc, out = run(self.tree(VERSION=f"{OTHER}\n")); self.assertNotEqual(rc, 0)

    def test_incorrect_package_metadata(self):
        rc, out = run(self.tree(**{"pyproject.toml": PYPROJECT.replace('dynamic = ["version"]', f'version = "{OTHER}"')})); self.assertNotEqual(rc, 0)
        self.assertIn("pyproject.toml", out)

    def test_malformed_toml_is_a_clean_failure(self):
        rc, out = run(self.tree(**{"pyproject.toml": "[project\n"})); self.assertNotEqual(rc, 0); self.assertIn("valid TOML", out); self.assertNotIn("Traceback", out)

    def test_wrong_sdk_version(self):
        rc, out = run(self.tree(**{"sdk/stokastisk/_version.py": f'__version__ = "{OTHER}"\n'})); self.assertNotEqual(rc, 0)

    def test_malformed_notebook_json(self):
        rc, out = run(self.tree(**{"notebooks/01.ipynb": "{not json"})); self.assertNotEqual(rc, 0); self.assertIn("unreadable notebook", out)

    def test_wrong_notebook_release(self):
        nb = {"cells": [{"cell_type": "code", "source": [f'RELEASE = "{OLD}"']}]}
        rc, out = run(self.tree(**{"notebooks/01.ipynb": json.dumps(nb)})); self.assertNotEqual(rc, 0); self.assertIn(OLD, out)

    def test_historical_reference_is_not_a_false_positive(self):
        t = self.tree(**{"legacy/removed/notes.md": f"old build {OLD}\n", "reports/old.md": f"# Stokastisk {OTHER}\n"})
        rc, out = run(t); self.assertEqual(rc, 0, out)

    def test_active_reference_to_other_release_fails(self):
        rc, out = run(self.tree(**{"README.md": f"install {OLD}\n"})); self.assertNotEqual(rc, 0)

    def test_exclusions_are_listed_with_reasons(self):
        p = subprocess.run([sys.executable, "-I", str(ROOT / "scripts/release/verify_project_version.py"), "--list-exclusions"], capture_output=True, text=True)
        lines = [l for l in p.stdout.splitlines() if l]
        self.assertTrue(lines and all("\t" in l and l.split("\t")[1] for l in lines))
        self.assertFalse(any(l.split("\t")[0] in ("", "/", "sdk/", "notebooks/", "scripts/", "VERSION", "README.md") for l in lines), "active paths must never be excluded")

    def test_missing_optional_artifacts_pass_in_local_mode_fail_in_strict_mode(self):
        t = self.tree()
        self.assertEqual(run(t)[0], 0)
        rc, out = run(t, "--strict-release"); self.assertNotEqual(rc, 0); self.assertIn("FAIL strict", out)

    def test_malformed_and_wrong_latest_json(self):
        t = self.tree(); bad = t / "latest.json"
        bad.write_text("{oops"); rc, out = run(t, "--latest-json", str(bad)); self.assertNotEqual(rc, 0); self.assertIn("valid JSON", out)
        bad.write_text(json.dumps({"project": "stokastisk", "version": OTHER, "release": OLD})); self.assertNotEqual(run(t, "--latest-json", str(bad))[0], 0)
        bad.write_text(json.dumps({"project": "stokastisk", "version": V, "release": REL})); self.assertEqual(run(t, "--latest-json", str(bad))[0], 0)
        self.assertNotEqual(run(t, "--latest-json", str(t / "nope.json"))[0], 0)

    def test_wrong_wheel_metadata(self):
        t = self.tree()
        for ver, ok in ((V, True), (OTHER, False)):
            w = t / f"stokastisk-{ver}-py3-none-any.whl"
            with zipfile.ZipFile(w, "w") as z:
                z.writestr(f"stokastisk-{ver}.dist-info/METADATA", f"Metadata-Version: 2.1\nName: stokastisk\nVersion: {ver}\n")
            self.assertEqual(run(t, "--wheel", str(w))[0] == 0, ok)

    def test_corrupt_wheel_is_a_clean_failure(self):
        t = self.tree(); w = t / "x.whl"; w.write_bytes(b"not a zip")
        rc, out = run(t, "--wheel", str(w)); self.assertNotEqual(rc, 0); self.assertNotIn("Traceback", out)

    def test_wrong_model_card(self):
        t = self.tree(); c = t / "card.md"
        c.write_text(f"# Stokastisk {OTHER}\n{OLD}\n"); self.assertNotEqual(run(t, "--model-card", str(c))[0], 0)
        c.write_text(f"# Stokastisk {V}\n{REL}\n"); self.assertEqual(run(t, "--model-card", str(c))[0], 0)

    def test_release_dir_with_wrong_installer_and_manifest(self):
        t = self.tree(); d = t / REL
        (d / "manifests").mkdir(parents=True); (d / "scripts/kaggle").mkdir(parents=True); (d / "wheels").mkdir()
        (d / "VERSION").write_text(f"{V}\n")
        (d / "manifests/RELEASE-MANIFEST.json").write_text(json.dumps({"name": REL, "version": V}))
        (d / "scripts/kaggle/02_install.py").write_text(f'LOCK = "{V}"\n')
        self.assertEqual(run(t, "--release-dir", str(d))[0], 0)
        (d / "scripts/kaggle/02_install.py").write_text(f'LOCK = "{OTHER}"\n'); self.assertNotEqual(run(t, "--release-dir", str(d))[0], 0)
        (d / "scripts/kaggle/02_install.py").write_text(f'LOCK = "{V}"\n')
        (d / "manifests/RELEASE-MANIFEST.json").write_text("{bad"); self.assertNotEqual(run(t, "--release-dir", str(d))[0], 0)
        (d / "manifests/RELEASE-MANIFEST.json").unlink(); self.assertNotEqual(run(t, "--release-dir", str(d))[0], 0)
        wrong = t / OLD; wrong.mkdir(); self.assertNotEqual(run(t, "--release-dir", str(wrong))[0], 0)


if __name__ == "__main__":
    unittest.main()
