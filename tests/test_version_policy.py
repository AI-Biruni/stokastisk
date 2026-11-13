"""Permanent project-version lock (VERSION_POLICY.md): the Stokastisk project version is exactly 0.1.0 everywhere. CPU-only."""
import importlib.util, pathlib, re, subprocess, sys, tempfile, unittest, zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verify_project_version", ROOT / "scripts/release/verify_project_version.py")
vpv = importlib.util.module_from_spec(spec); spec.loader.exec_module(vpv)


class VersionLock(unittest.TestCase):
    def test_locked_constant_and_files(self):
        self.assertEqual(vpv.LOCKED_VERSION, "0.1.0"); self.assertEqual((ROOT / "VERSION").read_bytes(), b"0.1.0\n")
        t = (ROOT / "VERSION_POLICY.md").read_text()
        for needle in ("0.1.0", "explicit", "owner"): self.assertIn(needle, t)

    def test_sdk_version_matches(self):
        sys.path.insert(0, str(ROOT / "sdk"))
        import stokastisk
        self.assertEqual(stokastisk.__version__, "0.1.0")

    def test_verifier_passes_on_the_repository(self):
        self.assertEqual(vpv.main([]), 0)

    def test_verifier_checks_wheel_metadata_and_tag(self):
        d = pathlib.Path(tempfile.mkdtemp())
        for ver, rc in (("0.1.0", 0), ("0.9.9", 1)):
            w = d / f"w{ver}.whl"
            with zipfile.ZipFile(w, "w") as z:
                z.writestr(f"stokastisk-{ver}.dist-info/METADATA", f"Name: stokastisk\nVersion: {ver}\n")
            self.assertEqual(vpv.main(["--wheel", str(w)]), rc, ver)
        self.assertEqual(vpv.main(["--tag", "v0.1.0"]), 0); self.assertEqual(vpv.main(["--tag", "v0.2.0"]), 1)

    def test_built_wheel_filename_keeps_the_version_and_only_the_build_tag_varies(self):
        s = (ROOT / "scripts/release/build_integrated_wheel.py").read_text()
        self.assertIn('LOCKED = "0.1.0"', s); self.assertIn("-{tag}-cp313-cp313-linux_x86_64.whl", s)


if __name__ == "__main__":
    unittest.main()
