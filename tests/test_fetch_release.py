#!/usr/bin/env python3.13
"""scripts/fetch_release.py (stdlib HF downloader): verified download, refuses tampered / unsafe / mis-named archives. Offline via file://."""
import hashlib, importlib.util, json, pathlib, tempfile, unittest, zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fetch_release", ROOT / "scripts/fetch_release.py")
fr = importlib.util.module_from_spec(spec); spec.loader.exec_module(fr)
REL = "stokastisk-9.9.9-test"


def make_hub(root, members=None, manifest_name=REL, sidecar_hash=None):
    d = pathlib.Path(root) / "releases" / REL; d.mkdir(parents=True)
    z = d / f"{REL}-kaggle-input.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for n, data in (members or {f"{REL}/manifests/RELEASE-MANIFEST.json": json.dumps({"name": manifest_name}), f"{REL}/wheels/x.whl": "w"}).items():
            zf.writestr(n, data)
    (d / (z.name + ".sha256")).write_text(f"{sidecar_hash or hashlib.sha256(z.read_bytes()).hexdigest()}  {z.name}\n")
    return "file://" + str(root)


class Fetch(unittest.TestCase):
    def test_verified_download_and_extract(self):
        with tempfile.TemporaryDirectory() as hub, tempfile.TemporaryDirectory() as dest:
            out = fr.fetch_release(REL, dest, base=make_hub(hub))
            self.assertEqual(out.name, REL); self.assertTrue((out / "wheels/x.whl").is_file())

    def test_sha256_mismatch_is_rejected_and_nothing_extracted(self):
        with tempfile.TemporaryDirectory() as hub, tempfile.TemporaryDirectory() as dest:
            base = make_hub(hub, sidecar_hash="0" * 64)
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"): fr.fetch_release(REL, dest, base=base)
            self.assertFalse((pathlib.Path(dest) / REL).exists())

    def test_path_traversal_and_foreign_prefix_rejected(self):
        for bad in ("../evil.txt", "other/evil.txt", "/abs.txt"):
            with tempfile.TemporaryDirectory() as hub, tempfile.TemporaryDirectory() as dest:
                base = make_hub(hub, members={bad: "x", f"{REL}/manifests/RELEASE-MANIFEST.json": json.dumps({"name": REL})})
                with self.assertRaisesRegex(RuntimeError, "unsafe or foreign"): fr.fetch_release(REL, dest, base=base)

    def test_no_partial_file_after_success_and_stale_part_removed(self):
        with tempfile.TemporaryDirectory() as hub, tempfile.TemporaryDirectory() as dest:
            (pathlib.Path(dest) / f"{REL}-kaggle-input.zip.part").write_bytes(b"stale")
            fr.fetch_release(REL, dest, base=make_hub(hub))
            self.assertEqual(list(pathlib.Path(dest).glob("*.part")), [])

    def test_failed_download_leaves_no_partial_and_no_final_file(self):
        import unittest.mock as m
        with tempfile.TemporaryDirectory() as d:
            dest = pathlib.Path(d) / "x.zip"
            def boom(url, part, workers, retries): part.write_bytes(b"half"); raise OSError("network down")
            with m.patch.object(fr, "_download_to", boom), m.patch.object(fr, "_range_ok", lambda u: None):
                with self.assertRaises(OSError): fr._download("http://x/y", dest)
            self.assertEqual(list(pathlib.Path(d).iterdir()), [])

    def test_short_download_is_rejected_atomically(self):
        import unittest.mock as m
        with tempfile.TemporaryDirectory() as d:
            dest = pathlib.Path(d) / "x.zip"
            with m.patch.object(fr, "_download_to", lambda u, p, w, r: p.write_bytes(b"12345")), m.patch.object(fr, "_range_ok", lambda u: 10):
                with self.assertRaisesRegex(RuntimeError, "incomplete download"): fr._download("http://x/y", dest)
            self.assertEqual(list(pathlib.Path(d).iterdir()), [])

    def test_insufficient_disk_space_refused_before_download(self):
        import unittest.mock as m, collections
        DU = collections.namedtuple("DU", "total used free")
        with tempfile.TemporaryDirectory() as d:
            called = []
            with m.patch.object(fr, "_range_ok", lambda u: 10 << 20), m.patch.object(fr.shutil, "disk_usage", lambda p: DU(1, 1, 1 << 20)), \
                    m.patch.object(fr, "_download_to", lambda *a: called.append(1)):
                with self.assertRaisesRegex(RuntimeError, "not enough free disk space"): fr._download("http://x/y", pathlib.Path(d) / "x.zip")
            self.assertFalse(called)

    def test_default_is_single_stream(self):
        import inspect
        self.assertEqual(inspect.signature(fr.fetch_release).parameters["workers"].default, 1)

    def test_manifest_name_must_match_release(self):
        with tempfile.TemporaryDirectory() as hub, tempfile.TemporaryDirectory() as dest:
            with self.assertRaisesRegex(RuntimeError, "manifest name"): fr.fetch_release(REL, dest, base=make_hub(hub, manifest_name="stokastisk-0.3.0-other"))


if __name__ == "__main__":
    unittest.main()
