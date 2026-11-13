"""Notebook-15 input discovery (regression for: the attached wheel-only ZIP was ignored and a DIFFERENT release was silently downloaded)."""
import hashlib, io, json, os, pathlib, sys, tempfile, unittest, zipfile
from unittest import mock
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "kaggle"))
import bundle as b

WHEEL = "stokastisk-0.1.0-3-cp313-cp313-linux_x86_64.whl"
sha = lambda x: hashlib.sha256(x).hexdigest()  # noqa: E731


def make_release(root: pathlib.Path, *, wheel_bytes=b"WHEEL-BYTES", extra=None, tamper=False, build="3", manifest_sha=None):
    files = {f"wheels/{WHEEL}": wheel_bytes, "requirements/core.lock": b"x==1 \\\n    --hash=sha256:" + b"0" * 64 + b"\n", "VERSION": b"0.1.0\n"}
    files.update(extra or {})
    for rel, data in files.items():
        p = root / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
    (root / "manifests").mkdir(exist_ok=True)
    man = {"version": "0.1.0", "integrated_wheel": {"filename": WHEEL, "sha256": manifest_sha or sha(wheel_bytes), "size": len(wheel_bytes), "build_tag": build}}
    (root / "manifests/RELEASE-MANIFEST.json").write_text(json.dumps(man))
    sums = {rel: sha(data) for rel, data in files.items()}
    sums["manifests/RELEASE-MANIFEST.json"] = sha((root / "manifests/RELEASE-MANIFEST.json").read_bytes())
    if tamper:
        sums["VERSION"] = "f" * 64
    (root / "manifests/SHA256SUMS").write_text("".join(f"{h}  {r}\n" for r, h in sorted(sums.items())))
    return root


def zip_dir(root: pathlib.Path, out: pathlib.Path, top="stokastisk-0.1.0-build3-kaggle-t4-sm75"):
    with zipfile.ZipFile(out, "w") as z:
        for p in sorted(root.rglob("*")):
            if p.is_file():
                z.write(p, f"{top}/{p.relative_to(root).as_posix()}")
    return out


class Discovery(unittest.TestCase):
    def setUp(self):
        self.td = pathlib.Path(tempfile.mkdtemp()); self.inp = self.td / "input"; self.inp.mkdir(); self.work = self.td / "work" / "release"

    def locate(self, **kw):
        return b.locate_and_verify(input_root=self.inp, work_dir=self.work, expected_wheel_name=WHEEL, **kw)

    def test_attached_zip_is_found_verified_and_extracted(self):
        rel = make_release(self.td / "rel"); ds = self.inp / "stokastisk-dataset"; ds.mkdir(); zip_dir(rel, ds / "bundle.zip")
        p = self.locate(); self.assertEqual(p["source"], "attached-zip"); self.assertEqual(p["wheel"], WHEEL); self.assertEqual(p["wheel_sha256"], sha(b"WHEEL-BYTES"))
        self.assertEqual(p["bundle_zip_sha256"], sha((ds / "bundle.zip").read_bytes())); self.assertTrue((pathlib.Path(p["release_root"]) / "wheels" / WHEEL).is_file())

    def test_the_real_kaggle_mount_layout_is_found(self):
        """Kaggle mounts a dataset at /kaggle/input/datasets/<user>/<slug>/<release>/manifests/... (5 levels below /kaggle/input): a depth limit of 4 found nothing."""
        make_release(self.inp / "datasets" / "waqasm86" / "stokastisk-0-1-0-build3-t4-sm75-kaggle-input" / "stokastisk-0.1.0-build3-kaggle-t4-sm75")
        p = self.locate(); self.assertEqual(p["source"], "attached-directory"); self.assertEqual(p["wheel"], WHEEL)
        z = self.td / "z"; z.mkdir(); zip_dir(make_release(self.td / "rel2"), z / "bundle.zip")
        deep = self.inp.parent / "input2" / "datasets" / "waqasm86" / "slug"; deep.mkdir(parents=True); (deep / "bundle.zip").write_bytes((z / "bundle.zip").read_bytes())
        self.assertEqual(len(b.find_candidates(self.inp.parent / "input2")["zips"]), 1)

    def test_already_extracted_directory_is_found(self):
        make_release(self.inp / "ds" / "stokastisk-0.1.0-build3-kaggle-t4-sm75")
        self.assertEqual(self.locate()["source"], "attached-directory")

    def test_wheel_only_zip_is_rejected_with_a_clear_message(self):
        ds = self.inp / "d"; ds.mkdir()
        with zipfile.ZipFile(ds / "w.zip", "w") as z: z.writestr(WHEEL, b"WHEEL-BYTES")
        with self.assertRaises(b.BundleError) as c: self.locate()
        self.assertIn("not a complete", str(c.exception).lower()); self.assertFalse(self.work.exists())

    def test_nothing_attached_is_an_error_and_never_a_download(self):
        with mock.patch.object(b.urllib.request, "urlopen", side_effect=AssertionError("network must not be touched")):
            with self.assertRaises(b.BundleError) as c: self.locate()
        self.assertIn("No network fallback", str(c.exception))

    def test_loose_wheel_is_not_a_release(self):
        (self.inp / "d").mkdir(); (self.inp / "d" / WHEEL).write_bytes(b"x")
        with self.assertRaises(b.BundleError) as c: self.locate()
        self.assertIn("Loose wheel", str(c.exception))

    def test_hf_fallback_is_opt_in_pinned_and_verified(self):
        rel = make_release(self.td / "rel"); z = zip_dir(rel, self.td / "remote.zip"); data = z.read_bytes()
        fb = {"url": "https://example.invalid/r.zip", "sha256": sha(data), "size": len(data)}
        with self.assertRaises(b.BundleError): self.locate(fallback=fb)                       # not enabled: never used
        class R(io.BytesIO):
            def __enter__(s): return s
            def __exit__(s, *a): return False
        with mock.patch.dict(os.environ, {"STOKASTISK_ALLOW_HF_FALLBACK": "1"}), mock.patch.object(b.urllib.request, "urlopen", return_value=R(data)):
            p = self.locate(fallback=fb)
        self.assertTrue(p["source"].startswith("HUGGING-FACE-FALLBACK"))
        bad = {**fb, "sha256": "0" * 64}
        with mock.patch.dict(os.environ, {"STOKASTISK_ALLOW_HF_FALLBACK": "1"}), mock.patch.object(b.urllib.request, "urlopen", return_value=R(data)):
            with self.assertRaises(b.BundleError): self.locate(fallback=bad)
        with mock.patch.dict(os.environ, {"STOKASTISK_ALLOW_HF_FALLBACK": "1"}):
            with self.assertRaises(b.BundleError): self.locate(fallback=None)                  # opt-in without a pinned release is an error

    def test_two_bundles_are_ambiguous(self):
        for n in "ab": make_release(self.inp / n / "stokastisk-0.1.0-build3-kaggle-t4-sm75")
        with self.assertRaises(b.BundleError) as c: self.locate()
        self.assertIn("more than one", str(c.exception))

    def test_expected_bundle_zip_hash_is_enforced(self):
        rel = make_release(self.td / "rel"); zip_dir(rel, self.inp / "b.zip")
        with self.assertRaises(b.BundleError): self.locate(expected_bundle_zip_sha256="0" * 64)


class Verification(unittest.TestCase):
    def setUp(self):
        self.td = pathlib.Path(tempfile.mkdtemp())

    def verify(self, root, **kw):
        return b.verify_release(root, expected_wheel_name=WHEEL, **kw)

    def test_valid(self):
        p = self.verify(make_release(self.td / "r")); self.assertEqual(p["build"], "3"); self.assertGreaterEqual(p["files_verified"], 4)

    def test_hash_mismatch(self):
        with self.assertRaises(b.BundleError): self.verify(make_release(self.td / "r", tamper=True))

    def test_unlisted_file(self):
        r = make_release(self.td / "r"); (r / "evil.py").write_text("x")
        with self.assertRaises(b.BundleError) as c: self.verify(r)
        self.assertIn("unlisted", str(c.exception))

    def test_wrong_wheel_sha_in_manifest(self):
        with self.assertRaises(b.BundleError): self.verify(make_release(self.td / "r", manifest_sha="0" * 64))

    def test_wheel_sha_compiled_into_the_notebook(self):
        r = make_release(self.td / "r"); self.verify(r, expected_wheel_sha256=sha(b"WHEEL-BYTES"))
        with self.assertRaises(b.BundleError): self.verify(r, expected_wheel_sha256="1" * 64)

    def test_a_stale_or_different_bundle_is_rejected_by_the_pinned_sums_hash(self):
        r = make_release(self.td / "r"); good = sha((r / "manifests/SHA256SUMS").read_bytes())
        self.verify(r, expected_sums_sha256=good)
        with self.assertRaises(b.BundleError) as c: self.verify(r, expected_sums_sha256="2" * 64)
        self.assertIn("stale", str(c.exception))

    def test_wrong_build_is_refused(self):
        with self.assertRaises(b.BundleError) as c: self.verify(make_release(self.td / "r", build="2"))
        self.assertIn("build", str(c.exception))

    def test_second_wheel_is_refused(self):
        r = make_release(self.td / "r", extra={"wheels/other-0.1.0-1-cp313-cp313-linux_x86_64.whl": b"o"})
        with self.assertRaises(b.BundleError): self.verify(r)

    def test_missing_manifest(self):
        r = make_release(self.td / "r"); (r / "manifests/RELEASE-MANIFEST.json").unlink()
        with self.assertRaises(b.BundleError): self.verify(r)


class ArchiveSafety(unittest.TestCase):
    def setUp(self):
        self.td = pathlib.Path(tempfile.mkdtemp())

    def zip_with(self, name, data=b"x", attr=None):
        p = self.td / "a.zip"
        with zipfile.ZipFile(p, "w") as z:
            zi = zipfile.ZipInfo(name)
            if attr is not None: zi.external_attr = attr << 16
            z.writestr(zi, data)
        return p

    def test_path_traversal_and_unsafe_names_are_rejected(self):
        for bad in ("../evil.txt", "a/../../evil", "/abs/evil", "C:/x", "a\\b"):
            with self.subTest(bad=bad), self.assertRaises(b.BundleError): b.inspect_zip(self.zip_with(bad))

    def test_symlink_member_is_rejected(self):
        with self.assertRaises(b.BundleError): b.inspect_zip(self.zip_with("link", b"/etc/passwd", attr=0o120777))

    def test_duplicate_member_is_rejected(self):
        p = self.td / "d.zip"
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("a", b"1"); z.writestr("a", b"2")
        with self.assertRaises(b.BundleError): b.inspect_zip(p)

    def test_corrupt_and_non_zip(self):
        p = self.td / "n.zip"; p.write_bytes(b"not a zip")
        with self.assertRaises(b.BundleError): b.inspect_zip(p)
        z = self.zip_with("a", b"hello world" * 100); raw = bytearray(z.read_bytes()); raw[40] ^= 0xFF; z.write_bytes(bytes(raw))
        with self.assertRaises(b.BundleError): b.inspect_zip(z)

    def test_oversize_is_rejected(self):
        with mock.patch.object(b, "MAX_UNCOMPRESSED_BYTES", 10), self.assertRaises(b.BundleError): b.inspect_zip(self.zip_with("a", b"x" * 100))

    def test_extraction_refuses_a_non_empty_destination_and_traversal(self):
        rel = make_release(self.td / "rel"); z = zip_dir(rel, self.td / "ok.zip"); dest = self.td / "dest"; dest.mkdir(); (dest / "x").write_text("1")
        with self.assertRaises(b.BundleError): b.extract_zip(z, dest)


class Embedding(unittest.TestCase):
    def test_notebook_embeds_bundle_py_verbatim_and_has_no_default_download(self):
        sys.path.insert(0, str(ROOT / "scripts" / "release"))
        import make_notebook15_build3 as mk
        nb = mk.build("a" * 64, "b" * 64)
        code = ["".join(c["source"]) for c in nb.cells if c.cell_type == "code"]
        self.assertIn((ROOT / "scripts/kaggle/bundle.py").read_text(), code[0])
        self.assertIn("EXPECTED_WHEEL_SHA256 = " + repr("a" * 64), code[0]); self.assertIn("fallback=None", code[0])
        self.assertNotIn("huggingface.co/", "\n".join(code))
        for s in code: compile(s, "cell", "exec")
        self.assertIn("assert summ.get('status') == 'PASS'", code[-1]); self.assertIn("ZipFile(ZIP", code[-1])
        self.assertLess(code[-1].index("ZipFile(ZIP"), code[-1].index("assert summ"))        # the archive is written BEFORE the notebook is failed

    def test_installer_prohibited_copy_is_identical_to_the_sdk_policy(self):
        self.assertEqual((ROOT / "scripts/kaggle/prohibited.py").read_bytes(), (ROOT / "sdk/stokastisk/prohibited.py").read_bytes())
        self.assertEqual((ROOT / "scripts/kaggle/structured_output_cpu_check.py").read_bytes(), (ROOT / "scripts/audit/structured_output_cpu_check.py").read_bytes())


if __name__ == "__main__":
    unittest.main()
