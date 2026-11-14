"""One wheel identity everywhere: RELEASE.json is the source of truth; the install command, the file table, the pip index generator and the notebooks must agree with it. CPU-only."""
import json, pathlib, re, sys, unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/release"))
import make_pip_index as mpi  # noqa: E402

REL = json.loads((ROOT / "RELEASE.json").read_text())
CUR = REL["builds"][str(REL["current_build"])]
WITHDRAWN = REL["withdrawn_builds"]["3"]
BUILD4_SHA = "052156d71adc6ad9966901565f86dbded9ff41c620c782db4ff377a1a2b4f62e"     # historical: withdrawn build 4
BUILD3_SHA = "72039ca4044e48fdf5edca8051dcb48b42fe61a69e07715fa8803e029581fe12"     # historical: the build the Kaggle evidence was produced on
PIN = f'python -m pip install --index-url https://pypi.org/simple "stokastisk @ {CUR["url"]}#sha256={CUR["sha256"]}"'
DOCS = ["README.md", "INSTALL.md", "docs/HUB_README.md", "docs/guide/getting-started.md"]


class ReleaseConsistency(unittest.TestCase):
    def test_project_version_and_build_tags(self):
        self.assertEqual(REL["project_version"], "0.1.0"); self.assertEqual((ROOT / "VERSION").read_text(), "0.1.0\n")
        self.assertRegex(CUR["filename"], r"^stokastisk-0\.1\.0-5-cp313-cp313-linux_x86_64\.whl$"); self.assertEqual(list(REL["builds"]), ["5"]); self.assertEqual(REL["current_build"], 5)
        self.assertEqual(sorted(REL["withdrawn_builds"]), ["3", "4"]); self.assertEqual(REL["withdrawn_builds"]["4"]["sha256"], BUILD4_SHA)
        self.assertEqual(WITHDRAWN["sha256"], BUILD3_SHA); self.assertIn("withdrawn", WITHDRAWN["status"])
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

    def test_no_document_offers_the_withdrawn_build_as_an_install_or_rollback(self):
        for d in ("README.md", "INSTALL.md", "RELEASE.md", "docs/HUB_README.md", "docs/guide/getting-started.md", "docs/guide/troubleshooting.md"):
            t = (ROOT / d).read_text()
            self.assertNotIn("stokastisk-0.1.0-3-cp313-cp313-linux_x86_64.whl#sha256", t, d); self.assertNotRegex(t, r"(?i)roll back to build 3|rollback wheel|stays (on hugging face|available) for rollback", d)

    def test_hash_and_size_tables_agree(self):
        for d in ("RELEASE.md", "docs/HUB_README.md", "docs/PRODUCTION_READINESS.md", "INSTALL.md"):
            t = (ROOT / d).read_text()
            self.assertIn(CUR["sha256"], t, d)
        for d in ("RELEASE.md", "docs/HUB_README.md", "docs/PRODUCTION_READINESS.md"):
            self.assertIn(f"{CUR['size']:,}", (ROOT / d).read_text(), d)
        self.assertIn(REL["pip_index"], (ROOT / "scripts/release/make_pip_index.py").read_text())
        self.assertEqual(mpi.INDEX, REL["pip_index"]); self.assertEqual(mpi.HF + CUR["filename"], CUR["url"])

    def test_current_build_is_never_called_kaggle_executed(self):
        self.assertFalse(CUR["executed_on_kaggle"])
        for d in ("README.md", "RELEASE.md", "docs/PRODUCTION_READINESS.md", "docs/HUB_README.md"):
            t = (ROOT / d).read_text()
            self.assertRegex(t, r"(?i)not (itself )?(separately )?(been )?(run|executed)( on Kaggle)?|not run", d)

    def test_wheel_description_has_no_hash(self):
        t = (ROOT / "README.wheel.md").read_text()
        self.assertNotIn(CUR["sha256"], t); self.assertNotIn(BUILD3_SHA, t); self.assertIn('readme = "README.wheel.md"', (ROOT / "pyproject.toml").read_text())


class ArchivedBuildReport(unittest.TestCase):
    """evidence/release/build-5/ is the archived audit of the current wheel; it must describe exactly the wheel RELEASE.json names."""
    D = ROOT / "evidence/release/build-5"

    def test_manifest_matches_release_json(self):
        m = json.loads((self.D / "release-manifest.json").read_text())
        self.assertEqual((m["project_version"], m["build_tag"]), ("0.1.0", "5"))
        self.assertEqual({k: m["wheel"][k] for k in ("filename", "size", "sha256")}, {k: CUR[k] for k in ("filename", "size", "sha256")})
        self.assertEqual(m["base_vllm_wheel"]["sha256"], REL["base_vllm_wheel"]["sha256"])
        v = m["verification"]
        self.assertFalse(v["this_build_executed_on_kaggle"]); self.assertTrue(v["build3_kaggle_acceptance"]["executed_on_kaggle"]); self.assertIn("NOT_RUN", v["this_build_kaggle_status"])
        self.assertTrue(v["inference_code_equivalent_to_previous_build"]["proven_by_delta_audit"])
        self.assertEqual((v["cpu_verification"]["wheel_audit"], v["cpu_verification"]["delta_audit"], v["cpu_verification"]["reproducible_build"]), ("PASS", "PASS", True))
        self.assertEqual(m["publication"]["huggingface_wheel"]["remote_redownloaded_sha256"], CUR["sha256"])
        self.assertEqual(m["audits"]["delta_vs_reference"]["reference"]["sha256"], BUILD4_SHA)          # audited against build 4 while it was still published

    def test_delta_report_shows_only_the_approved_changes(self):
        d = json.loads((self.D / "reports/delta-vs-reference.json").read_text())
        self.assertEqual(d["status"], "PASS"); s = d["summary"]
        self.assertEqual((s["added"], s["removed"], s["changed"]), (0, 0, 3)); self.assertEqual(s["vllm_members"], s["vllm_members_identical"]); self.assertEqual(s["native_binaries"], s["native_binaries_identical"])
        self.assertEqual(sorted(r["member"] for r in d["changed_members"]), sorted(["stokastisk/serving.py", "stokastisk-0.1.0.dist-info/RECORD", "stokastisk-0.1.0.dist-info/WHEEL"]))
        self.assertTrue(d["ast_analysis"]["stokastisk/serving.py"]["string_literals_only_change"])
        self.assertEqual(d["new_wheel"]["sha256"], CUR["sha256"]); self.assertTrue(all(c["status"] == "PASS" for c in d["checks"]))
        self.assertEqual(json.loads((self.D / "reports/verify.json").read_text())["status"], "PASS")


class WithdrawnBuild4Archive(unittest.TestCase):
    """evidence/release/build-4/ is history: the audit of the withdrawn build 4, still internally consistent and still naming the withdrawn identity."""
    D = ROOT / "evidence/release/build-4"

    def test_archive_describes_the_withdrawn_identity(self):
        m = json.loads((self.D / "release-manifest.json").read_text()); w = REL["withdrawn_builds"]["4"]
        self.assertEqual((m["build_tag"], m["wheel"]["sha256"], m["wheel"]["size"], m["wheel"]["filename"]), ("4", w["sha256"], w["size"], w["filename"]))
        self.assertFalse(m["verification"]["this_build_executed_on_kaggle"])
        d = json.loads((self.D / "reports/delta-vs-reference.json").read_text()); self.assertEqual((d["status"], d["summary"]["changed"]), ("PASS", 5))


if __name__ == "__main__":
    unittest.main()
