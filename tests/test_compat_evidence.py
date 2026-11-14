#!/usr/bin/env python3.13
"""CPU-only tests: the compatibility evidence shipped in the SDK (stokastisk.compat, stokastisk.serving) is correct, model-specific, offline, and agrees with the archived Kaggle
evidence manifest and with the documentation. No GPU, no vllm, no network, no model download."""
import hashlib, json, pathlib, re, sys, tempfile, unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sdk")); sys.path.insert(0, str(ROOT / "tests"))
from stokastisk import compat, StokastiskConfig  # noqa: E402
from stokastisk.errors import ConfigurationError, StokastiskCompatibilityError  # noqa: E402
from test_compat_serving import fake_vllm  # noqa: E402

MANIFEST = json.loads((ROOT / "evidence/kaggle/evidence-manifest.json").read_text())
IDS = ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct")
F = compat.FEATURES


class FeatureEvidence(unittest.TestCase):
    def test_server_streaming_metrics_are_kaggle_tested_not_stale(self):
        for name in ("openai_server", "streaming", "prometheus_metrics"):
            r = F[name]
            self.assertEqual(r["evidence"], "kaggle", name); self.assertIn("15", r["notebooks"].split(","), name)
            self.assertNotIn(r["status"], ("UNKNOWN_REQUIRES_EVIDENCE",), name)
            self.assertNotRegex(r["note"], r"build 2 FAILED|verification pending|CPU-tested against a local fake", name)
        self.assertIn("21/21", F["openai_server"]["note"])
        self.assertIn("SageMaker", F["openai_server"]["note"]); self.assertIn("NOT exercised", F["openai_server"]["note"]); self.assertIn("0.141.1", F["openai_server"]["note"])
        self.assertIn("[DONE]", F["streaming"]["note"]); self.assertIn("vllm: samples", F["prometheus_metrics"]["note"])

    def test_structured_outputs_keeps_the_pinned_backend_and_validation(self):
        r = F["structured_outputs"]
        self.assertEqual(r["evidence"], "kaggle"); self.assertEqual(r["status"], "SUPPORTED_WITH_CONDITIONS")
        for s in ("xgrammar 0.2.3", "never 'auto'", "no fallback", "server stage", "engine stage"):
            self.assertIn(s, r["note"])

    def test_chat_and_offline_generation_reflect_the_later_runs(self):
        self.assertNotIn("build2", F["chat_generation"]["evidence"]); self.assertEqual(F["chat_generation"]["status"], "SUPPORTED")
        for nb in ("15", "16", "18", "19"):
            self.assertIn(nb, F["chat_generation"]["notebooks"].split(","))
        for mid in IDS:
            self.assertIn(mid, F["offline_generation"]["note"])
        self.assertNotIn("0.5B/1.5B", F["offline_generation"]["note"])

    def test_genuinely_untested_features_stay_unknown_or_blocked(self):
        for n in ("prefix_caching", "chunked_prefill", "cuda_graphs", "custom_all_reduce", "lora", "multimodal", "speculative_decoding"):
            self.assertEqual(F[n]["status"], "UNKNOWN_REQUIRES_EVIDENCE", n)
        self.assertEqual(F["direct_upstream_vllm"]["status"], "UNSUPPORTED_ENTRY_POINT")
        for n in ("flash_attention", "flashinfer", "flashmla", "bfloat16"):
            with self.assertRaises(StokastiskCompatibilityError):
                compat.check_feature(n)

    def test_no_sdk_source_carries_a_stale_claim(self):
        bad = re.compile(r"verification pending|not yet verified on kaggle|build 2 FAILED|kaggle-build2|0\.5B/1\.5B|implemented-cpu-tested", re.I)
        for f in (ROOT / "sdk/stokastisk").rglob("*.py"):
            for i, line in enumerate(f.read_text().splitlines(), 1):
                self.assertIsNone(bad.search(line), f"{f.relative_to(ROOT)}:{i}: {line[:100]}")

    def test_serving_docstring_states_the_recorded_acceptance(self):
        doc = (ROOT / "sdk/stokastisk/serving.py").read_text().split('"""')[1]
        self.assertNotIn("Not yet verified", doc); self.assertIn("21/21", doc); self.assertIn("ServerProcess", doc); self.assertIn("not a supported entry point", doc)
        self.assertIn("SageMaker", doc)


class ModelEvidence(unittest.TestCase):
    def setUp(self):
        self.root = fake_vllm({"model_executor/models/qwen2.py": "from vllm.config import X\n", "__init__.py": "", "model_executor/__init__.py": "", "model_executor/models/__init__.py": "",
                               "config/__init__.py": ""})

    def test_registry_contains_exactly_the_evidenced_models(self):
        self.assertEqual(set(compat.MODEL_EVIDENCE), set(IDS))
        for mid, r in compat.MODEL_EVIDENCE.items():
            self.assertEqual((r["architecture"], r["dtype"], r["quantization"], r["max_tested_model_len"]), ("Qwen2ForCausalLM", "half", None, 2048), mid)
            self.assertTrue(r["evidence"] and r["limitations"] and r["tested_modes"], mid)
        self.assertEqual(compat.MODEL_EVIDENCE[IDS[0]]["tested_modes"], ("TP=1", "TP=2", "PP=2", "replicas=2"))
        self.assertEqual(compat.MODEL_EVIDENCE[IDS[1]]["tested_modes"], ("TP=2", "PP=2")); self.assertEqual(compat.MODEL_EVIDENCE[IDS[2]]["tested_modes"], ("TP=2", "PP=2"))
        self.assertNotIn("Qwen/Qwen2.5-1.5B-Instruct", compat.MODEL_EVIDENCE)

    def test_untested_qwen2_checkpoint_is_not_supported(self):
        for m in ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "someone/qwen2-finetune"):
            r = compat.check_model(m, architectures=["Qwen2ForCausalLM"], package_root=self.root)
            self.assertEqual((r["status"], r["model_status"]), ("UNKNOWN_REQUIRES_EVIDENCE",) * 2, m)
            self.assertEqual(r["architecture_status"], "SUPPORTED_WITH_CONDITIONS"); self.assertEqual(r["tested_modes"], []); self.assertIn("never run on Kaggle", r["note"])

    def test_known_model_reports_its_modes_offline(self):
        with mock.patch("socket.socket", side_effect=AssertionError("network access")), mock.patch("urllib.request.urlopen", side_effect=AssertionError("network access")):
            for mid in IDS:
                r = compat.check_model(mid, package_root=self.root)                    # architecture comes from the evidence record: no argument needed
                self.assertEqual((r["status"], r["model_status"], r["verified_model_id"]), ("SUPPORTED", "SUPPORTED", mid))
                self.assertEqual(r["tested_modes"], list(compat.MODEL_EVIDENCE[mid]["tested_modes"])); self.assertEqual(r["max_tested_model_len"], 2048)
                self.assertTrue(r["evidence"] and r["limitations"])
        r = compat.check_model(IDS[2].lower(), package_root=self.root); self.assertEqual(r["status"], "SUPPORTED")       # Hub ids are case-insensitive

    def test_mode_status(self):
        self.assertEqual(compat.check_model(IDS[2], package_root=self.root, mode="TP=2")["mode_status"], "TESTED")
        self.assertEqual(compat.check_model(IDS[2], package_root=self.root, mode="TP=1")["mode_status"], "UNTESTED")
        self.assertEqual(compat.check_model(IDS[1], package_root=self.root, mode="tp = 1")["mode_status"], "UNTESTED")
        self.assertEqual(compat.check_model(IDS[0], package_root=self.root, mode="replicas=2")["mode_status"], "TESTED")
        self.assertNotIn("mode_status", compat.check_model(IDS[0], package_root=self.root))

    def test_verified_model_with_a_different_configuration_is_not_supported(self):
        r = compat.check_model(IDS[0], package_root=self.root, dtype="auto"); self.assertEqual(r["status"], "UNKNOWN_REQUIRES_EVIDENCE"); self.assertIn("dtype", r["note"])
        r = compat.check_model(IDS[0], package_root=self.root, quantization="gptq"); self.assertEqual(r["status"], "UNKNOWN_REQUIRES_EVIDENCE"); self.assertIn("quantization", r["note"])
        r = compat.check_model(IDS[0], architectures=["Qwen2ForCausalLM", "BrandNewForCausalLM"], package_root=self.root); self.assertEqual(r["status"], "UNKNOWN_REQUIRES_EVIDENCE"); self.assertIn("differ", r["note"])
        self.assertEqual(compat.check_model(IDS[0], package_root=self.root, dtype="float16")["status"], "SUPPORTED")

    def test_local_directory_with_a_known_identity(self):
        d = pathlib.Path(tempfile.mkdtemp())
        (d / "config.json").write_text(json.dumps({"architectures": ["Qwen2ForCausalLM"], "_name_or_path": IDS[1]}))
        r = compat.check_model(str(d), package_root=self.root)
        self.assertEqual((r["status"], r["verified_model_id"], r["model"]), ("SUPPORTED", IDS[1], str(d)))
        (d / "config.json").write_text(json.dumps({"architectures": ["Qwen2ForCausalLM"], "_name_or_path": "/some/local/finetune"}))
        self.assertEqual(compat.check_model(str(d), package_root=self.root)["status"], "UNKNOWN_REQUIRES_EVIDENCE")
        (d / "config.json").write_text(json.dumps({"architectures": ["Qwen2ForCausalLM"], "_name_or_path": IDS[1], "quantization_config": {"quant_method": "gptq"}}))
        r = compat.check_model(str(d), package_root=self.root); self.assertEqual(r["status"], "UNKNOWN_REQUIRES_EVIDENCE")
        (d / "config.json").write_text(json.dumps({"architectures": ["Qwen2ForCausalLM"], "quantization_config": {"quant_method": "mxfp4"}}))
        with self.assertRaises(StokastiskCompatibilityError):
            compat.check_model(str(d), package_root=self.root)

    def test_unidentifiable_model_stays_unknown(self):
        r = compat.check_model("org/unknown", package_root=self.root)
        self.assertEqual((r["status"], r["model_status"], r["architecture_status"]), ("UNKNOWN_REQUIRES_EVIDENCE",) * 3)

    def test_rejections_are_preserved(self):
        for arch in ("MistralForCausalLM", "MixtralForCausalLM", "PixtralForConditionalGeneration", "VoxtralForConditionalGeneration"):
            with self.assertRaises(StokastiskCompatibilityError):
                compat.check_model("x/y", architectures=[arch], package_root=self.root)
        with self.assertRaises(StokastiskCompatibilityError):
            compat.check_model(IDS[0], package_root=self.root, dtype="bfloat16")
        with self.assertRaises(StokastiskCompatibilityError):
            compat.check_model(IDS[0], package_root=self.root, quantization="mxfp4")
        for backend in ("FLASH_ATTN", "FLASHINFER", "FA3"):
            with self.assertRaises(ConfigurationError):
                StokastiskConfig(attention_backend=backend)

    def test_backward_compatible_fields_remain(self):
        r = compat.check_model(IDS[0], package_root=self.root)
        for k in ("model", "dtype", "quantization", "architectures", "status", "note"):
            self.assertIn(k, r)
        self.assertEqual(r["architectures"][0]["architecture"], "Qwen2ForCausalLM")


class ManifestAgreement(unittest.TestCase):
    """The machine-readable manifest of the archived Kaggle artifacts and the SDK registry must describe the same runs."""

    def test_manifest_files_exist_and_hash(self):
        for a in MANIFEST["raw_artifacts"]:
            b = (ROOT / "evidence/kaggle" / a["path"]).read_bytes()
            self.assertEqual((len(b), hashlib.sha256(b).hexdigest()), (a["size"], a["sha256"]), a["path"])
        self.assertEqual({a["path"] for a in MANIFEST["raw_artifacts"]}, {str(p.relative_to(ROOT / "evidence/kaggle")) for p in (ROOT / "evidence/kaggle").rglob("*") if p.is_file() and p.name != "evidence-manifest.json"})

    def test_executed_notebooks_match_the_hashes_the_owner_supplied(self):
        want = {"18-stokastisk-0-1-0-qwen25-t4-direct-cells.ipynb": "3b571245f0f011b6b70c4ed9f768adf6020f8602cd603df3ccfc2031413f9065",
                "19-stokastisk-0-1-0-qwen2-5-coder-7b-dual-t4.ipynb": "015e6295fc8dce7eadc150566df25c17fece25d06649adac0a78fc356c1d6d50"}
        got = {pathlib.Path(a["path"]).name: a["sha256"] for a in MANIFEST["raw_artifacts"]}
        for n, h in want.items():
            self.assertEqual(got[n], h)

    def test_every_registry_mode_has_a_recorded_run_or_acceptance_stage(self):
        def norm(s):
            return s.replace("replicas=2", "REPLICAS=2")
        runs = {(r["model"], r["parallelism"]) for r in MANIFEST["runs"] if r["generation_completed"] and r["attention_backend"] == "TRITON_ATTN"}
        acc = MANIFEST["notebook_15_acceptance"]
        for mid, rec in compat.MODEL_EVIDENCE.items():
            for mode in rec["tested_modes"]:
                if (mid, mode) in runs:
                    continue
                self.assertEqual(mid, acc["models"]["parallel"], (mid, mode))                  # TP=2 / PP=2 / replicas of the acceptance model come from the notebook 15 parallel stage
                self.assertEqual(acc["stages"]["parallel"]["status"], "PASS")
                self.assertIn(mode, ("TP=2", "PP=2", "replicas=2", "TP=1"))
        for r in MANIFEST["runs"]:
            self.assertIn(r["model"], compat.MODEL_EVIDENCE)
            self.assertIn(r["parallelism"], compat.MODEL_EVIDENCE[r["model"]]["tested_modes"], r["run"])
            self.assertEqual((r["dtype"], r["max_model_len"]), ("half", 2048))
            self.assertTrue(r["generation_completed"], r["run"])

    def test_acceptance_counts_match_the_feature_notes_and_docs(self):
        s = MANIFEST["notebook_15_acceptance"]["stages"]
        self.assertEqual({k: (v["passed"], v["checks"]) for k, v in s.items()}, {"installer": (32, 32), "static": (79, 79), "engine": (44, 44), "parallel": (23, 23), "server": (21, 21), "summary": (15, 15)})
        text = " ".join((ROOT / p).read_text() for p in ("README.md", "RELEASE.md", "docs/PRODUCTION_READINESS.md", "docs/KAGGLE_ACCEPTANCE.md"))
        for frag in ("32/32", "79/79", "44/44", "23/23", "21/21", "15/15"):
            self.assertIn(frag, text)

    def test_memory_recovery_and_the_forced_cleanup_warning_are_recorded(self):
        for r in MANIFEST["runs"]:
            if r["notebook"] in ("18", "19"):
                self.assertTrue(r["gpu_memory_recovered"], r["run"]); self.assertEqual(r["free_gib_after_close"], [14.46, 14.46])
                self.assertEqual(r["process_manager_forced_cleanup_warning"], r["parallelism"] != "TP=1" or r["model"] != "Qwen/Qwen2.5-0.5B-Instruct", r["run"])


class DocumentationAgreement(unittest.TestCase):
    def test_diagnostics_guide_names_every_verified_model_and_no_stale_warning(self):
        t = (ROOT / "docs/guide/diagnostics.md").read_text()
        for mid in IDS:
            self.assertIn(mid, t)
        self.assertNotIn("was not rebuilt to mention them", t); self.assertNotIn("evidence note on `Qwen2ForCausalLM` was written at build time", t)

    def test_no_document_repeats_a_stale_claim(self):
        stale = ("0.5B/1.5B", "first run on Kaggle (notebook 16)", "Verification pending (Kaggle)")
        for f in [*ROOT.glob("*.md"), *(ROOT / "docs").rglob("*.md")]:
            if f.name == "CHANGELOG.md":
                continue
            for s in stale:
                self.assertNotIn(s, f.read_text(), f"{f.relative_to(ROOT)}: {s}")

    def test_feature_matrix_page_is_generated_from_the_registry(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("bds", ROOT / "scripts/release/build_docs_site.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        self.assertEqual(m.feature_matrix_page(), (ROOT / "docs/guide/feature-matrix.md").read_text(), "run scripts/release/build_docs_site.py")


if __name__ == "__main__":
    unittest.main()
