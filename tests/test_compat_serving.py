#!/usr/bin/env python3.13
"""CPU-only tests for stokastisk.compat and stokastisk.serving. They use a synthetic vllm package tree on disk and never import vllm, torch or a GPU.
These tests were written for future runs; they were NOT executed when they were added (no new test runs were requested)."""
import json, pathlib, sys, tempfile, textwrap, unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sdk"))
import stokastisk  # noqa: E402
from stokastisk import compat, serving  # noqa: E402
from stokastisk.errors import ConfigurationError, StokastiskCompatibilityError  # noqa: E402


def fake_vllm(files: dict[str, str]) -> pathlib.Path:
    td = pathlib.Path(tempfile.mkdtemp())
    reg = textwrap.dedent('''
        _TEXT_GENERATION_MODELS = {
            "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
            "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
            "OmniForCausalLM": ("omni_missing", "OmniForCausalLM"),
            "NeedsMissingForCausalLM": ("needs_missing", "NeedsMissingForCausalLM"),
        }
        _EMBEDDING_MODELS = {"BertModel": ("bert", "BertEmbeddingModel")}
        NOT_A_REGISTRY = {"X": ("a", "b")}
    ''')
    (td / "model_executor/models").mkdir(parents=True)
    (td / "model_executor/models/registry.py").write_text(reg)
    for rel, src in files.items():
        p = td / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(src)
    return td


class FeatureMatrix(unittest.TestCase):
    def test_every_feature_uses_a_defined_status_and_evidence(self):
        status = {"SUPPORTED", "SUPPORTED_WITH_CONDITIONS", "NEEDS_ADAPTATION", "NOT_SUPPORTED_SM75", "EXCLUDED_FLASH_DEPENDENCY", "NOT_IMPLEMENTED", "UNSUPPORTED_ENTRY_POINT", "UNKNOWN_REQUIRES_EVIDENCE"}
        ev = {"kaggle", "kaggle-build2", "kaggle-kernel", "cpu-tests", "static", "implemented-not-executed", "implemented-cpu-tested"}
        for name, r in compat.feature_matrix().items():
            self.assertIn(r["status"], status, name); self.assertIn(r["evidence"], ev, name); self.assertTrue(r["note"], name)

    def test_nothing_is_claimed_supported_without_kaggle_evidence(self):
        for name, r in compat.FEATURES.items():
            if r["status"] == "SUPPORTED":
                self.assertEqual(r["evidence"], "kaggle", name); self.assertTrue(r["notebooks"], name)

    def test_flash_and_sm89_features_raise_with_the_reason(self):
        for name in ("flash_attention", "flashinfer", "flashmla", "bfloat16", "l4_sm89", "qwen_omni_models"):
            with self.assertRaises(StokastiskCompatibilityError, msg=name):
                compat.check_feature(name)

    def test_untested_features_are_returned_not_raised(self):
        self.assertEqual(compat.check_feature("openai_server")["status"], "UNKNOWN_REQUIRES_EVIDENCE")
        self.assertEqual(compat.check_feature("offline_generation")["status"], "SUPPORTED")

    def test_unknown_feature_name_is_an_error(self):
        with self.assertRaises(StokastiskCompatibilityError):
            compat.get_feature("no_such_feature")

    def test_returned_copies_cannot_corrupt_the_table(self):
        compat.feature_matrix()["offline_generation"]["status"] = "NOT_IMPLEMENTED"; compat.get_feature("offline_generation")["status"] = "x"
        self.assertEqual(compat.FEATURES["offline_generation"]["status"], "SUPPORTED")


class DtypeAndQuantization(unittest.TestCase):
    def test_bfloat16_rejected_on_sm75_and_allowed_on_sm80(self):
        for d in ("bfloat16", "bf16", "torch.bfloat16"):
            with self.assertRaises(StokastiskCompatibilityError): compat.check_dtype(d)
        self.assertEqual(compat.check_dtype("bfloat16", (8, 0)), "bfloat16")

    def test_half_and_auto_are_allowed(self):
        self.assertEqual(compat.check_dtype("half"), "half"); self.assertEqual(compat.check_dtype("auto"), "auto"); self.assertEqual(compat.check_dtype(None), "half")

    def test_quantization_minimum_capability_from_the_source(self):
        for m in ("mxfp4", "experts_int8", "fbgemm_fp8", "fp_quant"):
            with self.assertRaises(StokastiskCompatibilityError, msg=m): compat.check_quantization(m)
        for m in ("awq", "gptq", "bitsandbytes", "torchao", "fp8"):
            r = compat.check_quantization(m); self.assertEqual(r["status"], "UNKNOWN_REQUIRES_EVIDENCE", m)     # allowed by capability, never run on Kaggle

    def test_unknown_method_is_not_assumed_to_work(self):
        self.assertEqual(compat.check_quantization("something_new")["status"], "UNKNOWN_REQUIRES_EVIDENCE")
        self.assertEqual(compat.check_quantization(None)["status"], "SUPPORTED")


class ModelCheck(unittest.TestCase):
    def setUp(self):
        self.root = fake_vllm({
            "model_executor/models/qwen2.py": "from vllm.config import X\n",
            "model_executor/models/llama.py": "from vllm.model_executor.models import qwen2\nfrom vllm.flashinfer_gone import thing\n",
            "model_executor/models/needs_missing.py": "from vllm.helper import thing\n",
            "model_executor/models/bert.py": "try:\n    from vllm.absent import q\nexcept ImportError:\n    q = None\n",
            "__init__.py": "", "model_executor/__init__.py": "", "model_executor/models/__init__.py": "", "config/__init__.py": "", "helper.py": "import os\n"})
        (self.root / "helper.py").write_text("from vllm.deeper_missing import z\n")

    def test_registry_is_read_without_importing(self):
        reg = compat.read_registry(self.root)
        self.assertEqual(reg["Qwen2ForCausalLM"], ("qwen2", "Qwen2ForCausalLM")); self.assertIn("BertModel", reg); self.assertNotIn("X", reg)

    def test_qwen2_is_supported_with_kaggle_evidence(self):
        r = compat.check_model(architectures=["Qwen2ForCausalLM"], package_root=self.root)
        self.assertEqual(r["status"], "SUPPORTED"); self.assertEqual(r["architectures"][0]["evidence"], "kaggle")

    def test_registered_but_module_removed_raises_a_clear_error(self):
        with self.assertRaises(StokastiskCompatibilityError) as c: compat.check_model(architectures=["OmniForCausalLM"], package_root=self.root)
        self.assertIn("not part of this T4/Triton-only build", str(c.exception))

    def test_unconditional_import_of_absent_module_raises_with_the_chain(self):
        with self.assertRaises(StokastiskCompatibilityError) as c: compat.check_model(architectures=["LlamaForCausalLM"], package_root=self.root)
        self.assertIn("flashinfer_gone", str(c.exception))
        with self.assertRaises(StokastiskCompatibilityError) as c2: compat.check_model(architectures=["NeedsMissingForCausalLM"], package_root=self.root)
        self.assertIn("deeper_missing", str(c2.exception))

    def test_guarded_import_does_not_block(self):
        r = compat.check_model(architectures=["BertModel"], package_root=self.root)
        self.assertEqual(r["architectures"][0]["status"], "UNKNOWN_REQUIRES_EVIDENCE")

    def test_unregistered_architecture_is_unknown_not_supported(self):
        r = compat.check_model(architectures=["BrandNewForCausalLM"], package_root=self.root)
        self.assertEqual(r["status"], "UNKNOWN_REQUIRES_EVIDENCE")

    def test_local_config_json_is_read_and_quantization_is_checked(self):
        d = pathlib.Path(tempfile.mkdtemp()); (d / "config.json").write_text(json.dumps({"architectures": ["Qwen2ForCausalLM"], "quantization_config": {"quant_method": "mxfp4"}}))
        with self.assertRaises(StokastiskCompatibilityError): compat.check_model(str(d), package_root=self.root)
        (d / "config.json").write_text(json.dumps({"architectures": ["Qwen2ForCausalLM"]}))
        self.assertEqual(compat.check_model(str(d), package_root=self.root)["status"], "SUPPORTED")

    def test_missing_architecture_information_is_reported_not_guessed(self):
        self.assertEqual(compat.check_model("org/some-model", package_root=self.root)["status"], "UNKNOWN_REQUIRES_EVIDENCE")

    def test_bfloat16_is_refused_before_the_registry_is_read(self):
        with self.assertRaises(StokastiskCompatibilityError): compat.check_model(architectures=["Qwen2ForCausalLM"], package_root=self.root, dtype="bfloat16")


class ServingCommand(unittest.TestCase):
    def cfg(self, **kw): return stokastisk.StokastiskConfig(**kw)

    def test_default_command_pins_triton_fp16_and_tp1(self):
        cmd = serving.build_serve_command("Qwen/Qwen2.5-0.5B-Instruct", self.cfg(), python="python")
        self.assertEqual(cmd[:4], ["python", "-c", "from vllm.entrypoints.cli.main import main; main()", "serve"])
        i = cmd.index("--attention-backend"); self.assertEqual(cmd[i + 1], "TRITON_ATTN")
        self.assertEqual(cmd[cmd.index("--dtype") + 1], "half"); self.assertEqual(cmd[cmd.index("--tensor-parallel-size") + 1], "1")
        self.assertIn("--enforce-eager", cmd); self.assertNotIn("--pipeline-parallel-size", cmd)

    def test_tp2_and_pp2(self):
        c = serving.build_serve_command("m", self.cfg(tensor_parallel_size=2, execution_strategy="tensor_parallel")); self.assertEqual(c[c.index("--tensor-parallel-size") + 1], "2")
        c = serving.build_serve_command("m", self.cfg(pipeline_parallel_size=2, execution_strategy="pipeline_parallel")); self.assertEqual(c[c.index("--pipeline-parallel-size") + 1], "2")

    def test_replica_and_auto_are_refused(self):
        for s in ("replica_parallel", "auto"):
            with self.assertRaises(ConfigurationError): serving.build_serve_command("m", self.cfg(execution_strategy=s))

    def test_attention_override_in_extra_args_is_refused(self):
        for a in ("--attention-backend=FLASH_ATTN", "--attention-config"):
            with self.assertRaises(StokastiskCompatibilityError): serving.build_serve_command("m", self.cfg(), extra_args=[a])

    def test_quantization_above_sm75_is_refused(self):
        with self.assertRaises(StokastiskCompatibilityError): serving.build_serve_command("m", self.cfg(), quantization="mxfp4")
        self.assertIn("--quantization", serving.build_serve_command("m", self.cfg(), quantization="awq"))

    def test_bad_port_is_refused(self):
        for p in (0, 70000, "80"):
            with self.assertRaises(ConfigurationError): serving.build_serve_command("m", self.cfg(), port=p)

    def test_start_reports_missing_server_packages_without_launching(self):
        sp = serving.ServerProcess("m", self.cfg())
        with mock.patch.object(serving, "check_server_requirements", return_value=["fastapi"]), mock.patch.object(serving.subprocess, "Popen") as popen:
            with self.assertRaises(stokastisk.MissingDependencyError) as c: sp.start()
        self.assertIn("fastapi", str(c.exception)); popen.assert_not_called()

    def test_check_server_requirements_lists_only_missing_names(self):
        self.assertEqual(serving.check_server_requirements(("json", "definitely_not_a_module_xyz")), ["definitely_not_a_module_xyz"])

    def test_stop_without_start_is_a_no_op(self):
        self.assertIsNone(serving.ServerProcess("m", self.cfg()).stop())


class StreamingParser(unittest.TestCase):
    def test_sse_lines_are_parsed_into_deltas(self):
        lines = [b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n', b"\n", b'data: {"choices":[{"delta":{"content":"Par"}}]}\n',
                 b'data: {"choices":[{"delta":{"content":"is"}}]}\n', b"data: [DONE]\n", b'data: {"choices":[{"delta":{"content":"ignored"}}]}\n']
        class R(list):
            def __enter__(self): return self
            def __exit__(self, *a): return False
        with mock.patch.object(serving, "_post_json", return_value=R(lines)):
            self.assertEqual("".join(serving.stream_chat("http://x", "m", [{"role": "user", "content": "hi"}])), "Paris")

    def test_metrics_text_is_parsed(self):
        class R:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return b'# HELP a b\n# TYPE a counter\nvllm:num_requests_running{model="m"} 3.0\nbad line here\nx 2\n'
        with mock.patch.object(serving.urllib.request, "urlopen", return_value=R()):
            self.assertEqual(serving.fetch_metrics("http://x"), {'vllm:num_requests_running{model="m"}': 3.0, "x": 2.0})


class PublicApi(unittest.TestCase):
    def test_existing_api_is_preserved_and_new_api_exported(self):
        for n in ("LLM", "StokastiskConfig", "ReplicaPool", "recommend_parallelism", "discover_topology", "validate_environment", "validate_kaggle_t4", "SamplingParams"):
            self.assertIn(n, stokastisk.__all__)
        for n in ("check_model", "check_feature", "check_dtype", "check_quantization", "feature_matrix", "ServerProcess", "build_serve_command", "StokastiskCompatibilityError"):
            self.assertIn(n, stokastisk.__all__)
        self.assertEqual(stokastisk.__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
