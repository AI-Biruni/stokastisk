#!/usr/bin/env python3.13
"""CPU-only tests: statistics, telemetry fallbacks, benchmark maths, evidence JSON, notebooks, release hygiene, native source."""
import ast, json, pathlib, re, sys, tempfile, unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sdk"))
from stokastisk import telemetry as T  # noqa: E402
from stokastisk.telemetry import cuda_profiler, nvml as tn  # noqa: E402
from stokastisk.benchmarks import baseline, communication, gpu_balance, report  # noqa: E402
from stokastisk.communication import list_transports  # noqa: E402


class Stats(unittest.TestCase):
    def test_percentiles_and_summary(self):
        v = [1, 2, 3, 4, 100]
        self.assertEqual(T.percentile(v, 50), 3); self.assertAlmostEqual(T.percentile(v, 95), 80.8)
        s = T.summarize(v); self.assertEqual((s["n"], s["min"], s["max"], s["median"]), (5, 1, 100, 3)); self.assertEqual(s["runs"], v)   # every run kept
        self.assertEqual(T.summarize([]), {"n": 0})
        with self.assertRaises(ValueError): T.percentile([], 50)

    def test_imbalance(self):
        self.assertEqual(T.imbalance_percent(10, 10), 0.0); self.assertAlmostEqual(T.imbalance_percent(10, 5), 50.0)
        self.assertEqual(T.imbalance_percent(5, 10), T.imbalance_percent(10, 5)); self.assertEqual(T.imbalance_percent(0, 0), 0.0)

    def test_throughput_and_latency(self):
        self.assertEqual(T.throughput(100, 2), 50); self.assertIsNone(T.throughput(1, 0))
        l = T.token_latencies(0.5, 2.5, 5); self.assertAlmostEqual(l["inter_token_latency_s"], 0.5); self.assertIsNone(T.token_latencies(None, 1, 1)["inter_token_latency_s"])

    def test_worker_metrics(self):
        a = T.WorkerMetrics("a", wall_s=10, compute_s=6, comm_wait_s=2); b = T.WorkerMetrics("b", wall_s=10, compute_s=3, comm_wait_s=2)
        self.assertEqual(a.idle_s, 2); r = T.balance_report(a, b)
        self.assertAlmostEqual(r["compute_imbalance_percent"], 50.0); self.assertAlmostEqual(r["busy_imbalance_percent"], (8 - 5) / 8 * 100)


class Fallbacks(unittest.TestCase):
    def test_sampler_without_nvidia_smi(self):
        with mock.patch.object(tn._n, "smi_query", return_value=None):
            s = tn.sample_once(); self.assertEqual((s["source"], s["gpus"]), ("unavailable", []))
            with T.GpuSampler(0.01) as g: pass
            self.assertEqual(g.summary()["source"], "unavailable")

    def test_smi_na_values_become_none(self):
        with mock.patch.object(tn._n, "smi_query", return_value=[["0", "[N/A]", "5", "100", "15000", "1590", "5001", "40", "[Not Supported]"]]):
            g = tn.sample_once()["gpus"][0]; self.assertIsNone(g["gpu_util_pct"]); self.assertIsNone(g["power_w"]); self.assertEqual(g["mem_used_mib"], 100.0)

    def test_nsight_absent_is_skipped(self):
        with mock.patch.object(cuda_profiler.shutil, "which", return_value=None):
            r = cuda_profiler.nsight_status(); self.assertEqual((r["nsys"]["status"], r["ncu"]["status"]), ("SKIPPED", "SKIPPED"))

    def test_otel_span_is_noop_without_package(self):
        with mock.patch.dict(sys.modules, {"opentelemetry": None}):
            with T.span("x", a=1) as s: self.assertIsNone(s)

    def test_json_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            p = T.write_json(pathlib.Path(d) / "a/b.json", {"x": (1, 2), "p": pathlib.Path("/tmp")}); self.assertEqual(json.loads(p.read_text())["x"], [1, 2])
        pr = T.provenance(); self.assertIn("python", pr); self.assertIn("git_commit", pr)

    def test_segment_timer(self):
        st = T.SegmentTimer()
        with st.wall("a"): pass
        with st.wall("a"): pass
        self.assertEqual(len(st.segments["a"]), 2); self.assertEqual(st.report()["a"]["n"], 2)


class BenchmarkMaths(unittest.TestCase):
    def test_workloads_are_deterministic_and_identical(self):
        self.assertEqual(baseline.SHORT.prompts(), baseline.SHORT.prompts()); self.assertEqual(len(baseline.LONG.prompts()), 8)
        self.assertEqual([w.batch_size for w in baseline.SCALING], [1, 8, 16, 32, 64]); self.assertGreater(len(baseline.LONG.prompts()[0]), len(baseline.SHORT.prompts()[0]) * 10)

    def test_communication_sizes_and_compare(self):
        self.assertEqual(list(communication.SIZES.values()), [1 << 10, 4 << 10, 64 << 10, 1 << 20, 4 << 20, 16 << 20, 64 << 20, 128 << 20])
        h = {"1KiB": {"latency_s": {"median": 2.0}}, "1MiB": {"latency_s": {"median": 1.0}}}; n = {"1KiB": {"latency_s": {"median": 1.0}}, "1MiB": {"latency_s": {"median": 2.0}}}
        c = communication.compare(h, n); self.assertEqual(c["host_staged_wins_at"], ["1MiB"]); self.assertEqual(c["per_size"]["1KiB"]["faster"], "nccl")
        slower = {"1KiB": {"latency_s": {"median": 2.0}}, "1MiB": {"latency_s": {"median": 2.0}}}; faster_nccl = {"1KiB": {"latency_s": {"median": 1.0}}, "1MiB": {"latency_s": {"median": 0.5}}}
        self.assertIn("NCCL is faster at every", communication.compare(slower, faster_nccl)["verdict"])

    def rec(self, tps):
        return {"status": "MEASURED", "results": {"short": {"decode_tok_per_s": {"median": tps}, "prefill_tok_per_s": {"median": 10.0}, "ttft_batch_seconds": {"median": 0.1}, "inter_token_latency_s": {"median": 0.01}}}}

    def test_tables_and_relative(self):
        recs = {"tp1": self.rec(949.0), "tp2": self.rec(521.3), "pp2": {"status": "ERROR"}}
        t = report.comparison_table(recs, "short"); self.assertIn("| pp2 | ERROR |", t); self.assertIn("949", t)
        rel = report.relative_to(recs, "short"); self.assertAlmostEqual(rel["tp2"], 521.3 / 949.0); self.assertIsNone(rel["pp2"]); self.assertEqual(rel["tp1"], 1.0)

    def test_replica_balance(self):
        rec = {"workers": {0: {"device": 0}, 1: {"device": 1}}, "gpu_telemetry": {}, "results": {"short": {
            "per_worker_busy_seconds": [{0: 4.0, 1: 2.0}, {0: 4.0, 1: 2.0}], "batch_latency_seconds": {"runs": [4.5, 4.5]}, "gpu_telemetry": {"per_gpu": {0: {"gpu_util_pct": {"mean": 80}}}}}}}
        b = gpu_balance.from_replica_record(rec, "short"); self.assertAlmostEqual(b["compute_imbalance_percent"], 50.0); self.assertEqual(b["workers"][0]["comm_wait_s"], 0.0)
        self.assertEqual(b["workers"][0]["gpu_util_pct_mean"], 80)


class Hygiene(unittest.TestCase):
    def test_transport_registry_is_honest(self):
        t = list_transports(); self.assertTrue(t["nccl"]["used_by_vllm"]); self.assertFalse(t["host_staged"]["used_by_vllm"]); self.assertEqual(t["host_staged"]["status"], "experimental")
        self.assertEqual(t["nvshmem"]["status"], "not_applicable");

    def test_native_source_uses_only_runtime_api(self):
        src = (ROOT / "native/host_staged_xfer.cpp").read_text()
        for must in ("cudaHostAlloc", "cudaMemcpyAsync", "cudaStreamWaitEvent", "cudaEventRecord", "cudaStreamCreateWithFlags", "cudaEventElapsedTime"): self.assertIn(must, src)
        for never in ("cuda.h>", "cuMemcpy", "nvshmem", "cublas", "<nccl", "__global__"): self.assertNotIn(never, src)

    def test_sdk_has_no_flash_or_cuda12_or_foreign_namespace(self):
        pat = re.compile(r"cuda[-_ ]?12\.[89]|cu12[89]|python3\.11|velocit|stochastisk", re.I)
        hits = [(f.relative_to(ROOT).as_posix(), m.group(0)) for f in (ROOT / "sdk").rglob("*.py") for m in pat.finditer(f.read_text())]
        self.assertEqual(hits, [])
        for f in (ROOT / "sdk").rglob("*.py"):
            self.assertNotRegex(f.read_text(), r"^\s*(import|from)\s+(flash_attn|flashinfer)", f.name)

    def test_cpu_only_modules_import_without_cuda(self):
        import subprocess
        code = "import sys; sys.path.insert(0, %r); import stokastisk, stokastisk.parallel, stokastisk.telemetry, stokastisk.topology, stokastisk.communication, stokastisk.cuda, stokastisk.benchmarks; print('ok')" % str(ROOT / "sdk")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin", "HOME": "/tmp"})
        self.assertIn("ok", r.stdout, r.stderr[-400:])

    def test_pyproject_packages_cover_every_sdk_package(self):
        toml = (ROOT / "pyproject.toml").read_text()
        pkgs = {"stokastisk." + p.parent.name for p in (ROOT / "sdk/stokastisk").glob("*/__init__.py")}
        for p in pkgs: self.assertIn(f'"{p}"', toml, p)
        self.assertIn('dynamic = ["version"]', toml); self.assertIn('dependencies = ["vllm==0.25.1"]', toml)

    def test_new_notebooks_are_valid_unexecuted_and_name_one_release(self):
        files = sorted((ROOT / "notebooks" / "kaggle-t4-sm75" / "notebooks").glob("*.ipynb"))
        if not files: self.fail("notebooks 01-14 missing; run scripts/make_notebooks.py")
        self.assertEqual([f.name[:2] for f in files], [f"{i:02d}" for i in range(1, 15)])
        rel = re.search(r'^RELEASE = "([^"]+)"', (ROOT / "scripts/make_notebooks.py").read_text(), re.M).group(1)
        self.assertEqual(rel, "stokastisk-0.1.0-kaggle-t4-sm75")
        for f in files:
            nb = json.loads(f.read_text()); code = [c for c in nb["cells"] if c["cell_type"] == "code"]
            for c in code:
                ast.parse("".join(c["source"])); self.assertEqual((c["outputs"], c["execution_count"]), ([], None), f.name)
            text = f.read_text(); self.assertIn(rel, text); self.assertNotIn("hits[0]", text)
            self.assertIn("assert all(v == 0 for v in rcs.values())" if f.name[:2] < "07" else "assert not bad", text)   # verdict cell present
            self.assertNotIn("kaggle kernels", text)
            others = set(re.findall(r"stokastisk-\d+\.\d+\.\d+-kaggle-t4-sm75(?:-rc\d+)?", text)) - {rel}; self.assertEqual(others, set(), f.name)

    def test_every_notebook_script_exists(self):
        for f in (ROOT / "notebooks" / "kaggle-t4-sm75" / "notebooks").glob("*.ipynb"):
            for s in set(re.findall(r'"(dual_[a-z0-9_]+\.py)"', f.read_text())): self.assertTrue((ROOT / "scripts/kaggle" / s).is_file(), s)

    def test_stage_scripts_parse_and_use_guards(self):
        for p in (ROOT / "scripts/kaggle").glob("dual_*.py"):
            tree = ast.parse(p.read_text()); self.assertIn("__main__", p.read_text(), p.name)
        for n in ("dual_08_pp_validation", "dual_09_replica_validation", "dual_bench", "dual_13_api_compat"):
            self.assertIn("freeze_support", (ROOT / f"scripts/kaggle/{n}.py").read_text(), n)

    def test_pp_script_never_converts_failure_to_pass(self):
        s = (ROOT / "scripts/kaggle/dual_08_pp_validation.py").read_text()
        self.assertIn('"status": "ERROR"', s); self.assertIn("return 1", s)


if __name__ == "__main__":
    unittest.main()
