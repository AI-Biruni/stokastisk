#!/usr/bin/env python3.13
"""CPU-only tests: config strategies, backward compatibility, topology parsing/serialisation, graceful degradation."""
import json, pathlib, sys, tempfile, unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sdk"))
import stokastisk  # noqa: E402
from stokastisk import StokastiskConfig, UnsupportedParallelismError, ConfigurationError  # noqa: E402
from stokastisk.topology import pcie, nvml, discovery  # noqa: E402
from stokastisk.topology.models import DeviceInfo, PeerAccess, Topology  # noqa: E402

PHB = "\x1b[4m\tGPU0\tGPU1\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\x1b[0m\nGPU0\t X \tPHB\t0-3\t0\t\tN/A\nGPU1\tPHB\t X \t0-3\t0\t\tN/A\n\nLegend:\n"
NV = "\tGPU0\tGPU1\tCPU Affinity\nGPU0\t X \tNV4\t0-7\nGPU1\tNV4\t X \t0-7\n"


class Config040(unittest.TestCase):
    def test_030_api_unchanged(self):
        c = StokastiskConfig(tensor_parallel_size=1)
        self.assertEqual((c.dtype, c.attention_backend, c.pipeline_parallel_size, c.strategy, c.world_size), ("half", "TRITON_ATTN", 1, "single_gpu", 1))
        self.assertNotIn("pipeline_parallel_size", c.engine_kwargs())          # engine call identical to the baseline at PP=1
        self.assertEqual(StokastiskConfig(tensor_parallel_size=2).strategy, "tensor_parallel")

    def test_valid_combinations(self):
        self.assertEqual(StokastiskConfig(tensor_parallel_size=1, pipeline_parallel_size=2).strategy, "pipeline_parallel")
        pp = StokastiskConfig(pipeline_parallel_size=2, execution_strategy="pipeline_parallel")
        self.assertEqual(pp.engine_kwargs()["pipeline_parallel_size"], 2); self.assertEqual(pp.world_size, 2); self.assertTrue(pp.is_experimental)
        r = StokastiskConfig(execution_strategy="replica_parallel", replica_devices=(0, 1)); self.assertEqual(r.replica_devices, (0, 1))

    def test_rejections(self):
        for kw in (dict(tensor_parallel_size=2, pipeline_parallel_size=2), dict(tensor_parallel_size=3), dict(pipeline_parallel_size=3),
                   dict(execution_strategy="bogus"), dict(execution_strategy="tensor_parallel"), dict(execution_strategy="pipeline_parallel"),
                   dict(execution_strategy="single_gpu", tensor_parallel_size=2), dict(execution_strategy="auto", tensor_parallel_size=2),
                   dict(replica_devices=(0, 1)), dict(execution_strategy="replica_parallel", replica_devices=(0, 0)),
                   dict(execution_strategy="replica_parallel", replica_devices=(0, 2)), dict(execution_strategy="replica_parallel", tensor_parallel_size=2)):
            with self.assertRaises(ConfigurationError, msg=str(kw)):
                StokastiskConfig(**kw)
        with self.assertRaises(UnsupportedParallelismError): StokastiskConfig(tensor_parallel_size=2, pipeline_parallel_size=2)
        with self.assertRaises(ConfigurationError): StokastiskConfig(dtype="bfloat16")
        with self.assertRaises(ConfigurationError): StokastiskConfig(attention_backend="FLASH_ATTN")

    def test_auto_is_recommendation_only(self):
        c = StokastiskConfig(execution_strategy="auto")
        with self.assertRaisesRegex(ConfigurationError, "recommendation-only"):
            stokastisk.LLM("m", config=c)
        c2 = StokastiskConfig(execution_strategy="replica_parallel")
        with self.assertRaisesRegex(ConfigurationError, "ReplicaPool"):
            stokastisk.LLM("m", config=c2)

    def test_env_overrides_and_dict_roundtrip(self):
        with mock.patch.dict("os.environ", {"STOKASTISK_PIPELINE_PARALLEL_SIZE": "2", "STOKASTISK_EXECUTION_STRATEGY": "pipeline_parallel"}):
            c = StokastiskConfig.from_env()
        self.assertEqual((c.pipeline_parallel_size, c.strategy), (2, "pipeline_parallel"))
        json.dumps(c.as_dict())

    def test_thread_budget_uses_world_size(self):
        from stokastisk.limits import threads_per_worker
        self.assertEqual(StokastiskConfig(pipeline_parallel_size=2).world_size, 2)
        self.assertLessEqual(threads_per_worker(2) * 2, max(2, threads_per_worker(1) * 2))


class TopologyParsing(unittest.TestCase):
    def test_phb(self): self.assertEqual(pcie.parse_topo_matrix(PHB), {"0-1": "PHB"})
    def test_nvlink(self): self.assertEqual(pcie.parse_topo_matrix(NV), {"0-1": "NVLINK"})
    def test_garbage(self): self.assertEqual(pcie.parse_topo_matrix("nothing useful"), {})
    def test_all_classes(self):
        for c in ("PIX", "PXB", "PHB", "NODE", "SYS"): self.assertEqual(pcie.classify_cell(c), c)
        self.assertEqual(pcie.classify_cell("NV12"), "NVLINK"); self.assertEqual(pcie.classify_cell("???"), "UNKNOWN")

    def test_pcie_sysfs(self):
        with tempfile.TemporaryDirectory() as d:
            dev = pathlib.Path(d) / "0000:00:04.0"; dev.mkdir()
            for f, v in (("current_link_speed", "8.0 GT/s PCIe"), ("max_link_speed", "8.0 GT/s PCIe"), ("current_link_width", "16"), ("max_link_width", "16")): (dev / f).write_text(v + "\n")
            with mock.patch.object(pcie, "SYSFS_PCI", pathlib.Path(d)):
                got = pcie.pcie_sysfs("00000000:00:04.0")
        self.assertEqual(got, {"pcie_gen_current": 3, "pcie_gen_max": 3, "pcie_width_current": 16, "pcie_width_max": 16})
        self.assertEqual(pcie.pcie_sysfs(None), {})


class TopologyDiscovery(unittest.TestCase):
    def two_t4(self):
        return [DeviceInfo(index=i, name="Tesla T4", compute_capability=(7, 5), total_vram_mib=14912.0, uuid=f"GPU-{i}") for i in range(2)]

    def run_discover(self, matrix=PHB, nvml_rows=None, smi_rows=None, peer=False):
        pa = PeerAccess({0: {1: peer}, 1: {0: peer}}, "mock")
        with mock.patch.object(discovery, "torch_devices", self.two_t4), mock.patch.object(nvml, "nvml_devices", return_value=nvml_rows), \
             mock.patch.object(nvml, "smi_query", return_value=smi_rows), mock.patch.object(nvml, "smi_topo_matrix", return_value=matrix), \
             mock.patch.object(discovery, "measure_peer_access", return_value=pa):
            return discovery.discover_topology()

    def test_phb_no_p2p_recommends_tp1(self):
        t = self.run_discover()
        self.assertEqual(t.pair_link(0, 1), "PHB"); self.assertFalse(t.nvlink_available); self.assertFalse(t.peer_access.any_pair_available)
        self.assertIn("TP=1", t.recommended_parallelism); self.assertIn("no NVLink", t.recommended_parallelism)

    def test_nvlink_detected_not_assumed(self):
        t = self.run_discover(matrix=NV, peer=True)
        self.assertTrue(t.nvlink_available); self.assertTrue(t.peer_access.all_pairs_available); self.assertEqual(t.pair_link(0, 1), "NVLINK")

    def test_missing_everything_degrades_to_unknown(self):
        t = self.run_discover(matrix=None)
        self.assertEqual(t.pair_link(0, 1), "UNKNOWN"); self.assertIsNone(t.nvlink_available); self.assertTrue(any("NVML" in n for n in t.notes))

    def test_p2p_directions_independent(self):
        pa = PeerAccess({0: {1: True}, 1: {0: False}}); self.assertTrue(pa.can_access(0, 1)); self.assertFalse(pa.can_access(1, 0)); self.assertFalse(pa.all_pairs_available); self.assertTrue(pa.any_pair_available)

    def test_serialisation_roundtrip(self):
        t = self.run_discover(); d = json.loads(json.dumps(t.to_dict(), default=str))
        self.assertEqual(d["link_class"], {"0-1": "PHB"}); self.assertEqual(len(d["devices"]), 2); self.assertIn("Topology", str(t))

    def test_unmeasured_values_stay_none(self):
        d = DeviceInfo(index=0); self.assertTrue(all(getattr(d, f) is None for f in ("uuid", "pcie_gen_current", "free_vram_mib", "total_vram_mib")))

    def test_smi_and_nvml_absent(self):
        with mock.patch.object(nvml.shutil, "which", return_value=None):
            self.assertIsNone(nvml.smi_query("index")); self.assertIsNone(nvml.smi_topo_matrix())
        with mock.patch.dict(sys.modules, {"pynvml": None}):
            self.assertIsNone(nvml.nvml_devices())


if __name__ == "__main__":
    unittest.main()
