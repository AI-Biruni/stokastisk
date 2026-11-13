"""The Flash policy (stokastisk.prohibited): real implementations are detected in every file form, ordinary text is not."""
import pathlib, sys, unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sdk"))
from stokastisk import prohibited as p


class Names(unittest.TestCase):
    def test_the_25_names_of_the_requirement_and_the_extra_integration_modules(self):
        for path in ("vllm/utils/flashinfer.py", "vllm/x/flashinfer_cutlass_moe.py", "vllm/v1/attention/backends/flash_attn.py", "vllm/v1/attention/backends/mla/prefill/aiter_flash_attn.py",
                     "vllm/v1/attention/backends/mla/flashmla.py", "vllm/v1/attention/backends/mla/flashmla_sparse.py", "vllm/v1/attention/backends/fa_utils.py",
                     "vllm/v1/attention/backends/rocm_aiter_fa.py", "vllm/model_executor/warmup/fa4_cutedsl_config.py", "vllm/vllm_flash_attn/__init__.py"):
            self.assertTrue(p.is_prohibited_path(path), path)

    def test_flash_named_but_not_flashattention(self):
        for path in ("vllm/v1/attention/ops/triton_reshape_and_cache_flash.py", "vllm/model_executor/models/longcat_flash.py", "vllm/model_executor/models/qwen3_dflash.py",
                     "vllm/v1/spec_decode/dflash.py", "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py", "vllm/v1/attention/backends/triton_attn.py", "vllm/v1/attention/backends/flex_attention.py"):
            self.assertFalse(p.is_prohibited_path(path), path)


class Scan(unittest.TestCase):
    def scan(self, files):
        return p.scan_members(list(files), files.__getitem__)

    def test_clean_tree_with_prose_is_clean(self):
        r = self.scan({"vllm/a.py": b'"""FlashAttention, FlashInfer and FlashMLA are not shipped. See flash_attn docs."""\n', "vllm/b.so": b"\x7fELF cuda kernels triton", "vllm/ops/triton_reshape_and_cache_flash.py": b"x"})
        self.assertTrue(r["clean"], r); self.assertEqual(len(r["allowed_flash_named"]), 1)

    def test_every_form_is_detected(self):
        cases = {"source": {"vllm/flashinfer.py": b""}, "bytecode": {"vllm/__pycache__/flash_attn.cpython-313.pyc": b"\0"}, "native by name": {"vllm/_vllm_fa2_C.abi3.so": b"\x7fELF"},
                 "native by content": {"vllm/innocent.so": b"\x7fELF ... flash_fwd_kernel ..."}, "cubin by content": {"vllm/k.cubin": b"FlashMLA"}, "ptx by content": {"vllm/k.ptx": b"// flashinfer"},
                 "binding": {"vllm/ok.py": b"y = torch.ops._vllm_fa3_C.fwd(q)\n"}, "directory": {"vllm/vllm_flash_attn/x.txt": b""}}
        for name, files in cases.items():
            self.assertFalse(self.scan(files)["clean"], name)

    def test_references_are_reported_but_are_not_implementations(self):
        r = self.scan({"vllm/a.py": b"def f():\n    import flashinfer\n    return flashinfer.x\n"})
        self.assertTrue(r["clean"]); self.assertEqual(list(r["references"]), ["vllm/a.py"])

    def test_pycache_is_listed(self):
        r = self.scan({"vllm/__pycache__/x.cpython-313.pyc": b"\0"}); self.assertEqual(r["pycache_dirs"], ["vllm/__pycache__"])

    def test_maps_parser(self):
        lines = ["7f00-7f01 r-xp 0 00:00 0 /usr/lib/libcudart.so.13", "7f02-7f03 r-xp 0 00:00 0 /x/site-packages/flashinfer/jit/libfoo.so", "7f04-7f05 r--p 0 00:00 0 /x/_vllm_fa2_C.abi3.so", "7f06-7f07 rw-p 0 00:00 0"]
        self.assertEqual(p.flash_in_maps(lines), ["/x/_vllm_fa2_C.abi3.so", "/x/site-packages/flashinfer/jit/libfoo.so"])
        self.assertEqual(p.flash_in_maps(lines[:1]), [])

    def test_this_process_maps_are_readable_and_clean(self):
        self.assertEqual(p.loaded_flash_libraries("self"), [])
        with self.assertRaises(OSError): p.loaded_flash_libraries(2 ** 22 + 12345)       # unreadable maps raise, they are never silently 'clean'


if __name__ == "__main__":
    unittest.main()
