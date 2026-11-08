# Release status

Project version: **0.1.0** (permanently locked — [VERSION_POLICY.md](VERSION_POLICY.md)). Release: `stokastisk-0.1.0-kaggle-t4-sm75`.
Git tag / GitHub release: `v0.1.0`. Hub: <https://huggingface.co/waqasm86/stokastisk> · archive `releases/stokastisk-0.1.0-kaggle-t4-sm75/stokastisk-0.1.0-kaggle-t4-sm75-kaggle-input.zip`
(SHA256 in `latest.json` and `releases/stokastisk-0.1.0-kaggle-t4-sm75/manifests/SHA256SUMS`).

## Validation status: `KAGGLE_GPU_VERIFICATION_PENDING`

| Item | State |
|---|---|
| CPU tests (existing + topology, scheduler, telemetry, transport, benchmarks, version lock) | pass; counts in `evidence/test-results.json` inside the release |
| Wheels | vLLM 0.25.1 wheel **byte-identical** to the earlier baseline build (`cc3bcafb…98f5e`, no native rebuild); `stokastisk-0.1.0` SDK wheel is pure Python, checked file-by-file against `sdk/` |
| Native binaries | 6 objects, `sm_75` cubins only (no PTX), `libcudart.so.13`, no `.so.12` (`scripts/audit/native_audit.py`) |
| Experimental C++ transport | CUDA-compiled for `sm_75` (links `libcudart.so.13` only); never executed on two GPUs |
| Local GPU | one GeForce 940M (SM50, not a T4); only topology discovery was exercised — no T4 evidence |
| Kaggle notebooks 01–12 | **NOT_RUN** for this release (all unexecuted) |
| PP=2, replica inference, host-staged transport | **not Kaggle-verified**; experimental |

This release is not marked verified. Only an owner-run Kaggle execution of notebooks 01–12 can change that.

## Historical engineering measurements (earlier internal baseline build — not validation of this release)

The vLLM wheel and native binaries are byte-identical to a baseline build that the owner ran on Kaggle (Python 3.13, torch 2.11.0+cu130, Triton 3.6.0, vLLM 0.25.1, 2 × T4, topology `PHB`, no P2P).
On that build, real inference with Qwen2.5-0.5B-Instruct worked at TP=1 and TP=2 (`TRITON_ATTN`), and Qwen2.5-1.5B-Instruct (4096 new tokens) measured TP=1 949.0 tok/s vs TP=2 521.3 tok/s (0.55×);
prefill 9648 tokens 97.5k vs 41.0k tok/s. The 0.1.0 SDK code around it (topology, strategies, replicas, scheduler, recommender, telemetry, transport, benchmarks) was written afterwards and has not run on Kaggle.

## Known facts
* **TP=2 is slower than TP=1** for models that fit one T4: the two T4s sit behind a PCIe host bridge with no peer access. Use TP=2 only when a model does not fit one GPU.
* **NCCL runtime:** Kaggle's image contains both `nvidia-nccl-cu13 2.28.9` and `nvidia-nccl-cu12 2.32.3`; both install `nvidia/nccl/lib/libnccl.so.2`, so the later one wins. This is a property of the Kaggle baseline (protected), not of our wheels.
* **Flash-free:** no FlashAttention/FlashInfer/FlashMLA native code or distribution. The vLLM wheel keeps upstream Python adapter modules named `flash*`; they are inert. `patches/0003` guards the two default-path FlashInfer imports (sampler, TP=2 communicator).
* **Reinstall behaviour:** the installer uses `--force-reinstall --no-deps` for the two wheels so a Kaggle session that already has another Stokastisk version ends with exactly 0.1.0; protected packages are never touched.
