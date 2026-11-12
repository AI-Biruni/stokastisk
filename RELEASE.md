# Release status

> **Two distributions, one project version (0.1.0).** The **integrated wheel** (`stokastisk-0.1.0-2-cp313-cp313-linux_x86_64.whl`, release `stokastisk-0.1.0-integrated-kaggle-t4-sm75`) is the canonical distribution: built, statically inspected and CPU-tested; **Kaggle verification pending (notebook 15)**. The **legacy two-wheel bundle** below is `KAGGLE_T4_VERIFIED`. Hashes, contents and limits of the integrated wheel: [docs/INTEGRATED_RELEASE.md](docs/INTEGRATED_RELEASE.md). The earlier build-tag-1 candidate (`sdk-candidate/stokastisk-0.1.0-build1/`) is superseded and kept as a historical artifact.

Project version: **0.1.0** (permanently locked — [VERSION_POLICY.md](VERSION_POLICY.md)). Release: `stokastisk-0.1.0-kaggle-t4-sm75`.
Git tag / GitHub release: `v0.1.0`. Hub: <https://huggingface.co/waqasm86/stokastisk> · archive `releases/stokastisk-0.1.0-kaggle-t4-sm75/stokastisk-0.1.0-kaggle-t4-sm75-kaggle-input.zip`
(SHA256 in `latest.json` and `releases/stokastisk-0.1.0-kaggle-t4-sm75/manifests/SHA256SUMS`).

## Validation status: `KAGGLE_T4_VERIFIED`

| Item | State |
|---|---|
| CPU tests (existing + topology, scheduler, telemetry, transport, benchmarks, version lock) | pass; counts in `evidence/test-results.json` inside the release |
| Wheels | vLLM 0.25.1 wheel **byte-identical** to the earlier baseline build (`cc3bcafb…98f5e`, no native rebuild); `stokastisk-0.1.0` SDK wheel is pure Python, checked file-by-file against `sdk/` (`29edaa55…ee49`) |
| Native binaries | 6 objects, `sm_75` cubins only (no PTX), `libcudart.so.13`, no `.so.12` (`scripts/audit/native_audit.py`) |
| Experimental C++ transport | CUDA-compiled for `sm_75` (links `libcudart.so.13` only); executed on Kaggle in notebook 10 (WARNING; experimental, not integrated into vLLM) |
| Local GPU | one GeForce 940M (SM50, not a T4); never used for GPU claims |
| Kaggle notebooks 01–14 | executed by the owner on 2 × T4: all PASS or WARNING, none FAIL — [docs/KAGGLE_T4_VERIFICATION.md](docs/KAGGLE_T4_VERIFICATION.md) |
| TP=2, PP=2, replicas | **Kaggle-verified** (notebooks 13, 14); slower than TP=1 for a model that fits one T4 |

Notebooks 13 and 14 were first run with SDK wheel `e17b99b4…` (bundle manifest `efe492a6…`) and **re-run on 2026-11-12 with the released SDK wheel `29edaa55…ee49` itself** (provenance printed by the executed notebooks); notebooks 01–12 ran the earlier SDK wheel `a6b8596b…`, with the identical vLLM wheel in all cases. The released SDK wheel predates `stokastisk.compat`, `stokastisk.serving` and `LLM.chat`, so those and the integrated wheel are not covered by any Kaggle run yet. See [docs/KAGGLE_T4_VERIFICATION.md](docs/KAGGLE_T4_VERIFICATION.md).

## Historical engineering measurements (earlier internal baseline build)

The vLLM wheel and native binaries are byte-identical to a baseline build that the owner ran on Kaggle (Python 3.13, torch 2.11.0+cu130, Triton 3.6.0, vLLM 0.25.1, 2 × T4, topology `PHB`).
On that build, real inference with Qwen2.5-0.5B-Instruct worked at TP=1 and TP=2 (`TRITON_ATTN`), and Qwen2.5-1.5B-Instruct (4096 new tokens) measured TP=1 949.0 tok/s vs TP=2 521.3 tok/s (0.55×);
prefill 9648 tokens 97.5k vs 41.0k tok/s. The SDK code around it (topology, strategies, replicas, scheduler, recommender, telemetry, transport, benchmarks) was exercised in notebooks 07–14; see the verification page.

## Known facts
* **TP=2 is slower than TP=1** for models that fit one T4: the two T4s sit behind a PCIe host bridge (peer access varies between Kaggle VMs). Use TP=2 only when a model does not fit one GPU.
* **NCCL runtime:** Kaggle's image contains both `nvidia-nccl-cu13 2.28.9` and `nvidia-nccl-cu12 2.32.3`; both install `nvidia/nccl/lib/libnccl.so.2`, so the later one wins. This is a property of the Kaggle baseline (protected), not of our wheels.
* **Flash-free:** no FlashAttention/FlashInfer/FlashMLA native code or distribution. The vLLM wheel keeps upstream Python adapter modules named `flash*`; they are inert. `patches/0003` guards the two default-path FlashInfer imports (sampler, TP=2 communicator).
* **Reinstall behaviour:** the installer uses `--force-reinstall --no-deps` for the two wheels so a Kaggle session that already has another Stokastisk version ends with exactly 0.1.0; protected packages are never touched.
