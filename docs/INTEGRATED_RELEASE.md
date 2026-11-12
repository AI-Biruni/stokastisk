# Integrated release — `stokastisk-0.1.0-integrated-kaggle-t4-sm75`

**Status: BUILT, STATICALLY INSPECTED, CPU-TESTED — KAGGLE VERIFICATION PENDING (notebook 15).**
The wheel's size and SHA-256 are in `manifests/RELEASE-MANIFEST.json` (inside the release ZIP), in `SHA256SUMS`, in the GitHub release notes and in the Hub README; they are deliberately not written into this file, which is itself packaged in the release.

## Distribution
| | |
|---|---|
| File | `stokastisk-0.1.0-<build tag>-cp313-cp313-linux_x86_64.whl` (build tag `2`) |
| Project version | exactly `0.1.0` — only the PEP 427 build tag differs from earlier artifacts |
| Python | `>=3.13,<3.14` (tag `cp313`), Linux x86_64 |
| Contents | `stokastisk/**` (fresh build of `sdk/`, byte-checked against it), `vllm/**` (every file byte-identical to the Kaggle-verified vLLM wheel), `vllm-0.25.1.dist-info` (installed through `*.data/platlib/`; vLLM calls `importlib.metadata.version("vllm")`), console script `vllm` and vLLM's plugin entry points |
| Native objects | 6: `_C_stable_libtorch`, `_moe_C_stable_libtorch` (the only two with GPU code: `sm_75` cubins, no PTX, `libcudart.so.13`), `cumem_allocator`, `fs_io_C`, `spinloop`, `_rust_tool_parser`; none needs a CUDA 12 library; max GLIBC 2.38 |
| Dependencies | vLLM's own list with the Kaggle-resolved core pins, `transformers>=5.5.3,<6`, `torch==2.11.0+cu130`, `triton==3.6.0`; **`vllm` is not a dependency** (it is built in) |
| Attention | `TRITON_ATTN` only. SDK: `StokastiskConfig` accepts nothing else, the engine passes `attention_config={"backend": "TRITON_ATTN"}`, `LLM.verify_attention_backend()` compares the live config, `serving` refuses overrides. vLLM source: no FlashAttention/FlashInfer/FlashMLA binaries; `FlashAttentionBackend` is invalid below compute capability 8.0 |
| Never together with | the standalone `vllm` wheel — both own `vllm/` |

## Release contents (ZIP, `releases/stokastisk-0.1.0-integrated-kaggle-t4-sm75/` on the Hub)
`wheels/` (the wheel) · `scripts/kaggle/02_install_integrated.py` (installer), `candidate_15_integration.py` (notebook-15 stages), `_common.py` · `requirements/kaggle-t4-core*.txt` (pins and sha256 hashes) · `notebooks/15_…ipynb` · `manifests/` (`RELEASE-MANIFEST.json`, `BUILD-MANIFEST.json`, `DEPENDENCY-MANIFEST.json`, `SHA256SUMS`) · `evidence/` (static verification, native audit, build manifest) · docs.

## Verification matrix
| Check | Result |
|---|---|
| Python sources compile; CPU tests (`python3.13 -m pytest tests`) | pass — counts in the final release report |
| `scripts/audit/verify_sdk_candidate.py`: RECORD hashes, `stokastisk/**` equals `sdk/`, `vllm/**` identical to the verified wheel, no Flash binaries, tags, metadata | pass |
| `scripts/audit/native_audit.py` (ELF, NEEDED, `cuobjdump`) | pass |
| `pip install --no-deps --target` of the wheel (no GPU): layout, `vllm 0.25.1` and `stokastisk 0.1.0` distributions, entry points, `import stokastisk` | pass (local, not Kaggle) |
| Installer decision functions (`tests/test_installer_integrated.py`) | pass |
| **Kaggle 2 × T4 install and run of the integrated wheel (notebook 15)** | **not run** |

## Evidence from earlier Kaggle runs and what it does not cover
Notebooks 01–14 ran the legacy two-wheel bundle ([KAGGLE_T4_VERIFICATION.md](KAGGLE_T4_VERIFICATION.md)); the vLLM tree and native objects there are byte-identical to those in the integrated wheel. They did not exercise the integrated installation, `LLM.chat`, `stokastisk.compat`, `stokastisk.serving`, the hardened `LLM`/`ReplicaPool` lifecycle ([SOURCE_DIFF_VS_KAGGLE_EXECUTED.md](SOURCE_DIFF_VS_KAGGLE_EXECUTED.md)) or GPU structured output.

## Exact remaining runtime limitations
Unverified until notebook 15 runs: install on Kaggle; `import vllm` through the integrated wheel; metadata resolution in the Kaggle image; the `_C_stable_libtorch` ABI against torch 2.11.0+cu130 via this install path; `LLM.chat`; `compat`; `serving` (HTTP, SSE, metrics); xgrammar JSON-schema decoding on SM75; CUDA graphs; fp8/bfloat16 (rejected by design); speculative decoding, LoRA, multimodal, MLA models.
Known measured facts (legacy bundle): TP=2, PP=2 and replicas are slower than TP=1 for a model that fits one T4. Known dependency finding: [DEPENDENCY_COMPATIBILITY.md](DEPENDENCY_COMPATIBILITY.md).

## Build and reproduce
`python3.13 scripts/release/assemble_integrated_release.py --build-tag 2` (reads the byte-pinned vLLM wheel from `dist-rc/stokastisk-0.1.0-kaggle-t4-sm75/wheels/`, never modifies it, refuses to overwrite a build). Rebuilds are not bit-reproducible across runs only if setuptools output changes; the vLLM and SDK payloads are checked byte-for-byte.
