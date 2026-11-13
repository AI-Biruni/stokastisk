# Architecture (Stokastisk 0.1.0, build 3)

## Target
Kaggle, Python 3.13, 2 × Tesla T4 (SM75, ~15 GiB each, PCIe host bridge, no NVLink, CUDA peer access varies between VMs), PyTorch 2.11.0+cu130 and Triton 3.6.0 provided by the image and **never replaced**. Kaggle limits baked into `stokastisk.LIMITS`: 4 CPU cores, ~30 GiB RAM, 20 GiB `/kaggle/working`.

## Distribution
One wheel `stokastisk-0.1.0-<build>-cp313-cp313-linux_x86_64.whl`:
* `stokastisk/` — the SDK, built from `sdk/`.
* `vllm/` — vLLM 0.25.1 from the byte‑pinned base wheel (SHA256 `cc3bcafb…98f5e`; natives compiled once for `sm_75`, CUDA 13.0 only, no PTX, `libcudart.so.13`) **through `scripts/release/t4_flash_free.py`**: every FlashAttention/FlashInfer/FlashMLA module is removed; imports of them are redirected to one explicit placeholder module (`vllm/utils/t4_prohibited_stubs.py`: predicates `False`, kernel classes "not supported", *warm‑up hooks no‑ops*, everything else raises `ProhibitedBackendError`); `AttentionBackendEnum`/`MLAPrefillBackendEnum` and `AttentionConfig` refuse Flash‑family backends; `vllm/t4_backend_policy.py` makes `import flashinfer`/`flash_attn`/`flash_mla` fail explicitly. No `.so` is touched.
* A bundled `vllm-0.25.1.dist-info` (vLLM resolves its own version through `importlib.metadata`).
Delivered as a Kaggle input bundle: wheel + installer/stage scripts + hash‑pinned locks + manifests (`SHA256SUMS`, `RELEASE-MANIFEST.json`) + docs + notebook 15.

## Dependency model
`requirements/<profile>.roots` (exact pins) → `scripts/release/lock_closure.py` → `<profile>.lock` (sha256 of every compatible wheel) + `closure/<profile>.closure.json`. The resolver treats Kaggle's inventory (`kaggle-baseline-2026-11-13.json`) as protected: a requirement the baseline does not satisfy is a conflict, never an upgrade. Profiles: `core`, `structured-output`, `serving` (installed by default), `models` (opt‑in). `stokastisk.dependencies` maps each profile to import names and is shared by the SDK, the installer and the tests. The installer uses `--no-deps --no-index` and `--require-hashes`; no resolver runs on Kaggle.

## SDK
| Module | Role |
|---|---|
| `config` | `StokastiskConfig`: strict types, T4 defaults (fp16, `TRITON_ATTN`, pinned xgrammar), NCCL env without P2P reliance |
| `engine` | `LLM`: policy kwargs enforced, serialised calls with `timeout=`, `generate/chat/complete/generate_structured`, close that reports failures after cleanup |
| `structured` | JSON‑schema validation independent of the backend |
| `serving` | `ServerProcess` (`vllm serve` child, persistent log, `ServerStartupError`, process‑group shutdown, restart), stdlib HTTP client with typed errors |
| `parallel` | `ReplicaPool` (one TP=1 engine per GPU, token‑aware scheduler, unique batch ids), recommender, memory model |
| `topology`, `telemetry`, `communication`, `cuda`, `benchmarks`, `nccl`, `limits`, `diagnostics` | discovery, NVML sampling, transports, resource limits, environment report |
| `compat` | static T4 compatibility: dtype/quantisation/feature matrix, `check_model` (refuses Flash‑built and Mistral‑family models) |
| `prohibited`, `dependencies` | the Flash policy and the profile registry (single sources of truth) |

## Verification design (notebook 15)
Cell 1 finds and verifies the attached bundle (embedded `bundle.py`); the installer; then separate processes for `static`, `engine`, `parallel`, `server`, `summary`, each writing JSON with provenance (wheel SHA, script SHAs, runtime). Only `PASS` passes; a missing report is `NOT_RUN`; the acceptance ZIP is always written.
