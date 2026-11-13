# Stokastisk 0.1.0

**A Python 3.13 SDK around a bundled, Flash‑free build of vLLM 0.25.1 (CUDA 13.0, `TRITON_ATTN`) for Kaggle's 2 × Tesla T4 (SM75).**

Stokastisk installs next to Kaggle's own PyTorch 2.11.0+cu130 / Triton 3.6.0 without replacing them and adds validated configuration, tensor/pipeline/replica parallelism, pinned structured output (xgrammar), an OpenAI‑compatible server launcher with real failure reporting, topology discovery, telemetry and Kaggle resource limits.

| | |
|---|---|
| Project version | **0.1.0** (locked — [VERSION_POLICY.md](VERSION_POLICY.md)); the wheel *build tag* distinguishes rebuilds |
| Runtime | Python 3.13 · vLLM 0.25.1 · PyTorch 2.11.0+cu130 · Triton 3.6.0 · CUDA 13.0 · 2 × Tesla T4 (SM75) |
| Attention | `TRITON_ATTN` only. No FlashAttention / FlashInfer / FlashMLA source or binary is shipped; requesting one raises `ProhibitedBackendError` |
| Distribution | one wheel `stokastisk-0.1.0-3-cp313-cp313-linux_x86_64.whl` + hash‑pinned dependency locks, delivered as a Kaggle input bundle — see [RELEASE.md](RELEASE.md) |
| Structured output | backend pinned to xgrammar 0.2.3 (never `auto`), every output validated against its schema |

## Status — read this first
* **CPU‑tested and independently verified:** the wheel (RECORD, byte‑equality with the pinned vLLM wheel plus the documented transform, strict Flash scan, reproducible build), the dependency locks (0 conflicts against Kaggle's real inventory) and the SDK (lifecycle, serving, replicas, configuration).
* **Kaggle‑tested on 2 × T4 (the owner's and maintainer's runs of notebook 15):** input discovery, the installer (32/32 checks, protected packages untouched, no new `pip check` line) and the static stage (79/79: Flash policy, inventory, configuration, backend registry on SM7.5, dependency completeness, CPU structured‑output stack).
* **Verification pending:** the GPU stages — `engine`, `parallel`, `server` — for the final wheel. The first full run failed in the engine because a Flash placeholder raised in vLLM's unconditional kernel warm‑up; that is fixed in the final wheel (see [docs/REMEDIATION.md](docs/REMEDIATION.md)) and must be re‑run by the owner. Until then nothing GPU‑side is claimed production‑verified. Details and every remaining limit: [docs/PRODUCTION_READINESS.md](docs/PRODUCTION_READINESS.md).

## Quick start (Kaggle: Internet ON, accelerator *GPU T4 × 2*, Python 3.13)
Attach the bundle as a dataset and run `notebooks/kaggle-t4-sm75/notebooks/15_stokastisk_candidate_sdk_integration.ipynb` ([how](docs/KAGGLE_BUILD3_VERIFICATION.md)). Manual install of an extracted bundle:
```bash
python3.13 scripts/kaggle/02_install_integrated.py <bundle dir> [--dry-run] [--with-models]
```
The installer verifies every file against `SHA256SUMS`, installs the wheel with `--no-deps --no-index` and the hash‑pinned locks (`core`, `structured-output`, `serving`) with `pip download --require-hashes --no-deps`, proves no pre‑installed package changed, and fails on any new `pip check` line. It never runs a resolver.

```python
import stokastisk
config = stokastisk.StokastiskConfig(tensor_parallel_size=1)
llm = stokastisk.LLM("Qwen/Qwen2.5-0.5B-Instruct", config=config)
print(llm.complete("The capital of France is"))
print(llm.generate_structured(["Describe Paris."], {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}))
llm.close()          # raises EngineCloseError if the engine's own shutdown failed (after finishing the cleanup)
```
Serving: `with stokastisk.serving.ServerProcess("Qwen/Qwen2.5-0.5B-Instruct") as srv: srv.wait_ready(); print(stokastisk.serving.chat(srv.base_url, srv.model, [{"role": "user", "content": "Hi"}]))`.
Parallelism: `StokastiskConfig(tensor_parallel_size=2)`, `StokastiskConfig(pipeline_parallel_size=2, execution_strategy="pipeline_parallel")` (experimental), `stokastisk.parallel.ReplicaPool(model, devices=[0, 1])`; `stokastisk.discover_topology()` / `recommend_parallelism(...)` choose between them. For a model that fits one T4 **TP=1 is fastest** (the two T4s share a PCIe host bridge, no NVLink). Scripts that start TP/PP/replica engines need an `if __name__ == "__main__":` guard (vLLM uses `spawn`).

## Not supported
bfloat16 (SM75) · FlashAttention/FlashInfer/FlashMLA · native FP8/FP4 · L4/SM89 · **Mistral‑family models** (`mistral_common` is not part of any dependency profile; `check_model` refuses them) · OpenTelemetry exporters (conflict with Kaggle's protobuf 7) · calling `vllm.LLM` / `vllm serve` directly (bypasses the SDK's checks; it cannot select Flash, but it is not a supported entry point).

## Public API
`LLM` (`generate`, `chat`, `complete`, `generate_structured`, `close`) · `StokastiskConfig` · `ReplicaPool` · `ServerProcess`/`build_serve_command` · `discover_topology` · `recommend_parallelism` · `check_model`/`check_dtype`/`check_feature`/`check_quantization` · `validate_json_output` · `prohibited_report` · `LIMITS`/`check_resources` · exceptions (`EngineClosedError`, `EngineTimeoutError`, `EngineCloseError`, `StructuredOutputError`, `ServerStartupError`, `ServerHTTPError`, `MissingDependencyError`, …).

## Repository layout
| Path | Content |
|---|---|
| `sdk/stokastisk/` | the SDK |
| `scripts/kaggle/` | installer, bundle verification, notebook‑15 stage scripts (shipped inside the bundle) |
| `scripts/release/` | wheel builder, Flash‑free transform, lock resolver, bundle assembler, notebook generator, Kaggle runner |
| `scripts/audit/` | wheel verifier, import sweep/closure, native audit, CPU structured‑output check |
| `requirements/` | `*.roots` (pinned inputs), `*.lock` (hash‑pinned closure), `closure/` (audit trail), the Kaggle baseline inventory |
| `notebooks/kaggle-t4-sm75/notebooks/` | notebook 15 (the acceptance notebook) |
| `docs/` | architecture, dependency compatibility, Kaggle verification, remediation, production readiness |
| `native/` | optional experimental host‑staged transport source (not packaged) |
| `tests/` | CPU tests (`CUDA_VISIBLE_DEVICES= python3.13 -m pytest tests`) |

Building from source needs the byte‑pinned vLLM base wheel (SHA256 `cc3bcafb…98f5e`, published as a GitHub release asset) at `inputs/vllm-0.25.1-cp38-abi3-linux_x86_64.whl`; see [BUILD.md](BUILD.md).

## License
Apache‑2.0. vLLM is Apache‑2.0 © the vLLM project.
