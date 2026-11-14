# Stokastisk 0.1.0

**A Python 3.13 SDK for Kaggle's 2 × Tesla T4 (SM75) with a built‑in vLLM 0.25.1, Triton attention (`TRITON_ATTN`) and CUDA 13.0 `sm_75` binaries — installed with one line.**

```bash
python -m pip install --index-url https://pypi.org/simple "stokastisk @ https://huggingface.co/waqasm86/stokastisk/resolve/main/wheels/stokastisk-0.1.0-4-cp313-cp313-linux_x86_64.whl#sha256=052156d71adc6ad9966901565f86dbded9ff41c620c782db4ff377a1a2b4f62e"
```

That downloads the wheel (SDK + vLLM 0.25.1 + CUDA binaries, 38 MB) from the Hugging Face CDN, **refuses it unless its SHA256 matches**, and installs 19 small pinned dependencies from PyPI next to Kaggle's own PyTorch 2.11.0+cu130 / Triton 3.6.0 — which it never touches. No dataset upload, no manual wheels.

| | |
|---|---|
| Project version | **0.1.0** (locked — [VERSION_POLICY.md](VERSION_POLICY.md)); current wheel build tag **4** (build 3 stays on Hugging Face for rollback — [INSTALL.md](INSTALL.md)) |
| Runtime | Python 3.13 · vLLM 0.25.1 · PyTorch 2.11.0+cu130 · Triton 3.6.0 · CUDA 13.0 · 2 × Tesla T4 |
| Attention | `TRITON_ATTN` only — no FlashAttention / FlashInfer / FlashMLA code is shipped (requesting one raises `ProhibitedBackendError`) |
| Structured output | xgrammar 0.2.3, pinned (never `auto`); every output validated against its schema |

## User guide (Kaggle)
**1. Notebook settings:** Accelerator **GPU T4 × 2**, **Internet ON**, a Python 3.13 image.

**2. Install** (one cell): `!python -m pip install --index-url https://pypi.org/simple "stokastisk @ https://huggingface.co/waqasm86/stokastisk/resolve/main/wheels/stokastisk-0.1.0-4-cp313-cp313-linux_x86_64.whl#sha256=052156d71adc6ad9966901565f86dbded9ff41c620c782db4ff377a1a2b4f62e"`

**3. Run.** Engine code works in ordinary notebook cells (notebooks 18 and 19) and in scripts; in a **script** keep it under a `__main__` guard, because vLLM starts its engine in a spawned process:
```python
%%writefile demo.py
import stokastisk

if __name__ == "__main__":
    schema = {"type": "object", "required": ["city", "population"], "additionalProperties": False,
              "properties": {"city": {"type": "string", "maxLength": 40}, "population": {"type": "integer", "minimum": 0, "maximum": 100000000}}}
    with stokastisk.LLM("Qwen/Qwen2.5-0.5B-Instruct", config=stokastisk.StokastiskConfig(max_model_len=2048)) as llm:
        print(llm.verify_attention_backend())                                    # TRITON_ATTN
        print(llm.complete("The capital of France is", max_tokens=16))
        print(llm.chat([{"role": "user", "content": "Say hello in five words."}])[0].outputs[0].text)
        print(llm.generate_structured(["Describe Paris."], schema, max_tokens=96))   # JSON that satisfies the schema, or an error
```
then `!python demo.py`. Ready‑made notebooks (plain cells, no script needed): [notebooks/18_stokastisk_qwen25_t4.ipynb](notebooks/18_stokastisk_qwen25_t4.ipynb) (Qwen2.5‑0.5B TP=1, Qwen2.5‑3B TP=2 and PP=2) and [notebooks/19_stokastisk_qwen25_coder_7b_dual_t4.ipynb](notebooks/19_stokastisk_qwen25_coder_7b_dual_t4.ipynb) (Qwen2.5‑Coder‑7B‑Instruct, TP=2 and PP=2).

**4. OpenAI‑compatible server** (also from a script):
```python
import stokastisk
from stokastisk import serving
with serving.ServerProcess("Qwen/Qwen2.5-0.5B-Instruct") as srv:
    srv.wait_ready()
    print(serving.chat(srv.base_url, srv.served_model_name, [{"role": "user", "content": "Hi"}]))
    print("".join(serving.stream_chat(srv.base_url, srv.served_model_name, [{"role": "user", "content": "Count to five."}])))
```
If the server fails to start, `ServerStartupError` carries the exit code, the command, the log path and the last lines of the server's output.

**5. Two GPUs.** `StokastiskConfig(tensor_parallel_size=2)` for a model too large for one T4, `StokastiskConfig(pipeline_parallel_size=2, execution_strategy="pipeline_parallel")` (experimental), or `stokastisk.parallel.ReplicaPool(model, devices=[0, 1])` for two independent copies. `stokastisk.discover_topology()` and `recommend_parallelism(model)` choose for you. For a model that fits one T4, **TP=1 is fastest** (the T4s share a PCIe host bridge, no NVLink).

**Tips.** Bound open‑ended schema fields (`maximum`, `maxLength`, `maxItems`): a grammar keeps the model inside the schema but cannot make it stop, and an unbounded integer can run to `max_tokens`. `llm.close()` (or `with`) releases the GPUs; it raises `EngineCloseError` if the engine's own shutdown failed.

| Problem | Cause / fix |
|---|---|
| `No matching distribution` / `is not a supported wheel` | the notebook is not Python 3.13, or Internet is OFF |
| `pip` wants to change torch / transformers | should never happen; report it with the `pip install` log |
| engine start fails with no GPU | accelerator is not *GPU T4 × 2* |
| `ProhibitedBackendError` | a Flash backend was requested; only `TRITON_ATTN` exists in this build |

**Security note.** The command above uses PyPI as its only index and pins the wheel by URL and SHA256. The convenience index (`--extra-index-url https://stokastisk-ai.github.io/stokastisk/simple/`) makes pip also consider PyPI, where a different project of the same name could in principle win; see [INSTALL.md](INSTALL.md). Uninstall: `pip uninstall stokastisk`.

## Not supported
bfloat16 · FlashAttention/FlashInfer/FlashMLA · native FP8/FP4 · L4/SM89 · **Mistral‑family models** (`check_model` refuses them) · OpenTelemetry exporters (conflict with Kaggle's protobuf 7) · calling `vllm.LLM` / `vllm serve` directly (it cannot select Flash, but bypasses the SDK's checks).

## Verification status
Three separate classes of evidence — details in [RELEASE.md](RELEASE.md), [docs/PRODUCTION_READINESS.md](docs/PRODUCTION_READINESS.md), raw artifacts in [evidence/kaggle/](evidence/kaggle/):

| What | State |
|---|---|
| **Build 3 on Kaggle 2 × T4** — notebook 15 acceptance (2026‑11‑13): installer 32/32 · static 79/79 · engine 44/44 · parallel 23/23 · server 21/21 · summary 15/15 | **Kaggle‑tested** (acceptance wheel `b31588c9…`; the published build‑3 wheel differs only in `METADATA`/`RECORD`) |
| **Build 3, end‑user notebooks 16, 18, 19** (Python 3.13.16, PyTorch 2.11.0+cu130, Triton 3.6.0): quickstart; Qwen2.5‑0.5B‑Instruct TP=1; Qwen2.5‑3B‑Instruct TP=2 and PP=2; Qwen2.5‑Coder‑7B‑Instruct TP=2 and PP=2 (FP16, `max_model_len` 2048, `TRITON_ATTN`; GPUs back to 14.46 of 14.56 GiB free after each engine) | **Kaggle‑tested** — only these exact models and modes; `stokastisk.check_model()` reports them |
| **Build 4 (current)** — compatibility‑reporting correction; 2,577 of 2,582 members byte‑identical to build 3 (all vLLM files, all native binaries, every dependency pin) | **verified on CPU** (wheel audit, delta audit, reproducible build, tests). **Not itself run on Kaggle**; build 3's evidence is inherited for the unchanged inference/server code |

A checkpoint that is not in that list — even another Qwen2 model — is reported as `UNKNOWN_REQUIRES_EVIDENCE`. Details and limits: [docs/PRODUCTION_READINESS.md](docs/PRODUCTION_READINESS.md), [docs/KAGGLE_ACCEPTANCE.md](docs/KAGGLE_ACCEPTANCE.md).

## Documentation
Full user guide (Kaggle): **<https://stokastisk-ai.github.io/stokastisk/docs/>** — source in [docs/guide/](docs/guide/index.md): [getting started](docs/guide/getting-started.md) · [configuration](docs/guide/configuration.md) · [LLM engine](docs/guide/engine.md) · [structured output](docs/guide/structured-output.md) · [server](docs/guide/server.md) · [two GPUs](docs/guide/multi-gpu.md) · [diagnostics](docs/guide/diagnostics.md) · [errors](docs/guide/errors.md) · [advanced](docs/guide/advanced.md) · [troubleshooting](docs/guide/troubleshooting.md) · [API reference](docs/guide/api-reference.md).

## Public API
`LLM` (`generate`, `chat`, `complete`, `generate_structured`, `close`) · `StokastiskConfig` · `ReplicaPool` · `serving.ServerProcess` · `discover_topology` · `recommend_parallelism` · `check_model`/`check_dtype`/`check_feature` · `validate_json_output` · `prohibited_report` · `LIMITS` · exceptions (`EngineClosedError`, `EngineTimeoutError`, `EngineCloseError`, `StructuredOutputError`, `ServerStartupError`, `ServerHTTPError`, `MissingDependencyError`, …).

## Repository
`sdk/` the SDK · `scripts/release/` wheel builder, Flash‑free transform, lock resolver, index generator · `scripts/kaggle/` + `scripts/audit/` acceptance and audit tools · `requirements/` hash‑pinned locks · `notebooks/` Kaggle user‑guide notebooks 18 and 19 · `evidence/kaggle/` archived Kaggle reports and executed notebooks · `docs/guide/` user guide (built into the GitHub Pages site by `scripts/release/build_docs_site.py`) · `docs/` maintainer notes · `tests/`. Build from source: [BUILD.md](BUILD.md).

## License
Apache‑2.0. vLLM is Apache‑2.0 © the vLLM project.
