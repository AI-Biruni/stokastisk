# Stokastisk 0.1.0

**A Python 3.13 SDK for Kaggle's 2 × Tesla T4 (SM75) with a built‑in vLLM 0.25.1, Triton attention (`TRITON_ATTN`) and CUDA 13.0 `sm_75` binaries — installed with one line.**

```bash
pip install stokastisk --extra-index-url https://stokastisk-ai.github.io/stokastisk/simple/
```

That downloads the wheel (SDK + vLLM 0.25.1 + CUDA binaries, 38 MB) from the Hugging Face CDN, verifies its SHA256, and installs 19 small pinned dependencies next to Kaggle's own PyTorch 2.11.0+cu130 / Triton 3.6.0 — which it never touches. No dataset upload, no manual wheels.

| | |
|---|---|
| Project version | **0.1.0** (locked — [VERSION_POLICY.md](VERSION_POLICY.md)); the wheel build tag is 3 |
| Runtime | Python 3.13 · vLLM 0.25.1 · PyTorch 2.11.0+cu130 · Triton 3.6.0 · CUDA 13.0 · 2 × Tesla T4 |
| Attention | `TRITON_ATTN` only — no FlashAttention / FlashInfer / FlashMLA code is shipped (requesting one raises `ProhibitedBackendError`) |
| Structured output | xgrammar 0.2.3, pinned (never `auto`); every output validated against its schema |

## User guide (Kaggle)
**1. Notebook settings:** Accelerator **GPU T4 × 2**, **Internet ON**, a Python 3.13 image.

**2. Install** (one cell): `!pip install stokastisk --extra-index-url https://stokastisk-ai.github.io/stokastisk/simple/`

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
| `No matching distribution found for stokastisk` | the notebook is not Python 3.13, or Internet is OFF |
| `pip` wants to change torch / transformers | should never happen; report it with the `pip install` log |
| engine start fails with no GPU | accelerator is not *GPU T4 × 2* |
| `ProhibitedBackendError` | a Flash backend was requested; only `TRITON_ATTN` exists in this build |

**Security note.** `--extra-index-url` searches PyPI as well; the name `stokastisk` is not registered there today. pip verifies the wheel against the SHA256 in the index page. Uninstall: `pip uninstall stokastisk`.

## Not supported
bfloat16 · FlashAttention/FlashInfer/FlashMLA · native FP8/FP4 · L4/SM89 · **Mistral‑family models** (`check_model` refuses them) · OpenTelemetry exporters (conflict with Kaggle's protobuf 7) · calling `vllm.LLM` / `vllm serve` directly (it cannot select Flash, but bypasses the SDK's checks).

## Verification status
| What | State |
|---|---|
| Kaggle 2 × T4 acceptance (notebook 15, 2026‑11‑13): installer 32/32 · static 79/79 · engine 44/44 · parallel 23/23 · server 21/21 · summary 15/15, one wheel SHA throughout | **Kaggle‑tested** (wheel `b31588c9…`) |
| The published wheel (hash: [RELEASE.md](RELEASE.md)) | differs from that wheel **only** in `METADATA`/`RECORD` (two extra exact pins and this README as description); all 2,580 other members — every line of code and every CUDA binary — are byte‑identical |
| The one‑line `pip install` on Kaggle (2026‑11‑13/14, notebooks 16, 18, 19; Python 3.13.16, PyTorch 2.11.0+cu130, Triton 3.6.0) | installed `stokastisk` + 19 pinned packages from the published index; `prohibited_report()['clean']` is `True`; **Kaggle‑tested** |
| Notebook 16 — quickstart script | `TRITON_ATTN`; completion, chat and schema‑constrained JSON returned; engine shut down cleanly — **Kaggle‑tested** |
| Notebook 18 — Qwen2.5 (FP16, eager, `max_model_len` 2048) | 0.5B‑Instruct TP=1; 3B‑Instruct TP=2 and PP=2: `TRITON_ATTN` verified, completion + chat returned, both T4s back to 14.46 GiB free after close — **Kaggle‑tested** |
| Notebook 19 — Qwen2.5‑Coder‑7B‑Instruct (original unquantized checkpoint, FP16) | TP=2 and PP=2 each generated a Python answer; `TRITON_ATTN` verified; GPUs fully released after each engine — **Kaggle‑tested** (PP=2 stays experimental) |
Details and limits: [docs/PRODUCTION_READINESS.md](docs/PRODUCTION_READINESS.md), [docs/KAGGLE_ACCEPTANCE.md](docs/KAGGLE_ACCEPTANCE.md).

## Documentation
Full user guide (Kaggle): **<https://stokastisk-ai.github.io/stokastisk/docs/>** — source in [docs/guide/](docs/guide/index.md): [getting started](docs/guide/getting-started.md) · [configuration](docs/guide/configuration.md) · [LLM engine](docs/guide/engine.md) · [structured output](docs/guide/structured-output.md) · [server](docs/guide/server.md) · [two GPUs](docs/guide/multi-gpu.md) · [diagnostics](docs/guide/diagnostics.md) · [errors](docs/guide/errors.md) · [advanced](docs/guide/advanced.md) · [troubleshooting](docs/guide/troubleshooting.md) · [API reference](docs/guide/api-reference.md).

## Public API
`LLM` (`generate`, `chat`, `complete`, `generate_structured`, `close`) · `StokastiskConfig` · `ReplicaPool` · `serving.ServerProcess` · `discover_topology` · `recommend_parallelism` · `check_model`/`check_dtype`/`check_feature` · `validate_json_output` · `prohibited_report` · `LIMITS` · exceptions (`EngineClosedError`, `EngineTimeoutError`, `EngineCloseError`, `StructuredOutputError`, `ServerStartupError`, `ServerHTTPError`, `MissingDependencyError`, …).

## Repository
`sdk/` the SDK · `scripts/release/` wheel builder, Flash‑free transform, lock resolver, index generator · `scripts/kaggle/` + `scripts/audit/` acceptance and audit tools · `requirements/` hash‑pinned locks · `notebooks/` Kaggle user‑guide notebooks 18 and 19 · `docs/guide/` user guide (built into the GitHub Pages site by `scripts/release/build_docs_site.py`) · `docs/` maintainer notes · `tests/`. Build from source: [BUILD.md](BUILD.md).

## License
Apache‑2.0. vLLM is Apache‑2.0 © the vLLM project.
