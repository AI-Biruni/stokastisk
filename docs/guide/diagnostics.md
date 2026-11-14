# 7. Checks and diagnostics

Everything on this page is safe to run **before** starting an engine, and most of it works without a GPU.

## Environment

```python
import stokastisk
print(stokastisk.report())                  # never raises; safe on CPU hosts
stokastisk.validate_environment()           # raises EnvironmentValidationError on a mismatch
```

| Function | Purpose |
|---|---|
| `report()` | non-raising summary of Python, torch, CUDA, GPUs, native extensions |
| `validate_environment(*, require_gpus=2, strict_torch=True)` | checks the build target: CUDA 13.0, SM75, torch 2.11 cu130. `validate_kaggle_t4` is an alias |
| `discover_gpus()` | list of visible GPUs (name, capability, memory) |
| `check_native_extensions()` | imports each bundled extension; value is `"ok"` or the error text |

## Backend policy

```python
rep = stokastisk.prohibited_report()
assert rep["clean"], rep
```

Keys: `distributions`, `modules`, `files`, `native_hits`, `binaries`, `clean`. All lists are empty when the install is Flash-free (as verified in every Kaggle notebook). Requesting a Flash backend raises `ConfigurationError` (SDK) or `ProhibitedBackendError` (an `ImportError` from the bundled vLLM).

`stokastisk.prohibited` holds the scanners used to audit a wheel or tree (`scan_wheel`, `scan_tree`, `is_prohibited_path`, `loaded_flash_libraries`).

## Is my model supported? — `check_model`

```python
import stokastisk
r = stokastisk.check_model("Qwen/Qwen2.5-Coder-7B-Instruct", mode="TP=2")
print(r["status"], r["model_status"], r["architecture_status"], r["tested_modes"], r["mode_status"])
# SUPPORTED SUPPORTED SUPPORTED_WITH_CONDITIONS ['TP=2', 'PP=2'] TESTED
```

`check_model` is **offline**: it never downloads anything. It reads the model registry of the *installed* package (run it after installing Stokastisk) and a small table of the checkpoints that were actually run on Kaggle. It answers two different questions:

| Field | Question | Values |
|---|---|---|
| `architecture_status` | Is the architecture in the shipped vLLM, and has any verified checkpoint exercised it? | `SUPPORTED_WITH_CONDITIONS` (exercised, for the listed checkpoints only) · `UNKNOWN_REQUIRES_EVIDENCE` |
| `model_status` | Was **this exact checkpoint** (FP16, unquantized) run on Kaggle? | `SUPPORTED` · `UNKNOWN_REQUIRES_EVIDENCE` |
| `status` | the overall answer; equal to `model_status` | `SUPPORTED` only for a verified checkpoint |
| `tested_modes`, `max_tested_model_len`, `evidence`, `limitations` | for a verified checkpoint: what was run and the caveats | |
| `mode_status` | only if you pass `mode="TP=2"` etc. | `TESTED` · `UNTESTED` |

An architecture being present in vLLM — or another checkpoint of it having been run — is **never** treated as evidence for your checkpoint.

### The verified models

| Model id | Modes run | Evidence |
|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | TP=1, TP=2, PP=2, two replicas | notebook 15 acceptance (TP=1 engine and server; parallel stage), notebooks 16 and 18 |
| `Qwen/Qwen2.5-3B-Instruct` | TP=2, PP=2 | notebook 18 |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | TP=2, PP=2 | notebook 19 (original unquantized checkpoint) |

All: FP16, `max_model_len` 2048, eager execution, `TRITON_ATTN`, Kaggle 2 × T4. The full registry is `stokastisk.compat.MODEL_EVIDENCE`; the evidence behind it is in `evidence/kaggle/` in the repository.

### Examples

```python
stokastisk.check_model("Qwen/Qwen2.5-0.5B-Instruct")["status"]                  # SUPPORTED
stokastisk.check_model("Qwen/Qwen2.5-3B-Instruct", mode="TP=1")["mode_status"]  # UNTESTED  (3B was run with TP=2 and PP=2 only)
stokastisk.check_model("Qwen/Qwen2.5-Coder-7B-Instruct", dtype="auto")["status"]  # UNKNOWN_REQUIRES_EVIDENCE (only FP16 was run)

# an untested Qwen2-family checkpoint: the architecture is fine, the checkpoint is unproven
r = stokastisk.check_model("Qwen/Qwen2.5-7B-Instruct", architectures=["Qwen2ForCausalLM"])
r["status"], r["architecture_status"]       # ('UNKNOWN_REQUIRES_EVIDENCE', 'SUPPORTED_WITH_CONDITIONS')
```

For a model you have downloaded, pass its directory: `check_model("/kaggle/working/my-model")`. The directory's `config.json` supplies the architecture and quantization; it is matched to a verified checkpoint only through its `_name_or_path` entry (which identifies the configuration, not the weights).

### When it raises

`StokastiskCompatibilityError` is raised, instead of a status being returned, when the architecture cannot run here: its module is missing from the shipped vLLM (e.g. FlashAttention-based models), it is a Mistral‑family architecture, the dtype is bfloat16, or the quantization needs a newer GPU (e.g. `mxfp4`, minimum capability 8.0). Without a model id, architecture or directory, the status is `UNKNOWN_REQUIRES_EVIDENCE` and the note says what to pass.

`UNKNOWN_REQUIRES_EVIDENCE` is not an error and not a prohibition: it means no one has run that checkpoint on these GPUs yet. Related: `check_dtype(dtype)`, `check_quantization(method)`, `check_feature(name)`, `feature_matrix()`.

### Feature matrix

The same module records which *features* were exercised on Kaggle (OpenAI server, streaming, metrics, structured output, parallelism, …) and which were not. The table is on its own page, generated from the code so it cannot drift: [Feature matrix](feature-matrix.md).

## Kaggle limits

```python
print(stokastisk.LIMITS)                    # 4 CPU, 30 GiB RAM, 20 GiB /kaggle/working, 2 × T4 (SM75, ~15 GiB)
print(stokastisk.check_resources())         # PASS / WARNING per limit; never raises
print(stokastisk.working_dir_status())      # /kaggle/working usage vs the 20 GiB quota
print(stokastisk.cpu_budget())              # CPU cores to plan with
```

Model downloads go to the Hugging Face cache; large models can fill `/kaggle/working` if you save outputs there.

## Dependencies

`stokastisk.dependencies` lists which third-party imports each feature needs (`LOCKED`, `BASELINE`, `PROFILES`), and `missing_for("core", "serving")` returns the import names that cannot be found. Useful when an import fails.

Next: [Errors](errors.md).
