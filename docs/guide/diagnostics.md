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
stokastisk.check_model("Qwen/Qwen2.5-Coder-7B-Instruct", architectures=["Qwen2ForCausalLM"])
```

`check_model` is **offline**: it never downloads the model; it reads the model registry of the *installed* package (so run it after `pip install stokastisk`). Therefore:

* pass `architectures=[...]` (read it from the model's `config.json` → `architectures`), **or** pass a local directory containing `config.json`;
* without either, the status is `UNKNOWN_REQUIRES_EVIDENCE` and the note tells you to pass the architecture.

Result keys: `model`, `dtype`, `quantization`, `architectures`, `note`, `status`.

| Status | Meaning |
|---|---|
| `SUPPORTED` | the architecture is shipped and has been run on Kaggle |
| `UNKNOWN_REQUIRES_EVIDENCE` | shipped or unknown, but never run on Kaggle |

It raises `StokastiskCompatibilityError` when the architecture's module is missing from this wheel, the dtype is bfloat16, or the quantization cannot work on SM75 (Mistral-family models are refused).

!!! The evidence note on `Qwen2ForCausalLM` was written at build time and names only the 0.5B/1.5B runs. The 3B and Coder-7B runs described in this guide happened afterwards; the shipped wheel (frozen at 0.1.0) was not rebuilt to mention them.

Related: `check_dtype(dtype)`, `check_quantization(method)`, `check_feature(name)`, `feature_matrix()` (a copy of the feature table: `SUPPORTED`, `SUPPORTED_WITH_CONDITIONS`, `UNKNOWN_REQUIRES_EVIDENCE`, `EXCLUDED_FLASH_DEPENDENCY`, …). Feature evidence strings reflect the state when the wheel was built; for example the OpenAI server is marked "unknown" in the table although it later passed the Kaggle server stage (21/21) — see [KAGGLE_ACCEPTANCE.md](../KAGGLE_ACCEPTANCE.md).

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
