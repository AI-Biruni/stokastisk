"""Feature -> third-party import names. The single registry that the SDK (`serving`, `engine`), the Kaggle installer and the unit tests share.

Each profile corresponds to one hash-pinned lock file in the release bundle (`requirements/<profile>.lock`) and, for the part that is ABSENT from the protected Kaggle
baseline, lists the import names that lock provides. ``BASELINE`` lists names that the protected Kaggle image provides and Stokastisk never installs.
Nothing here installs anything: `missing()` only asks the import system (``find_spec``; no import side effects)."""
from __future__ import annotations

import importlib.util
from typing import Iterable

# provided by the lock (absent from Kaggle) -------------------------------------------------------------------------------------------------------------
LOCKED: dict[str, tuple[str, ...]] = {
    # imported by vllm.LLM / vllm serve at module level (xgrammar and friends are in vllm/v1/structured_output, tool parsers, chat_utils, hashing, serial_utils)
    "core": ("xgrammar", "tvm_ffi", "llguidance", "cbor2", "msgspec", "openai_harmony", "partial_json_parser", "pybase64"),
    # `SamplingParams` structured-output validation imports backend_lm_format_enforcer, whose dataclass annotation evaluates lmformatenforcer at IMPORT time:
    # lmformatenforcer is therefore required by EVERY structured-output request, whatever backend is selected.
    "structured-output": ("lmformatenforcer", "outlines_core", "diskcache", "interegular"),
    # `vllm serve`: fastapi_cli (CLI), sse_starlette (streaming), pydantic_settings + model_hosting_container_standards (SageMaker routes, bootstrapped unconditionally
    # by build_app), prometheus_fastapi_instrumentator (/metrics), rich_toolkit (fastapi_cli), supervisor (model_hosting_container_standards)
    "serving": ("fastapi_cli", "sse_starlette", "pydantic_settings", "model_hosting_container_standards", "prometheus_fastapi_instrumentator", "rich_toolkit", "supervisor"),
    # optional, NOT installed by default (`--with-models`): quantised checkpoints, Mistral tokenizers, torch.compile debugging
    "models": ("compressed_tensors", "mistral_common", "depyf", "ijson", "setproctitle"),
}
# provided by the protected Kaggle baseline (never installed by Stokastisk) ----------------------------------------------------------------------------------
BASELINE: dict[str, tuple[str, ...]] = {
    "core": ("torch", "triton", "numpy", "transformers", "tokenizers", "safetensors", "huggingface_hub", "pydantic", "regex", "zmq", "psutil", "cloudpickle", "einops",
             "tiktoken", "sentencepiece", "cpuinfo", "cachetools", "requests", "tqdm", "yaml", "PIL", "uvloop", "filelock", "aiohttp", "openai", "fastapi", "starlette",
             "prometheus_client", "opentelemetry"),
    "structured-output": ("jsonschema", "packaging"),
    "serving": ("uvicorn", "httpx", "watchfiles", "click", "rich", "typer", "dotenv", "anyio"),
}
PROFILES = ("core", "structured-output", "serving")                    # installed by the canonical installer
OPTIONAL_PROFILES = ("models",)


def missing(names: Iterable[str]) -> list[str]:
    """Import names that cannot be found (``find_spec`` returning None, or raising for a half-installed namespace)."""
    out = []
    for n in names:
        try:
            if importlib.util.find_spec(n) is None:
                out.append(n)
        except (ImportError, ValueError):
            out.append(n)
    return out


def required_imports(*profiles: str) -> tuple[str, ...]:
    """Locked + baseline import names for the given profiles (order preserved, no duplicates)."""
    seen: dict[str, None] = {}
    for p in profiles:
        for n in (*BASELINE.get(p, ()), *LOCKED.get(p, ())):
            seen.setdefault(n)
    return tuple(seen)


def missing_for(*profiles: str) -> list[str]:
    return missing(required_imports(*profiles))


__all__ = ["LOCKED", "BASELINE", "PROFILES", "OPTIONAL_PROFILES", "missing", "required_imports", "missing_for"]
