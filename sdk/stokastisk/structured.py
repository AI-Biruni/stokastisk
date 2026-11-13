"""Structured (JSON-schema) output helpers: pure validation that needs no GPU, plus the sampling-parameter builder.

Two independent defences, because "the grammar backend was asked to constrain the output" is not the same as "the output is valid":

1. **Pinned backend.** :class:`stokastisk.StokastiskConfig` pins ``structured_outputs_config.backend='xgrammar'``. vLLM's default ``'auto'`` silently falls back from xgrammar to
   guidance to outlines; with a pinned backend an unsupported schema is *rejected* instead (``ValueError`` from vLLM, surfaced as :class:`~stokastisk.errors.EngineError`).
   Bound open-ended fields (integer ``minimum``/``maximum``, string ``maxLength``, array ``maxItems``): a grammar cannot force a model to stop, only to stay inside the schema, so an
   unbounded integer can run to ``max_tokens`` and leave truncated JSON (observed on Kaggle with Qwen2.5-0.5B). Bounded fields make termination a property of the grammar.
2. **Output validation.** :func:`validate_json_output` parses the generated text and checks it against the schema with ``jsonschema``; a violation raises
   :class:`~stokastisk.errors.StructuredOutputError`. This is what detects an output that a backend failed to constrain (for example one cut off by ``max_tokens``).
"""
from __future__ import annotations

import json
from typing import Any

from .errors import MissingDependencyError, StructuredOutputError


def _jsonschema():
    try:
        import jsonschema
    except ImportError as e:             # pragma: no cover - jsonschema is part of the Kaggle baseline
        raise MissingDependencyError("structured-output validation", ["jsonschema"], "it is preinstalled on Kaggle; install it elsewhere") from e
    return jsonschema


def check_schema(schema: Any) -> dict:
    """Return ``schema`` (a dict, or a JSON string parsed to one) if it is a valid JSON Schema, else raise :class:`StructuredOutputError`."""
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except json.JSONDecodeError as e:
            raise StructuredOutputError(f"schema is not valid JSON: {e}") from e
    if not isinstance(schema, dict) or not schema:
        raise StructuredOutputError("schema must be a non-empty JSON object (dict)")
    js = _jsonschema()
    try:
        js.Draft202012Validator.check_schema(schema)
    except js.SchemaError as e:
        raise StructuredOutputError(f"schema is not a valid JSON Schema: {e.message}") from e
    return schema


def validate_json_output(text: str, schema: Any) -> Any:
    """Parse ``text`` as JSON and validate it against ``schema``. Returns the parsed object; raises :class:`StructuredOutputError` (never returns invalid data)."""
    schema = check_schema(schema)
    if not isinstance(text, str):
        raise StructuredOutputError(f"generated output must be a str, got {type(text).__name__}")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise StructuredOutputError(f"output is not valid JSON ({e}): {text[:200]!r}") from e
    js = _jsonschema()
    errors = sorted(js.Draft202012Validator(schema).iter_errors(obj), key=lambda e: list(e.path))
    if errors:
        e = errors[0]
        where = "/".join(str(p) for p in e.path) or "<root>"
        raise StructuredOutputError(f"output violates the schema at {where}: {e.message} ({len(errors)} violation(s)); output: {text[:200]!r}")
    return obj


def json_schema_params(schema: Any, *, max_tokens: int = 64, temperature: float = 0.0):
    """``vllm.SamplingParams`` that constrain the output to ``schema``. The backend is the engine's (pinned) one; a request cannot choose another."""
    schema = check_schema(schema)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive int")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or temperature < 0:
        raise ValueError("temperature must be a number >= 0")
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    return SamplingParams(max_tokens=max_tokens, temperature=temperature, structured_outputs=StructuredOutputsParams(json=schema))


__all__ = ["check_schema", "validate_json_output", "json_schema_params"]
