"""LLM facade over the bundled vLLM engine (Tesla T4 / SM75 / TRITON_ATTN / fp16).

Lifecycle contract (each point is covered by CPU tests with a fake engine; the GPU behaviour is verified by Kaggle notebook 15):

* ``LLM(...)`` either returns a live engine or raises. A failed construction leaves ``closed == True`` and never a half-alive GPU engine.
* ``generate`` / ``chat`` / ``complete`` / ``generate_structured`` validate their input *before* touching the engine, are serialised (vLLM's offline engine is not
  thread-safe: concurrent callers queue, they never interleave) and accept ``timeout=`` seconds. On timeout :class:`EngineTimeoutError` is raised; the request may still be
  running inside vLLM (offline generation cannot be cancelled), so the next call waits for it - the engine is never driven by two requests at once.
* ``close()`` is idempotent. It always finishes every cleanup step; if the engine's own shutdown failed it then raises :class:`EngineCloseError` (with the original
  exception chained). Nothing is swallowed. ``with LLM(...)`` never lets a close error hide an exception that is already propagating.
* After ``close()`` every inference method raises :class:`EngineClosedError`.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any

from .compat import check_dtype
from .config import ALLOWED_STRUCTURED_OUTPUT_BACKENDS, StokastiskConfig
from .errors import (AttentionBackendError, ConfigurationError, EngineClosedError, EngineCloseError, EngineError, EngineTimeoutError, StokastiskCompatibilityError,
                     StructuredOutputError)
from . import structured

log = logging.getLogger("stokastisk")

# keyword arguments that would change a policy the SDK enforces; they are accepted only when they repeat the enforced value
_POLICY_KWARGS = ("attention_backend", "attention_config", "structured_outputs_config")


def _check_policy_kwargs(kwargs: dict[str, Any], config: StokastiskConfig) -> None:
    if "attention_backend" in kwargs:
        got = str(kwargs["attention_backend"]).upper()
        if got != config.attention_backend:
            raise StokastiskCompatibilityError(f"attention_backend={kwargs['attention_backend']!r}: this build supports {config.attention_backend} only")
    ac = kwargs.get("attention_config")
    if ac is not None:
        backend = ac.get("backend") if isinstance(ac, dict) else getattr(ac, "backend", None)
        name = str(getattr(backend, "name", backend)).upper() if backend is not None else None
        if name is not None and name != config.attention_backend:
            raise StokastiskCompatibilityError(f"attention_config backend {backend!r}: this build supports {config.attention_backend} only")
    sc = kwargs.get("structured_outputs_config")
    if sc is not None:
        backend = sc.get("backend") if isinstance(sc, dict) else getattr(sc, "backend", None)
        if backend is not None and str(backend).lower() not in ALLOWED_STRUCTURED_OUTPUT_BACKENDS:
            raise ConfigurationError(f"structured_outputs_config backend {backend!r} is refused (hidden fallbacks); allowed: {ALLOWED_STRUCTURED_OUTPUT_BACKENDS}")


class LLM:
    """Thin wrapper: applies the T4 defaults (NCCL configured not to depend on peer access, TRITON_ATTN, pinned structured-output backend), then delegates to ``vllm.LLM``.

    Explicit keyword arguments override the config defaults, except the ones that would change an enforced policy (see ``_check_policy_kwargs``)."""

    def __init__(self, model: str, *, config: StokastiskConfig | None = None, **kwargs: Any) -> None:
        self._engine = None
        self._state_lock = threading.RLock()
        self._call_lock = threading.Lock()           # held for the whole duration of one engine call
        self._close_error: BaseException | None = None
        if not isinstance(model, str) or not model.strip():
            raise ConfigurationError("model must be a non-empty str (a Hugging Face id or a local path)")
        self.config = config or StokastiskConfig.from_env()
        strat = self.config.strategy
        if strat == "auto":
            raise ConfigurationError("execution_strategy='auto' is recommendation-only: call stokastisk.recommend_parallelism(...) "
                                     "and pass its to_config() (or an explicit strategy) to LLM")
        if strat == "replica_parallel" and not os.environ.get("STOKASTISK_REPLICA_WORKER"):
            raise ConfigurationError("replica_parallel runs one engine per GPU: use stokastisk.parallel.ReplicaPool(model, devices=[0, 1])")
        if self.config.is_experimental and strat == "pipeline_parallel":
            log.warning("pipeline_parallel is EXPERIMENTAL: Kaggle-verified on 2xT4 (notebooks 08, 13) but slower than TP=1 for a model that fits one T4; use it only for models that need two GPUs")
        _check_policy_kwargs(kwargs, self.config)
        self.config.apply_environment()
        merged = {**self.config.engine_kwargs(), **kwargs}
        check_dtype(str(merged.get("dtype", "half")))      # clear StokastiskCompatibilityError for an explicit bfloat16 instead of vLLM's late ValueError
        try:
            from vllm import LLM as VllmLLM
            self._engine = VllmLLM(model=model, **merged)
        except Exception as e:  # noqa: BLE001
            self._engine = None
            raise EngineError(f"vLLM engine initialisation failed: {type(e).__name__}: {e}") from e
        try:
            self.verify_attention_backend()
        except AttentionBackendError:
            self._close_quietly()          # never leave a GPU engine alive behind a failed construction; the AttentionBackendError is the error that matters
            raise
        log.info("engine ready: model=%s tp=%s pp=%s", model, merged.get("tensor_parallel_size"), merged.get("pipeline_parallel_size", 1))

    # ------------------------------------------------------------------ introspection
    def selected_attention_backend(self) -> str | None:
        """Backend recorded in the live engine's VllmConfig (config-level evidence; the worker's own log line and the Triton cache are the kernel-level evidence)."""
        try:
            b = self._engine.llm_engine.vllm_config.attention_config.backend
        except AttributeError:
            return None
        return getattr(b, "name", str(b)) if b is not None else None

    def verify_attention_backend(self) -> str:
        want = self.config.attention_backend
        got = self.selected_attention_backend()
        if got != want:
            raise AttentionBackendError(f"requested attention backend {want}, engine config has {got!r}")
        return got

    def structured_output_backend(self) -> str | None:
        """The structured-output backend recorded in the live engine's VllmConfig (pinned to 'xgrammar'; never 'auto')."""
        try:
            return str(self._engine.llm_engine.vllm_config.structured_outputs_config.backend)
        except AttributeError:
            return None

    def verify_structured_output_backend(self) -> str:
        want, got = self.config.structured_output_backend, self.structured_output_backend()
        if got != want:
            raise StructuredOutputError(f"requested structured-output backend {want!r}, engine config has {got!r}")
        return got

    @property
    def engine(self):
        return self._engine

    @property
    def closed(self) -> bool:
        """True once :meth:`close` has run, or when the engine failed to start."""
        return getattr(self, "_engine", None) is None

    def _require_engine(self):
        eng = getattr(self, "_engine", None)
        if eng is None:
            raise EngineClosedError("the engine is closed; create a new stokastisk.LLM")
        return eng

    # ------------------------------------------------------------------ input validation (pure; runs before the engine is touched)
    @staticmethod
    def _check_prompts(prompts) -> list[str]:
        if isinstance(prompts, str):
            prompts = [prompts]
        if not isinstance(prompts, (list, tuple)) or not all(isinstance(p, str) for p in prompts):
            raise TypeError("prompts must be a str or a list/tuple of str")
        if any(not p.strip() for p in prompts):
            raise ValueError("prompts must not be empty or whitespace-only")
        return list(prompts)

    @staticmethod
    def _check_timeout(timeout) -> float | None:
        if timeout is None:
            return None
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not timeout > 0:
            raise ValueError("timeout must be None or a number of seconds > 0")
        return float(timeout)

    @staticmethod
    def _check_messages(messages) -> None:
        if not isinstance(messages, (list, tuple)) or not messages:
            raise ValueError("messages must be a non-empty list of {'role', 'content'} dicts (or a list of such lists)")
        flat = messages if isinstance(messages[0], (list, tuple)) else [messages]
        for conv in flat:
            if not conv or not all(isinstance(m, dict) and "role" in m and "content" in m for m in conv):
                raise ValueError("every message must be a dict with 'role' and 'content'")

    # ------------------------------------------------------------------ serialised engine calls
    def _call(self, name: str, args: tuple, kwargs: dict, timeout: float | None, what: str):
        engine = self._require_engine()
        if not self._call_lock.acquire(timeout=-1 if timeout is None else timeout):
            raise EngineTimeoutError(f"{what}: the engine is busy with an earlier request (calls are serialised); waited {timeout}s")
        released = False
        try:
            self._require_engine()                                 # closed while we were queued
            if timeout is None:
                return self._invoke(engine, name, args, kwargs, what)
            box: dict[str, Any] = {}

            def run() -> None:
                try:
                    box["value"] = self._invoke(engine, name, args, kwargs, what)
                except BaseException as e:  # noqa: BLE001 - handed back to the caller's thread
                    box["error"] = e
                finally:
                    self._call_lock.release()

            t = threading.Thread(target=run, name=f"stokastisk-{name}", daemon=True)
            released = True                                        # the helper thread releases the lock when the engine call really ends
            t.start()
            t.join(timeout)
            if t.is_alive():
                raise EngineTimeoutError(f"{what} did not finish within {timeout}s; the request may still be running inside vLLM (offline generation cannot be cancelled), "
                                         "so the next call waits for it")
            if "error" in box:
                raise box["error"]
            return box["value"]
        finally:
            if not released:
                self._call_lock.release()

    @staticmethod
    def _invoke(engine, name: str, args: tuple, kwargs: dict, what: str):
        try:
            return getattr(engine, name)(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            raise EngineError(f"{what} failed: {type(e).__name__}: {e}") from e

    # ------------------------------------------------------------------ inference
    def generate(self, prompts, sampling_params=None, *, timeout: float | None = None, **kwargs):
        """Pass-through to ``vllm.LLM.generate``. An empty prompt list returns ``[]`` without touching the engine; non-str or empty prompts raise ``TypeError`` /
        ``ValueError``; a closed engine raises :class:`EngineClosedError`; ``timeout`` raises :class:`EngineTimeoutError`."""
        self._require_engine()
        prompts = self._check_prompts(prompts) if not (isinstance(prompts, (list, tuple)) and len(prompts) == 0) else []
        timeout = self._check_timeout(timeout)
        if not prompts:
            return []
        return self._call("generate", (prompts, sampling_params), kwargs, timeout, "generation")

    def chat(self, messages, sampling_params=None, *, timeout: float | None = None, **kwargs):
        """Pass-through to ``vllm.LLM.chat`` (the chat template is applied by vLLM). ``messages`` is a list of ``{"role", "content"}`` dicts or a list of such lists."""
        self._require_engine()
        self._check_messages(messages)
        timeout = self._check_timeout(timeout)
        return self._call("chat", (messages, sampling_params), kwargs, timeout, "chat generation")

    def complete(self, prompt: str, *, max_tokens: int = 64, temperature: float = 0.0, timeout: float | None = None) -> str:
        """Generate one completion for ``prompt`` and return its text."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty str")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive int")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or temperature < 0:
            raise ValueError("temperature must be a number >= 0")
        from vllm import SamplingParams
        out = self.generate([prompt], SamplingParams(max_tokens=max_tokens, temperature=temperature), timeout=timeout)
        return out[0].outputs[0].text

    def generate_structured(self, prompts, schema, *, max_tokens: int = 64, temperature: float = 0.0, timeout: float | None = None) -> list[Any]:
        """Generate JSON constrained to ``schema`` by the engine's pinned structured-output backend, then validate every output against the schema.

        Returns the parsed objects. Raises :class:`StructuredOutputError` if any output is not valid JSON or violates the schema (for example because ``max_tokens`` cut it
        off) and :class:`EngineError` if the backend rejected the schema/request - there is no fallback to another backend."""
        self._require_engine()
        prompts = self._check_prompts(prompts)
        params = structured.json_schema_params(schema, max_tokens=max_tokens, temperature=temperature)
        self.verify_structured_output_backend()
        outs = self.generate(prompts, params, timeout=timeout)
        return [structured.validate_json_output(o.outputs[0].text, schema) for o in outs]

    # ------------------------------------------------------------------ lifecycle
    def __enter__(self) -> "LLM":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.close()
        except EngineCloseError:
            if exc_type is None:
                raise
            log.exception("engine shutdown failed while another exception was propagating; the original exception is kept")

    def _close_quietly(self) -> None:
        try:
            self.close()
        except EngineCloseError:
            log.exception("engine shutdown failed while cleaning up a failed construction")

    def close(self) -> None:
        """Shut the engine (and, for TP>1, its worker processes) down explicitly, then release cached CUDA memory. Idempotent.

        Every step runs even if an earlier one failed. If the engine's own shutdown (or the CUDA cache release) raised, :class:`EngineCloseError` is raised AFTER the
        cleanup, with the original exception as ``__cause__``; the engine is closed either way."""
        with self._state_lock:
            eng, self._engine = getattr(self, "_engine", None), None
        if eng is None:
            return
        errors: list[tuple[str, BaseException]] = []
        try:
            core = getattr(getattr(eng, "llm_engine", None), "engine_core", None)
            if core is not None and hasattr(core, "shutdown"):
                core.shutdown()
        except Exception as e:  # noqa: BLE001
            log.exception("engine shutdown raised; continuing with memory release")
            errors.append(("engine shutdown", e))
        del eng
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
        except ImportError:
            log.debug("torch is not importable; no CUDA cache to release")
        except Exception as e:  # noqa: BLE001
            log.exception("torch.cuda.empty_cache() failed")
            errors.append(("CUDA cache release", e))
        if errors:
            self._close_error = errors[0][1]
            raise EngineCloseError("; ".join(f"{what} failed: {type(e).__name__}: {e}" for what, e in errors)) from errors[0][1]
