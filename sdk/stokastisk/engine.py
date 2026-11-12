"""LLM facade over the bundled, unmodified vLLM engine."""
from __future__ import annotations

import logging
import os
from typing import Any

from .config import StokastiskConfig
from .compat import check_dtype
from .errors import AttentionBackendError, ConfigurationError, EngineError

log = logging.getLogger("stokastisk")


class LLM:
    """Thin wrapper: applies T4 defaults (NCCL configured not to depend on peer access), then delegates to ``vllm.LLM``.

    Explicit keyword arguments always override the config defaults.
    """

    def __init__(self, model: str, *, config: StokastiskConfig | None = None, **kwargs: Any) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ConfigurationError("model must be a non-empty str (a Hugging Face id or a local path)")
        self._engine = None
        self.config = config or StokastiskConfig.from_env()
        strat = self.config.strategy
        if strat == "auto":
            raise ConfigurationError("execution_strategy='auto' is recommendation-only: call stokastisk.recommend_parallelism(...) "
                                     "and pass its to_config() (or an explicit strategy) to LLM")
        if strat == "replica_parallel" and not os.environ.get("STOKASTISK_REPLICA_WORKER"):
            raise ConfigurationError("replica_parallel runs one engine per GPU: use stokastisk.parallel.ReplicaPool(model, devices=[0, 1])")
        if self.config.is_experimental and strat == "pipeline_parallel":
            log.warning("pipeline_parallel is EXPERIMENTAL: Kaggle-verified on 2xT4 (notebooks 08, 13) but slower than TP=1 for a model that fits one T4; use it only for models that need two GPUs")
        self.config.apply_environment()
        merged = {**self.config.engine_kwargs(), **kwargs}
        check_dtype(str(merged.get("dtype", "half")))      # clear StokastiskCompatibilityError for an explicit bfloat16 instead of vLLM's late ValueError
        try:
            from vllm import LLM as VllmLLM
            self._engine = VllmLLM(model=model, **merged)
        except Exception as e:  # noqa: BLE001
            raise EngineError(f"vLLM engine initialisation failed: {e}") from e
        try:
            self.verify_attention_backend()
        except AttentionBackendError:
            self.close()          # never leave a GPU engine alive behind a failed construction
            raise
        log.info("engine ready: model=%s tp=%s pp=%s", model, merged.get("tensor_parallel_size"), merged.get("pipeline_parallel_size", 1))

    def selected_attention_backend(self) -> str | None:
        """Backend recorded in the live engine's VllmConfig (config-level evidence, not kernel-level)."""
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

    @property
    def engine(self):
        return self._engine

    # ------------------------------------------------------------------ inference
    @property
    def closed(self) -> bool:
        """True once :meth:`close` has run (or the engine failed to start)."""
        return getattr(self, "_engine", None) is None

    def _require_engine(self):
        if self.closed:
            raise EngineError("the engine is closed; create a new stokastisk.LLM")
        return self._engine

    @staticmethod
    def _check_prompts(prompts) -> list:
        if isinstance(prompts, str):
            return [prompts]
        if not isinstance(prompts, (list, tuple)) or not all(isinstance(p, str) for p in prompts):
            raise TypeError("prompts must be a str or a list/tuple of str")
        return list(prompts)

    def generate(self, prompts, sampling_params=None, **kwargs):
        """Pass-through to ``vllm.LLM.generate``. An empty prompt list returns ``[]`` without touching the engine; non-str prompts raise ``TypeError``;
        a closed engine raises :class:`EngineError`."""
        engine = self._require_engine()
        prompts = self._check_prompts(prompts)
        if not prompts:
            return []
        try:
            return engine.generate(prompts, sampling_params, **kwargs)
        except Exception as e:  # noqa: BLE001 - re-raised with context below
            raise EngineError(f"generation failed: {e}") from e

    def chat(self, messages, sampling_params=None, **kwargs):
        """Pass-through to ``vllm.LLM.chat`` (chat template applied by vLLM). ``messages`` is a list of ``{"role", "content"}`` dicts or a list of such lists.
        Implemented and packaged, NOT executed on Kaggle (see stokastisk.compat.FEATURES['chat_generation'])."""
        engine = self._require_engine()
        if not isinstance(messages, (list, tuple)) or not messages:
            raise ValueError("messages must be a non-empty list of {'role', 'content'} dicts (or a list of such lists)")
        flat = messages if isinstance(messages[0], (list, tuple)) else [messages]
        for conv in flat:
            if not conv or not all(isinstance(m, dict) and "role" in m and "content" in m for m in conv):
                raise ValueError("every message must be a dict with 'role' and 'content'")
        try:
            return engine.chat(messages, sampling_params, **kwargs)
        except Exception as e:  # noqa: BLE001
            raise EngineError(f"chat generation failed: {e}") from e

    def complete(self, prompt: str, *, max_tokens: int = 64, temperature: float = 0.0) -> str:
        """Generate one completion for ``prompt`` and return its text."""
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a non-empty str")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive int")
        from vllm import SamplingParams
        out = self.generate([prompt], SamplingParams(max_tokens=max_tokens, temperature=temperature))
        return out[0].outputs[0].text

    def __enter__(self) -> "LLM":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        """Shut the engine (and, for TP>1, its worker processes) down explicitly, then release cached CUDA memory. Idempotent.
        A failing engine shutdown is logged with its traceback (never silent) and the memory release is still attempted."""
        eng, self._engine = getattr(self, "_engine", None), None
        if eng is None:
            return
        try:
            core = getattr(getattr(eng, "llm_engine", None), "engine_core", None)
            if core is not None and hasattr(core, "shutdown"):
                core.shutdown()
        except Exception:  # noqa: BLE001
            log.exception("engine shutdown raised; continuing with memory release")
        del eng
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            log.debug("torch is not importable; no CUDA cache to release")
        except Exception:  # noqa: BLE001
            log.exception("torch.cuda.empty_cache() failed")
