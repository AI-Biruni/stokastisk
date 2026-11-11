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
        self.verify_attention_backend()
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

    def generate(self, prompts, sampling_params=None, **kwargs):
        try:
            return self._engine.generate(prompts, sampling_params, **kwargs)
        except Exception as e:  # noqa: BLE001
            raise EngineError(f"generation failed: {e}") from e

    def chat(self, messages, sampling_params=None, **kwargs):
        """Pass-through to ``vllm.LLM.chat`` (chat template applied by vLLM). Implemented, NOT executed on Kaggle (see stokastisk.compat.FEATURES['chat_generation'])."""
        try:
            return self._engine.chat(messages, sampling_params, **kwargs)
        except Exception as e:  # noqa: BLE001
            raise EngineError(f"chat generation failed: {e}") from e

    def complete(self, prompt: str, *, max_tokens: int = 64, temperature: float = 0.0) -> str:
        from vllm import SamplingParams
        out = self.generate([prompt], SamplingParams(max_tokens=max_tokens, temperature=temperature))
        return out[0].outputs[0].text

    def close(self) -> None:
        """Shut the engine (and, for TP>1, its worker processes) down explicitly, then release cached CUDA memory."""
        eng, self._engine = self._engine, None
        if eng is None:
            return
        try:
            core = getattr(getattr(eng, "llm_engine", None), "engine_core", None)
            if core is not None and hasattr(core, "shutdown"):
                core.shutdown()
        except Exception as e:  # noqa: BLE001
            log.warning("engine shutdown raised %s: %s", type(e).__name__, e)
        del eng
        try:
            import gc, torch
            gc.collect(); torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
