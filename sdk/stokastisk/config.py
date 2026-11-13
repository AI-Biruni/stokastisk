"""Typed configuration with T4-safe defaults and environment overrides."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any

from .errors import ConfigurationError, UnsupportedParallelismError
from .limits import LIMITS, cpu_thread_env

ALLOWED_ATTENTION_BACKENDS = ("TRITON_ATTN",)
FORBIDDEN_ATTENTION_MARKERS = ("FLASH", "FA2", "FA3", "FA4", "AITER")
# Structured outputs: the backend is chosen once per ENGINE (vLLM rejects request-level overrides). vLLM's default "auto" silently falls back xgrammar -> guidance -> outlines,
# which is exactly the hidden fallback this SDK refuses: the backend is pinned, and "auto" is an error.
ALLOWED_STRUCTURED_OUTPUT_BACKENDS = ("xgrammar",)
EXECUTION_STRATEGIES = ("single_gpu", "tensor_parallel", "pipeline_parallel", "replica_parallel", "auto")
EXPERIMENTAL_STRATEGIES = ("pipeline_parallel", "replica_parallel")   # Kaggle-verified on 2xT4 (notebooks 08, 09, 13, 14) but still experimental: slower than TP=1 for models that fit one GPU
MAX_GPUS = LIMITS.gpu_count
_BOOL = {"1": True, "true": True, "yes": True, "on": True, "0": False, "false": False, "no": False, "off": False}

# Conservative NCCL settings for 2x T4 behind a PCIe host bridge (PHB). CUDA peer access is a per-VM property on Kaggle (measured true in one
# session, false in another), so P2P is not relied on; shared memory / host paths work in both cases.
NCCL_ENV_NO_P2P: dict[str, str] = {
    "NCCL_P2P_DISABLE": "1",   # do not depend on peer access (it varies between Kaggle VMs) -> shared memory / host
    "NCCL_IB_DISABLE": "1",    # no InfiniBand on Kaggle
    "NCCL_CUMEM_ENABLE": "0",  # vLLM cumem sleep-mode allocator is not used here
}

# vLLM FlashInfer integrations are not shipped in this build; pin them off (user-set values are never overridden).
VLLM_ENV_NO_FLASHINFER: dict[str, str] = {
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "VLLM_ALLREDUCE_USE_FLASHINFER": "0",
}


@dataclass
class StokastiskConfig:
    tensor_parallel_size: int = 1                # 1 or 2; TP=2 adds NCCL traffic over PCIe (T4 has no NVLink)
    pipeline_parallel_size: int = 1              # 1 or 2 (experimental until Kaggle notebook 08 passes)
    execution_strategy: str | None = None        # None = inferred from TP (baseline behaviour); "auto" is recommendation-only
    replica_devices: tuple[int, ...] | None = None   # replica_parallel only: one TP=1 engine per listed GPU index
    dtype: str = "half"                       # T4 has no bf16 tensor cores
    kv_cache_dtype: str = "auto"
    attention_backend: str = "TRITON_ATTN"    # vLLM built-in Triton backend; FlashAttention is never used
    enforce_eager: bool = True
    disable_custom_all_reduce: bool = True    # custom all-reduce requires P2P
    gpu_memory_utilization: float = 0.88
    max_model_len: int | None = 4096
    structured_output_backend: str = "xgrammar"   # pinned (never "auto"): see ALLOWED_STRUCTURED_OUTPUT_BACKENDS
    apply_nccl_env: bool = True
    nccl_env: dict[str, str] = field(default_factory=lambda: dict(NCCL_ENV_NO_P2P))

    def __post_init__(self) -> None:
        self.validate()

    @property
    def strategy(self) -> str:
        """Effective strategy. ``None`` keeps the baseline meaning: TP=2 -> tensor_parallel, otherwise single_gpu (PP>1 -> pipeline_parallel)."""
        if self.execution_strategy is not None:
            return self.execution_strategy
        if self.pipeline_parallel_size > 1:
            return "pipeline_parallel"
        return "tensor_parallel" if self.tensor_parallel_size > 1 else "single_gpu"

    @property
    def world_size(self) -> int:
        return self.tensor_parallel_size * self.pipeline_parallel_size

    @property
    def is_experimental(self) -> bool:
        return self.strategy in EXPERIMENTAL_STRATEGIES

    @staticmethod
    def _int(v: Any) -> bool:
        """A real int. ``bool`` subclasses ``int`` in Python, so ``True`` would otherwise pass as the number 1."""
        return isinstance(v, int) and not isinstance(v, bool)

    def _validate_types(self) -> None:
        for f in ("tensor_parallel_size", "pipeline_parallel_size"):
            if not self._int(getattr(self, f)):
                raise ConfigurationError(f"{f} must be an int, got {getattr(self, f)!r} ({type(getattr(self, f)).__name__})")
        if self.max_model_len is not None and not self._int(self.max_model_len):
            raise ConfigurationError(f"max_model_len must be None or an int, got {self.max_model_len!r} ({type(self.max_model_len).__name__})")
        g = self.gpu_memory_utilization
        if isinstance(g, bool) or not isinstance(g, (int, float)) or g != g:
            raise ConfigurationError(f"gpu_memory_utilization must be a number, got {g!r} ({type(g).__name__})")
        for f in ("enforce_eager", "disable_custom_all_reduce", "apply_nccl_env"):
            if not isinstance(getattr(self, f), bool):
                raise ConfigurationError(f"{f} must be a bool, got {getattr(self, f)!r} ({type(getattr(self, f)).__name__})")
        for f in ("dtype", "kv_cache_dtype", "attention_backend", "structured_output_backend"):
            v = getattr(self, f)
            if not isinstance(v, str) or not v.strip():
                raise ConfigurationError(f"{f} must be a non-empty str, got {v!r}")
        if self.execution_strategy is not None and not isinstance(self.execution_strategy, str):
            raise ConfigurationError(f"execution_strategy must be None or a str, got {self.execution_strategy!r}")
        if not isinstance(self.nccl_env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in self.nccl_env.items()):
            raise ConfigurationError("nccl_env must be a dict[str, str]")
        if self.replica_devices is not None:
            if not isinstance(self.replica_devices, (tuple, list)) or not all(self._int(d) for d in self.replica_devices):
                raise ConfigurationError(f"replica_devices must be a tuple/list of int GPU indices (bool is not an index), got {self.replica_devices!r}")

    def validate(self) -> None:
        self._validate_types()
        b = str(self.attention_backend).upper()
        if any(m in b for m in FORBIDDEN_ATTENTION_MARKERS):
            raise ConfigurationError(f"FlashAttention/FlashInfer/FlashMLA backends are not supported by this release: {self.attention_backend!r}")
        if b not in ALLOWED_ATTENTION_BACKENDS:
            raise ConfigurationError(f"attention_backend must be one of {ALLOWED_ATTENTION_BACKENDS}, got {self.attention_backend!r}")
        self.attention_backend = b
        sb = self.structured_output_backend.strip().lower()
        if sb == "auto":
            raise ConfigurationError("structured_output_backend='auto' is refused: vLLM's auto mode silently falls back between xgrammar, guidance and outlines. "
                                     f"Pin one of {ALLOWED_STRUCTURED_OUTPUT_BACKENDS}.")
        if sb not in ALLOWED_STRUCTURED_OUTPUT_BACKENDS:
            raise ConfigurationError(f"structured_output_backend must be one of {ALLOWED_STRUCTURED_OUTPUT_BACKENDS}, got {self.structured_output_backend!r}")
        self.structured_output_backend = sb
        if self.tensor_parallel_size not in (1, 2):
            raise ConfigurationError("tensor_parallel_size must be 1 or 2 (two Tesla T4 target)")
        if self.pipeline_parallel_size not in (1, 2):
            raise ConfigurationError("pipeline_parallel_size must be 1 or 2 (two Tesla T4 target)")
        if self.tensor_parallel_size * self.pipeline_parallel_size > MAX_GPUS:
            raise UnsupportedParallelismError(
                f"TP={self.tensor_parallel_size} x PP={self.pipeline_parallel_size} needs "
                f"{self.tensor_parallel_size * self.pipeline_parallel_size} GPUs; the target has {MAX_GPUS}")
        strat = self.strategy
        if strat not in EXECUTION_STRATEGIES:
            raise UnsupportedParallelismError(f"execution_strategy must be one of {EXECUTION_STRATEGIES}, got {self.execution_strategy!r}")
        tp, pp = self.tensor_parallel_size, self.pipeline_parallel_size
        want = {"single_gpu": (1, 1), "tensor_parallel": (2, 1), "pipeline_parallel": (1, 2), "replica_parallel": (1, 1)}
        if strat in want and (tp, pp) != want[strat]:
            raise UnsupportedParallelismError(f"execution_strategy={strat!r} requires TP={want[strat][0]}, PP={want[strat][1]}; got TP={tp}, PP={pp}")
        if strat == "auto" and (tp, pp) != (1, 1):
            raise UnsupportedParallelismError("execution_strategy='auto' cannot be combined with explicit TP/PP sizes")
        if self.replica_devices is not None:
            devs = tuple(self.replica_devices)
            if strat != "replica_parallel":
                raise UnsupportedParallelismError("replica_devices is only valid with execution_strategy='replica_parallel'")
            if not devs or len(set(devs)) != len(devs) or any(d < 0 or d >= MAX_GPUS for d in devs):
                raise UnsupportedParallelismError(f"replica_devices must be distinct GPU indices in [0, {MAX_GPUS - 1}], got {devs!r}")
            self.replica_devices = devs
        if self.max_model_len is not None and self.max_model_len < 16:
            raise ConfigurationError(f"max_model_len must be None or an int >= 16, got {self.max_model_len!r}")
        if not 0.1 <= self.gpu_memory_utilization <= 0.98:
            raise ConfigurationError("gpu_memory_utilization must be within [0.1, 0.98]")
        if self.dtype.lower() in ("bfloat16", "bf16"):
            raise ConfigurationError("T4 (SM75) has no bfloat16 support; use 'half'")

    @classmethod
    def from_env(cls, prefix: str = "STOKASTISK_") -> "StokastiskConfig":
        def b(x: str) -> bool:
            if x.lower() not in _BOOL:
                raise ConfigurationError(f"invalid boolean {x!r}")
            return _BOOL[x.lower()]
        cfg = cls()
        for name, ftype in (("tensor_parallel_size", int), ("pipeline_parallel_size", int), ("execution_strategy", str), ("gpu_memory_utilization", float),
                            ("max_model_len", int), ("dtype", str), ("attention_backend", str), ("structured_output_backend", str),
                            ("enforce_eager", b), ("disable_custom_all_reduce", b)):
            raw = os.environ.get(prefix + name.upper())
            if raw is not None:
                try:
                    setattr(cfg, name, ftype(raw))
                except ValueError as e:
                    raise ConfigurationError(f"{prefix}{name.upper()}={raw!r}: {e}") from e
        cfg.validate()
        return cfg

    def apply_environment(self) -> dict[str, str]:
        """Export NCCL settings (never overriding values the user already set)."""
        applied = {}
        for k, v in VLLM_ENV_NO_FLASHINFER.items():
            os.environ.setdefault(k, v)
        for k, v in cpu_thread_env(self.world_size).items():   # 4 Kaggle cores shared by the TP worker processes
            os.environ.setdefault(k, v)
        if self.apply_nccl_env:
            for k, v in self.nccl_env.items():
                if k not in os.environ:
                    os.environ[k] = v
                    applied[k] = v
        return applied

    def engine_kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = dict(
            tensor_parallel_size=self.tensor_parallel_size, dtype=self.dtype,
            kv_cache_dtype=self.kv_cache_dtype, enforce_eager=self.enforce_eager,
            disable_custom_all_reduce=self.disable_custom_all_reduce,
            gpu_memory_utilization=self.gpu_memory_utilization,
            attention_config={"backend": self.attention_backend},
            structured_outputs_config={"backend": self.structured_output_backend},   # pinned: no silent xgrammar -> guidance -> outlines fallback
        )
        if self.pipeline_parallel_size > 1:   # kwarg is omitted at PP=1 so the baseline engine call stays byte-for-byte identical
            kw["pipeline_parallel_size"] = self.pipeline_parallel_size
        if self.max_model_len is not None:
            kw["max_model_len"] = self.max_model_len
        return kw

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
