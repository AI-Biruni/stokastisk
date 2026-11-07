"""Weight / KV-cache memory estimation from a model's config.json (an ESTIMATE, labelled as such everywhere it is shown)."""
from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass
from typing import Any

from ..errors import ConfigurationError

UNSUPPORTED_ARCH_MARKERS = ("deepseek", "mla", "mamba", "jamba", "falconh1", "nemotronh", "bamba", "zamba", "granitemoehybrid", "minimax", "qwen3next", "qwen3_next")
CUDA_CONTEXT_OVERHEAD_MIB = 1024.0     # CUDA context + activations/workspace reserve per GPU (conservative estimate)


@dataclass
class ModelSpec:
    name: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    intermediate_size: int
    vocab_size: int
    head_dim: int | None = None
    tie_word_embeddings: bool = False
    architecture: str | None = None
    bytes_per_param: int = 2                 # fp16 inference dtype
    kv_bytes_per_elem: int = 2
    param_count_override: float | None = None

    @property
    def head_dim_(self) -> int:
        return self.head_dim or self.hidden_size // self.num_attention_heads

    @property
    def params(self) -> float:
        if self.param_count_override:
            return float(self.param_count_override)
        hd, h = self.head_dim_, self.hidden_size
        attn = h * self.num_attention_heads * hd + 2 * h * self.num_kv_heads * hd + self.num_attention_heads * hd * h
        mlp = 3 * h * self.intermediate_size
        emb = self.vocab_size * h
        head = 0 if self.tie_word_embeddings else emb
        return float(self.num_layers * (attn + mlp + 2 * h) + emb + head + h)

    @property
    def weight_mib(self) -> float:
        return self.params * self.bytes_per_param / 2**20

    def kv_mib(self, tokens: int) -> float:
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim_ * self.kv_bytes_per_elem * tokens / 2**20

    @property
    def unsupported_reason(self) -> str | None:
        a = (self.architecture or "").lower()
        hit = next((m for m in UNSUPPORTED_ARCH_MARKERS if m in a), None)
        return f"architecture {self.architecture!r} (MLA/hybrid/state-space family) needs a non-Triton-unified attention backend" if hit else None

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "estimated_params": self.params, "estimated_weight_mib": self.weight_mib}

    @classmethod
    def from_hf_config(cls, name: str, cfg: dict[str, Any], **kw) -> "ModelSpec":
        try:
            heads = int(cfg["num_attention_heads"])
            return cls(name=name, num_layers=int(cfg["num_hidden_layers"]), hidden_size=int(cfg["hidden_size"]), num_attention_heads=heads,
                       num_kv_heads=int(cfg.get("num_key_value_heads", heads)), intermediate_size=int(cfg.get("intermediate_size", 4 * cfg["hidden_size"])),
                       vocab_size=int(cfg["vocab_size"]), head_dim=cfg.get("head_dim"), tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
                       architecture=(cfg.get("architectures") or [cfg.get("model_type")])[0], **kw)
        except KeyError as e:
            raise ConfigurationError(f"model config for {name!r} lacks {e.args[0]!r}; pass a ModelSpec explicitly") from e


def load_model_spec(model: "str | ModelSpec", *, local_files_only: bool = False) -> ModelSpec:
    """ModelSpec from a ModelSpec, a local model directory, or a Hub repo id (downloads ONLY config.json)."""
    if isinstance(model, ModelSpec):
        return model
    p = pathlib.Path(str(model))
    if p.is_dir() and (p / "config.json").is_file():
        return ModelSpec.from_hf_config(p.name, json.loads((p / "config.json").read_text()))
    try:
        from huggingface_hub import hf_hub_download
        f = hf_hub_download(str(model), "config.json", local_files_only=local_files_only)
    except Exception as e:  # noqa: BLE001
        raise ConfigurationError(f"cannot read config.json for {model!r}: {type(e).__name__}: {e}") from e
    return ModelSpec.from_hf_config(str(model), json.loads(pathlib.Path(f).read_text()))
