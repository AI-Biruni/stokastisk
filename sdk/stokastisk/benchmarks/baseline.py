"""Shared engine benchmark: identical workload for every strategy, ALL runs recorded (median/p95/min/max, never only the best)."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from ..telemetry import GpuSampler, provenance, summarize

TOPIC_WORDS = ("tensor", "kernel", "warp", "stream", "cache", "memory", "latency", "bandwidth", "pipeline", "scheduler")


@dataclass
class Workload:
    name: str = "default"
    batch_size: int = 8
    input_tokens: int = 128          # approximate: prompts are built from a fixed word list (1 word ~ 1-2 tokens)
    output_tokens: int = 128
    warmup: int = 1
    repeats: int = 5
    temperature: float = 0.0

    def prompts(self) -> list[str]:
        out = []
        for i in range(self.batch_size):      # deterministic: no RNG, identical across strategies
            words = [TOPIC_WORDS[(i + j) % len(TOPIC_WORDS)] for j in range(max(1, self.input_tokens // 2))]
            out.append(f"Request {i}. Explain briefly how these relate on a GPU: " + " ".join(words))
        return out


SHORT = Workload("short", batch_size=8, input_tokens=16, output_tokens=64)
LONG = Workload("long", batch_size=8, input_tokens=1024, output_tokens=64)
SCALING = [Workload(f"b{b}", batch_size=b, input_tokens=64, output_tokens=64) for b in (1, 8, 16, 32, 64)]


def _model_revision(model: str) -> str | None:
    try:
        import pathlib, os
        root = pathlib.Path(os.environ.get("HF_HOME", pathlib.Path.home() / ".cache/huggingface")) / "hub" / ("models--" + model.replace("/", "--")) / "refs" / "main"
        return root.read_text().strip() if root.exists() else None
    except Exception:  # noqa: BLE001
        return None


def measure(llm, workload: Workload, sampler_interval: float = 0.5) -> dict[str, Any]:
    """Time ``workload`` on an already-loaded engine. Warm-up runs are discarded; every measured repeat is kept."""
    from vllm import SamplingParams
    prompts = workload.prompts()
    sp = SamplingParams(max_tokens=workload.output_tokens, temperature=workload.temperature, ignore_eos=True)
    sp1 = SamplingParams(max_tokens=1, temperature=workload.temperature, ignore_eos=True)
    for _ in range(workload.warmup):
        llm.generate(prompts, sp)
    with GpuSampler(sampler_interval) as gs:
        full, prefill, toks, in_toks = [], [], 0, 0
        for _ in range(workload.repeats):
            a = time.perf_counter(); llm.generate(prompts, sp1); b = time.perf_counter(); prefill.append(b - a)
            a = time.perf_counter(); out = llm.generate(prompts, sp); b = time.perf_counter(); full.append(b - a)
            toks = sum(len(x.outputs[0].token_ids) for x in out); in_toks = sum(len(x.prompt_token_ids) for x in out)
    n = max(1, workload.output_tokens)
    return {"workload": asdict(workload), "output_tokens_per_run": toks, "input_tokens_per_run": in_toks,
            "full_run_seconds": summarize(full), "ttft_batch_seconds": summarize(prefill),   # TTFT of the batch: time until every request has its first token
            "decode_tok_per_s": summarize([toks / t for t in full]), "prefill_tok_per_s": summarize([in_toks / t for t in prefill]),
            "inter_token_latency_s": summarize([(f - p) / max(1, n - 1) for f, p in zip(full, prefill)]), "gpu_telemetry": gs.summary(), "status": "MEASURED"}


def run_engine(config, model: str, workloads: "Workload | list[Workload]", *, label: str | None = None, sampler_interval: float = 0.5) -> dict[str, Any]:
    """Load ONE in-process stokastisk.LLM and measure every workload on it. Call from a ``__main__``-guarded script (vLLM spawns)."""
    import stokastisk
    wls = [workloads] if isinstance(workloads, Workload) else list(workloads)
    rec: dict[str, Any] = {"label": label or config.strategy, "strategy": config.strategy, "config": config.as_dict(), "model": model,
                           "model_revision": _model_revision(model), "provenance": provenance(), "results": {}}
    t0 = time.perf_counter(); llm = stokastisk.LLM(model, config=config); rec["load_seconds"] = time.perf_counter() - t0
    rec["attention_backend"] = llm.selected_attention_backend()
    try:
        for w in wls:
            rec["results"][w.name] = measure(llm, w, sampler_interval)
        rec["status"] = "MEASURED"
    finally:
        llm.close()
    return rec
