"""Local JSON evidence + optional OpenTelemetry spans. All metrics are available as JSON without any network telemetry."""
from __future__ import annotations

import logging

import json
import pathlib
import time
from contextlib import contextmanager
from typing import Any

log = logging.getLogger("stokastisk")


def write_json(path: str | pathlib.Path, obj: Any) -> pathlib.Path:
    p = pathlib.Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str)); return p


@contextmanager
def span(name: str, **attrs: Any):
    """OpenTelemetry span if ``opentelemetry`` is installed AND a provider is configured; otherwise a no-op. Never required."""
    tracer = None
    try:
        from opentelemetry import trace  # type: ignore
        tracer = trace.get_tracer("stokastisk")
    except Exception:  # noqa: BLE001
        log.debug("ignored Exception (best-effort probe)", exc_info=True)
    if tracer is None:
        yield None; return
    with tracer.start_as_current_span(name) as s:
        for k, v in attrs.items():
            try:
                s.set_attribute(k, v)
            except Exception:  # noqa: BLE001
                log.debug("ignored Exception (best-effort probe)", exc_info=True)
        yield s


def provenance() -> dict[str, Any]:
    """Versions + identity every benchmark records."""
    import importlib.metadata as md, subprocess, sys
    def ver(n):
        try:
            return md.version(n)
        except md.PackageNotFoundError:
            return None
    d: dict[str, Any] = {"time": time.strftime("%FT%T%z"), "python": sys.version.split()[0], "stokastisk": ver("stokastisk"), "vllm": ver("vllm"),
                         "torch": ver("torch"), "triton": ver("triton"), "transformers": ver("transformers")}
    try:
        import torch
        d["cuda"] = torch.version.cuda; d["nccl_compiled"] = ".".join(map(str, torch.cuda.nccl.version()))
    except Exception:  # noqa: BLE001
        log.debug("ignored Exception (best-effort probe)", exc_info=True)
    try:
        d["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        d["git_commit"] = None
    return d
