"""OpenAI-compatible serving for the T4 build: launch the bundled ``vllm serve`` with the Stokastisk T4 defaults, wait for /health, read /metrics, stream tokens.

STATUS: implemented, NOT executed. The OpenAI server was never run on Kaggle in this project (see ``stokastisk.compat.FEATURES['openai_server']``), so this module
only builds the command, checks that the server's third-party requirements are importable, supervises the process and speaks plain HTTP with the standard library.
It does not install anything (the Kaggle baseline is protected) and it does not import vllm or torch in the calling process, so it is safe to use from a notebook that
has not initialised CUDA.

One server owns the engine it starts: TP=1, TP=2 or PP=2 on the visible GPUs. ``replica_parallel`` is not served by a single process: start one server per GPU with
``cuda_visible_devices`` (two ServerProcess objects, two ports) or use ``stokastisk.parallel.ReplicaPool`` for offline work.
"""
from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Iterator, Sequence

from .compat import check_dtype, check_feature, check_quantization
from .config import VLLM_ENV_NO_FLASHINFER, StokastiskConfig
from .errors import ConfigurationError, EngineError, StokastiskCompatibilityError

# import names of the third-party packages that vllm/entrypoints/openai/api_server.py needs (from the wheel's Requires-Dist); checked, never installed
SERVER_REQUIREMENTS = ("fastapi", "starlette", "uvicorn", "openai", "pydantic", "prometheus_client", "zmq", "msgspec", "cbor2", "aiohttp")


def check_server_requirements(names: Sequence[str] = SERVER_REQUIREMENTS) -> list[str]:
    """Return the names that cannot be imported. Empty list = nothing missing. Uses find_spec only (no import side effects)."""
    return [n for n in names if importlib.util.find_spec(n) is None]


def build_serve_command(model: str, config: StokastiskConfig | None = None, *, host: str = "127.0.0.1", port: int = 8000, served_model_name: str | None = None,
                        quantization: str | None = None, extra_args: Sequence[str] = (), python: str | None = None) -> list[str]:
    """The exact argv that ServerProcess runs. Pure function: no process, no import of vllm.

    T4 rules enforced here: the attention backend is always ``--attention-backend TRITON_ATTN``; the dtype must be fp16; replica_parallel is refused (one engine per
    process); a quantization method whose declared minimum capability exceeds SM7.5 is refused before launch."""
    cfg = config or StokastiskConfig.from_env()
    if cfg.strategy == "auto":
        raise ConfigurationError("execution_strategy='auto' is recommendation-only: call stokastisk.recommend_parallelism(...) first")
    if cfg.strategy == "replica_parallel":
        raise ConfigurationError("replica_parallel is not served by one process: start one server per GPU with cuda_visible_devices, or use stokastisk.parallel.ReplicaPool")
    check_dtype(cfg.dtype)
    if quantization:
        check_quantization(quantization)
    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise ConfigurationError(f"invalid port {port!r}")
    cmd = [python or sys.executable, "-c", "from vllm.entrypoints.cli.main import main; main()", "serve", model,
           "--host", host, "--port", str(port),
           "--dtype", cfg.dtype, "--kv-cache-dtype", cfg.kv_cache_dtype,
           "--tensor-parallel-size", str(cfg.tensor_parallel_size),
           "--attention-backend", cfg.attention_backend,
           "--gpu-memory-utilization", str(cfg.gpu_memory_utilization),
           "--enforce-eager" if cfg.enforce_eager else "--no-enforce-eager",
           "--disable-custom-all-reduce" if cfg.disable_custom_all_reduce else "--no-disable-custom-all-reduce"]
    if cfg.pipeline_parallel_size > 1:
        cmd += ["--pipeline-parallel-size", str(cfg.pipeline_parallel_size)]
    if cfg.max_model_len is not None:
        cmd += ["--max-model-len", str(cfg.max_model_len)]
    if served_model_name:
        cmd += ["--served-model-name", served_model_name]
    if quantization:
        cmd += ["--quantization", quantization]
    bad = [a for a in extra_args if a.lower().startswith("--attention-backend") or a.lower().startswith("--attention-config")]
    if bad:
        raise StokastiskCompatibilityError(f"the attention backend is fixed to TRITON_ATTN on this build; remove {bad[0]!r} from extra_args")
    return cmd + list(extra_args)


class ServerProcess:
    """Supervise one ``vllm serve`` child. Use as a context manager. Never started implicitly, never left running by ``with``."""

    def __init__(self, model: str, config: StokastiskConfig | None = None, *, host: str = "127.0.0.1", port: int = 8000, served_model_name: str | None = None,
                 quantization: str | None = None, extra_args: Sequence[str] = (), cuda_visible_devices: str | None = None, log_path: str | None = None) -> None:
        self.config = config or StokastiskConfig.from_env()
        self.model, self.host, self.port = model, host, port
        self.served_model_name = served_model_name or model
        self.command = build_serve_command(model, self.config, host=host, port=port, served_model_name=self.served_model_name, quantization=quantization, extra_args=extra_args)
        self.cuda_visible_devices, self.log_path = cuda_visible_devices, log_path
        self._proc: subprocess.Popen | None = None
        self._log = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> "ServerProcess":
        check_feature("openai_server")            # returns the UNKNOWN_REQUIRES_EVIDENCE record; raises only if the feature became blocking
        missing = check_server_requirements()
        if missing:
            raise StokastiskCompatibilityError("the OpenAI server needs packages that are not importable here: " + ", ".join(missing)
                                               + ". Stokastisk does not install packages on Kaggle (protected baseline); install them yourself if your notebook allows it.")
        if self._proc is not None:
            raise EngineError("server already started")
        env = dict(os.environ)                      # the child gets the same defaults StokastiskConfig.apply_environment() gives an in-process engine; user values win
        for k, v in VLLM_ENV_NO_FLASHINFER.items():
            env.setdefault(k, v)
        if self.config.apply_nccl_env:
            for k, v in self.config.nccl_env.items():
                env.setdefault(k, v)
        if self.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        out = open(self.log_path, "ab") if self.log_path else subprocess.DEVNULL
        self._log = out if self.log_path else None
        self._proc = subprocess.Popen(self.command, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        return self

    def wait_ready(self, timeout: float = 600.0, interval: float = 2.0) -> None:
        """Poll GET /health until HTTP 200. Raises EngineError if the child exits first or the timeout passes."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc is None or self._proc.poll() is not None:
                code = None if self._proc is None else self._proc.returncode
                raise EngineError(f"server process exited before it was ready (exit code {code}); see {self.log_path or 'the log (pass log_path=...)'}")
            try:
                with urllib.request.urlopen(self.base_url + "/health", timeout=5) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(interval)
        raise EngineError(f"server not healthy after {timeout:.0f}s")

    def stop(self, grace: float = 30.0) -> int | None:
        """SIGINT the process group (vLLM's own shutdown), then SIGTERM, then SIGKILL. Returns the exit code."""
        p, self._proc = self._proc, None
        if p is None:
            return None
        for sig, wait in ((signal.SIGINT, grace), (signal.SIGTERM, 10.0), (signal.SIGKILL, 5.0)):
            if p.poll() is not None:
                break
            try:
                os.killpg(os.getpgid(p.pid), sig)
            except ProcessLookupError:
                break
            try:
                p.wait(timeout=wait)
            except subprocess.TimeoutExpired:
                continue
        if self._log:
            self._log.close(); self._log = None
        return p.returncode

    def __enter__(self) -> "ServerProcess":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()


def _post_json(url: str, body: dict[str, Any], timeout: float):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


def chat(base_url: str, model: str, messages: list[dict[str, str]], *, max_tokens: int = 64, temperature: float = 0.0, timeout: float = 120.0) -> str:
    """Non-streaming /v1/chat/completions; returns the assistant text."""
    with _post_json(base_url + "/v1/chat/completions", {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}, timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def stream_chat(base_url: str, model: str, messages: list[dict[str, str]], *, max_tokens: int = 64, temperature: float = 0.0, timeout: float = 120.0) -> Iterator[str]:
    """Yield content deltas from /v1/chat/completions with stream=true (server-sent events: ``data: {...}`` lines, terminated by ``data: [DONE]``)."""
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": True}
    with _post_json(base_url + "/v1/chat/completions", body, timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                return
            delta = json.loads(payload)["choices"][0].get("delta", {}).get("content")
            if delta:
                yield delta


def list_models(base_url: str, timeout: float = 10.0) -> list[str]:
    with urllib.request.urlopen(base_url + "/v1/models", timeout=timeout) as r:
        return [m["id"] for m in json.loads(r.read())["data"]]


def fetch_metrics(base_url: str, timeout: float = 10.0) -> dict[str, float]:
    """Parse the Prometheus text at /metrics into {sample_name{labels}: value} (comment lines skipped; non-numeric samples ignored)."""
    out: dict[str, float] = {}
    with urllib.request.urlopen(base_url + "/metrics", timeout=timeout) as r:
        for line in r.read().decode("utf-8", "replace").splitlines():
            if not line or line.startswith("#"):
                continue
            name, _, val = line.rpartition(" ")
            try:
                out[name] = float(val)
            except ValueError:
                continue
    return out


__all__ = ["SERVER_REQUIREMENTS", "check_server_requirements", "build_serve_command", "ServerProcess", "chat", "stream_chat", "list_models", "fetch_metrics"]
