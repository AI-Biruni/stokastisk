"""OpenAI-compatible serving for the T4 build: launch the bundled ``vllm serve`` with the Stokastisk T4 defaults, wait for /health, read /metrics, stream tokens.

STATUS: implemented, CPU-tested (command construction, validation, supervision, error reporting, HTTP client against a local fake server) and **verified on Kaggle 2 x Tesla T4**:
the notebook 15 ``server`` stage on integrated build 3 passed 21/21 checks with Qwen/Qwen2.5-0.5B-Instruct at TP=1 (health, model list, chat, SSE streaming, error mapping, structured
output over HTTP, /metrics, restart on the same port, graceful SIGINT shutdown, GPU memory released). Not covered by that run: other models, TP=2/PP=2 serving, and the SageMaker
``/ping`` and ``/invocations`` routes of the container-standards package (Kaggle's fastapi 0.141.1 is newer than upstream vLLM's ``<0.137`` cap).

The supported public API is :class:`ServerProcess` (and the client helpers below). The process it launches is the bundled ``vllm serve``, started with the Stokastisk T4 defaults;
running upstream ``vllm serve`` yourself is not a supported entry point because it bypasses these checks. This module only builds the command, supervises the child and speaks
plain HTTP with the standard library. It does not import vllm or torch in the calling process, so it is safe to use from a notebook that has not initialised CUDA.

Failure reporting (Part 6 of the build-3 requirements): the child's stdout+stderr always go to a PERSISTENT log file (a temporary directory is created when none is
given and its path is reported). If the server exits, or does not become ready, :class:`~stokastisk.errors.ServerStartupError` carries the exit code, the exact command, the log
path, the last lines of the child's output and the timeout - never a bare "not ready".

One server owns the engine it starts: TP=1, TP=2 or PP=2 on the visible GPUs. ``replica_parallel`` is not served by a single process: start one server per GPU with
``cuda_visible_devices`` (two ServerProcess objects, two ports) or use ``stokastisk.parallel.ReplicaPool`` for offline work.
"""
from __future__ import annotations

import logging

import ipaddress
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from typing import Any, Iterator, Sequence

from . import dependencies
from .compat import check_dtype, check_feature, check_quantization
from .config import VLLM_ENV_NO_FLASHINFER, StokastiskConfig
from .errors import (ConfigurationError, EngineError, MissingDependencyError, ServerError, ServerHTTPError, ServerShutdownError, ServerStartupError,
                     StokastiskCompatibilityError)

log = logging.getLogger("stokastisk")

# Import names of every third-party package `vllm serve` and the pinned structured-output path need: baseline + lock (see stokastisk.dependencies / requirements/*.lock).
SERVER_REQUIREMENTS = dependencies.required_imports("core", "structured-output", "serving")
_HOSTNAME = re.compile(r"^(?=.{1,253}$)([A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?)(\.[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*$")


def check_server_requirements(names: Sequence[str] | None = None) -> list[str]:
    """Return the import names that cannot be found (empty list = nothing missing). Uses find_spec only (no import side effects)."""
    return dependencies.missing(SERVER_REQUIREMENTS if names is None else names)


def validate_model_id(model: Any) -> str:
    """A Hugging Face id (``org/name``) or a path. Rejects empty strings, whitespace/control characters and anything that vLLM's argument parser would read as an option."""
    if not isinstance(model, str) or not model.strip():
        raise ConfigurationError("model must be a non-empty str (a Hugging Face id or a local path)")
    if model != model.strip() or re.search(r"[\s\x00-\x1f\x7f]", model):
        raise ConfigurationError(f"model identifier contains whitespace or control characters: {model!r}")
    if model.startswith("-"):
        raise ConfigurationError(f"model identifier {model!r} starts with '-' and would be parsed as a command-line option")
    return model


def validate_host(host: Any) -> str:
    if not isinstance(host, str) or not host:
        raise ConfigurationError(f"invalid host {host!r}")
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        if _HOSTNAME.match(host):
            return host
    raise ConfigurationError(f"invalid host {host!r}: expected an IPv4/IPv6 address or a hostname")


def validate_port(port: Any) -> int:
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ConfigurationError(f"invalid port {port!r}: expected an int in [1, 65535]")
    return port


def port_is_free(host: str, port: int) -> bool:
    """True if nothing is listening on (host, port) and the address can be bound."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)      # connections in TIME_WAIT from a stopped server must not make a restart on the same port fail
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def find_free_port(host: str = "127.0.0.1") -> int:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def build_serve_command(model: str, config: StokastiskConfig | None = None, *, host: str = "127.0.0.1", port: int = 8000, served_model_name: str | None = None,
                        quantization: str | None = None, extra_args: Sequence[str] = (), python: str | None = None) -> list[str]:
    """The exact argv that ServerProcess runs. Pure function: no process, no import of vllm.

    T4 rules enforced here: the attention backend is always ``--attention-backend TRITON_ATTN``; the dtype must be fp16; the structured-output backend is pinned
    (``--structured-outputs-config {"backend": "xgrammar"}``, never 'auto'); replica_parallel is refused (one engine per process); a quantization method whose declared
    minimum capability exceeds SM7.5 is refused before launch."""
    cfg = config or StokastiskConfig.from_env()
    validate_model_id(model)
    validate_host(host)
    validate_port(port)
    if cfg.strategy == "auto":
        raise ConfigurationError("execution_strategy='auto' is recommendation-only: call stokastisk.recommend_parallelism(...) first")
    if cfg.strategy == "replica_parallel":
        raise ConfigurationError("replica_parallel is not served by one process: start one server per GPU with cuda_visible_devices, or use stokastisk.parallel.ReplicaPool")
    check_dtype(cfg.dtype)
    if quantization:
        check_quantization(quantization)
    bad = [a for a in extra_args if a.lower().startswith(("--attention-backend", "--attention-config", "--structured-outputs-config", "--structured_outputs_config"))]
    if bad:
        raise StokastiskCompatibilityError(f"the attention backend is fixed to TRITON_ATTN and the structured-output backend to {cfg.structured_output_backend} on this build; "
                                           f"remove {bad[0]!r} from extra_args")
    cmd = [python or sys.executable, "-c", "from vllm.entrypoints.cli.main import main; main()", "serve", model,
           "--host", host, "--port", str(port),
           "--dtype", cfg.dtype, "--kv-cache-dtype", cfg.kv_cache_dtype,
           "--tensor-parallel-size", str(cfg.tensor_parallel_size),
           "--attention-backend", cfg.attention_backend,
           "--structured-outputs-config", json.dumps({"backend": cfg.structured_output_backend}),
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
    return cmd + list(extra_args)


def _default_sigint() -> None:
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def _tail(path: str | None, max_lines: int = 40, max_chars: int = 6000) -> str:
    if not path:
        return "(no log file)"
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 64 * 1024))
            text = f.read().decode("utf-8", "replace")
    except OSError as e:
        return f"(cannot read {path}: {e})"
    lines = [l for l in text.splitlines() if l.strip()]
    return "\n".join(lines[-max_lines:])[-max_chars:] or "(the server wrote nothing)"


class ServerProcess:
    """Supervise one ``vllm serve`` child. Use as a context manager. Never started implicitly, never left running by ``with``."""

    def __init__(self, model: str, config: StokastiskConfig | None = None, *, host: str = "127.0.0.1", port: int = 8000, served_model_name: str | None = None,
                 quantization: str | None = None, extra_args: Sequence[str] = (), cuda_visible_devices: str | None = None, log_path: str | None = None,
                 python: str | None = None) -> None:
        self.config = config or StokastiskConfig.from_env()
        self.model, self.host, self.port = validate_model_id(model), validate_host(host), validate_port(port)
        self.served_model_name = served_model_name or model
        self.command = build_serve_command(model, self.config, host=host, port=port, served_model_name=self.served_model_name, quantization=quantization,
                                           extra_args=extra_args, python=python)
        self.cuda_visible_devices = cuda_visible_devices
        self.log_path = log_path or os.path.join(tempfile.mkdtemp(prefix="stokastisk-server-"), "server.log")      # always persistent: kept after stop()
        self._proc: subprocess.Popen | None = None
        self._pgid: int | None = None
        self._log = None
        self.shutdown_report: dict[str, Any] = {}

    @property
    def base_url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}"

    @property
    def pid(self) -> int | None:
        return None if self._proc is None else self._proc.pid

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def log_tail(self, max_lines: int = 40) -> str:
        return _tail(self.log_path, max_lines)

    def _startup_error(self, reason: str, *, exit_code, timeout=None, waited=None) -> ServerStartupError:
        return ServerStartupError(reason, exit_code=exit_code, command=self.command, log_path=self.log_path, log_excerpt=_tail(self.log_path), timeout=timeout, waited=waited)

    def start(self) -> "ServerProcess":
        check_feature("openai_server")            # returns the feature record; raises only if the feature became blocking
        missing = check_server_requirements()
        if missing:
            raise MissingDependencyError("the OpenAI server", missing, "Re-run the bundle installer (scripts/kaggle/02_install_integrated.py), which installs the serving and "
                                         "structured-output locks; Stokastisk never installs packages on its own (protected Kaggle baseline).")
        if self._proc is not None:
            raise EngineError("server already started")
        if not port_is_free(self.host, self.port):
            raise ServerError(f"port {self.port} on {self.host} is already in use; choose another port (stokastisk.serving.find_free_port())")
        env = dict(os.environ)                      # the child gets the same defaults StokastiskConfig.apply_environment() gives an in-process engine; user values win
        for k, v in VLLM_ENV_NO_FLASHINFER.items():
            env.setdefault(k, v)
        if self.config.apply_nccl_env:
            for k, v in self.config.nccl_env.items():
                env.setdefault(k, v)
        if self.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
        self._log = open(self.log_path, "ab")
        self._log.write(f"# stokastisk server start {time.strftime('%FT%T%z')}\n# command: {' '.join(self.command)}\n".encode())
        self._log.flush()
        try:
            self._proc = subprocess.Popen(self.command, env=env, stdout=self._log, stderr=subprocess.STDOUT, start_new_session=True,
                                         preexec_fn=_default_sigint)      # a parent started as a background job ignores SIGINT, and that is inherited: restore it so shutdown stays graceful
        except OSError as e:
            self._log.close(); self._log = None
            raise self._startup_error(f"could not launch the server process: {e}", exit_code=None) from e
        self._pgid = self._proc.pid                 # start_new_session=True: the child leads its own process group
        return self

    def health(self, timeout: float = 5.0) -> bool:
        """True if ``GET /health`` answers 200 right now."""
        try:
            with urllib.request.urlopen(self.base_url + "/health", timeout=timeout) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def wait_ready(self, timeout: float = 600.0, interval: float = 2.0) -> None:
        """Poll ``GET /health`` until HTTP 200. Raises :class:`ServerStartupError` if the child exits first or the timeout passes (with the exit code, the command,
        the log path and the tail of the server output)."""
        if self._proc is None:
            raise EngineError("server was not started")
        t0 = time.monotonic()
        deadline = t0 + timeout
        while time.monotonic() < deadline:
            code = self._proc.poll()
            if code is not None:
                raise self._startup_error("server process exited before it was ready", exit_code=code, timeout=timeout, waited=time.monotonic() - t0)
            if self.health():
                return
            time.sleep(interval)
        raise self._startup_error("server did not become healthy in time (the process is still running)", exit_code=None, timeout=timeout, waited=time.monotonic() - t0)

    def start_and_wait(self, timeout: float = 600.0) -> "ServerProcess":
        self.start()
        try:
            self.wait_ready(timeout)
        except BaseException:
            self.stop()
            raise
        return self

    def restart(self, timeout: float = 600.0) -> "ServerProcess":
        """Stop the server (graceful first) and start a fresh one on the same port; returns once it is healthy."""
        self.stop()
        return self.start_and_wait(timeout)

    def _group_alive(self) -> bool:
        if self._pgid is None:
            return False
        try:
            os.killpg(self._pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:           # pragma: no cover - the group exists but is not ours
            return True

    def stop(self, grace: float = 30.0) -> int | None:
        """SIGINT the process group (vLLM's own graceful shutdown), then SIGTERM, then SIGKILL; then make sure no process of the group survives.

        Returns the exit code of the server process. ``shutdown_report`` records which signals were needed and whether any child survived the leader (those are SIGKILLed).
        Raises :class:`ServerShutdownError` only if a process of the group could not be killed."""
        p, self._proc = self._proc, None
        pgid, self._pgid = self._pgid, None
        if p is None:
            return None
        sent: list[str] = []
        for sig, wait in ((signal.SIGINT, grace), (signal.SIGTERM, 10.0), (signal.SIGKILL, 5.0)):
            if p.poll() is not None:
                break
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                break
            sent.append(sig.name)
            try:
                p.wait(timeout=wait)
            except subprocess.TimeoutExpired:
                continue
        survivors = False
        end = time.monotonic() + 5.0
        self._pgid = pgid
        while self._group_alive() and time.monotonic() < end:         # children of the leader (engine core, TP workers) may outlive it for a moment
            time.sleep(0.2)
        if self._group_alive():
            survivors = True
            try:
                os.killpg(pgid, signal.SIGKILL); sent.append("SIGKILL(group)")
            except ProcessLookupError:
                log.debug("ignored ProcessLookupError (best-effort probe)", exc_info=True)
            end = time.monotonic() + 5.0
            while self._group_alive() and time.monotonic() < end:
                time.sleep(0.2)
        still = self._group_alive()
        self._pgid = None
        if self._log:
            self._log.close(); self._log = None
        self.shutdown_report = {"signals": sent, "graceful": sent in ([], ["SIGINT"]), "exit_code": p.returncode, "children_survived_leader": survivors, "group_gone": not still}
        if still:
            raise ServerShutdownError(f"processes of the server's process group {pgid} survived SIGKILL; log: {self.log_path}")
        return p.returncode

    def __enter__(self) -> "ServerProcess":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()


# ---------------------------------------------------------------------------------------------------------------------------------- HTTP client (standard library)
def _post_json(url: str, body: dict[str, Any], timeout: float):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise ServerHTTPError(f"POST {url} failed", status=e.code, body=_read_error(e)) from e
    except (urllib.error.URLError, OSError) as e:
        raise ServerHTTPError(f"POST {url} could not reach the server: {e}") from e


def _read_error(e: urllib.error.HTTPError) -> str:
    try:
        return e.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return ""


def _get(url: str, timeout: float) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise ServerHTTPError(f"GET {url} failed", status=e.code, body=_read_error(e)) from e
    except (urllib.error.URLError, OSError) as e:
        raise ServerHTTPError(f"GET {url} could not reach the server: {e}") from e


def _json(raw: bytes, what: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ServerHTTPError(f"{what}: the response is not valid JSON ({e})", body=raw[:300].decode("utf-8", "replace")) from e


def chat(base_url: str, model: str, messages: list[dict[str, str]], *, max_tokens: int = 64, temperature: float = 0.0, timeout: float = 120.0) -> str:
    """Non-streaming /v1/chat/completions; returns the assistant text."""
    with _post_json(base_url + "/v1/chat/completions", {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}, timeout) as r:
        data = _json(r.read(), "chat completion")
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise ServerHTTPError("chat completion: unexpected response shape", body=json.dumps(data)[:300]) from e


def stream_chat(base_url: str, model: str, messages: list[dict[str, str]], *, max_tokens: int = 64, temperature: float = 0.0, timeout: float = 120.0) -> Iterator[str]:
    """Yield content deltas from /v1/chat/completions with stream=true (server-sent events: ``data: {...}`` lines, terminated by ``data: [DONE]``).

    Raises :class:`ServerHTTPError` if the server sends an error event or the stream ends without ``[DONE]`` (a truncated stream is not a success)."""
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": True}
    with _post_json(base_url + "/v1/chat/completions", body, timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                return
            event = _json(payload.encode(), "stream event")
            if isinstance(event, dict) and "error" in event:
                raise ServerHTTPError("the server sent an error event in the stream", body=json.dumps(event["error"])[:300])
            delta = event["choices"][0].get("delta", {}).get("content")
            if delta:
                yield delta
    raise ServerHTTPError("the stream ended without a [DONE] marker (truncated)")


def list_models(base_url: str, timeout: float = 10.0) -> list[str]:
    return [m["id"] for m in _json(_get(base_url + "/v1/models", timeout), "models")["data"]]


def fetch_metrics(base_url: str, timeout: float = 10.0) -> dict[str, float]:
    """Parse the Prometheus text at /metrics into {sample_name{labels}: value} (comment lines skipped; non-numeric samples ignored)."""
    out: dict[str, float] = {}
    for line in _get(base_url + "/metrics", timeout).decode("utf-8", "replace").splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, val = line.rpartition(" ")
        try:
            out[name] = float(val)
        except ValueError:
            continue
    return out


__all__ = ["SERVER_REQUIREMENTS", "check_server_requirements", "validate_model_id", "validate_host", "validate_port", "port_is_free", "find_free_port", "build_serve_command",
           "ServerProcess", "chat", "stream_chat", "list_models", "fetch_metrics"]
