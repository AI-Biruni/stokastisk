"""Stage `server`: the real `vllm serve` started by stokastisk.serving on a Tesla T4: readiness, OpenAI-compatible API, streaming, structured output over HTTP, error handling,
metrics, backend selection, startup-failure reporting, port collision, restart, graceful shutdown, process-group cleanup and GPU memory release."""
import json, os, shutil, socket, sys, time, traceback, urllib.request
from _common import OUT, check, error, finish, gpu_used_mib, process_tree, wait_children_gone
from candidate_15_common import *   # noqa: F401,F403


def server() -> int:
    cache = OUT / "triton-cache-15c-server"; shutil.rmtree(cache, ignore_errors=True); cache.mkdir(parents=True); os.environ["TRITON_CACHE_DIR"] = str(cache)
    import stokastisk
    from stokastisk import serving
    R = []; stage = "15c_server"; extra = {"model": MODEL}
    before = gpu_used_mib(); check(R, "both GPUs idle before the server starts", before is not None and all(x < IDLE_MIB for x in before), before)
    miss = serving.check_server_requirements(); extra["server_requirements_missing"] = miss
    check(R, "server requirements are importable (installed by the bundle's locks)", not miss, miss or "complete")
    if miss:
        finish(stage, R, extra, script_file=__file__); return 1
    cfg = stokastisk.StokastiskConfig(max_model_len=2048)
    # ---- startup failure must be diagnosable from the exception alone (a model that does not exist -> the child exits with an error)
    bad_log = str(OUT / "15c_server_startfail.log"); port_bad = serving.find_free_port()
    bad = serving.ServerProcess("stokastisk-test/this-model-does-not-exist", cfg, port=port_bad, log_path=bad_log)
    try:
        bad.start(); bad.wait_ready(timeout=300); check(R, "startup failure is reported (a non-existent model must not become ready)", False, "became ready")
    except stokastisk.ServerStartupError as e:
        ok = e.exit_code not in (None, 0) and e.command and os.path.isfile(e.log_path) and bool(e.log_excerpt.strip()) and e.log_path in str(e) and "exit code" in str(e)
        check(R, "ServerStartupError carries exit code, command, persistent log path and a log excerpt", ok, {"exit_code": e.exit_code, "log": e.log_path, "excerpt_chars": len(e.log_excerpt)})
    finally:
        bad.stop()
    time.sleep(2)
    # ---- port collision
    s = socket.socket(); s.bind(("127.0.0.1", 0)); busy = s.getsockname()[1]; s.listen(1)
    check(R, "a port that is already in use is refused before launching", raises(lambda: serving.ServerProcess(MODEL, cfg, port=busy).start(), stokastisk.ServerError)); s.close()
    # ---- the real server
    log = str(OUT / "15c_server.log"); port = serving.find_free_port()
    srv = serving.ServerProcess(MODEL, cfg, port=port, log_path=log); extra["command"] = srv.command
    rc = None
    try:
        srv.start(); srv.wait_ready(timeout=900)
        check(R, "server became healthy (GET /health 200)", srv.health())
        base = srv.base_url
        check(R, "GET /v1/models lists the served model", MODEL in serving.list_models(base), serving.list_models(base))
        txt = serving.chat(base, MODEL, [{"role": "user", "content": "Say hello in five words."}], max_tokens=24); check(R, "POST /v1/chat/completions returns non-empty text", bool(txt.strip()), txt[:80]); extra["chat_text"] = txt
        chunks = list(serving.stream_chat(base, MODEL, [{"role": "user", "content": "Count from one to five."}], max_tokens=32))
        check(R, "streaming (SSE) yields several non-empty deltas and ends with [DONE]", len(chunks) >= 2 and bool("".join(chunks).strip()), {"n": len(chunks), "text": "".join(chunks)[:80]})
        # error handling
        check(R, "unknown model -> ServerHTTPError with status 404", raises_status(lambda: serving.chat(base, "no/such-model", [{"role": "user", "content": "hi"}]), 404))
        check(R, "malformed request body -> a 4xx ServerHTTPError", raises_status(lambda: post_raw(base + "/v1/chat/completions", b"{not json"), None, lo=400, hi=499))
        check(R, "missing endpoint -> 404 ServerHTTPError", raises_status(lambda: serving._get(base + "/v1/nope", 5), 404))
        # structured output over HTTP (response_format -> the pinned xgrammar backend)
        body = {"model": MODEL, "messages": [{"role": "user", "content": "Give a JSON object describing Paris."}], "max_tokens": 96, "temperature": 0.0,
                "response_format": {"type": "json_schema", "json_schema": {"name": "city", "schema": ENGINE_SCHEMA, "strict": True}}}
        with serving._post_json(base + "/v1/chat/completions", body, 120) as r:
            content = json.loads(r.read())["choices"][0]["message"]["content"]
        try:
            stokastisk.validate_json_output(content, ENGINE_SCHEMA); check(R, "structured output over HTTP is schema-valid JSON", True, content[:100])
        except stokastisk.StructuredOutputError as e:
            check(R, "structured output over HTTP is schema-valid JSON", False, e)
        met = serving.fetch_metrics(base); check(R, "/metrics exposes vllm: samples", any(k.startswith("vllm:") for k in met), len(met))
        peak = gpu_used_mib() or []; check(R, "exactly one GPU holds the model while serving", len(peak) == 2 and sum(x > IDLE_MIB for x in peak) == 1, peak)
        pid = srv.pid; libs = flash_libs_tree([pid, *process_tree(pid)]); badl = flat(libs)
        check(R, "no Flash library mapped in the server process tree", not badl, badl or f"{len(libs)} processes inspected"); extra.update(gpu_used_MiB={"before": before, "peak": peak})
        arts = triton_artifacts(cache); check(R, "Triton cache holds compiled kernel_unified_attention (server)", bool(arts), arts[:3] or "none")
        text = open(log, errors="replace").read().upper()
        check(R, "server log shows TRITON_ATTN and no Flash backend selection", "TRITON_ATTN" in text and "USING ATTENTIONBACKENDENUM.FLASH" not in text and "BACKEND FLASH" not in text, "TRITON_ATTN" in text)
        # restart
        srv.restart(timeout=900); check(R, "restart: a fresh server becomes healthy on the same port and answers", bool(serving.chat(srv.base_url, MODEL, [{"role": "user", "content": "Hi"}], max_tokens=8).strip()))
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc(); print(tb, file=sys.stderr)
        error(R, "server start / requests", f"{type(e).__name__}: {str(e)[:1500]}"); extra["traceback"] = tb; extra["log_tail"] = tail(log, 60)
    finally:
        try: rc = srv.stop()
        except stokastisk.ServerShutdownError as e: error(R, "server shutdown", e)
        extra["server_exit_code"] = rc; extra["shutdown_report"] = srv.shutdown_report
    check(R, "server stopped gracefully (SIGINT only) and no process of its group survived", bool(srv.shutdown_report.get("graceful")) and srv.shutdown_report.get("group_gone") is True, srv.shutdown_report)
    time.sleep(3); live, ignored, waited = wait_children_gone(timeout=30); after = gpu_used_mib() or []
    check(R, "no live engine process after stop", not live, live); check(R, "GPU memory released after stop (<500 MiB each)", all(x < IDLE_MIB for x in after), after)
    extra["gpu_after_close"] = after
    finish(stage, R, extra, script_file=__file__); return 0


def post_raw(url, data):
    from stokastisk import serving
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        return urllib.request.urlopen(req, timeout=10)
    except urllib.error.HTTPError as e:
        from stokastisk.errors import ServerHTTPError
        raise ServerHTTPError("POST failed", status=e.code, body="")


def raises_status(fn, status, lo=None, hi=None):
    from stokastisk.errors import ServerHTTPError
    try:
        fn()
    except ServerHTTPError as e:
        return (e.status == status) if status is not None else (e.status is not None and lo <= e.status <= hi)
    except BaseException:  # noqa: BLE001
        return False
    return False
