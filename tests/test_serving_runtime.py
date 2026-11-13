"""stokastisk.serving at runtime with FAKE servers (CPU): readiness, error reporting with the child's output, port collision, graceful stop, process-group cleanup, restart, HTTP client errors.
The fake 'vllm serve' is a python script behind a fake interpreter; it proves the supervision code, not vLLM."""
import json, os, pathlib, signal, socket, stat, subprocess, sys, tempfile, textwrap, threading, time, unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sdk"))
import stokastisk
from stokastisk import serving
from stokastisk.errors import MissingDependencyError, ServerError, ServerHTTPError, ServerShutdownError, ServerStartupError

FAKE = textwrap.dedent('''
    import json, os, signal, subprocess, sys, time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    argv = sys.argv; port = int(argv[argv.index("--port") + 1]); mode = os.environ.get("FAKE_MODE", "ok")
    print("fake server argv:", " ".join(argv), flush=True)
    if mode == "crash":
        sys.stderr.write("Traceback (most recent call last):\\nModuleNotFoundError: No module named 'model_hosting_container_standards'\\n"); sys.exit(1)
    if mode == "hang":
        time.sleep(600)
    if mode == "child":
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])           # an engine-core-like child
    if mode == "stubborn":
        signal.signal(signal.SIGINT, signal.SIG_IGN); signal.signal(signal.SIGTERM, signal.SIG_IGN)
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            if self.path == "/health": self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
            else: self.send_response(404); self.end_headers()
    s = ThreadingHTTPServer(("127.0.0.1", port), H)
    try:
        s.serve_forever()
    except KeyboardInterrupt:          # default SIGINT handling; vLLM also exits 0 on SIGINT
        sys.exit(0)
''')


class Fake(unittest.TestCase):
    def setUp(self):
        self.td = pathlib.Path(tempfile.mkdtemp()); (self.td / "fake_server.py").write_text(FAKE)
        py = self.td / "fakepython"; py.write_text(f'#!/bin/bash\nshift 2   # drop: -c "<code>"\nexec {sys.executable} {self.td}/fake_server.py "$@"\n'); py.chmod(py.stat().st_mode | stat.S_IEXEC)
        self.py = str(py); self.cfg = stokastisk.StokastiskConfig(max_model_len=2048)
        for p in (mock.patch.object(serving, "check_server_requirements", return_value=[]),): p.start(); self.addCleanup(p.stop)

    def srv(self, mode="ok", **kw):
        os.environ["FAKE_MODE"] = mode; self.addCleanup(lambda: os.environ.pop("FAKE_MODE", None))
        s = serving.ServerProcess("m", self.cfg, port=kw.pop("port", serving.find_free_port()), python=self.py, log_path=kw.pop("log_path", str(self.td / "s.log")), **kw); self.addCleanup(s.stop); return s


class Startup(Fake):
    def test_ready_then_graceful_stop(self):
        s = self.srv(); s.start(); s.wait_ready(20, 0.1); self.assertTrue(s.health()); self.assertTrue(s.running)
        rc = s.stop(grace=10); self.assertEqual(rc, 0); self.assertTrue(s.shutdown_report["graceful"]); self.assertTrue(s.shutdown_report["group_gone"]); self.assertFalse(s.health())

    def test_startup_failure_reports_exit_code_command_log_and_excerpt(self):
        s = self.srv("crash"); s.start()
        with self.assertRaises(ServerStartupError) as c: s.wait_ready(20, 0.1)
        e = c.exception
        self.assertEqual(e.exit_code, 1); self.assertEqual(e.command, s.command); self.assertEqual(e.log_path, str(self.td / "s.log")); self.assertTrue(os.path.isfile(e.log_path))
        self.assertIn("model_hosting_container_standards", e.log_excerpt)                                      # the child's REAL error, not a generic 'not ready'
        for needle in ("exit code : 1", "--attention-backend", e.log_path, "model_hosting_container_standards", "startup timeout: 20"): self.assertIn(needle, str(e))

    def test_timeout_while_the_process_is_still_running(self):
        s = self.srv("hang"); s.start()
        with self.assertRaises(ServerStartupError) as c: s.wait_ready(1.0, 0.1)
        self.assertIsNone(c.exception.exit_code); self.assertEqual(c.exception.timeout, 1.0); self.assertGreaterEqual(c.exception.waited, 1.0); self.assertIn("fake server argv", c.exception.log_excerpt)
        self.assertTrue(s.running); s.stop(grace=5)

    def test_log_is_persistent_and_defaults_to_a_temp_file(self):
        s = serving.ServerProcess("m", self.cfg, python=self.py, port=serving.find_free_port()); self.assertTrue(os.path.isabs(s.log_path) and "stokastisk-server-" in s.log_path)
        os.environ["FAKE_MODE"] = "ok"; self.addCleanup(lambda: os.environ.pop("FAKE_MODE", None)); self.addCleanup(s.stop)
        s.start(); s.wait_ready(20, 0.1); s.stop(); self.assertIn("fake server argv", open(s.log_path).read())                  # still there after stop

    def test_port_collision_is_refused_before_launch(self):
        sock = socket.socket(); sock.bind(("127.0.0.1", 0)); sock.listen(1); port = sock.getsockname()[1]; self.addCleanup(sock.close)
        s = self.srv(port=port)
        with mock.patch.object(serving.subprocess, "Popen") as popen, self.assertRaises(ServerError) as c: s.start()
        self.assertIn("already in use", str(c.exception)); popen.assert_not_called()

    def test_cannot_start_twice_and_missing_dependencies_block_launch(self):
        s = self.srv(); s.start()
        with self.assertRaises(stokastisk.EngineError): s.start()
        with mock.patch.object(serving, "check_server_requirements", return_value=["model_hosting_container_standards"]), mock.patch.object(serving.subprocess, "Popen") as popen:
            with self.assertRaises(MissingDependencyError) as c: serving.ServerProcess("m", self.cfg).start()
        self.assertEqual(c.exception.missing, ("model_hosting_container_standards",)); popen.assert_not_called()

    def test_unlaunchable_command(self):
        s = serving.ServerProcess("m", self.cfg, python="/nonexistent/python", log_path=str(self.td / "x.log"), port=serving.find_free_port())
        with self.assertRaises(ServerStartupError) as c: s.start()
        self.assertIsNone(c.exception.exit_code)


class Shutdown(Fake):
    def test_children_of_the_server_are_killed_with_it(self):
        s = self.srv("child"); s.start(); s.wait_ready(20, 0.1); pgid = s._pgid; time.sleep(0.5)
        self.assertTrue(s._group_alive()); s.stop(grace=10)
        self.assertFalse(s._group_alive()); self.assertTrue(s.shutdown_report["group_gone"])
        with self.assertRaises(ProcessLookupError): os.killpg(pgid, 0)

    def test_a_server_that_ignores_sigint_and_sigterm_is_killed_and_reported(self):
        s = self.srv("stubborn"); s.start(); s.wait_ready(20, 0.1)
        with mock.patch.object(serving.signal, "SIGINT", signal.SIGINT):
            t0 = time.time(); s.stop(grace=0.5)
        self.assertFalse(s.shutdown_report["graceful"]); self.assertIn("SIGKILL", s.shutdown_report["signals"]); self.assertTrue(s.shutdown_report["group_gone"]); self.assertLess(time.time() - t0, 30)

    def test_stop_is_idempotent_and_stop_without_start_is_a_noop(self):
        s = self.srv(); self.assertIsNone(s.stop()); s.start(); s.wait_ready(20, 0.1); s.stop(); self.assertIsNone(s.stop())

    def test_restart_gives_a_fresh_healthy_process_on_the_same_port(self):
        s = self.srv(); s.start_and_wait(20); first = s.pid; s.restart(20); self.assertNotEqual(first, s.pid); self.assertTrue(s.health()); s.stop()

    def test_context_manager_never_leaves_a_server_running(self):
        os.environ["FAKE_MODE"] = "ok"; self.addCleanup(lambda: os.environ.pop("FAKE_MODE", None))
        with serving.ServerProcess("m", self.cfg, python=self.py, log_path=str(self.td / "c.log"), port=serving.find_free_port()) as s:
            s.wait_ready(20, 0.1); pid = s.pid
        self.assertFalse(s.running)
        with self.assertRaises(ProcessLookupError): os.kill(pid, 0) if not os.path.exists(f"/proc/{pid}/stat") or open(f"/proc/{pid}/stat").read().split()[2] != "Z" else (_ for _ in ()).throw(ProcessLookupError())


class Validation(unittest.TestCase):
    cfg = stokastisk.StokastiskConfig(max_model_len=2048)

    def test_model_host_port(self):
        for bad in ("", " ", "a b", "x\ny", "--help", " m", None, 3):
            with self.subTest(model=bad), self.assertRaises(stokastisk.ConfigurationError): serving.build_serve_command(bad, self.cfg)
        for bad in ("bad host", "", "a..b", "-x.com", None):
            with self.subTest(host=bad), self.assertRaises(stokastisk.ConfigurationError): serving.build_serve_command("m", self.cfg, host=bad)
        for bad in (0, -1, 65536, True, "80", 8000.0):
            with self.subTest(port=bad), self.assertRaises(stokastisk.ConfigurationError): serving.build_serve_command("m", self.cfg, port=bad)
        for ok in ("127.0.0.1", "0.0.0.0", "::1", "localhost", "my-host.example.com"): serving.build_serve_command("org/model", self.cfg, host=ok)
        self.assertEqual(serving.ServerProcess("m", self.cfg, host="::1", port=9).base_url, "http://[::1]:9")

    def test_command_pins_everything(self):
        c = serving.build_serve_command("org/m", self.cfg, port=1234); g = lambda f: c[c.index(f) + 1]
        self.assertEqual((g("--attention-backend"), g("--dtype"), g("--tensor-parallel-size"), g("--port")), ("TRITON_ATTN", "half", "1", "1234")); self.assertEqual(json.loads(g("--structured-outputs-config")), {"backend": "xgrammar"})
        self.assertIn("--enforce-eager", c); self.assertIn("--disable-custom-all-reduce", c); self.assertEqual(c[4], "org/m")
        for ex in (["--attention-backend", "FLASH_ATTN"], ["--attention-config.backend=FLASHINFER"], ["--structured-outputs-config", '{"backend":"auto"}'], ["--ATTENTION-BACKEND=x"]):
            with self.subTest(ex=ex), self.assertRaises(stokastisk.StokastiskCompatibilityError): serving.build_serve_command("m", self.cfg, extra_args=ex)
        with self.assertRaises(stokastisk.ConfigurationError): serving.build_serve_command("m", stokastisk.StokastiskConfig(execution_strategy="replica_parallel"))
        with self.assertRaises(stokastisk.StokastiskCompatibilityError): serving.build_serve_command("m", stokastisk.StokastiskConfig(), quantization="mxfp4")


class Api(unittest.TestCase):
    """HTTP client against a local fake OpenAI server."""

    @classmethod
    def setUpClass(cls):
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def _send(self, code, body, ctype="application/json"): self.send_response(code); self.send_header("Content-Type", ctype); self.end_headers(); self.wfile.write(body if isinstance(body, bytes) else json.dumps(body).encode())
            def do_GET(self):
                if self.path == "/v1/models": self._send(200, {"data": [{"id": "m"}]})
                elif self.path == "/metrics": self._send(200, b"# HELP x\nvllm:num_requests_running 2\nbad line here\nprocess_cpu 1.5\n", "text/plain")
                elif self.path == "/v1/bad-json": self._send(200, b"<html>")
                else: self._send(404, {"error": "nope"})
            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0)); body = json.loads(self.rfile.read(n) or b"{}")
                if body.get("model") != "m": return self._send(404, {"error": {"message": "model not found"}})
                if body.get("max_tokens") == 999: return self._send(200, {"unexpected": True})
                if body.get("stream"):
                    self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
                    mode = body.get("messages")[0]["content"]
                    for t in ("Hel", "lo", " there"): self.wfile.write(f'data: {json.dumps({"choices": [{"delta": {"content": t}}]})}\n\n'.encode())
                    if mode == "error": self.wfile.write(b'data: {"error": {"message": "engine died"}}\n\n'); return
                    if mode != "truncate": self.wfile.write(b"data: [DONE]\n\n")
                    return
                self._send(200, {"choices": [{"message": {"content": "hello"}}]})
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H); cls.url = f"http://127.0.0.1:{cls.httpd.server_port}"; threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls): cls.httpd.shutdown()

    msgs = lambda self, c="hi": [{"role": "user", "content": c}]  # noqa: E731

    def test_chat_models_metrics_stream(self):
        self.assertEqual(serving.chat(self.url, "m", self.msgs()), "hello"); self.assertEqual(serving.list_models(self.url), ["m"])
        self.assertEqual("".join(serving.stream_chat(self.url, "m", self.msgs())), "Hello there")
        self.assertEqual(serving.fetch_metrics(self.url), {"vllm:num_requests_running": 2.0, "process_cpu": 1.5})

    def test_http_errors_are_typed(self):
        with self.assertRaises(ServerHTTPError) as c: serving.chat(self.url, "other", self.msgs())
        self.assertEqual(c.exception.status, 404); self.assertIn("model not found", c.exception.body)
        with self.assertRaises(ServerHTTPError): serving.chat(self.url, "m", self.msgs(), max_tokens=999)
        with self.assertRaises(ServerHTTPError) as c: serving._json(serving._get(self.url + "/v1/bad-json", 5), "x")
        self.assertIsNone(c.exception.status)
        with self.assertRaises(ServerHTTPError): serving.list_models("http://127.0.0.1:9")

    def test_a_truncated_or_failing_stream_is_an_error_not_a_success(self):
        with self.assertRaises(ServerHTTPError) as c: list(serving.stream_chat(self.url, "m", self.msgs("truncate")))
        self.assertIn("[DONE]", str(c.exception))
        with self.assertRaises(ServerHTTPError) as c: list(serving.stream_chat(self.url, "m", self.msgs("error")))
        self.assertIn("engine died", c.exception.body)


if __name__ == "__main__":
    unittest.main()
