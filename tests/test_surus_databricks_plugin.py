"""The Surus Databricks plugin: cluster execution per chat, usage logging rows,
and registration into the staff/optimizer profiles. Runs against a fake
Databricks workspace (tests/fake_databricks.py); no Hermes needed."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(Path(__file__).parent)):
    if path not in sys.path:
        sys.path.insert(0, path)

from fake_databricks import FakeDatabricks  # noqa: E402
from hermes_plugins.surus_databricks import chatlog, executor, usage_log  # noqa: E402
import hermes_plugins.surus_databricks as plugin  # noqa: E402

VOLUME = "/Volumes/universal/chatbot/files"


class TempDirTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="surus-plugin-")
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)


class ExecutorTests(TempDirTest):
    def make(self, fake: FakeDatabricks, **kwargs):
        return executor.ClusterExecutor(fake.client(), "cluster-1", VOLUME, self.dir / "contexts.json",
                                        poll_s=0, sleep=lambda s: None, **kwargs)

    def test_first_call_starts_cluster_and_bootstraps_then_reuses_the_session(self):
        fake = FakeDatabricks(cluster_state="TERMINATED",
                              respond=lambda cmd: {"resultType": "text", "data": "rows: 42"})
        ex = self.make(fake)
        out = ex.run("chat-1", "print('rows:', 42)", timeout_s=60)
        self.assertEqual((out.status, out.text), ("ok", "rows: 42"))
        self.assertEqual(fake.started_cluster, 1)
        self.assertIn("def _surus_before", fake.executed[0])                   # the prelude
        self.assertIn(f"_surus_init('{VOLUME}/chats/chat-1'", fake.executed[0])
        self.assertIn("exec(compile(", fake.executed[1])                        # the wrapped code
        ex.run("chat-1", "x = 1", timeout_s=60)
        self.assertEqual(len(fake.contexts), 1, "a second call reuses the chat's context")
        ex2 = self.make(fake)                                                   # after a restart
        ex2.run("chat-1", "x", timeout_s=60)
        self.assertEqual(len(fake.contexts), 1, "the context id survives a restart")

    def test_each_chat_gets_its_own_context(self):
        fake = FakeDatabricks()
        ex = self.make(fake)
        ex.run("chat-a", "1", timeout_s=60)
        ex.run("chat-b", "1", timeout_s=60)
        self.assertEqual(len(fake.contexts), 2)

    def test_lost_context_is_rebuilt(self):
        fake = FakeDatabricks()
        ex = self.make(fake)
        ex.run("chat-1", "1", timeout_s=60)
        fake.contexts.clear()                                                   # cluster restarted
        out = ex.run("chat-1", "print('again')", timeout_s=60)
        self.assertEqual(out.status, "ok")
        self.assertEqual(len(fake.contexts), 1)

    def test_stop_cancels_the_command_on_the_cluster(self):
        fake = FakeDatabricks(polls_to_finish=50)
        ex = self.make(fake)
        ex.run("chat-1", "1", timeout_s=60)       # create the context first
        calls = {"n": 0}

        def interrupted():
            calls["n"] += 1
            return calls["n"] > 2

        out = ex.run("chat-1", "import time; time.sleep(999)", timeout_s=60, interrupted=interrupted)
        self.assertEqual(out.status, "cancelled")
        self.assertEqual(len(fake.cancelled), 1)
        self.assertEqual(executor.format_result(out, 8000), "Execution stopped by the user.")

    def test_time_limit_cancels_and_tells_the_model(self):
        fake = FakeDatabricks(polls_to_finish=1000)
        clock = {"t": 0.0}

        def tick():
            clock["t"] += 10
            return clock["t"]

        ex = executor.ClusterExecutor(fake.client(), "cluster-1", VOLUME, self.dir / "c.json", poll_s=0,
                                      sleep=lambda s: None, clock=tick)
        fake.polls_to_finish = 0
        ex.run("chat-1", "1", timeout_s=60)
        fake.polls_to_finish = 1000
        out = ex.run("chat-1", "while True: pass", timeout_s=60)
        self.assertEqual(out.status, "timeout")
        self.assertTrue(fake.cancelled)
        self.assertIn("time limit", executor.format_result(out, 8000))

    def test_cluster_error_is_reported(self):
        fake = FakeDatabricks()
        ex = self.make(fake)
        ex.run("chat-1", "1", timeout_s=60)
        fake.respond = lambda cmd: {"resultType": "error", "cause": "AnalysisException: no such table"}
        out = ex.run("chat-1", "spark.table('nope')", timeout_s=60)
        self.assertEqual(out.status, "error")
        self.assertEqual(executor.format_result(out, 8000), "ERROR:\nAnalysisException: no such table")

    def test_exports_are_announced_and_long_output_is_shortened(self):
        export = f"{VOLUME}/chats/chat-1/exports/precincts.csv"
        fake = FakeDatabricks(respond=lambda cmd: {
            "resultType": "text", "data": "x" * 20000 + f"\n{executor.EXPORT_SENTINEL}{export}"})
        out = self.make(fake).run("chat-1", "export_csv(df, 'precincts.csv')", timeout_s=60)
        self.assertEqual(out.exports, [export])
        shown = executor.format_result(out, 8000)
        self.assertLessEqual(len(shown), 8100)
        self.assertIn("characters omitted", shown)
        self.assertIn("precincts.csv to this chat's files", shown)

    def test_trailing_expression_is_displayed(self):
        self.assertIn("__surus_last_expr = (df.head())", executor.prepare_code("df = 1\ndf.head()"))
        self.assertEqual(executor.prepare_code("print(1)"), "print(1)")

    def test_chat_ids_are_made_path_safe(self):
        self.assertEqual(executor.safe_chat_id("a/b c"), "a_b_c")


class ClientAuthTests(unittest.TestCase):
    def test_service_principal_gets_and_caches_an_oauth_token(self):
        fake = FakeDatabricks()
        client = fake.client(token="", client_id="sp-id", client_secret="sp-secret")
        client.cluster_state("c")
        client.cluster_state("c")
        self.assertEqual(fake.token_requests, 1)
        self.assertEqual(fake.auth_headers[-1], "Bearer oauth-1")

    def test_from_env_prefers_the_service_principal(self):
        from hermes_plugins.surus_databricks.dbx import DatabricksClient
        self.assertIsNone(DatabricksClient.from_env({"DATABRICKS_HOST": "x"}))
        pat = DatabricksClient.from_env({"DATABRICKS_HOST": "adb.example", "DATABRICKS_TOKEN": "t"})
        self.assertEqual((pat.host, pat._token), ("https://adb.example", "t"))
        sp = DatabricksClient.from_env({"DATABRICKS_HOST": "h", "DATABRICKS_TOKEN": "t",
                                        "DATABRICKS_CLIENT_ID": "i", "DATABRICKS_CLIENT_SECRET": "s"})
        self.assertEqual(sp._client_id, "i")

    def test_volume_listing_and_downloads(self):
        fake = FakeDatabricks()
        client = fake.client()
        self.assertEqual(client.list_directory(f"{VOLUME}/chats/none/exports"), [])
        client.upload(f"{VOLUME}/chats/c/exports/a.csv", b"x,y\n1,2\n")
        entries = client.list_directory(f"{VOLUME}/chats/c/exports")
        self.assertEqual([e["name"] for e in entries], ["a.csv"])
        self.assertEqual(client.download(f"{VOLUME}/chats/c/exports/a.csv"), b"x,y\n1,2\n")
        with self.assertRaises(ValueError):
            client.download("/etc/passwd")


class ChatLogTests(TempDirTest):
    def make(self, *, running=True, ok=True):
        self.writes = []

        def writer(kind, command):
            self.writes.append((kind, command))
            return ok

        return chatlog.ChatLog(self.dir / "inbox", "universal.logging.chatbot", writer=writer,
                               cluster_running=lambda: running)

    def test_rows_wait_on_disk_until_the_cluster_is_running(self):
        log = self.make(running=False)
        log.enqueue("llm", {"email": "a@surusenterprises.com", "input_token_count": "12"})
        self.assertEqual(log.flush(), 0)
        self.assertEqual(self.writes, [])
        log._cluster_running = lambda: True
        self.assertEqual(log.flush(), 1)
        kind, command = self.writes[0]
        self.assertEqual(kind, "llm")
        self.assertIn("saveAsTable('universal.logging.chatbot_logs')", command)
        self.assertEqual(list((self.dir / "inbox").glob("batch-*")), [])

    def test_old_rows_are_written_even_if_the_cluster_is_off(self):
        log = self.make(running=False)
        log.max_wait_s = 0
        log.enqueue("turn", {"chat_id": "c"})
        self.assertEqual(log.flush(), 1)

    def test_failed_writes_are_kept_then_set_aside(self):
        log = self.make(ok=False)
        log.enqueue("tool", {"chat_id": "c"})
        for _ in range(4):
            self.assertEqual(log.flush(), 0)
            self.assertEqual(len(list((self.dir / "inbox").glob("batch-*"))), 1)
        log.flush()
        self.assertEqual(list((self.dir / "inbox").glob("batch-*")), [])
        self.assertEqual(len(list((self.dir / "inbox" / "failed").iterdir())), 1)

    def test_mercury_feedback_rows_are_picked_up(self):
        log = self.make()
        chatlog.drop_in_inbox(self.dir / "inbox", "feedback", {"chat_id": "c", "rating": "up"})
        self.assertEqual(log.flush(), 1)
        self.assertIn("chatbot_feedback", self.writes[0][1])

    def test_rows_are_cast_to_the_table_schema(self):
        row = chatlog.coerce_row("llm", {"status_code": "200", "cost_usd": "0.01", "degenerate": 0,
                                         "email": "a@b.c", "unknown_column": 1})
        names = [n for n, _ in chatlog.SCHEMAS["llm"]]
        self.assertEqual(len(row), len(names))
        self.assertEqual(row[names.index("status_code")], 200)
        self.assertEqual(row[names.index("cost_usd")], 0.01)
        self.assertIs(row[names.index("degenerate")], False)

    def test_write_command_appends_with_the_old_schema(self):
        """Run the generated command against a stand-in Spark to check what it writes."""
        captured = {}

        class FakeWriter:
            def format(self, f): captured["format"] = f; return self
            def option(self, k, v): captured[k] = v; return self
            def mode(self, m): captured["mode"] = m; return self
            def saveAsTable(self, t): captured["table"] = t

        class FakeDF:
            write = FakeWriter()

        class FakeSpark:
            def createDataFrame(self, rows, schema):
                captured["rows"], captured["schema"] = rows, schema
                return FakeDF()

        types_mod = types.ModuleType("pyspark.sql.types")
        types_mod.StructType = types.SimpleNamespace(fromJson=lambda d: d)
        modules = {"pyspark": types.ModuleType("pyspark"), "pyspark.sql": types.ModuleType("pyspark.sql"),
                   "pyspark.sql.types": types_mod}
        rows = [chatlog.coerce_row("feedback", {"chat_id": "c1", "rating": "down", "message_index": 3})]
        with patch.dict(sys.modules, modules):
            exec(chatlog.write_command("feedback", "universal.logging.chatbot_feedback", rows),
                 {"spark": FakeSpark(), "print": lambda *a: None})
        self.assertEqual(captured["table"], "universal.logging.chatbot_feedback")
        self.assertEqual((captured["format"], captured["mode"], captured["mergeSchema"]), ("delta", "append", "true"))
        self.assertEqual([f["name"] for f in captured["schema"]["fields"]],
                         [n for n, _ in chatlog.SCHEMAS["feedback"]])
        self.assertEqual(captured["rows"][0][0], "c1")


class UsageLoggerTests(TempDirTest):
    def setUp(self):
        super().setUp()
        self.rows = []
        log = types.SimpleNamespace(enqueue=lambda kind, row: self.rows.append((kind, row)))
        self.clock = {"t": 1000.0}
        self.logger = usage_log.UsageLogger(log, prompt_name="default",
                                            email_for=lambda sid, fallback: "pat@surusenterprises.com",
                                            cost_for=lambda model, usage: 0.002, clock=lambda: self.clock["t"])

    def test_a_turn_produces_call_tool_and_turn_rows(self):
        L = self.logger
        L.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="How many voters in Kane?",
                          model="x-ai/grok-4.7", platform="api_server")
        L.on_post_api_request(session_id="s1", turn_id="t1", api_request_id="r1", model="x-ai/grok-4.7",
                              response_model="x-ai/grok-4.7", api_duration=1.5, started_at=1000.0,
                              first_chunk_at=1000.4, finish_reason="tool_calls",
                              usage={"prompt_tokens": 900, "output_tokens": 40}, assistant_content_chars=0,
                              assistant_tool_call_count=1)
        L.on_post_tool_call(tool_name="execute_python", args={"code": "spark.table('x').count()"},
                            result="312,904", session_id="s1", turn_id="t1", duration_ms=4200, status="ok")
        L.on_post_tool_call(tool_name="memory", args={}, result="", session_id="s1", turn_id="t1")
        L.on_post_api_request(session_id="s1", turn_id="t1", api_request_id="r2", model="x-ai/grok-4.7",
                              api_duration=0.8, started_at=1006.0, usage={"prompt_tokens": 1000, "output_tokens": 60},
                              response={"assistant_message": {"content": "Kane County has 312,904 voters."}},
                              assistant_content_chars=31, assistant_tool_call_count=0, finish_reason="stop")
        self.clock["t"] = 1010.0
        L.on_post_llm_call(session_id="s1", turn_id="t1", assistant_response="Kane County has 312,904 voters.")
        L.on_session_end(session_id="s1", turn_id="t1", completed=True)

        kinds = [k for k, _ in self.rows]
        self.assertEqual(kinds, ["llm", "tool", "llm", "turn"])
        first, tool, second, turn = (r for _, r in self.rows)
        self.assertEqual((first["email"], first["first_token_ms"], first["cost_usd"]),
                         ("pat@surusenterprises.com", 400, 0.002))
        self.assertEqual((tool["status"], tool["step"], tool["duration_ms"]), ("ok", 1, 4200))
        self.assertEqual(second["response"], "Kane County has 312,904 voters.")
        self.assertEqual((turn["llm_calls"], turn["tool_calls"], turn["prompt_tokens"],
                          turn["completion_tokens"], turn["cost_usd"]), (2, 1, 1900, 100, 0.004))
        self.assertEqual((turn["models"], turn["prompt_name"], turn["stopped"], turn["duration_ms"]),
                         ("x-ai/grok-4.7", "default", False, 10000))

    def test_a_stopped_turn_is_still_recorded(self):
        L = self.logger
        L.on_pre_llm_call(session_id="s2", turn_id="t9", user_message="long query")
        L.on_post_tool_call(tool_name="execute_python", args={"code": "x"}, result="Execution stopped by the user.",
                            session_id="s2", turn_id="t9", duration_ms=100)
        L.on_session_end(session_id="s2", turn_id="t9", completed=False, interrupted=True)
        tool, turn = (r for _, r in self.rows)
        self.assertEqual(tool["status"], "cancelled")
        self.assertTrue(turn["stopped"])

    def test_failed_calls_carry_the_error(self):
        L = self.logger
        L.on_pre_llm_call(session_id="s3", turn_id="t1", user_message="hi")
        L.on_api_request_error(session_id="s3", turn_id="t1", model="x-ai/grok-4.7", status_code=401,
                               error={"message": "Missing Authentication header"}, retry_count=0)
        L.on_session_end(session_id="s3", turn_id="t1", completed=False, failed=True)
        call, turn = (r for _, r in self.rows)
        self.assertEqual((call["status_code"], call["error"]), (401, "Missing Authentication header"))
        self.assertEqual(turn["error"], "The turn failed")

    def test_email_comes_from_mercury_chat_records(self):
        db = self.dir / "chats.sqlite3"
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE chats (session_id TEXT PRIMARY KEY, email TEXT)")
        conn.execute("INSERT INTO chats VALUES ('s1', 'pat@americanstewardship.org')")
        conn.commit()
        conn.close()
        lookup = usage_log.EmailLookup(db)
        self.assertEqual(lookup("s1"), "pat@americanstewardship.org")
        self.assertEqual(lookup("s-unknown", "+16305550100"), "+16305550100")
        self.assertEqual(usage_log.EmailLookup(self.dir / "missing.sqlite3")("s1"), "unknown")


class RegistrationTests(TempDirTest):
    def register(self, profile: str, env: dict):
        registered = {"tools": {}, "hooks": []}
        ctx = types.SimpleNamespace(
            profile_name=profile,
            register_tool=lambda **kw: registered["tools"].__setitem__(kw["name"], kw),
            register_hook=lambda name, cb: registered["hooks"].append(name))
        hermes_constants = types.ModuleType("hermes_constants")
        hermes_constants.get_hermes_home = lambda: self.dir / "home"
        with patch.dict(sys.modules, {"hermes_constants": hermes_constants}), \
                patch.dict(os.environ, env, clear=False):
            for key in ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_CLIENT_ID", "DATABRICKS_CLIENT_SECRET"):
                if key not in env:
                    os.environ.pop(key, None)
            plugin.register(ctx)
        return registered

    def test_staff_gets_execute_python_and_the_logging_hooks(self):
        reg = self.register("staff", {"DATABRICKS_HOST": "h", "DATABRICKS_TOKEN": "t",
                                      "MERCURY_DIR": str(self.dir / "mercury")})
        self.assertEqual(set(reg["tools"]), {"execute_python"})
        self.assertEqual(reg["tools"]["execute_python"]["toolset"], "surus_databricks")
        self.assertEqual(set(reg["hooks"]), {"pre_llm_call", "post_api_request", "api_request_error",
                                             "post_tool_call", "post_llm_call", "on_session_end"})

    def test_optimizer_also_gets_the_procedure_tool(self):
        reg = self.register("optimizer", {"DATABRICKS_HOST": "h", "DATABRICKS_TOKEN": "t"})
        self.assertEqual(reg["tools"]["surus_procedure"]["toolset"], "surus_optimizer")
        text = plugin.surus_procedure({"name": "optimization_injection"})
        self.assertIn("CRITICAL RULES", text)
        self.assertTrue(plugin.surus_procedure({"name": "../../etc/passwd"}).startswith("ERROR"))

    def test_tool_is_hidden_until_databricks_is_configured(self):
        reg = self.register("staff", {})
        self.assertFalse(reg["tools"]["execute_python"]["check_fn"]())

    def test_execute_python_without_databricks_explains_why(self):
        state = plugin._PluginState(self.dir / "state")
        with patch.dict(os.environ, {}, clear=False):
            for key in ("DATABRICKS_HOST", "DATABRICKS_TOKEN"):
                os.environ.pop(key, None)
            out = plugin.make_execute_python(state)({"code": "1"}, session_id="s1")
        self.assertIn("Databricks is not configured", out)


if __name__ == "__main__":
    unittest.main()
