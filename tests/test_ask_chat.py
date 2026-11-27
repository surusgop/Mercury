"""Mercury's staff chat at /ask (mercury_chat.py and its wiring in server.py):
private chats, turns that run in the background, feedback into the usage-log
inbox, and the chat's exported files. Hermes is faked; no gateway needed."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import surus_auth  # noqa: E402  (vendored)
import mercury_chat as mc  # noqa: E402
from hermes_plugins.surus_databricks.usage_log import EmailLookup  # noqa: E402

STAFF = "analyst@americanstewardship.org"
OTHER = "someone@surusenterprises.com"
ADMIN = "admin@surusenterprises.com"
GOOGLE_ENV = {"GOOGLE_CLIENT_ID": "client-id.apps.googleusercontent.com", "GOOGLE_CLIENT_SECRET": "client-secret",
              "APP_URL": "https://mercury.example", "ADMIN_BOOTSTRAP_EMAILS": ADMIN}
JSON_HEADERS = {"accept": "application/json"}

TRANSCRIPT = [
    {"role": "user", "content": "How many voters are in DuPage County, IL?"},
    {"role": "assistant", "content": "Let me count them.",
     "tool_calls": [{"id": "call_1", "type": "function",
                     "function": {"name": "execute_python", "arguments": json.dumps({"code": "spark.sql('...')"})}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "count\n612,345"},
    {"role": "assistant", "content": "There are **612,345** registered voters in DuPage County."},
]


class FakeHermes:
    """Stands in for HermesAgentAPI. `script` is the next turn's stream:
    (event, data) pairs, plus ("wait-stop", None) to block until Stop and
    ("cost", 0.01) to add to the session's estimated cost."""

    def __init__(self):
        self.sessions: dict[str, dict] = {}
        self.script: list = []
        self.stopped: list = []
        self.deleted: list = []
        self.stop_event = asyncio.Event()

    async def create_session(self, agent, session_id):
        self.sessions[session_id] = {"agent": agent, "cost": 0.0, "messages": []}

    async def session(self, agent, session_id):
        return {"estimated_cost_usd": self.sessions.get(session_id, {}).get("cost", 0.0)}

    async def messages(self, agent, session_id):
        return self.sessions.get(session_id, {}).get("messages", [])

    async def delete_session(self, agent, session_id):
        self.deleted.append(session_id)

    async def stop_run(self, agent, run_id):
        self.stopped.append(run_id)
        self.stop_event.set()

    async def stream_turn(self, agent, session_id, text):
        for name, data in self.script:
            if name == "raise":
                raise data
            if name == "wait-stop":
                await self.stop_event.wait()
                continue
            if name == "cost":
                self.sessions.setdefault(session_id, {"cost": 0.0})["cost"] += data
                continue
            yield name, data


def answer_script(text="There are **612,345** registered voters.", session_id=None):
    done = {"session_id": session_id} if session_id else {}
    return [
        ("run.started", {"run_id": "run_1"}),
        ("assistant.delta", {"delta": "Let me count them."}),
        ("tool.started", {"tool_name": "execute_python", "args": {"code": "spark.sql('...')"}}),
        ("tool.completed", {"tool_name": "execute_python"}),
        ("cost", 0.0123),
        ("assistant.delta", {"delta": text}),
        ("assistant.completed", {"content": text}),
        ("run.completed", {"usage": {"input_tokens": 10}, "messages": [{"huge": "x" * 1000}], **done}),
        ("done", {}),
    ]


class TempDir(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ask-chat-")
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)


class ChatStoreTests(TempDir):
    def test_chats_belong_to_their_owner(self):
        store = mc.ChatStore(self.dir / "chats.sqlite3")
        store.create("mercury_" + "a" * 32, STAFF, "staff", "  How many   voters?  ")
        store.create("mercury_" + "b" * 32, OTHER, "staff", "Mine")
        store.create("mercury_" + "c" * 32, STAFF, "optimizer", "Notebook")
        self.assertEqual(store.get("mercury_" + "a" * 32, STAFF)["title"], "How many voters?")
        self.assertIsNone(store.get("mercury_" + "a" * 32, OTHER))
        self.assertEqual([c["title"] for c in store.list(STAFF, ["staff"])], ["How many voters?"])
        self.assertEqual(len(store.list(STAFF, ["staff", "optimizer"])), 2)
        self.assertEqual(store.list(STAFF, []), [])

    def test_titles_are_shortened(self):
        self.assertEqual(len(mc.clean_title("x" * 300)), mc.TITLE_MAX)
        self.assertTrue(mc.clean_title("x" * 300).endswith("…"))

    def test_the_usage_logs_find_the_owner_of_every_session_a_chat_used(self):
        db = self.dir / "chats.sqlite3"
        store = mc.ChatStore(db)
        chat_id = mc.new_chat_id()
        store.create(chat_id, STAFF, "staff", "Q")
        store.add_session(chat_id, "compressed_session_2")
        self.assertEqual(store.get(chat_id, STAFF)["session_id"], "compressed_session_2")
        self.assertEqual(store.session_ids(chat_id), [chat_id, "compressed_session_2"])
        lookup = EmailLookup(db)
        self.assertEqual((lookup(chat_id), lookup("compressed_session_2")), (STAFF, STAFF))
        store.delete(chat_id)
        self.assertIsNone(store.get(chat_id, STAFF))
        self.assertEqual(EmailLookup(db)(chat_id), STAFF)  # late log rows keep the email

    def test_feedback_and_turn_stats(self):
        store = mc.ChatStore(self.dir / "chats.sqlite3")
        chat_id = mc.new_chat_id()
        store.create(chat_id, STAFF, "staff", "Q")
        store.begin_turn(chat_id, 1)
        store.record_turn(chat_id, 1, model="x-ai/grok-4.7", steps=2, duration_ms=1500, cost_usd=0.01,
                          status="completed")
        self.assertEqual(store.get(chat_id, STAFF)["turns"], 1)
        self.assertEqual(store.turn_stats(chat_id)[1]["steps"], 2)
        store.set_feedback(chat_id, 1, "up")
        self.assertEqual(store.feedback(chat_id), {1: "up"})
        store.set_feedback(chat_id, 1, "cleared")
        self.assertEqual(store.feedback(chat_id), {})


class TranscriptTests(unittest.TestCase):
    def test_a_transcript_becomes_question_steps_and_answer(self):
        (ex,) = mc.history_view(TRANSCRIPT)
        self.assertEqual(ex["turn"], 1)
        self.assertEqual(ex["user"], "How many voters are in DuPage County, IL?")
        note, code = ex["steps"]
        self.assertEqual(note, {"kind": "note", "text": "Let me count them."})
        self.assertEqual((code["code"], code["output"], code["status"]), ("spark.sql('...')", "count\n612,345", "done"))
        self.assertIn("612,345", ex["answer"])

    def test_errors_long_output_and_multimodal_messages(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}]},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c", "function": {"name": "execute_python", "arguments": "{\"code\": \"1/0\"}"}}]},
            {"role": "tool", "tool_call_id": "c", "content": "ERROR:\nZeroDivisionError" + "x" * 5000},
        ]
        (ex,) = mc.history_view(messages)
        self.assertEqual(ex["user"], "part one\npart two")
        step = ex["steps"][0]
        self.assertEqual(step["status"], "error")
        self.assertLess(len(step["output"]), 4100)
        self.assertEqual(ex["answer"], "")


class SseTests(unittest.IsolatedAsyncioTestCase):
    async def test_events_keepalives_and_multiline_data(self):
        async def lines():
            for line in [": keepalive", "", "event: run.started", 'data: {"run_id": "r1"}', "",
                         "event: assistant.delta", 'data: {"delta":', 'data: "hi"}', "", "data: [1]"]:
                yield line
        events = [e async for e in mc.parse_sse(lines())]
        self.assertEqual(events, [("run.started", {"run_id": "r1"}), ("assistant.delta", {"delta": "hi"}),
                                  ("message", {"value": [1]})])


class RunManagerTests(TempDir):
    def setUp(self):
        super().setUp()
        self.store = mc.ChatStore(self.dir / "chats.sqlite3")
        self.hermes = FakeHermes()
        self.runs = mc.RunManager(self.store, self.hermes, model_for=lambda a: "x-ai/grok-4.7", stop_grace_s=5)

    async def chat(self):
        chat_id = mc.new_chat_id()
        await self.hermes.create_session("staff", chat_id)
        return self.store.create(chat_id, STAFF, "staff", "Q")

    async def finish(self, run):
        await asyncio.wait_for(run.task, 5)
        return {e["type"]: e for e in run.events}

    async def test_a_turn_streams_and_records_its_stats(self):
        chat = await self.chat()
        self.hermes.script = answer_script()
        run = self.runs.start(chat, "How many?")
        with self.assertRaises(mc.Busy):
            self.runs.start(chat, "Again?")
        events = await self.finish(run)
        types = [e["type"] for e in run.events]
        self.assertEqual(types, ["turn.started", "assistant.delta", "tool.started", "tool.completed",
                                 "assistant.delta", "assistant.completed", "turn.finished"])
        self.assertEqual(events["tool.started"]["code"], "spark.sql('...')")
        done = events["turn.finished"]
        self.assertEqual((done["status"], done["turn"], done["steps"], done["cost_usd"]), ("completed", 1, 1, 0.0123))
        self.assertTrue(run.done)
        self.assertEqual(self.store.get(chat["chat_id"], STAFF)["turns"], 1)
        self.assertEqual(self.store.turn_stats(chat["chat_id"])[1]["status"], "completed")
        self.assertNotIn("huge", json.dumps(run.events))  # Hermes' full turn transcript is not kept

    async def test_long_polling_waits_for_new_events(self):
        chat = await self.chat()
        self.hermes.script = [("run.started", {"run_id": "r"}), ("wait-stop", None),
                              ("run.cancelled", {}), ("done", {})]
        run = self.runs.start(chat, "Slow one")
        first = await self.runs.events(chat["chat_id"], 0, 1.0)
        self.assertEqual([e["type"] for e in first["events"]], ["turn.started"])
        waiting = asyncio.create_task(self.runs.events(chat["chat_id"], 1, 5.0))
        await asyncio.sleep(0.05)
        self.assertFalse(waiting.done())
        self.assertTrue(await self.runs.stop(chat["chat_id"]))
        result = await asyncio.wait_for(waiting, 5)
        self.assertTrue(result["done"])
        self.assertEqual(result["events"][-1]["status"], "cancelled")
        self.assertEqual(self.hermes.stopped, ["r"])
        # A page following an older turn's key starts this turn from its first event.
        stale = await self.runs.events(chat["chat_id"], 99, 0, key="old")
        self.assertEqual(stale["events"][0]["type"], "turn.started")
        self.assertEqual((await self.runs.events("mercury_" + "0" * 32, 0, 0))["active"], False)

    async def test_stop_before_hermes_started_the_turn(self):
        chat = await self.chat()
        self.hermes.script = answer_script()
        run = self.runs.start(chat, "Never mind")
        await self.runs.stop(chat["chat_id"])
        events = await self.finish(run)
        self.assertEqual(events["turn.finished"]["status"], "cancelled")
        self.assertIsNone(events["turn.finished"]["turn"])
        self.assertEqual(self.store.get(chat["chat_id"], STAFF)["turns"], 0)

    async def test_hermes_errors_end_the_turn_with_a_message(self):
        chat = await self.chat()
        self.hermes.script = [("raise", mc.HermesError("Hermes answered HTTP 429: too many runs", status=429))]
        run = self.runs.start(chat, "Q")
        events = await self.finish(run)
        self.assertIn("429", events["error"]["message"])
        self.assertEqual(events["turn.finished"]["status"], "failed")
        self.assertEqual(self.store.turn_stats(chat["chat_id"]), {})
        # A stream that just stops is a failure too, not a turn left running forever.
        self.hermes.script = [("run.started", {"run_id": "r2"}), ("assistant.delta", {"delta": "Par"})]
        run = self.runs.start(self.store.get(chat["chat_id"], STAFF), "Q2")
        events = await self.finish(run)
        self.assertEqual(events["turn.finished"]["status"], "failed")
        self.assertEqual(events["error"]["message"], "The reply ended unexpectedly.")

    async def test_a_compressed_session_keeps_the_chat_together(self):
        chat = await self.chat()
        self.hermes.script = answer_script(session_id="rotated_session")
        run = self.runs.start(chat, "Q")
        await self.finish(run)
        self.assertEqual(self.store.get(chat["chat_id"], STAFF)["session_id"], "rotated_session")
        self.assertEqual(EmailLookup(self.dir / "chats.sqlite3")("rotated_session"), STAFF)


class AskRouteTests(unittest.IsolatedAsyncioTestCase):
    """The /ask pages and API through the real app, with Google sign-in on."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ask-routes-")
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name) / ".hermes"
        self.home.mkdir()

    def load(self):
        env = {"HERMES_HOME": str(self.home), "ADMIN_PASSWORD": "test-password", **GOOGLE_ENV}
        patcher = patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in ("SESSION_SECRET", "ALLOWED_EMAIL_DOMAIN", "ALLOWED_EMAIL_DOMAINS", "GOOGLE_WORKSPACE_DOMAIN",
                     "ALLOWED_EMAILS", "CHAT_LOGGING"):
            os.environ.pop(name, None)
        spec = importlib.util.spec_from_file_location(f"ask_routes_{uuid.uuid4().hex}", ROOT / "server.py")
        server = importlib.util.module_from_spec(spec)
        if not hasattr(os, "fchmod"):  # Windows test hosts only; Linux has it
            no_fchmod = patch.object(os, "fchmod", lambda fd, mode: None, create=True)
            no_fchmod.start()
            self.addCleanup(no_fchmod.stop)
        spec.loader.exec_module(server)
        self.hermes = FakeHermes()
        server.ASK.d.api = self.hermes
        server.ask_runs.api = self.hermes
        server.surus_agents_status.update(enabled=True, agents={
            "staff": {"ready": True, "openrouter_key": "the Setup key (…a1b2)"},
            "optimizer": {"ready": True, "openrouter_key": "the Setup key (…a1b2)"}})
        server.gw.state = "running"
        self.server = server
        return server

    def client(self):
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=self.server.asgi_app),
                                 base_url="https://mercury.example")

    @staticmethod
    def as_user(email: str) -> dict:
        token = surus_auth.serialize_session({"email": email, "name": email.split("@")[0], "first_name": "Pat",
                                              "session_id": "s1"})
        return {"cookie": f"mercury_session={token}", **JSON_HEADERS}

    async def ask(self, c, user, text="How many voters are in DuPage County, IL?", agent="staff"):
        r = await c.post("/ask/api/chats", json={"agent": agent, "message": text}, headers=user)
        self.assertEqual(r.status_code, 202, r.text)
        return r.json()

    async def follow(self, c, user, chat_id):
        after, events = 0, []
        for _ in range(50):
            r = await c.get(f"/ask/api/chats/{chat_id}/events", params={"after": after, "wait": 2}, headers=user)
            data = r.json()
            events += data["events"]
            after = events[-1]["seq"] if events else 0
            if data["done"]:
                return events
        self.fail("the turn never finished")

    async def test_staff_chat_from_question_to_rated_answer(self):
        server = self.load()
        user = self.as_user(STAFF)
        async with self.client() as c:
            me = (await c.get("/ask/api/me", headers=user)).json()
            self.assertEqual([a["id"] for a in me["agents"]], ["staff"])  # no optimizer for staff
            self.hermes.script = answer_script()
            started = await self.ask(c, user)
            chat_id = started["chat"]["id"]
            self.assertTrue(mc.CHAT_ID_RE.match(chat_id))
            self.assertEqual(self.hermes.sessions[chat_id]["agent"], "staff")
            events = await self.follow(c, user, chat_id)
            self.assertEqual(events[-1]["type"], "turn.finished")
            self.assertEqual(events[-1]["model"], "x-ai/grok-4.7")

            self.hermes.sessions[chat_id]["messages"] = TRANSCRIPT
            chat = (await c.get(f"/ask/api/chats/{chat_id}", headers=user)).json()
            (ex,) = chat["exchanges"]
            self.assertEqual((ex["stats"]["steps"], ex["stats"]["cost_usd"], ex["rating"]), (1, 0.0123, None))
            self.assertFalse(chat["run"]["active"])
            listed = (await c.get("/ask/api/chats", headers=user)).json()["chats"]
            self.assertEqual([x["title"] for x in listed], ["How many voters are in DuPage County, IL?"])

            r = await c.post(f"/ask/api/chats/{chat_id}/feedback", headers=user,
                             json={"turn": 1, "rating": "down", "snippet": "There are 612,345"})
            self.assertEqual(r.status_code, 200)
            (inbox_file,) = (server.MERCURY_DIR / "log-inbox" / "staff").glob("batch-*.jsonl")
            item = json.loads(inbox_file.read_text(encoding="utf-8"))
            self.assertEqual(item["kind"], "feedback")
            row = item["row"]
            self.assertEqual((row["chat_id"], row["email"], row["rating"], row["turn_number"], row["message_index"],
                              row["model"], row["response_snippet"]),
                             (chat_id, STAFF, "down", 1, 1, "x-ai/grok-4.7", "There are 612,345"))
            again = (await c.get(f"/ask/api/chats/{chat_id}", headers=user)).json()
            self.assertEqual(again["exchanges"][0]["rating"], "down")
            bad = await c.post(f"/ask/api/chats/{chat_id}/feedback", headers=user, json={"turn": 7, "rating": "up"})
            self.assertEqual(bad.status_code, 400)

            r = await c.patch(f"/ask/api/chats/{chat_id}", headers=user, json={"title": "DuPage count"})
            self.assertEqual(r.json()["chat"]["title"], "DuPage count")
            r = await c.delete(f"/ask/api/chats/{chat_id}", headers=user)
            self.assertEqual(r.json(), {"deleted": True})
            self.assertEqual(self.hermes.deleted, [chat_id])
            self.assertEqual((await c.get(f"/ask/api/chats/{chat_id}", headers=user)).status_code, 404)

    async def test_chats_are_private(self):
        self.load()
        owner, other = self.as_user(STAFF), self.as_user(OTHER)
        async with self.client() as c:
            self.hermes.script = answer_script()
            chat_id = (await self.ask(c, owner))["chat"]["id"]
            await self.follow(c, owner, chat_id)
            self.assertEqual((await c.get("/ask/api/chats", headers=other)).json()["chats"], [])
            for method, path, body in [
                    ("GET", f"/ask/api/chats/{chat_id}", None), ("PATCH", f"/ask/api/chats/{chat_id}", {"title": "x"}),
                    ("DELETE", f"/ask/api/chats/{chat_id}", None),
                    ("POST", f"/ask/api/chats/{chat_id}/messages", {"message": "hi"}),
                    ("GET", f"/ask/api/chats/{chat_id}/events", None),
                    ("POST", f"/ask/api/chats/{chat_id}/stop", {}),
                    ("POST", f"/ask/api/chats/{chat_id}/feedback", {"turn": 1, "rating": "up"}),
                    ("GET", f"/ask/api/chats/{chat_id}/files", None),
                    ("GET", f"/ask/api/chats/{chat_id}/files/matches.csv", None)]:
                r = await c.request(method, path, json=body, headers=other)
                self.assertEqual(r.status_code, 404, f"{method} {path}")
            self.assertEqual(self.hermes.deleted, [])

    async def test_the_optimizer_is_for_admins_only(self):
        self.load()
        async with self.client() as c:
            r = await c.post("/ask/api/chats", json={"agent": "optimizer", "message": "hi"}, headers=self.as_user(STAFF))
            self.assertEqual(r.status_code, 403)
            admin = self.as_user(ADMIN)
            self.assertEqual([a["id"] for a in (await c.get("/ask/api/me", headers=admin)).json()["agents"]],
                             ["staff", "optimizer"])
            self.hermes.script = answer_script()
            started = await self.ask(c, admin, "Which tables need OPTIMIZE?", agent="optimizer")
            self.assertEqual(started["chat"]["agent"], "optimizer")
            # The emergency username/password admin can use it too.
            emergency = {"cookie": f"hermes_auth={self.server._make_auth_token()}", **JSON_HEADERS}
            self.assertTrue((await c.get("/ask/api/me", headers=emergency)).json()["admin"])

    async def test_requests_need_json_and_a_ready_assistant(self):
        server = self.load()
        user = self.as_user(STAFF)
        async with self.client() as c:
            r = await c.post("/ask/api/chats", content="message=hi", headers={**user, "content-type": "text/plain"})
            self.assertEqual(r.status_code, 415)
            r = await c.post("/ask/api/chats", json={"message": "   "}, headers=user)
            self.assertEqual(r.status_code, 400)
            server.surus_agents_status["agents"]["staff"]["openrouter_key"] = "none: set an OpenRouter key in Setup"
            r = await c.post("/ask/api/chats", json={"message": "hi"}, headers=user)
            self.assertEqual(r.status_code, 503)
            self.assertIn("OpenRouter", r.json()["error"])
            server.gw.state = "starting"
            me = (await c.get("/ask/api/me", headers=user)).json()
            self.assertFalse(me["agents"][0]["ready"])
            r = await c.get("/ask/api/me", headers=JSON_HEADERS)  # signed out
            self.assertEqual(r.status_code, 302)

    async def test_a_second_question_waits_for_the_first(self):
        self.load()
        user = self.as_user(STAFF)
        async with self.client() as c:
            self.hermes.script = [("run.started", {"run_id": "r"}), ("wait-stop", None), ("run.cancelled", {})]
            chat_id = (await self.ask(c, user))["chat"]["id"]
            r = await c.post(f"/ask/api/chats/{chat_id}/messages", json={"message": "and Kane?"}, headers=user)
            self.assertEqual(r.status_code, 409)
            chat = (await c.get(f"/ask/api/chats/{chat_id}", headers=user)).json()
            self.assertTrue(chat["run"]["active"])
            self.assertTrue(chat["chat"]["running"])
            r = await c.post(f"/ask/api/chats/{chat_id}/stop", json={}, headers=user)
            self.assertEqual(r.json(), {"stopping": True})
            events = await self.follow(c, user, chat_id)
            self.assertEqual(events[-1]["status"], "cancelled")

    async def test_files_are_listed_and_downloaded_from_the_chat_folder(self):
        server = self.load()
        user = self.as_user(STAFF)
        listed: list[str] = []

        class FakeClient:
            def list_directory(self, path):
                listed.append(path)
                if path.endswith("/exports") and "mercury_" in path:
                    return [{"path": f"{path}/matches.csv", "name": "matches.csv", "file_size": 12,
                             "last_modified": 1}, {"path": f"{path}/sub", "name": "sub", "is_directory": True}]
                return []

            def file_url(self, path):
                return f"https://dbc.example/api/2.0/fs/files{path}", {"Authorization": "Bearer t"}

            def close(self):
                pass

        server._ask_databricks = lambda: (FakeClient(), "/Volumes/universal/chatbot/files")

        class Body(httpx.AsyncByteStream):  # a streamed body, as Databricks sends it
            async def __aiter__(self):
                yield b"a,b\n"
                yield b"1,2\n"

        async def fake_send(request, **kwargs):
            self.assertEqual(request.url.path.split("/")[-1], "matches.csv")
            self.assertTrue(kwargs.get("stream"))
            return httpx.Response(200, stream=Body(), headers={"content-length": "8"}, request=request)

        async with self.client() as c:
            self.hermes.script = answer_script()
            chat_id = (await self.ask(c, user))["chat"]["id"]
            await self.follow(c, user, chat_id)
            files = (await c.get(f"/ask/api/chats/{chat_id}/files", headers=user)).json()["files"]
            self.assertEqual(files, [{"name": "matches.csv", "size": 12, "modified": 1}])
            self.assertEqual(listed, [f"/Volumes/universal/chatbot/files/chats/{chat_id}/exports"])
            with patch.object(server.get_http_client(), "send", fake_send):
                r = await c.get(f"/ask/api/chats/{chat_id}/files/matches.csv", headers=user)
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.content, b"a,b\n1,2\n")
                self.assertIn('filename="matches.csv"', r.headers["content-disposition"])
                for name in ("other.csv", "sub"):  # only files in the chat's own listing
                    r = await c.get(f"/ask/api/chats/{chat_id}/files/{name}", headers=user)
                    self.assertEqual(r.status_code, 404, name)
                r = await c.get(f"/ask/api/chats/{chat_id}/files/..%2F..%2Fsecret.csv", headers=user)
                self.assertNotEqual(r.status_code, 200)

    async def test_the_page_shows_who_is_signed_in(self):
        server = self.load()
        async with self.client() as c:
            page = await c.get("/ask", headers={**self.as_user(STAFF), "accept": "text/html"})
            self.assertEqual(page.status_code, 200)
            self.assertIn(STAFF, page.text)
            self.assertNotIn("Open Setup", page.text)
            self.assertIn('"admin": false', page.text)
            page = await c.get("/ask", headers={**self.as_user(ADMIN), "accept": "text/html"})
            self.assertIn("Open Setup", page.text)
        self.assertEqual(server._script_json({"x": "</script><b>&"}),
                         '{"x": "\\u003c/script\\u003e\\u003cb\\u003e\\u0026"}')


if __name__ == "__main__":
    unittest.main()
