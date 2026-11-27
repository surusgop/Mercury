"""Mercury's staff chat at /ask: private chats with the Surus agents.

The agents are Hermes profiles (staff, optimizer) served by the one
multiplexing gateway's API server on loopback (/p/<agent>/api/sessions/...).
Mercury adds what Hermes does not have: who owns each chat, a page for staff,
and turns that keep running when the browser tab closes.

  ChatStore       chat ownership, titles and per-turn stats in
                  MERCURY_DIR/chats.sqlite3. The Surus Databricks plugin reads
                  `chats(session_id, email)` to put the user's email on every
                  usage-log row, so every Hermes session id a chat uses is there.
  HermesAgentAPI  the Hermes Sessions API: create, transcript, streamed turn, stop.
  RunManager      runs each turn in a background task and keeps its events, so
                  the page can long-poll them and pick a turn up again after a
                  reload or a closed tab.
  AskAPI          the /ask/api/* routes.
"""
from __future__ import annotations

import asyncio
import datetime
import json
import re
import sqlite3
import threading
import time
import uuid
from contextlib import closing, contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

import httpx
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    session_id TEXT PRIMARY KEY,   -- every Hermes session id a chat has used
    chat_id    TEXT NOT NULL,
    email      TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS chats_by_chat ON chats (chat_id);
CREATE TABLE IF NOT EXISTS chat_meta (
    chat_id    TEXT PRIMARY KEY,   -- also the chat's first Hermes session id
    email      TEXT NOT NULL,
    agent      TEXT NOT NULL,
    title      TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL,      -- the Hermes session new turns go to
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    turns      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS chat_meta_by_owner ON chat_meta (email, updated_at);
CREATE TABLE IF NOT EXISTS turns (
    chat_id     TEXT NOT NULL,
    turn        INTEGER NOT NULL,
    model       TEXT,
    steps       INTEGER,
    duration_ms INTEGER,
    cost_usd    REAL,
    status      TEXT,
    created_at  REAL NOT NULL,
    PRIMARY KEY (chat_id, turn)
);
CREATE TABLE IF NOT EXISTS feedback (
    chat_id    TEXT NOT NULL,
    turn       INTEGER NOT NULL,
    rating     TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (chat_id, turn)
);
"""

TITLE_MAX = 80
MESSAGE_MAX = 20_000
STEP_OUTPUT_MAX = 4_000
SNIPPET_MAX = 500
RUN_RETAIN_S = 600.0          # a finished turn's events stay available this long
POLL_WAIT_MAX_S = 25.0
STOP_GRACE_S = 10.0           # then the upstream connection is dropped instead
STREAM_READ_TIMEOUT_S = 120.0  # Hermes sends a keepalive every 10 s
CHAT_ID_RE = re.compile(r"^mercury_[0-9a-f]{32}$")
FEEDBACK_RATINGS = ("up", "down", "cleared")


def new_chat_id() -> str:
    return f"mercury_{uuid.uuid4().hex}"


def clean_title(text: str) -> str:
    title = re.sub(r"\s+", " ", text or "").strip()
    if len(title) > TITLE_MAX:
        title = title[:TITLE_MAX - 1].rstrip() + "…"
    return title


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ── chat records ──────────────────────────────────────────────────────────────
class ChatStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._schema_ready = False

    @contextmanager
    def _db(self):
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # closing(): sqlite3's own context manager commits but never closes.
            with closing(sqlite3.connect(self.path, timeout=10)) as db:
                db.row_factory = sqlite3.Row
                if not self._schema_ready:
                    db.executescript(SCHEMA)
                    self._schema_ready = True
                with db:
                    yield db

    def create(self, chat_id: str, email: str, agent: str, title: str) -> dict:
        now = time.time()
        with self._db() as db:
            db.execute("INSERT INTO chat_meta (chat_id, email, agent, title, session_id, created_at, updated_at)"
                       " VALUES (?, ?, ?, ?, ?, ?, ?)", (chat_id, email, agent, clean_title(title), chat_id, now, now))
            db.execute("INSERT OR REPLACE INTO chats (session_id, chat_id, email, created_at) VALUES (?, ?, ?, ?)",
                       (chat_id, chat_id, email, now))
        return self.get(chat_id, email)

    def get(self, chat_id: str, email: str) -> Optional[dict]:
        """The chat, only for its owner."""
        with self._db() as db:
            row = db.execute("SELECT * FROM chat_meta WHERE chat_id = ? AND email = ?", (chat_id, email)).fetchone()
        return dict(row) if row else None

    def list(self, email: str, agents: list[str], limit: int = 200) -> list[dict]:
        if not agents:
            return []
        marks = ",".join("?" * len(agents))
        with self._db() as db:
            rows = db.execute(f"SELECT * FROM chat_meta WHERE email = ? AND agent IN ({marks})"
                              " ORDER BY updated_at DESC LIMIT ?", (email, *agents, limit)).fetchall()
        return [dict(r) for r in rows]

    def rename(self, chat_id: str, title: str) -> None:
        with self._db() as db:
            db.execute("UPDATE chat_meta SET title = ? WHERE chat_id = ?", (clean_title(title), chat_id))

    def session_ids(self, chat_id: str) -> list[str]:
        with self._db() as db:
            rows = db.execute("SELECT session_id FROM chats WHERE chat_id = ? ORDER BY created_at", (chat_id,)).fetchall()
        return [r[0] for r in rows]

    def add_session(self, chat_id: str, session_id: str) -> None:
        """Hermes moved the chat to a new session (context compression): keep both."""
        with self._db() as db:
            row = db.execute("SELECT email FROM chat_meta WHERE chat_id = ?", (chat_id,)).fetchone()
            if row is None:
                return
            db.execute("INSERT OR IGNORE INTO chats (session_id, chat_id, email, created_at) VALUES (?, ?, ?, ?)",
                       (session_id, chat_id, row[0], time.time()))
            db.execute("UPDATE chat_meta SET session_id = ? WHERE chat_id = ?", (session_id, chat_id))

    def begin_turn(self, chat_id: str, turn: int) -> None:
        with self._db() as db:
            db.execute("UPDATE chat_meta SET turns = MAX(turns, ?), updated_at = ? WHERE chat_id = ?",
                       (turn, time.time(), chat_id))

    def record_turn(self, chat_id: str, turn: int, *, model: Optional[str], steps: int, duration_ms: int,
                    cost_usd: Optional[float], status: str) -> None:
        with self._db() as db:
            db.execute("INSERT OR REPLACE INTO turns (chat_id, turn, model, steps, duration_ms, cost_usd, status,"
                       " created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       (chat_id, turn, model, steps, duration_ms, cost_usd, status, time.time()))
            db.execute("UPDATE chat_meta SET updated_at = ? WHERE chat_id = ?", (time.time(), chat_id))

    def turn_stats(self, chat_id: str) -> dict[int, dict]:
        with self._db() as db:
            rows = db.execute("SELECT turn, model, steps, duration_ms, cost_usd, status FROM turns WHERE chat_id = ?",
                              (chat_id,)).fetchall()
        return {r["turn"]: {k: r[k] for k in ("model", "steps", "duration_ms", "cost_usd", "status")} for r in rows}

    def set_feedback(self, chat_id: str, turn: int, rating: str) -> None:
        with self._db() as db:
            if rating == "cleared":
                db.execute("DELETE FROM feedback WHERE chat_id = ? AND turn = ?", (chat_id, turn))
            else:
                db.execute("INSERT OR REPLACE INTO feedback (chat_id, turn, rating, created_at) VALUES (?, ?, ?, ?)",
                           (chat_id, turn, rating, time.time()))

    def feedback(self, chat_id: str) -> dict[int, str]:
        with self._db() as db:
            rows = db.execute("SELECT turn, rating FROM feedback WHERE chat_id = ?", (chat_id,)).fetchall()
        return {r[0]: r[1] for r in rows}

    def delete(self, chat_id: str) -> None:
        """Forget the chat. Its rows in `chats` stay, so usage logs written later
        for a turn still running in Hermes keep the owner's email."""
        with self._db() as db:
            for table in ("chat_meta", "turns", "feedback"):
                db.execute(f"DELETE FROM {table} WHERE chat_id = ?", (chat_id,))


# ── Hermes ────────────────────────────────────────────────────────────────────
class HermesError(RuntimeError):
    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


def _hermes_error_text(status: int, body: bytes | str) -> str:
    text = body.decode("utf-8", "replace") if isinstance(body, bytes) else str(body)
    message = text.strip()[:300]
    with suppress(ValueError, AttributeError):
        payload = json.loads(text)
        err = payload.get("error")
        message = (err.get("message") if isinstance(err, dict) else err) or payload.get("message") or message
    return f"Hermes answered HTTP {status}: {message}" if message else f"Hermes answered HTTP {status}"


async def parse_sse(lines: AsyncIterator[str]) -> AsyncIterator[tuple[str, dict]]:
    """(event name, JSON payload) pairs from server-sent event lines."""
    name, data = "", []
    async for line in lines:
        if line == "":
            if data:
                try:
                    payload = json.loads("\n".join(data))
                except ValueError:
                    payload = {}
                yield (name or "message"), (payload if isinstance(payload, dict) else {"value": payload})
            name, data = "", []
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        with suppress(ValueError):
            payload = json.loads("\n".join(data))
            yield (name or "message"), (payload if isinstance(payload, dict) else {"value": payload})


class HermesAgentAPI:
    """The agents' Hermes API server, addressed per profile as /p/<agent>/..."""

    def __init__(self, base_url: str, key_for: Callable[[str], str], client: Callable[[], httpx.AsyncClient]):
        self.base_url = base_url.rstrip("/")
        self.key_for = key_for
        self.client = client

    def _url(self, agent: str, path: str) -> str:
        return f"{self.base_url}/p/{agent}{path}"

    def _headers(self, agent: str) -> dict:
        key = self.key_for(agent)
        if not key:
            raise HermesError(f"The {agent} agent is not set up yet (it has no API key).")
        return {"Authorization": f"Bearer {key}"}

    async def _request(self, method: str, agent: str, path: str, *, body: Any = None,
                       ok: tuple = (200,)) -> dict:
        try:
            resp = await self.client().request(method, self._url(agent, path), json=body,
                                               headers=self._headers(agent), timeout=30.0)
        except httpx.HTTPError as exc:
            raise HermesError(f"Hermes is not reachable ({type(exc).__name__}).") from exc
        if resp.status_code not in ok:
            raise HermesError(_hermes_error_text(resp.status_code, resp.content), status=resp.status_code)
        try:
            data = resp.json()
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}

    async def create_session(self, agent: str, session_id: str) -> None:
        await self._request("POST", agent, "/api/sessions", body={"id": session_id}, ok=(200, 201))

    async def session(self, agent: str, session_id: str) -> dict:
        return (await self._request("GET", agent, f"/api/sessions/{session_id}")).get("session") or {}

    async def messages(self, agent: str, session_id: str) -> list[dict]:
        """The transcript (the latest 500 messages, oldest first)."""
        data = (await self._request("GET", agent, f"/api/sessions/{session_id}/messages")).get("data")
        return [m for m in data if isinstance(m, dict)] if isinstance(data, list) else []

    async def delete_session(self, agent: str, session_id: str) -> None:
        await self._request("DELETE", agent, f"/api/sessions/{session_id}", ok=(200, 404))

    async def stop_run(self, agent: str, run_id: str) -> None:
        await self._request("POST", agent, f"/v1/runs/{run_id}/stop", body={}, ok=(200, 202))

    async def stream_turn(self, agent: str, session_id: str, text: str) -> AsyncIterator[tuple[str, dict]]:
        headers = {**self._headers(agent), "Accept": "text/event-stream"}
        timeout = httpx.Timeout(connect=5.0, read=STREAM_READ_TIMEOUT_S, write=30.0, pool=30.0)
        try:
            async with self.client().stream("POST", self._url(agent, f"/api/sessions/{session_id}/chat/stream"),
                                            json={"message": text}, headers=headers, timeout=timeout) as resp:
                if resp.status_code != 200:
                    raise HermesError(_hermes_error_text(resp.status_code, await resp.aread()),
                                      status=resp.status_code)
                async for item in parse_sse(resp.aiter_lines()):
                    yield item
        except httpx.HTTPError as exc:
            raise HermesError(f"Lost the connection to Hermes ({type(exc).__name__}).") from exc


# ── turns ─────────────────────────────────────────────────────────────────────
FORWARDED_EVENTS = ("assistant.delta", "assistant.commentary", "tool.started", "tool.completed",
                    "tool.failed", "assistant.completed", "error")


def _tool_code(args: Any) -> str:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return ""
    code = args.get("code") if isinstance(args, dict) else None
    return code if isinstance(code, str) else ""


def slim_event(name: str, data: dict) -> dict:
    """Only what the page needs from one Hermes stream event."""
    if name == "assistant.delta":
        return {"delta": str(data.get("delta") or "")}
    if name == "assistant.commentary":
        return {"text": str(data.get("text") or ""), "already_streamed": bool(data.get("already_streamed"))}
    if name == "tool.started":
        return {"tool": str(data.get("tool_name") or ""), "code": _tool_code(data.get("args"))}
    if name in ("tool.completed", "tool.failed"):
        return {"tool": str(data.get("tool_name") or "")}
    if name == "assistant.completed":
        return {"content": str(data.get("content") or ""), "interrupted": bool(data.get("interrupted"))}
    if name == "error":
        return {"message": str(data.get("message") or "Something went wrong.")}
    return {}


@dataclass
class Run:
    chat_id: str
    agent: str
    session_id: str
    turn: int
    email: str
    text: str
    key: str = field(default_factory=lambda: uuid.uuid4().hex)
    started: float = field(default_factory=time.monotonic)
    events: list = field(default_factory=list)
    done: bool = False
    status: str = "running"
    run_id: str = ""
    stop_requested: bool = False
    task: Optional[asyncio.Task] = None
    # Set and replaced on every new event, so waiters never need a lock (and a
    # cancelled turn can always record that it ended).
    changed: asyncio.Event = field(default_factory=asyncio.Event)


class Busy(Exception):
    """The chat already has a turn running."""


class RunManager:
    def __init__(self, store: ChatStore, api: HermesAgentAPI, *, model_for: Callable[[str], str],
                 retain_s: float = RUN_RETAIN_S, stop_grace_s: float = STOP_GRACE_S):
        self.store = store
        self.api = api
        self.model_for = model_for
        self.retain_s = retain_s
        self.stop_grace_s = stop_grace_s
        self.runs: dict[str, Run] = {}

    def get(self, chat_id: str) -> Optional[Run]:
        return self.runs.get(chat_id)

    def active(self, chat_id: str) -> Optional[Run]:
        run = self.runs.get(chat_id)
        return run if run is not None and not run.done else None

    def start(self, chat: dict, text: str) -> Run:
        if self.active(chat["chat_id"]):
            raise Busy()
        run = Run(chat_id=chat["chat_id"], agent=chat["agent"], session_id=chat["session_id"],
                  turn=int(chat.get("turns") or 0) + 1, email=chat["email"], text=text)
        self.runs[run.chat_id] = run
        run.task = asyncio.create_task(self._drive(run))
        run.task.add_done_callback(lambda task: self._ended(run, task))
        return run

    def _ended(self, run: Run, task: asyncio.Task) -> None:
        """Safety net: a turn whose task died without saying so still ends, so
        the page never waits on it forever."""
        if run.done:
            return
        cancelled = run.stop_requested or task.cancelled()
        run.status = "cancelled" if cancelled else "failed"
        self._emit(run, "turn.finished", {
            "status": run.status, "turn": None, "model": self.model_for(run.agent), "steps": 0,
            "duration_ms": int((time.monotonic() - run.started) * 1000), "cost_usd": None,
            "error": None if cancelled else "The reply ended unexpectedly."}, done=True)
        self._forget_later(run)

    @staticmethod
    def _emit(run: Run, name: str, data: dict, *, done: bool = False) -> None:
        run.events.append({**data, "seq": len(run.events) + 1, "type": name})
        if done:
            run.done = True
        changed, run.changed = run.changed, asyncio.Event()
        changed.set()

    async def _session_cost(self, run: Run) -> Optional[float]:
        try:
            session = await self.api.session(run.agent, run.session_id)
        except (HermesError, asyncio.CancelledError):
            return None
        for key in ("actual_cost_usd", "estimated_cost_usd"):
            value = session.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    async def _drive(self, run: Run) -> None:
        turn = {"model": self.model_for(run.agent), "steps": 0, "status": "", "error": None,
                "error_shown": False, "started": False, "cost_before": None}
        self._emit(run, "turn.started", {"turn": run.turn, "text": run.text})
        try:
            await self._stream(run, turn)
        except asyncio.CancelledError:
            turn["status"] = "cancelled"
        except HermesError as exc:
            turn["error"] = str(exc)
        except Exception as exc:  # never leave the page waiting on a turn that died
            turn["error"] = f"{type(exc).__name__}: {exc}"
        await self._finish(run, turn)

    async def _stream(self, run: Run, turn: dict) -> None:
        turn["cost_before"] = await self._session_cost(run)
        if run.stop_requested:  # stopped before Hermes was even asked
            turn["status"] = "cancelled"
            return
        async for name, data in self.api.stream_turn(run.agent, run.session_id, run.text):
            if name == "run.started":
                run.run_id = str(data.get("run_id") or "")
                if not turn["started"]:
                    turn["started"] = True
                    self.store.begin_turn(run.chat_id, run.turn)
                if run.stop_requested and run.run_id:
                    await self._stop_upstream(run)
                continue
            if name in ("run.completed", "run.failed", "run.cancelled"):
                turn["status"] = name.split(".", 1)[1]
                session_id = data.get("session_id")
                if isinstance(session_id, str) and session_id and session_id != run.session_id:
                    self.store.add_session(run.chat_id, session_id)
                    run.session_id = session_id
                continue
            if name not in FORWARDED_EVENTS:
                continue
            event = slim_event(name, data)
            if name == "tool.started":
                turn["steps"] += 1
            elif name == "error":
                turn["error"], turn["error_shown"] = event["message"], True
            self._emit(run, name, event)

    async def _finish(self, run: Run, turn: dict) -> None:
        status, error = turn["status"], turn["error"]
        if run.stop_requested and status in ("", "failed"):
            status = "cancelled"
        if not status:
            status = "failed"
            error = error or "The reply ended unexpectedly."
        if status == "failed" and error and not turn["error_shown"]:
            self._emit(run, "error", {"message": error})
        duration_ms = int((time.monotonic() - run.started) * 1000)
        cost_usd = None
        try:
            if turn["started"]:
                cost_after = await self._session_cost(run)
                before = turn["cost_before"]
                if before is not None and cost_after is not None and cost_after >= before:
                    cost_usd = round(cost_after - before, 6)
                self.store.record_turn(run.chat_id, run.turn, model=turn["model"], steps=turn["steps"],
                                       duration_ms=duration_ms, cost_usd=cost_usd, status=status)
        except Exception as exc:  # the stats are optional; the end of the turn is not
            print(f"[ask] could not record turn {run.turn} of {run.chat_id}: {exc}", flush=True)
        finally:
            run.status = status
            self._emit(run, "turn.finished", {
                "status": status, "turn": run.turn if turn["started"] else None, "model": turn["model"],
                "steps": turn["steps"], "duration_ms": duration_ms, "cost_usd": cost_usd, "error": error}, done=True)
            self._forget_later(run)

    def _forget_later(self, run: Run) -> None:
        def forget():
            if self.runs.get(run.chat_id) is run:
                del self.runs[run.chat_id]
        asyncio.get_running_loop().call_later(self.retain_s, forget)

    async def _stop_upstream(self, run: Run) -> None:
        try:
            await self.api.stop_run(run.agent, run.run_id)
        except HermesError:
            if run.task is not None:
                run.task.cancel()  # dropping the stream interrupts the turn in Hermes

    async def stop(self, chat_id: str) -> bool:
        run = self.active(chat_id)
        if run is None:
            return False
        run.stop_requested = True
        if run.run_id:
            await self._stop_upstream(run)
        # Otherwise Hermes has not started the turn yet; _stream stops it as soon
        # as it does (or before asking at all).

        def give_up():
            if not run.done and run.task is not None:
                run.task.cancel()
        asyncio.get_running_loop().call_later(self.stop_grace_s, give_up)
        return True

    async def wait_done(self, chat_id: str, timeout: float) -> None:
        run = self.runs.get(chat_id)
        if run is not None and run.task is not None and not run.done:
            with suppress(asyncio.TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(run.task), timeout)

    async def events(self, chat_id: str, after: int, wait_s: float, key: str = "") -> dict:
        run = self.runs.get(chat_id)
        if run is None:
            return {"active": False, "done": True, "events": []}
        if key and key != run.key:
            after = 0  # a newer turn than the one the page was following
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait_s
        while len(run.events) <= after and not run.done:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(run.changed.wait(), remaining)
        if len(run.events) > after and not run.done:
            await asyncio.sleep(0.1)  # let a few more stream deltas arrive in the same answer
        return {"active": True, "key": run.key, "turn": run.turn, "done": run.done, "events": run.events[after:]}


# ── transcripts ───────────────────────────────────────────────────────────────
def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    return ""


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n… [{len(text) - limit:,} more characters]"


def history_view(messages: list[dict]) -> list[dict]:
    """A Hermes transcript → exchanges: the question, the steps (code run and
    notes), and the answer."""
    exchanges: list[dict] = []
    current: Optional[dict] = None
    steps_by_call: dict[str, dict] = {}
    for message in messages:
        role = message.get("role")
        if role == "user":
            current = {"turn": len(exchanges) + 1, "user": _text_of(message.get("content")),
                       "steps": [], "answer": ""}
            exchanges.append(current)
            continue
        if current is None:
            continue
        if role == "assistant":
            text = _text_of(message.get("content"))
            calls = message.get("tool_calls") or []
            if calls:
                if text.strip():
                    current["steps"].append({"kind": "note", "text": text})
                for call in calls:
                    if not isinstance(call, dict):
                        continue
                    fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                    step = {"kind": "code", "tool": fn.get("name") or call.get("name") or "",
                            "code": _tool_code(fn.get("arguments", call.get("arguments"))),
                            "output": None, "status": "done"}
                    current["steps"].append(step)
                    if call.get("id"):
                        steps_by_call[str(call["id"])] = step
            elif text.strip():
                current["answer"] = f"{current['answer']}\n\n{text}" if current["answer"] else text
        elif role == "tool":
            step = steps_by_call.get(str(message.get("tool_call_id") or ""))
            if step is not None:
                output = _text_of(message.get("content"))
                step["output"] = _clip(output, STEP_OUTPUT_MAX)
                if output.startswith("ERROR"):
                    step["status"] = "error"
                elif output.startswith("Execution stopped by the user"):
                    step["status"] = "stopped"
    return exchanges


# ── /ask/api routes ───────────────────────────────────────────────────────────
@dataclass
class AskDeps:
    identity: Callable[[Request], Optional[dict]]   # {"email", "name", "admin"} or None
    agents: Callable[[dict], list[dict]]            # agents this user may use: {"id", "ready", "reason", ...}
    store: ChatStore
    runs: RunManager
    api: HermesAgentAPI
    list_files: Callable[[list[str]], Awaitable[list[dict]]]
    open_file: Callable[[list[str], str], Awaitable[Optional[tuple[httpx.Response, str]]]]
    feedback_sink: Callable[[str, dict], None]      # (agent, feedback row)


def _error(message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


class AskAPI:
    def __init__(self, deps: AskDeps):
        self.d = deps

    def routes(self) -> list[Route]:
        return [
            Route("/ask/api/me", self.me, methods=["GET"]),
            Route("/ask/api/chats", self.chats, methods=["GET", "POST"]),
            Route("/ask/api/chats/{chat_id}", self.chat, methods=["GET", "PATCH", "DELETE"]),
            Route("/ask/api/chats/{chat_id}/messages", self.send, methods=["POST"]),
            Route("/ask/api/chats/{chat_id}/events", self.events, methods=["GET"]),
            Route("/ask/api/chats/{chat_id}/stop", self.stop, methods=["POST"]),
            Route("/ask/api/chats/{chat_id}/feedback", self.feedback, methods=["POST"]),
            Route("/ask/api/chats/{chat_id}/files", self.files, methods=["GET"]),
            Route("/ask/api/chats/{chat_id}/files/{name}", self.file, methods=["GET"]),
        ]

    # ── helpers ───────────────────────────────────────────────────────────────
    def _who(self, request: Request) -> tuple[Optional[dict], Optional[Response]]:
        user = self.d.identity(request)
        if user is None:
            return None, _error("Sign in again.", 401)
        return user, None

    def _agents(self, user: dict) -> dict[str, dict]:
        return {a["id"]: a for a in self.d.agents(user)}

    def _owned(self, request: Request, user: dict) -> tuple[Optional[dict], Optional[Response]]:
        chat_id = request.path_params["chat_id"]
        chat = self.d.store.get(chat_id, user["email"]) if CHAT_ID_RE.match(chat_id) else None
        # A chat with an agent this user can no longer use (e.g. no longer an admin) is hidden too.
        if chat is None or chat["agent"] not in self._agents(user):
            return None, _error("Chat not found.", 404)
        return chat, None

    @staticmethod
    async def _json(request: Request) -> tuple[dict, Optional[Response]]:
        # JSON only: with SameSite=Lax cookies this keeps other sites from posting here.
        if "application/json" not in request.headers.get("content-type", ""):
            return {}, _error("Send JSON.", 415)
        try:
            body = await request.json()
        except ValueError:
            return {}, _error("Invalid JSON.", 400)
        return (body, None) if isinstance(body, dict) else ({}, _error("Invalid JSON.", 400))

    @staticmethod
    def _message(body: dict) -> tuple[str, Optional[Response]]:
        text = body.get("message")
        if not isinstance(text, str) or not text.strip():
            return "", _error("Type a question first.", 400)
        if len(text) > MESSAGE_MAX:
            return "", _error(f"Messages are limited to {MESSAGE_MAX:,} characters.", 400)
        return text.strip(), None

    def _summary(self, chat: dict) -> dict:
        return {"id": chat["chat_id"], "title": chat["title"] or "New chat", "agent": chat["agent"],
                "updated_at": chat["updated_at"], "running": self.d.runs.active(chat["chat_id"]) is not None}

    def _start(self, chat: dict, text: str) -> Response:
        try:
            run = self.d.runs.start(chat, text)
        except Busy:
            return _error("This chat is still answering. Stop it or wait for the reply.", 409)
        return JSONResponse({"chat": self._summary(chat), "run": {"key": run.key, "turn": run.turn}},
                            status_code=202)

    # ── endpoints ─────────────────────────────────────────────────────────────
    async def me(self, request: Request) -> Response:
        user, err = self._who(request)
        if err:
            return err
        return JSONResponse({"email": user["email"], "name": user.get("name") or "", "admin": bool(user.get("admin")),
                             "agents": self.d.agents(user)})

    async def chats(self, request: Request) -> Response:
        user, err = self._who(request)
        if err:
            return err
        agents = self._agents(user)
        if request.method == "GET":
            rows = self.d.store.list(user["email"], list(agents))
            return JSONResponse({"chats": [self._summary(c) for c in rows]})
        body, err = await self._json(request)
        if err:
            return err
        text, err = self._message(body)
        if err:
            return err
        agent = agents.get(str(body.get("agent") or "staff"))
        if agent is None:
            return _error("That assistant is not available to you.", 403)
        if not agent.get("ready"):
            return _error(agent.get("reason") or "That assistant is not ready yet.", 503)
        chat_id = new_chat_id()
        try:
            await self.d.api.create_session(agent["id"], chat_id)
        except HermesError as exc:
            return _error(str(exc), 503)
        chat = self.d.store.create(chat_id, user["email"], agent["id"], text)
        return self._start(chat, text)

    async def chat(self, request: Request) -> Response:
        user, err = self._who(request)
        if err:
            return err
        chat, err = self._owned(request, user)
        if err:
            return err
        chat_id = chat["chat_id"]
        if request.method == "PATCH":
            body, err = await self._json(request)
            if err:
                return err
            title = clean_title(str(body.get("title") or ""))
            if not title:
                return _error("Give the chat a name.", 400)
            self.d.store.rename(chat_id, title)
            return JSONResponse({"chat": self._summary(self.d.store.get(chat_id, user["email"]))})
        if request.method == "DELETE":
            if await self.d.runs.stop(chat_id):
                await self.d.runs.wait_done(chat_id, STOP_GRACE_S + 5)
            for session_id in self.d.store.session_ids(chat_id):
                with suppress(HermesError):
                    await self.d.api.delete_session(chat["agent"], session_id)
            self.d.store.delete(chat_id)
            return JSONResponse({"deleted": True})
        history_error = ""
        try:
            exchanges = history_view(await self.d.api.messages(chat["agent"], chat["session_id"]))
        except HermesError as exc:
            exchanges, history_error = [], str(exc)
        run = self.d.runs.active(chat_id)
        if run is not None and exchanges and exchanges[-1]["user"].strip() == run.text.strip():
            exchanges.pop()  # the turn in progress comes from the run's events instead
        stats, ratings = self.d.store.turn_stats(chat_id), self.d.store.feedback(chat_id)
        for exchange in exchanges:
            exchange["stats"] = stats.get(exchange["turn"])
            exchange["rating"] = ratings.get(exchange["turn"])
        return JSONResponse({
            "chat": self._summary(chat), "exchanges": exchanges, "history_error": history_error,
            "run": {"active": True, "key": run.key, "turn": run.turn} if run is not None else {"active": False}})

    async def send(self, request: Request) -> Response:
        user, err = self._who(request)
        if err:
            return err
        chat, err = self._owned(request, user)
        if err:
            return err
        body, err = await self._json(request)
        if err:
            return err
        text, err = self._message(body)
        if err:
            return err
        agent = self._agents(user)[chat["agent"]]
        if not agent.get("ready"):
            return _error(agent.get("reason") or "That assistant is not ready yet.", 503)
        return self._start(chat, text)

    async def events(self, request: Request) -> Response:
        user, err = self._who(request)
        if err:
            return err
        chat, err = self._owned(request, user)
        if err:
            return err
        try:
            after = max(0, int(request.query_params.get("after", "0")))
            wait = min(POLL_WAIT_MAX_S, max(0.0, float(request.query_params.get("wait", "0"))))
        except ValueError:
            return _error("after and wait must be numbers.", 400)
        result = await self.d.runs.events(chat["chat_id"], after, wait, request.query_params.get("key", ""))
        return JSONResponse(result, headers={"Cache-Control": "no-store"})

    async def stop(self, request: Request) -> Response:
        user, err = self._who(request)
        if err:
            return err
        chat, err = self._owned(request, user)
        if err:
            return err
        _body, err = await self._json(request)
        if err:
            return err
        return JSONResponse({"stopping": await self.d.runs.stop(chat["chat_id"])})

    async def feedback(self, request: Request) -> Response:
        user, err = self._who(request)
        if err:
            return err
        chat, err = self._owned(request, user)
        if err:
            return err
        body, err = await self._json(request)
        if err:
            return err
        rating, turn = body.get("rating"), body.get("turn")
        if rating not in FEEDBACK_RATINGS or not isinstance(turn, int) or isinstance(turn, bool) \
                or not 1 <= turn <= max(1, int(chat["turns"] or 0)):
            return _error("rating must be up, down or cleared, for one of this chat's answers.", 400)
        snippet = body.get("snippet") if isinstance(body.get("snippet"), str) else ""
        self.d.store.set_feedback(chat["chat_id"], turn, rating)
        stats = self.d.store.turn_stats(chat["chat_id"]).get(turn) or {}
        # The old chatbot's feedback columns; message_index counted user and
        # assistant messages from 0, so answer N was message 2N - 1.
        self.d.feedback_sink(chat["agent"], {
            "chat_id": chat["session_id"], "email": user["email"], "created_at": now_iso(),
            "message_index": 2 * turn - 1, "turn_number": turn, "rating": rating,
            "model": stats.get("model"), "response_snippet": snippet[:SNIPPET_MAX] or None})
        return JSONResponse({"rating": rating})

    async def files(self, request: Request) -> Response:
        user, err = self._who(request)
        if err:
            return err
        chat, err = self._owned(request, user)
        if err:
            return err
        try:
            files = await self.d.list_files(self.d.store.session_ids(chat["chat_id"]))
        except Exception as exc:  # Databricks down or misconfigured: the chat still works
            return JSONResponse({"files": [], "error": f"Could not list this chat's files ({type(exc).__name__})."})
        return JSONResponse({"files": [{k: f.get(k) for k in ("name", "size", "modified")} for f in files]})

    async def file(self, request: Request) -> Response:
        user, err = self._who(request)
        if err:
            return err
        chat, err = self._owned(request, user)
        if err:
            return err
        name = request.path_params["name"]
        try:
            opened = await self.d.open_file(self.d.store.session_ids(chat["chat_id"]), name)
        except Exception as exc:
            return _error(f"Could not download {name} ({type(exc).__name__}).", 502)
        if opened is None:
            return _error("File not found.", 404)
        upstream, filename = opened
        safe = re.sub(r'[^A-Za-z0-9._-]', "_", filename) or "download"
        media = "text/csv; charset=utf-8" if safe.lower().endswith(".csv") else "application/octet-stream"
        headers = {"Content-Disposition": f'attachment; filename="{safe}"', "Cache-Control": "no-store",
                   "X-Content-Type-Options": "nosniff"}
        if upstream.headers.get("content-length") and not upstream.headers.get("content-encoding"):
            headers["Content-Length"] = upstream.headers["content-length"]
            body = upstream.aiter_raw()
        else:
            body = upstream.aiter_bytes()  # decoded, so its length is unknown
        return StreamingResponse(body, media_type=media, headers=headers,
                                 background=BackgroundTask(upstream.aclose))
