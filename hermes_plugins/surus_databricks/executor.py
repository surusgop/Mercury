"""Run model-written Python on the Databricks cluster, one session per chat.

Ported from the Surus Intelligence chatbot's cluster_exec.py (dev branch): the
same command wrapping, notebook-style display of a trailing expression, and
sentinel lines for exports and charts, over the Command Execution API.

  * Each chat (Hermes session id) gets its own execution context on the
    all-purpose cluster; variables persist between calls like a notebook.
  * Context ids are remembered on disk, so they survive a Mercury restart
    (the contexts themselves live on the cluster).
  * Stop (Hermes' per-thread interrupt) and the time limit cancel the running
    command on the cluster.
"""
from __future__ import annotations

import ast
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .dbx import DatabricksClient, DatabricksError

PRELUDE_SOURCE = (Path(__file__).parent / "prelude" / "surus_tools.py").read_text(encoding="utf-8")
CHART_SENTINEL = "__SURUS_CHART__:"
EXPORT_SENTINEL = "__SURUS_EXPORT__:"
_NOISE_LINES = {"sc will be removed in future DBR versions"}
_CONTEXT_STATE_TTL_S = 14 * 24 * 3600


class Cancelled(Exception):
    """Stop was pressed while waiting on the cluster."""


# ── the command protocol (unchanged from cluster_exec.py) ─────────────────────
_COMMAND_TEMPLATE = '''__surus_before_state = _surus_before()
try:
    exec(compile({code!r}, "<chatbot>", "exec"), globals())
except BaseException:
    import traceback as __surus_tb
    print("ERROR:")
    print(__surus_tb.format_exc())
finally:
    _surus_after(__surus_before_state)
'''
_SILENT_CALLS = {"print", "display", "show", "printSchema", "info", "describe_", "savefig"}


def prepare_code(code: str) -> str:
    """Notebook semantics: a trailing bare expression is displayed (see cluster_exec.py)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body:
        return code
    last = tree.body[-1]
    if not isinstance(last, ast.Expr) or isinstance(last.value, (ast.Constant, ast.JoinedStr)):
        return code
    value = last.value
    if isinstance(value, ast.Call):
        fn = value.func
        name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else None)
        if name in _SILENT_CALLS:
            return code
    if isinstance(value, ast.Await):
        return code
    raw = code.encode("utf-8")
    raw_lines = raw.splitlines(keepends=True)
    try:
        start = sum(len(line) for line in raw_lines[: last.lineno - 1]) + last.col_offset
        end = sum(len(line) for line in raw_lines[: last.end_lineno - 1]) + last.end_col_offset
    except TypeError:
        return code
    if not (0 <= start < end <= len(raw)):
        return code
    return (raw[:start] + b"__surus_last_expr = (" + raw[start:end] + b")" + raw[end:]).decode("utf-8")


def wrap_command(code: str) -> str:
    return _COMMAND_TEMPLATE.format(code=prepare_code(code))


def bootstrap_code(chat_dir: str, match_index_schema: Optional[str] = None) -> str:
    """Prelude for a fresh context: helper library and imports."""
    return "\n".join([
        PRELUDE_SOURCE,
        "",
        f"_surus_init({chat_dir!r}, match_index_schema={match_index_schema!r})",
        "try:",
        "    import pandas as pd",
        "    import numpy as np",
        "except Exception as _surus_e:",
        "    print('(warning: pandas/numpy unavailable: ' + str(_surus_e) + ')')",
        "try:",
        "    import plotly.express as px",
        "    import plotly.graph_objects as go",
        "    import plotly.io as pio",
        "    pio.show = lambda *a, **k: None",
        "except Exception as _surus_e:",
        "    print('(warning: plotly unavailable: ' + str(_surus_e) + ')')",
        "print('environment ready')",
    ])


def parse_sentinels(text: str) -> tuple[str, list[str], list[str]]:
    """Split command output into (visible text, chart paths, export paths)."""
    charts, exports, keep = [], [], []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith(CHART_SENTINEL):
            charts.append(stripped[len(CHART_SENTINEL):].strip())
        elif stripped.startswith(EXPORT_SENTINEL):
            exports.append(stripped[len(EXPORT_SENTINEL):].strip())
        elif stripped not in _NOISE_LINES:
            keep.append(line)
    return "\n".join(keep).strip("\n"), charts, exports


def truncate_text(text: str, max_chars: int) -> str:
    """Keep the head and tail of long output with a marker in between."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = int(max_chars * 0.7)
    tail = max_chars - head
    omitted = len(text) - head - tail
    return text[:head] + f"\n... [output truncated: {omitted:,} characters omitted] ...\n" + (text[-tail:] if tail else "")


def safe_chat_id(session_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", session_id or "no-session")[:120] or "no-session"


@dataclass
class Outcome:
    status: str                      # ok | error | cancelled | timeout
    text: str = ""
    error: Optional[str] = None
    exports: list = field(default_factory=list)
    charts: list = field(default_factory=list)
    truncated: bool = False


class ContextStore:
    """Chat → execution-context ids, kept in a small JSON file."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def _read(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (FileNotFoundError, ValueError):
            return {}

    def get(self, chat_id: str) -> Optional[str]:
        with self._lock:
            entry = self._read().get(chat_id) or {}
            return entry.get("context_id")

    def put(self, chat_id: str, context_id: Optional[str]) -> None:
        with self._lock:
            data = self._read()
            now = time.time()
            data = {k: v for k, v in data.items() if now - float(v.get("used_at", 0)) < _CONTEXT_STATE_TTL_S}
            if context_id:
                data[chat_id] = {"context_id": context_id, "used_at": now}
            else:
                data.pop(chat_id, None)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            os.replace(tmp, self.path)


class ClusterExecutor:
    def __init__(self, client: DatabricksClient, cluster_id: str, volume_root: str, state_path: Path, *,
                 match_index_schema: Optional[str] = None, poll_s: float = 1.0,
                 cluster_timeout_s: float = 1200.0, context_timeout_s: float = 1200.0,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic):
        self.client = client
        self.cluster_id = cluster_id
        self.volume_root = volume_root.rstrip("/")
        self.contexts = ContextStore(state_path)
        self.match_index_schema = match_index_schema
        self.poll_s = poll_s
        self.cluster_timeout_s = cluster_timeout_s
        self.context_timeout_s = context_timeout_s
        self._sleep = sleep
        self._clock = clock
        self._chat_locks: dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()

    def chat_dir(self, chat_id: str) -> str:
        return f"{self.volume_root}/chats/{chat_id}"

    def _chat_lock(self, chat_id: str) -> threading.Lock:
        with self._locks_lock:
            return self._chat_locks.setdefault(chat_id, threading.Lock())

    # ── cluster + context lifecycle ───────────────────────────────────────────
    def ensure_cluster_running(self, interrupted: Callable[[], bool]) -> None:
        deadline = self._clock() + self.cluster_timeout_s
        started = False
        while True:
            state = self.client.cluster_state(self.cluster_id)
            if state in ("RUNNING", "RESIZING"):
                return
            if interrupted():
                raise Cancelled()
            if state in ("TERMINATED", "TERMINATING", "ERROR", "UNKNOWN", None) and not started:
                self.client.start_cluster(self.cluster_id)
                started = True
            if self._clock() > deadline:
                raise DatabricksError(f"The Databricks cluster did not start within "
                                      f"{int(self.cluster_timeout_s)} seconds (state {state}).")
            self._sleep(5.0)

    def _new_context(self, chat_id: str, interrupted: Callable[[], bool]) -> str:
        self.ensure_cluster_running(interrupted)
        context_id = self.client.create_context(self.cluster_id)
        deadline = self._clock() + self.context_timeout_s
        while True:
            status = self.client.context_status(self.cluster_id, context_id)
            if status == "Running":
                break
            if status in ("Error", None) or self._clock() > deadline:
                self.client.destroy_context(self.cluster_id, context_id)
                raise DatabricksError(f"Could not create an execution context (status {status}).")
            if interrupted():
                self.client.destroy_context(self.cluster_id, context_id)
                raise Cancelled()
            self._sleep(self.poll_s)
        boot = self._run_command(context_id, bootstrap_code(self.chat_dir(chat_id), self.match_index_schema),
                                 timeout_s=900, interrupted=interrupted)
        if boot.status != "ok":
            self.client.destroy_context(self.cluster_id, context_id)
            raise DatabricksError(f"Could not prepare the execution environment: {boot.error or boot.text}")
        self.contexts.put(chat_id, context_id)
        return context_id

    def _context_for(self, chat_id: str, interrupted: Callable[[], bool]) -> tuple[str, bool]:
        context_id = self.contexts.get(chat_id)
        if context_id and self.client.cluster_state(self.cluster_id) in ("RUNNING", "RESIZING") \
                and self.client.context_status(self.cluster_id, context_id) == "Running":
            return context_id, False
        return self._new_context(chat_id, interrupted), True

    # ── commands ──────────────────────────────────────────────────────────────
    def _run_command(self, context_id: str, command: str, *, timeout_s: Optional[float],
                     interrupted: Callable[[], bool]) -> Outcome:
        command_id = self.client.execute(self.cluster_id, context_id, command)
        started = self._clock()
        failures = 0
        cancel_sent = False
        while True:
            try:
                status = self.client.command_status(self.cluster_id, context_id, command_id)
                failures = 0
            except DatabricksError:
                failures += 1
                if failures >= 5:
                    raise
                self._sleep(self.poll_s)
                continue
            state = status.get("status")
            if state in ("Finished", "Error", "Cancelled"):
                break
            if not cancel_sent and interrupted():
                self.client.cancel(self.cluster_id, context_id, command_id)
                cancel_sent = True
            if timeout_s and self._clock() - started > timeout_s:
                self.client.cancel(self.cluster_id, context_id, command_id)
                return Outcome("timeout")
            self._sleep(self.poll_s)
        results = status.get("results") or {}
        if state == "Cancelled" or cancel_sent:
            return Outcome("cancelled", text=str(results.get("data") or ""))
        if state == "Error" or results.get("resultType") == "error":
            cause = results.get("cause") or results.get("summary") or "The command failed without details."
            return Outcome("error", error=str(cause))
        data = results.get("data")
        return Outcome("ok", text="" if data is None else str(data), truncated=bool(results.get("truncated")))

    def run(self, chat_id: str, code: str, *, timeout_s: Optional[float],
            interrupted: Callable[[], bool] = lambda: False) -> Outcome:
        """Run model code in the chat's context (created or rebuilt as needed)."""
        with self._chat_lock(chat_id):
            wrapped = wrap_command(code)
            rebootstrapped = False
            for _attempt in range(3):
                try:
                    context_id, _created = self._context_for(chat_id, interrupted)
                    if interrupted():
                        return Outcome("cancelled")
                    outcome = self._run_command(context_id, wrapped, timeout_s=timeout_s, interrupted=interrupted)
                except Cancelled:
                    return Outcome("cancelled")
                except DatabricksError as exc:
                    if exc.status in (400, 404) and "context" in str(exc).lower():
                        self.contexts.put(chat_id, None)  # context vanished: rebuild it
                        continue
                    raise
                if outcome.status == "error" and "_surus_before" in (outcome.error or "") and not rebootstrapped:
                    # Prelude missing (context recreated elsewhere): drop it and rebuild.
                    rebootstrapped = True
                    self.contexts.put(chat_id, None)
                    continue
                break
            else:
                return Outcome("error", error="The execution environment could not be reached after several attempts.")
            self.contexts.put(chat_id, context_id)
            if outcome.status == "ok":
                outcome.text, outcome.charts, outcome.exports = parse_sentinels(outcome.text)
            return outcome

    def run_raw(self, chat_id: str, command: str, *, timeout_s: float = 600.0) -> Outcome:
        """Run an internal command (no wrapping) in a dedicated context, e.g. log writes."""
        with self._chat_lock(chat_id):
            for _attempt in range(2):
                context_id, _ = self._context_for(chat_id, lambda: False)
                try:
                    outcome = self._run_command(context_id, command, timeout_s=timeout_s, interrupted=lambda: False)
                except DatabricksError as exc:
                    if exc.status in (400, 404) and "context" in str(exc).lower():
                        self.contexts.put(chat_id, None)
                        continue
                    raise
                self.contexts.put(chat_id, context_id)
                return outcome
            return Outcome("error", error="The logging execution environment could not be reached.")


def format_result(outcome: Outcome, max_chars: int) -> str:
    """What the model sees for one execute_python call."""
    if outcome.status == "cancelled":
        return "Execution stopped by the user."
    if outcome.status == "timeout":
        return "ERROR: execution exceeded the time limit and was cancelled on the cluster. Break the task into smaller steps."
    if outcome.status == "error":
        return "ERROR:\n" + (outcome.error or "unknown error")
    text = outcome.text or "(code ran with no output)"
    notes = []
    for path in outcome.exports:
        notes.append(f"Saved {path.rsplit('/', 1)[-1]} to this chat's files; the user can download it "
                     "from the chat page.")
    if outcome.truncated:
        notes.append("(output truncated by Databricks; print less or write results to a file)")
    if notes:
        text += "\n" + "\n".join(notes)
    return truncate_text(text, max_chars)
