"""Surus Databricks plugin for Hermes (installed by Mercury into the staff and
optimizer profiles).

Tools
  execute_python    run Python/PySpark in this chat's session on the chatbot's
                    Databricks cluster (toolset `surus_databricks`)
  surus_procedure   load a long procedure document on demand (toolset
                    `surus_optimizer`, optimizer profile only)

Hooks log usage to the chatbot's Delta tables (see usage_log.py / chatlog.py).

Settings are the chatbot's own environment variables: DATABRICKS_HOST,
DATABRICKS_TOKEN or DATABRICKS_CLIENT_ID/SECRET, DATABRICKS_CLUSTER_ID,
SURUS_VOLUME, MATCH_INDEX_SCHEMA, LOG_TABLE_PREFIX, CHAT_LOGGING,
TOOL_TIMEOUT_SECONDS, TOOL_OUTPUT_MAX_CHARS, COMMAND_POLL_SECONDS,
CONTEXT_TIMEOUT_SECONDS.

Module imports stay light (stdlib + httpx) so Mercury itself can import the
helpers; Hermes-only modules are imported inside functions.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Optional

DEFAULT_CLUSTER_ID = "0323-203611-a2zxrlbg"        # the chatbot's all-purpose cluster
DEFAULT_VOLUME = "/Volumes/universal/chatbot/files"
DEFAULT_MATCH_INDEX_SCHEMA = "universal.chatbot"
DEFAULT_LOG_TABLE_PREFIX = "universal.logging.chatbot"
PROMPT_NAMES = {"staff": "default", "optimizer": "opti_bot"}  # the old chatbot's prompt names
PROCEDURES_DIR = Path(__file__).parent / "procedures"

EXECUTE_PYTHON_SCHEMA = {
    "name": "execute_python",
    "description": (
        "Run Python (PySpark) code on the Surus Databricks cluster in this chat's persistent session. "
        "Variables persist between calls, like a notebook; if the cluster restarted, recreate them. "
        "`spark` is the cluster SparkSession; pandas (pd) and numpy (np) are imported. Print what you "
        "need to see: output is returned as text (long output is shortened, so print summaries, "
        "counts and .limit()-ed samples rather than whole tables). A bare expression on the last line "
        "is displayed, e.g. `df.limit(10).toPandas()`. To give the user a file, call "
        "`export_csv(df, \"name.csv\")` with a pandas or Spark DataFrame."
    ),
    "parameters": {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Python code to run on the cluster."}},
        "required": ["code"],
    },
}

SURUS_PROCEDURE_SCHEMA = {
    "name": "surus_procedure",
    "description": (
        "Load a Surus procedure document. 'optimization_injection' is the full step-by-step procedure "
        "for annotating, analyzing, scoring or injecting optimization tracking cells into a Databricks "
        "notebook. Load it before doing any of those tasks and follow it exactly."
    ),
    "parameters": {
        "type": "object",
        "properties": {"name": {"type": "string", "enum": ["optimization_injection"]}},
        "required": ["name"],
    },
}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _interrupted() -> bool:
    try:
        from tools.interrupt import is_interrupted
    except Exception:
        return False
    return is_interrupted()


class _PluginState:
    """Per-profile singletons, built on first use so loading never needs Databricks."""

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self._executor = None
        self._lock = threading.Lock()
        self._cluster_seen = (0.0, False)

    def executor(self):
        with self._lock:
            if self._executor is None:
                from .dbx import DatabricksClient
                from .executor import ClusterExecutor
                client = DatabricksClient.from_env()
                if client is None:
                    return None
                self._executor = ClusterExecutor(
                    client, os.environ.get("DATABRICKS_CLUSTER_ID", "").strip() or DEFAULT_CLUSTER_ID,
                    os.environ.get("SURUS_VOLUME", "").strip() or DEFAULT_VOLUME,
                    self.state_dir / "contexts.json",
                    match_index_schema=os.environ.get("MATCH_INDEX_SCHEMA", "").strip() or DEFAULT_MATCH_INDEX_SCHEMA,
                    poll_s=_float_env("COMMAND_POLL_SECONDS", 1.0),
                    context_timeout_s=_float_env("CONTEXT_TIMEOUT_SECONDS", 1200.0),
                )
            return self._executor

    def cluster_running(self) -> bool:
        """Cached for 30 s so the log flusher does not poll the cluster API every cycle."""
        checked_at, running = self._cluster_seen
        if time.time() - checked_at < 30:
            return running
        ex = self.executor()
        running = bool(ex) and ex.client.cluster_state(ex.cluster_id) in ("RUNNING", "RESIZING")
        self._cluster_seen = (time.time(), running)
        return running

    def write_log(self, kind: str, command: str) -> bool:
        from .chatlog import LOG_CONTEXT_ID
        ex = self.executor()
        if ex is None:
            return False
        outcome = ex.run_raw(LOG_CONTEXT_ID, command)
        return outcome.status == "ok" and "SURUS_LOG_OK" in (outcome.text or "")


def databricks_configured() -> bool:
    from .dbx import DatabricksClient
    return DatabricksClient.from_env() is not None


def make_execute_python(state: _PluginState):
    def execute_python(args: dict, session_id: Optional[str] = None, task_id: Optional[str] = None, **_) -> str:
        from .dbx import DatabricksError
        from .executor import format_result, safe_chat_id
        code = str((args or {}).get("code") or "")
        if not code.strip():
            return "ERROR: no code was given."
        ex = state.executor()
        if ex is None:
            return "ERROR: Databricks is not configured for Mercury (DATABRICKS_HOST and a token are missing)."
        try:
            outcome = ex.run(safe_chat_id(session_id or task_id or ""), code,
                             timeout_s=_float_env("TOOL_TIMEOUT_SECONDS", 1800.0), interrupted=_interrupted)
        except DatabricksError as exc:
            return f"ERROR: Databricks request failed: {exc}"
        return format_result(outcome, _int_env("TOOL_OUTPUT_MAX_CHARS", 8000))
    return execute_python


def surus_procedure(args: dict, **_) -> str:
    name = str((args or {}).get("name") or "")
    path = PROCEDURES_DIR / f"{name}.md"
    if name != "optimization_injection" or not path.is_file():
        return "ERROR: unknown procedure. Available: optimization_injection."
    return path.read_text(encoding="utf-8")


def register(ctx) -> None:
    from hermes_constants import get_hermes_home
    from .chatlog import ChatLog
    from .usage_log import EmailLookup, UsageLogger

    profile = getattr(ctx, "profile_name", "") or "default"
    home = Path(get_hermes_home())
    state = _PluginState(home / "plugin-data" / "surus_databricks")

    ctx.register_tool(name="execute_python", toolset="surus_databricks", schema=EXECUTE_PYTHON_SCHEMA,
                      handler=make_execute_python(state), check_fn=databricks_configured)
    if profile == "optimizer":
        ctx.register_tool(name="surus_procedure", toolset="surus_optimizer", schema=SURUS_PROCEDURE_SCHEMA,
                          handler=surus_procedure)

    mercury_dir = Path(os.environ.get("MERCURY_DIR") or home / "plugin-data" / "surus_databricks")
    logging_on = (os.environ.get("CHAT_LOGGING", "true").strip().lower() not in ("false", "0", "no", "off")
                  and databricks_configured())
    log = ChatLog(mercury_dir / "log-inbox" / profile,
                  os.environ.get("LOG_TABLE_PREFIX", "").strip() or DEFAULT_LOG_TABLE_PREFIX,
                  writer=state.write_log, cluster_running=state.cluster_running, enabled=logging_on)
    UsageLogger(log, prompt_name=PROMPT_NAMES.get(profile, profile),
                email_for=EmailLookup(mercury_dir / "chats.sqlite3")).register(ctx)
