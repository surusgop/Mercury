"""Usage logging to the chatbot's Delta tables, same tables and columns as before.

Ported from the Surus Intelligence chatbot's chatlog.py. Tables (prefix
LOG_TABLE_PREFIX, default `universal.logging.chatbot`):

  <prefix>_logs        one row per LLM API call
  <prefix>_tool_runs   one row per code execution
  <prefix>_turns       one row per assistant turn
  <prefix>_feedback    one row per thumbs up/down (written by Mercury's chat page)

Rows go into an on-disk inbox (JSON lines) first, so a restart never loses
them, and Mercury can drop feedback rows into the same inbox. A background
thread writes them with one Spark append per table per flush, run on the
chatbot's cluster in a dedicated execution context. It only writes while the
cluster is already running, unless rows have waited longer than
`max_wait_s`, so a chat that runs no code never starts the cluster just to log.
"""
from __future__ import annotations

import datetime
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

S, I, D, B = "string", "integer", "double", "boolean"

SCHEMAS: dict[str, list[tuple[str, str]]] = {
    "llm": [
        ("databricks_request_id", S), ("model_response_id", S), ("email", S), ("status_code", I),
        ("request_time", S), ("input_token_count", I), ("output_token_count", I),
        ("input_character_count", I), ("output_character_count", I), ("endpoint_name", S),
        ("execution_duration_ms", I), ("request", S), ("response", S), ("error", S),
        ("chat_id", S), ("turn_id", S), ("model_requested", S), ("execution_mode", S), ("attempt", I),
        ("cost_usd", D), ("first_token_ms", I), ("finish_reason", S), ("tool_call_count", I),
        ("degenerate", B),
    ],
    "tool": [
        ("chat_id", S), ("turn_id", S), ("email", S), ("model", S), ("execution_mode", S), ("step", I),
        ("started_at", S), ("duration_ms", I), ("status", S), ("code", S), ("output_chars", I),
        ("error", S), ("chart", B), ("export", B),
    ],
    "turn": [
        ("chat_id", S), ("turn_id", S), ("email", S), ("prompt_name", S), ("execution_mode", S),
        ("started_at", S), ("duration_ms", I), ("first_token_ms", I), ("models", S), ("model_reason", S),
        ("llm_calls", I), ("tool_calls", I), ("tool_seconds", D), ("prompt_tokens", I),
        ("completion_tokens", I), ("cost_usd", D), ("stopped", B), ("degenerate_count", I), ("error", S),
        ("user_message", S), ("response_chars", I),
    ],
    "feedback": [
        ("chat_id", S), ("email", S), ("created_at", S), ("message_index", I), ("turn_number", I),
        ("rating", S), ("model", S), ("response_snippet", S),
    ],
}
TABLE_SUFFIX = {"llm": "_logs", "tool": "_tool_runs", "turn": "_turns", "feedback": "_feedback"}
_CASTS = {S: str, I: int, D: float, B: bool}
LOG_CONTEXT_ID = "__surus_logging__"


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def clip(value: Any, limit: int = 20_000) -> Optional[str]:
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    return text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} more chars]"


def coerce_row(kind: str, row: dict) -> list:
    """Row dict → list in schema order, with every value cast to its column type."""
    out = []
    for name, typ in SCHEMAS[kind]:
        value = row.get(name)
        if value is not None:
            try:
                value = _CASTS[typ](value)
            except (TypeError, ValueError):
                value = None
        out.append(value)
    return out


def schema_json(kind: str) -> str:
    return json.dumps({"type": "struct", "fields": [
        {"name": n, "type": t, "nullable": True, "metadata": {}} for n, t in SCHEMAS[kind]]})


def write_command(kind: str, table: str, rows: list[list]) -> str:
    """Python run on the cluster: append rows with an explicit schema (as chatlog.py did)."""
    return "\n".join([
        "import json as _sj",
        "from pyspark.sql.types import StructType as _SST",
        f"_srows = _sj.loads({json.dumps(rows)!r})",
        f"_sschema = _SST.fromJson(_sj.loads({schema_json(kind)!r}))",
        "(spark.createDataFrame([tuple(r) for r in _srows], schema=_sschema)",
        "   .write.format('delta').option('mergeSchema', 'true').mode('append')",
        f"   .saveAsTable({table!r}))",
        "print('SURUS_LOG_OK', len(_srows))",
    ])


def drop_in_inbox(inbox: Path, kind: str, row: dict) -> None:
    """Atomically add one ready-to-write file to an inbox (used by Mercury for feedback)."""
    inbox.mkdir(parents=True, exist_ok=True)
    name = f"batch-{time.time():.6f}-{uuid.uuid4().hex}.jsonl"
    tmp = inbox / (name + ".tmp")
    tmp.write_text(json.dumps({"kind": kind, "row": row}) + "\n", encoding="utf-8")
    os.replace(tmp, inbox / name)


class ChatLog:
    def __init__(self, inbox: Path, table_prefix: str, *, writer: Callable[[str, str], bool],
                 cluster_running: Callable[[], bool], enabled: bool = True,
                 flush_s: float = 5.0, max_wait_s: float = 6 * 3600):
        self.inbox = inbox
        self.tables = {k: f"{table_prefix}{suffix}" for k, suffix in TABLE_SUFFIX.items()}
        self._writer = writer                  # (kind, command) -> success
        self._cluster_running = cluster_running
        self.enabled = enabled
        self.flush_s = flush_s
        self.max_wait_s = max_wait_s
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._wake = threading.Event()
        self._failures: dict[str, int] = {}
        self.stats = {"written": 0, "failed_batches": 0, "set_aside": 0, "last_error": None}

    @property
    def _current(self) -> Path:
        return self.inbox / "current.jsonl"

    def enqueue(self, kind: str, row: dict) -> None:
        if not self.enabled:
            return
        line = json.dumps({"kind": kind, "row": row}, default=str) + "\n"
        with self._lock:
            self.inbox.mkdir(parents=True, exist_ok=True)
            with open(self._current, "a", encoding="utf-8") as fh:
                fh.write(line)
        self._ensure_thread()

    # ── flushing ──────────────────────────────────────────────────────────────
    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._loop, name="surus-chatlog", daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        while True:
            self._wake.wait(self.flush_s)
            self._wake.clear()
            try:
                self.flush()
            except Exception as exc:  # the flusher must never die
                self.stats["last_error"] = f"{type(exc).__name__}: {exc}"

    def _rotate(self) -> None:
        with self._lock:
            if self._current.exists() and self._current.stat().st_size:
                os.replace(self._current, self.inbox / f"batch-{time.time():.6f}-{uuid.uuid4().hex}.jsonl")

    def _batches(self) -> list[Path]:
        return sorted(self.inbox.glob("batch-*.jsonl")) if self.inbox.is_dir() else []

    def pending_age_s(self) -> float:
        files = self._batches() + ([self._current] if self._current.exists() else [])
        if not files:
            return 0.0
        return max(0.0, time.time() - min(f.stat().st_mtime for f in files))

    def flush(self, *, force: bool = False) -> int:
        """Write every queued row. Returns rows written; failed rows stay queued."""
        self._rotate()
        batches = self._batches()
        if not batches:
            return 0
        if not force and not self._cluster_running() and self.pending_age_s() < self.max_wait_s:
            return 0
        written = 0
        for path in batches:
            by_kind: dict[str, list[dict]] = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                    if item.get("kind") in SCHEMAS and isinstance(item.get("row"), dict):
                        by_kind.setdefault(item["kind"], []).append(item["row"])
                except ValueError:
                    continue
            failed: dict[str, list[dict]] = {}
            for kind, rows in by_kind.items():
                ok = False
                try:
                    ok = self._writer(kind, write_command(kind, self.tables[kind],
                                                          [coerce_row(kind, r) for r in rows]))
                except Exception as exc:
                    self.stats["last_error"] = f"{type(exc).__name__}: {exc}"
                if ok:
                    written += len(rows)
                else:
                    failed[kind] = rows
            if failed:
                self.stats["failed_batches"] += 1
                path.write_text("".join(json.dumps({"kind": k, "row": r}, default=str) + "\n"
                                        for k, rows in failed.items() for r in rows), encoding="utf-8")
                self._failures[path.name] = self._failures.get(path.name, 0) + 1
                if self._failures[path.name] >= 5:
                    # A batch that keeps failing (e.g. a permissions error) must not block
                    # every later row: keep it for inspection and move on.
                    set_aside = self.inbox / "failed"
                    set_aside.mkdir(exist_ok=True)
                    os.replace(path, set_aside / path.name)
                    self._failures.pop(path.name, None)
                    self.stats["set_aside"] += 1
                    continue
                break  # keep order; retry on the next cycle
            self._failures.pop(path.name, None)
            path.unlink(missing_ok=True)
        self.stats["written"] += written
        return written
