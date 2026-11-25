"""Hermes lifecycle hooks → rows for the chatbot's Delta tables (see chatlog.py).

  pre_llm_call       a turn starts (user message, model)
  post_api_request   one LLM API call finished        → <prefix>_logs row
  api_request_error  one LLM API call failed          → <prefix>_logs row (status/error)
  post_tool_call     execute_python finished          → <prefix>_tool_runs row
  post_llm_call      the turn finished successfully   → <prefix>_turns row
  on_session_end     the turn ended (stopped/failed)  → <prefix>_turns row if not yet written

`email` comes from Mercury's chat records (MERCURY_DIR/chats.sqlite3), which
map each chat (Hermes session id) to the signed-in user who owns it.
`cost_usd` is Hermes' own estimate from OpenRouter's per-token prices: Hermes
does not pass the provider's billed cost to plugins.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Callable, Optional

from .chatlog import ChatLog, clip, now_iso

EXECUTION_MODE = "cluster"
TOOL_NAME = "execute_python"


def _num(value: Any, cast=float) -> Any:
    try:
        return None if value is None else cast(value)
    except (TypeError, ValueError):
        return None


def estimate_cost(model: str, usage: Optional[dict], provider: str = "openrouter") -> Optional[float]:
    """Hermes' cost estimate for one call (None when pricing is unknown)."""
    if not usage:
        return None
    try:
        from agent.usage_pricing import CanonicalUsage, estimate_usage_cost
        canonical = CanonicalUsage(**{k: int(usage.get(k) or 0) for k in (
            "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens")})
        result = estimate_usage_cost(model, canonical, provider=provider)
        amount = getattr(result, "amount_usd", None)
        if amount is None or getattr(result, "status", "") in ("unknown", "included"):
            return None
        return float(amount)
    except Exception:
        return None


class EmailLookup:
    """session id → owner email from Mercury's chat records (cached)."""

    def __init__(self, db_path: Optional[Path]):
        self.db_path = db_path
        self._cache: dict[str, str] = {}

    def __call__(self, session_id: str, fallback: str = "") -> str:
        if session_id in self._cache:
            return self._cache[session_id]
        email = ""
        if self.db_path and self.db_path.exists():
            try:
                # closing(): sqlite3's own context manager commits but never closes.
                with closing(sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=2)) as db:
                    row = db.execute("SELECT email FROM chats WHERE session_id = ?", (session_id,)).fetchone()
                email = (row[0] or "") if row else ""
            except sqlite3.Error:
                email = ""
        if email:
            self._cache[session_id] = email
        return email or fallback or "unknown"


class UsageLogger:
    def __init__(self, log: ChatLog, *, prompt_name: str, email_for: Callable[[str, str], str],
                 cost_for: Callable[[str, Optional[dict]], Optional[float]] = estimate_cost,
                 clock: Callable[[], float] = time.time):
        self.log = log
        self.prompt_name = prompt_name
        self.email_for = email_for
        self.cost_for = cost_for
        self.clock = clock
        self._turns: dict[tuple, dict] = {}
        self._lock = threading.Lock()

    def _turn(self, session_id: str, turn_id: str) -> dict:
        key = (session_id or "", turn_id or "")
        with self._lock:
            turn = self._turns.get(key)
            if turn is None:
                turn = self._turns[key] = {
                    "started": self.clock(), "started_at": now_iso(), "user_message": "", "sender_id": "",
                    "models": [], "llm_calls": 0, "tool_calls": 0, "tool_seconds": 0.0,
                    "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0, "cost_known": False,
                    "first_token_ms": None, "error": None, "steps": 0}
            return turn

    def _pop_turn(self, session_id: str, turn_id: str) -> Optional[dict]:
        with self._lock:
            return self._turns.pop((session_id or "", turn_id or ""), None)

    # ── hooks ─────────────────────────────────────────────────────────────────
    def on_pre_llm_call(self, session_id="", turn_id="", user_message="", sender_id="", model="", **_):
        turn = self._turn(session_id, turn_id)
        turn["user_message"] = user_message or ""
        turn["sender_id"] = sender_id or ""
        turn["model_requested"] = model or ""

    def on_post_api_request(self, session_id="", turn_id="", api_request_id="", model="", response_model=None,
                            api_duration=None, started_at=None, first_chunk_at=None, finish_reason=None,
                            usage=None, response=None, assistant_content_chars=None,
                            assistant_tool_call_count=None, **_):
        turn = self._turn(session_id, turn_id)
        usage = usage or {}
        served = response_model or model
        prompt_tokens = _num(usage.get("prompt_tokens"), int)
        output_tokens = _num(usage.get("output_tokens"), int)
        cost = self.cost_for(served or model, usage)
        first_ms = None
        if first_chunk_at and started_at:
            first_ms = round((float(first_chunk_at) - float(started_at)) * 1000)
        text = ((response or {}).get("assistant_message") or {}).get("content")
        with self._lock:
            turn["llm_calls"] += 1
            if served and served not in turn["models"]:
                turn["models"].append(served)
            turn["prompt_tokens"] += prompt_tokens or 0
            turn["completion_tokens"] += output_tokens or 0
            if cost is not None:
                turn["cost"] += cost
                turn["cost_known"] = True
            if turn["first_token_ms"] is None and first_ms is not None:
                turn["first_token_ms"] = round((float(started_at) - turn["started"]) * 1000) + first_ms
        self.log.enqueue("llm", {
            "databricks_request_id": clip(api_request_id), "model_response_id": None,
            "email": self.email_for(session_id, turn["sender_id"]), "status_code": 200,
            "request_time": now_iso(), "input_token_count": prompt_tokens, "output_token_count": output_tokens,
            "input_character_count": len(turn["user_message"]),
            "output_character_count": _num(assistant_content_chars, int),
            "endpoint_name": clip(served), "execution_duration_ms": _num((api_duration or 0) * 1000, int),
            "request": clip(turn["user_message"]), "response": clip(text), "error": None,
            "chat_id": clip(session_id), "turn_id": clip(turn_id), "model_requested": clip(model),
            "execution_mode": EXECUTION_MODE, "attempt": 1, "cost_usd": cost, "first_token_ms": first_ms,
            "finish_reason": clip(finish_reason), "tool_call_count": _num(assistant_tool_call_count, int),
            "degenerate": False,
        })

    def on_api_request_error(self, session_id="", turn_id="", api_request_id="", model="", status_code=None,
                             retry_count=None, error=None, reason=None, started_at=None, ended_at=None, **_):
        turn = self._turn(session_id, turn_id)
        message = (error or {}).get("message") if isinstance(error, dict) else error
        message = message or reason or "LLM request failed"
        with self._lock:
            turn["error"] = clip(message, 4000)
        duration = (float(ended_at) - float(started_at)) * 1000 if ended_at and started_at else None
        self.log.enqueue("llm", {
            "databricks_request_id": clip(api_request_id), "email": self.email_for(session_id, turn["sender_id"]),
            "status_code": _num(status_code, int), "request_time": now_iso(),
            "input_character_count": len(turn["user_message"]), "endpoint_name": clip(model),
            "execution_duration_ms": _num(duration, int), "request": clip(turn["user_message"]),
            "error": clip(message), "chat_id": clip(session_id), "turn_id": clip(turn_id),
            "model_requested": clip(model), "execution_mode": EXECUTION_MODE,
            "attempt": (_num(retry_count, int) or 0) + 1, "degenerate": False,
        })

    def on_post_tool_call(self, tool_name="", args=None, result=None, session_id="", turn_id="", duration_ms=None,
                          status=None, error_message=None, **_):
        if tool_name != TOOL_NAME:
            return
        turn = self._turn(session_id, turn_id)
        text = result if isinstance(result, str) else ("" if result is None else str(result))
        outcome = "ok"
        if text.startswith("Execution stopped by the user"):
            outcome = "cancelled"
        elif text.startswith("ERROR: execution exceeded the time limit"):
            outcome = "timeout"
        elif text.startswith("ERROR") or status == "error":
            outcome = "error"
        seconds = (_num(duration_ms, float) or 0.0) / 1000.0
        with self._lock:
            turn["tool_calls"] += 1
            turn["steps"] += 1
            turn["tool_seconds"] += seconds
            step = turn["steps"]
        self.log.enqueue("tool", {
            "chat_id": clip(session_id), "turn_id": clip(turn_id),
            "email": self.email_for(session_id, turn["sender_id"]),
            "model": clip(",".join(turn["models"]) or turn.get("model_requested")), "execution_mode": EXECUTION_MODE,
            "step": step, "started_at": now_iso(), "duration_ms": _num(duration_ms, int), "status": outcome,
            "code": clip((args or {}).get("code"), 8000), "output_chars": len(text),
            "error": clip(text if outcome != "ok" else error_message, 4000),
            "chart": False, "export": "to this chat's files" in text,
        })

    def _finish(self, session_id: str, turn_id: str, *, response: str = "", stopped: bool = False,
                error: Optional[str] = None) -> None:
        turn = self._pop_turn(session_id, turn_id)
        if turn is None:
            return
        self.log.enqueue("turn", {
            "chat_id": clip(session_id), "turn_id": clip(turn_id),
            "email": self.email_for(session_id, turn["sender_id"]), "prompt_name": self.prompt_name,
            "execution_mode": EXECUTION_MODE, "started_at": turn["started_at"],
            "duration_ms": int((self.clock() - turn["started"]) * 1000), "first_token_ms": turn["first_token_ms"],
            "models": ",".join(turn["models"]) or turn.get("model_requested"), "model_reason": "default",
            "llm_calls": turn["llm_calls"], "tool_calls": turn["tool_calls"],
            "tool_seconds": round(turn["tool_seconds"], 3), "prompt_tokens": turn["prompt_tokens"],
            "completion_tokens": turn["completion_tokens"],
            "cost_usd": round(turn["cost"], 6) if turn["cost_known"] else None,
            "stopped": stopped, "degenerate_count": 0, "error": clip(error or turn["error"], 4000),
            "user_message": clip(turn["user_message"], 4000), "response_chars": len(response or ""),
        })

    def on_post_llm_call(self, session_id="", turn_id="", assistant_response="", **_):
        self._finish(session_id, turn_id, response=assistant_response or "")

    def on_session_end(self, session_id="", turn_id="", completed=True, failed=False, interrupted=False, **_):
        # Successful turns were already written by post_llm_call; this catches the rest.
        self._finish(session_id, turn_id, stopped=bool(interrupted),
                     error=("The turn failed" if failed else None))

    def register(self, ctx) -> None:
        ctx.register_hook("pre_llm_call", self.on_pre_llm_call)
        ctx.register_hook("post_api_request", self.on_post_api_request)
        ctx.register_hook("api_request_error", self.on_api_request_error)
        ctx.register_hook("post_tool_call", self.on_post_tool_call)
        ctx.register_hook("post_llm_call", self.on_post_llm_call)
        ctx.register_hook("on_session_end", self.on_session_end)
