"""A fake Databricks workspace for the Surus plugin tests (httpx.MockTransport).

Covers the calls hermes_plugins/surus_databricks/dbx.py makes: cluster
get/start, Command Execution API 1.2 contexts and commands, the Files API
and the OAuth M2M token endpoint. Commands "run" for `polls_to_finish`
status polls, then finish with whatever `respond(command)` returns.
"""
from __future__ import annotations

import json
import urllib.parse
from typing import Callable, Optional

import httpx


class FakeDatabricks:
    def __init__(self, *, cluster_state: str = "RUNNING", polls_to_finish: int = 1,
                 respond: Optional[Callable[[str], dict]] = None):
        self.cluster_state = cluster_state
        self.polls_to_finish = polls_to_finish
        self.respond = respond or (lambda command: {"resultType": "text", "data": "ok"})
        self.contexts: dict[str, str] = {}          # id -> status
        self.commands: dict[str, dict] = {}         # id -> {command, polls, status, cancelled}
        self.files: dict[str, bytes] = {}
        self.calls: list[tuple[str, str]] = []
        self.executed: list[str] = []
        self.cancelled: list[str] = []
        self.started_cluster = 0
        self.token_requests = 0
        self.auth_headers: list[str] = []
        self._next = 0

    def _id(self, prefix: str) -> str:
        self._next += 1
        return f"{prefix}{self._next}"

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def client(self, **kwargs):
        from hermes_plugins.surus_databricks.dbx import DatabricksClient
        kwargs.setdefault("token", "dapi-test")
        return DatabricksClient("https://adb-test.azuredatabricks.net",
                                http=httpx.Client(transport=self.transport()), **kwargs)

    # ── request handling ──────────────────────────────────────────────────────
    def handle(self, request: httpx.Request) -> httpx.Response:
        path = urllib.parse.unquote(request.url.path)
        self.calls.append((request.method, path))
        if path == "/oidc/v1/token":
            self.token_requests += 1
            return httpx.Response(200, json={"access_token": f"oauth-{self.token_requests}", "expires_in": 3600})
        self.auth_headers.append(request.headers.get("authorization", ""))
        params = dict(request.url.params)
        body = json.loads(request.content) if request.content and request.method == "POST" else {}

        if path == "/api/2.1/clusters/get":
            return httpx.Response(200, json={"state": self.cluster_state})
        if path == "/api/2.1/clusters/start":
            self.started_cluster += 1
            self.cluster_state = "RUNNING"
            return httpx.Response(200, json={})
        if path == "/api/1.2/contexts/create":
            cid = self._id("ctx")
            self.contexts[cid] = "Pending"
            return httpx.Response(200, json={"id": cid})
        if path == "/api/1.2/contexts/status":
            cid = params["contextId"]
            if cid not in self.contexts:
                return httpx.Response(404, json={"error_code": "RESOURCE_DOES_NOT_EXIST",
                                                 "message": f"context {cid} not found"})
            status = self.contexts[cid]
            if status == "Pending":
                self.contexts[cid] = "Running"
            return httpx.Response(200, json={"id": cid, "status": status})
        if path == "/api/1.2/contexts/destroy":
            self.contexts.pop(body["contextId"], None)
            return httpx.Response(200, json={})
        if path == "/api/1.2/commands/execute":
            if body["contextId"] not in self.contexts:
                return httpx.Response(400, json={"error_code": "INVALID_PARAMETER_VALUE",
                                                 "message": "context does not exist"})
            cmd = self._id("cmd")
            self.commands[cmd] = {"command": body["command"], "polls": 0, "cancelled": False}
            self.executed.append(body["command"])
            return httpx.Response(200, json={"id": cmd})
        if path == "/api/1.2/commands/status":
            cmd = self.commands[params["commandId"]]
            cmd["polls"] += 1
            if cmd["cancelled"]:
                return httpx.Response(200, json={"status": "Cancelled", "results": {"data": "partial"}})
            if cmd["polls"] <= self.polls_to_finish:
                return httpx.Response(200, json={"status": "Running"})
            results = self.respond(cmd["command"])
            status = "Error" if results.get("resultType") == "error" else "Finished"
            return httpx.Response(200, json={"status": status, "results": results})
        if path == "/api/1.2/commands/cancel":
            self.cancelled.append(body["commandId"])
            self.commands[body["commandId"]]["cancelled"] = True
            return httpx.Response(200, json={})
        if path.startswith("/api/2.0/fs/files/"):
            fpath = path[len("/api/2.0/fs/files"):]
            if request.method == "PUT":
                self.files[fpath] = request.content
                return httpx.Response(204)
            if fpath not in self.files:
                return httpx.Response(404, json={"error_code": "NOT_FOUND", "message": "no such file"})
            return httpx.Response(200, content=self.files[fpath])
        if path.startswith("/api/2.0/fs/directories/"):
            dpath = path[len("/api/2.0/fs/directories"):].rstrip("/") + "/"
            entries = [{"path": p, "name": p.rsplit("/", 1)[-1], "is_directory": False,
                        "file_size": len(v)} for p, v in self.files.items() if p.startswith(dpath)]
            if not entries:
                return httpx.Response(404, json={"error_code": "NOT_FOUND", "message": "no such directory"})
            return httpx.Response(200, json={"contents": entries})
        return httpx.Response(404, json={"error_code": "ENDPOINT_NOT_FOUND", "message": path})
