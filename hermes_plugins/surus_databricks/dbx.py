"""Minimal Databricks REST client (httpx only).

Mercury's image is sealed (no pip), so this replaces databricks-sdk for the few
calls the chatbot needs: cluster state/start, Command Execution API 1.2 contexts
and commands, and the Files API for Unity Catalog Volumes.

Auth matches the old chatbot: a personal access token (DATABRICKS_TOKEN), or a
service principal via OAuth M2M (DATABRICKS_CLIENT_ID / DATABRICKS_CLIENT_SECRET).
"""
from __future__ import annotations

import os
import threading
import time
import urllib.parse
from typing import Any, Mapping, Optional

import httpx


class DatabricksError(RuntimeError):
    def __init__(self, message: str, *, status: Optional[int] = None, code: str = ""):
        super().__init__(message)
        self.status = status
        self.code = code


def _env(env: Mapping[str, str], name: str) -> str:
    return (env.get(name) or "").strip()


class DatabricksClient:
    def __init__(self, host: str, *, token: str = "", client_id: str = "", client_secret: str = "",
                 http: Optional[httpx.Client] = None, timeout: float = 60.0):
        host = host.strip().rstrip("/")
        if host and not host.startswith(("https://", "http://")):
            host = "https://" + host
        self.host = host
        self._token = token
        self._client_id = client_id
        self._client_secret = client_secret
        self._http = http or httpx.Client(timeout=timeout)
        self._oauth_token = ""
        self._oauth_expires = 0.0
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ, **kwargs) -> Optional["DatabricksClient"]:
        """A client from DATABRICKS_* variables, or None when they are incomplete."""
        host = _env(env, "DATABRICKS_HOST")
        token = _env(env, "DATABRICKS_TOKEN")
        cid, secret = _env(env, "DATABRICKS_CLIENT_ID"), _env(env, "DATABRICKS_CLIENT_SECRET")
        if not host or not (token or (cid and secret)):
            return None
        # Service principal wins when both are set, as in the old chatbot.
        if cid and secret:
            return cls(host, client_id=cid, client_secret=secret, **kwargs)
        return cls(host, token=token, **kwargs)

    # ── auth ──────────────────────────────────────────────────────────────────
    def _bearer(self) -> str:
        if not self._client_id:
            return self._token
        with self._lock:
            if self._oauth_token and time.time() < self._oauth_expires - 60:
                return self._oauth_token
            resp = self._http.post(
                f"{self.host}/oidc/v1/token",
                data={"grant_type": "client_credentials", "scope": "all-apis"},
                auth=(self._client_id, self._client_secret),
            )
            if resp.status_code != 200:
                raise DatabricksError(f"OAuth token request failed (HTTP {resp.status_code})",
                                      status=resp.status_code)
            body = resp.json()
            self._oauth_token = body["access_token"]
            self._oauth_expires = time.time() + float(body.get("expires_in") or 3600)
            return self._oauth_token

    # ── transport ─────────────────────────────────────────────────────────────
    def request(self, method: str, path: str, *, params: Optional[dict] = None, json: Any = None,
                content: Optional[bytes] = None, raw: bool = False) -> Any:
        headers = {"Authorization": f"Bearer {self._bearer()}"}
        resp = self._http.request(method, f"{self.host}{path}", params=params, json=json,
                                  content=content, headers=headers)
        if resp.status_code >= 400:
            code, message = "", resp.text[:500]
            try:
                body = resp.json()
                code = str(body.get("error_code") or "")
                message = str(body.get("message") or body.get("error") or message)
            except Exception:
                pass
            raise DatabricksError(f"{method} {path}: HTTP {resp.status_code} {code} {message}".strip(),
                                  status=resp.status_code, code=code)
        if raw:
            return resp.content
        if not resp.content:
            return {}
        return resp.json()

    # ── clusters ──────────────────────────────────────────────────────────────
    def cluster_state(self, cluster_id: str) -> Optional[str]:
        try:
            return self.request("GET", "/api/2.1/clusters/get", params={"cluster_id": cluster_id}).get("state")
        except DatabricksError:
            return None

    def start_cluster(self, cluster_id: str) -> None:
        try:
            self.request("POST", "/api/2.1/clusters/start", json={"cluster_id": cluster_id})
        except DatabricksError as exc:
            # Already starting/running: someone else got there first.
            if exc.code != "INVALID_STATE":
                raise

    # ── Command Execution API 1.2 ─────────────────────────────────────────────
    def create_context(self, cluster_id: str) -> str:
        body = self.request("POST", "/api/1.2/contexts/create",
                            json={"clusterId": cluster_id, "language": "python"})
        return body["id"]

    def context_status(self, cluster_id: str, context_id: str) -> Optional[str]:
        """'Running' / 'Pending' / 'Error', or None when the context is gone."""
        try:
            return self.request("GET", "/api/1.2/contexts/status",
                                params={"clusterId": cluster_id, "contextId": context_id}).get("status")
        except DatabricksError:
            return None

    def destroy_context(self, cluster_id: str, context_id: str) -> None:
        try:
            self.request("POST", "/api/1.2/contexts/destroy",
                         json={"clusterId": cluster_id, "contextId": context_id})
        except DatabricksError:
            pass

    def execute(self, cluster_id: str, context_id: str, command: str) -> str:
        body = self.request("POST", "/api/1.2/commands/execute", json={
            "clusterId": cluster_id, "contextId": context_id, "language": "python", "command": command})
        return body["id"]

    def command_status(self, cluster_id: str, context_id: str, command_id: str) -> dict:
        return self.request("GET", "/api/1.2/commands/status", params={
            "clusterId": cluster_id, "contextId": context_id, "commandId": command_id})

    def cancel(self, cluster_id: str, context_id: str, command_id: str) -> None:
        try:
            self.request("POST", "/api/1.2/commands/cancel", json={
                "clusterId": cluster_id, "contextId": context_id, "commandId": command_id})
        except DatabricksError:
            pass

    # ── Files API (Unity Catalog Volumes) ─────────────────────────────────────
    @staticmethod
    def _fs_path(path: str) -> str:
        if not path.startswith("/Volumes/"):
            raise ValueError(f"not a Unity Catalog Volume path: {path!r}")
        return urllib.parse.quote(path)

    def download(self, path: str) -> bytes:
        return self.request("GET", "/api/2.0/fs/files" + self._fs_path(path), raw=True)

    def file_url(self, path: str) -> tuple[str, dict]:
        """URL and auth header for downloading a Volume file with another (streaming) client."""
        return f"{self.host}/api/2.0/fs/files{self._fs_path(path)}", {"Authorization": f"Bearer {self._bearer()}"}

    def close(self) -> None:
        self._http.close()

    def upload(self, path: str, data: bytes) -> None:
        self.request("PUT", "/api/2.0/fs/files" + self._fs_path(path),
                     params={"overwrite": "true"}, content=data)

    def list_directory(self, path: str) -> list[dict]:
        """Entries of a Volume directory ([] when it does not exist yet)."""
        entries, token = [], None
        while True:
            try:
                body = self.request("GET", "/api/2.0/fs/directories" + self._fs_path(path),
                                    params={"page_token": token} if token else None)
            except DatabricksError as exc:
                if exc.status == 404:
                    return []
                raise
            entries.extend(body.get("contents") or [])
            token = body.get("next_page_token")
            if not token:
                return entries
