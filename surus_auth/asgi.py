"""Universal ASGI adapter: one middleware for every async framework.

FastAPI, Starlette, Quart, Litestar, Sanic-ASGI, async Django -- if it speaks
ASGI, this gates it. Like `wsgi.py`, the whole file is protocol translation
over `_engine.AuthEngine`; nothing here knows what FastAPI is.

    from surus_auth.asgi import protect

    app = FastAPI()
    app = protect(app)                       # wrap it last, after the routes

    # Starlette / Litestar middleware stack:
    from surus_auth.asgi import AuthMiddleware
    app.add_middleware(AuthMiddleware)

Read the signed-in user downstream:

    from surus_auth.asgi import current_user
    user = current_user(request.scope)       # {email, name, ..., role} or None

FastAPI apps that prefer dependency injection can keep using
`fastapi.get_current_user` / `require_admin`, which are unchanged.

Two notes specific to ASGI:

* `websocket` and `lifespan` scopes pass straight through. Gating lifespan
  would stop the app booting; gating websockets here would break Streamlit,
  Dash and anything else whose transport is a socket, and the socket is
  opened by a page that is already behind the guard.
* Google's token exchange is two blocking HTTPS calls. They run in a thread
  so the event loop is not stalled for up to 20 seconds per sign-in.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Callable, Dict, Iterable, Optional

from ._engine import AuthEngine, Response

__all__ = ["protect", "AuthMiddleware", "current_user"]

SCOPE_KEY = "surus_auth.user"


def current_user(scope: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The signed-in user for this request, or None.

    Populated by the middleware before the app runs; it does not re-verify
    the cookie.
    """
    return scope.get(SCOPE_KEY)


def _cookie_header(scope: Dict[str, Any]) -> str:
    for key, value in scope.get("headers") or ():
        if key == b"cookie":
            return value.decode("latin-1")
    return ""


class AuthMiddleware:
    """ASGI middleware that gates an application with Google sign-in.

    Takes the same keyword arguments as `AuthEngine`; see its docstring.
    `public_url` defaults to $PUBLIC_URL, then $APP_URL.
    """

    def __init__(
        self,
        app: Callable,
        *,
        public_url: Optional[str] = None,
        session_secret: Optional[str] = None,
        announce: bool = True,
        **engine_kwargs: Any,
    ):
        self.app = app
        public_url = (
            public_url
            or os.environ.get("PUBLIC_URL")
            or os.environ.get("APP_URL")
            or "http://localhost:8000"
        )
        self.engine = AuthEngine(
            public_url=public_url,
            session_secret=session_secret or os.environ.get("SESSION_SECRET", ""),
            **engine_kwargs,
        )
        if announce:
            self.engine.announce()

    async def __call__(self, scope: Dict[str, Any], receive: Callable, send: Callable):
        # lifespan must reach the app or it never boots; websockets are opened
        # by a page that already passed the guard.
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)

        cookie_header = _cookie_header(scope)
        # handle() does a blocking token exchange on the callback path only.
        resp = await asyncio.to_thread(
            self.engine.handle,
            scope.get("method", "GET"),
            scope.get("path", "/") or "/",
            (scope.get("query_string") or b"").decode("latin-1"),
            cookie_header,
        )
        if resp is not None:
            return await self._send(resp, send)

        scope[SCOPE_KEY] = self.engine.user_from_cookies(cookie_header)
        return await self.app(scope, receive, send)

    @staticmethod
    async def _send(resp: Response, send: Callable):
        # ASGI requires lowercase header names. Servers vary in how tolerant
        # they are of "Location", and the ones that are not simply drop it,
        # which turns every redirect into a blank 302.
        headers = [
            (k.lower().encode("latin-1"), v.encode("latin-1"))
            for k, v in resp.headers
        ]
        headers.append((b"content-length", str(len(resp.body)).encode()))
        await send({
            "type": "http.response.start",
            "status": resp.status,
            "headers": headers,
        })
        await send({"type": "http.response.body", "body": resp.body})


def protect(
    app: Callable,
    *,
    public_url: Optional[str] = None,
    session_secret: Optional[str] = None,
    cookie_name: str = "surus_session",
    public_prefixes: Iterable[str] = (),
    framework: Optional[str] = None,
    **kwargs: Any,
) -> AuthMiddleware:
    """Wrap an ASGI app so every request needs a signed-in Google account.

    framework       "streamlit", "gradio", … to leave that framework's own
                    asset routes ungated. See _engine.FRAMEWORK_PREFIXES.
    public_prefixes Extra path prefixes that skip the guard.
    cookie_name     Rename per app, so two apps on one host do not collide.
    """
    return AuthMiddleware(
        app,
        public_url=public_url,
        session_secret=session_secret,
        cookie_name=cookie_name,
        public_prefixes=public_prefixes,
        framework=framework,
        **kwargs,
    )
