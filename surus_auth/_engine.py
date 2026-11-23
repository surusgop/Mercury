"""The framework-free half of every adapter.

`flask.py` and `fastapi.py` each re-implemented the same four things against a
different API: parse a cookie, run the login/callback/logout endpoints, guard
everything else, and hand the app a user dict. That is why adding Streamlit
meant writing a third copy.

This module is that logic expressed against nothing at all. It speaks in
plain strings and bytes:

    engine = AuthEngine(public_url=..., session_secret=...)
    resp = engine.handle(method, path, query_string, cookie_header)
    if resp is None:
        ...  # let the real app serve this request
    else:
        ...  # send resp.status / resp.headers / resp.body

`wsgi.py` and `asgi.py` are ~120 lines each of pure protocol translation on
top of this. Anything that can report a method, a path and a Cookie header can
be gated by it, including frameworks neither of us has heard of.

Dependency-free on purpose: signing goes through the HMAC helpers in
`surus_auth`, not itsdangerous, so the universal adapters pull in nothing.
"""

from __future__ import annotations

import http.cookies
import urllib.parse
import uuid
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Tuple

from . import (
    AuthError,
    authorization_url,
    deserialize_session,
    exchange_code,
    get_config,
    is_email_allowed,
    new_state,
    role_for,
    serialize_session,
    warn_if_unrestricted,
)

__all__ = ["AuthEngine", "Response", "FRAMEWORK_PREFIXES"]

_STATE_COOKIE = "surus_oauth_state"
_STATE_MAX_AGE = 600  # 10 minutes to get through Google's screens

# Frameworks that serve their own machinery under a fixed prefix. Gating these
# breaks the page before the user can ever reach a login link.
FRAMEWORK_PREFIXES = {
    "dash":      ("/_", "/assets/"),
    "streamlit": ("/_stcore", "/static/", "/media/", "/component/"),
    "gradio":    ("/gradio_api", "/theme.css", "/assets/", "/static/"),
    "django":    ("/static/", "/media/"),
}


class Response(NamedTuple):
    """What to send instead of calling the app. Protocol-neutral."""

    status: int
    headers: List[Tuple[str, str]]
    body: bytes = b""


def _safe_next(value: Optional[str]) -> str:
    """Only accept a relative path, so `next` cannot become an open redirect.

    "//evil.com" is rejected alongside "https://evil.com" -- a protocol-
    relative URL is an off-site redirect that happens to start with a slash.
    """
    if not value or not value.startswith("/") or value.startswith("//"):
        return "/"
    return value


def _parse_cookies(header: str) -> Dict[str, str]:
    """Cookie header -> dict. Never raises; the value comes from the browser."""
    if not header:
        return {}
    try:
        parsed = http.cookies.SimpleCookie()
        parsed.load(header)
        return {key: morsel.value for key, morsel in parsed.items()}
    except Exception:
        # A malformed header is a signed-out user, not a 500.
        return {}


def _set_cookie(name, value, *, max_age: int, secure: bool) -> Tuple[str, str]:
    parts = [f"{name}={value}", "Path=/", f"Max-Age={max_age}", "HttpOnly", "SameSite=Lax"]
    if secure:
        parts.append("Secure")
    return ("Set-Cookie", "; ".join(parts))


def _clear_cookie(name: str, *, secure: bool) -> Tuple[str, str]:
    return _set_cookie(name, "", max_age=0, secure=secure)


def _html(status: int, message: str) -> Response:
    body = (
        "<!doctype html><meta charset=utf-8><title>Sign-in</title>"
        "<body style=\"font-family:system-ui,sans-serif;max-width:34rem;"
        "margin:20vh auto;padding:0 1.5rem;line-height:1.6\">"
        "<p>" + message + "</p></body>"
    ).encode()
    return Response(status, [("Content-Type", "text/html; charset=utf-8")], body)


class AuthEngine:
    """Google sign-in as a pure request -> optional-response function.

    public_url      Where the browser reaches the app. redirect_uri is
                    public_url + callback_path and must match the Google
                    console byte-for-byte.
    session_secret  Unique per app. A shared one lets a cookie minted by one
                    app authenticate against this one.
    framework       Key into FRAMEWORK_PREFIXES, to leave that framework's own
                    asset/transport routes ungated. None for a plain app.
    on_login        Called with the user dict after a successful sign-in.
                    Exceptions are swallowed -- analytics must never be able
                    to block a login.
    role_resolver   Override role lookup, e.g. to read a database instead of
                    ADMIN_BOOTSTRAP_EMAILS. Called per request with the email.
    """

    def __init__(
        self,
        *,
        public_url: str,
        session_secret: str = "",
        cookie_name: str = "surus_session",
        session_max_age: int = 30 * 24 * 3600,
        login_path: str = "/auth/login",
        callback_path: str = "/auth/callback",
        logout_path: str = "/auth/logout",
        public_prefixes: Iterable[str] = (),
        framework: Optional[str] = None,
        healthz: bool = True,
        healthz_path: str = "/healthz",
        on_login: Optional[Callable[[Dict[str, Any]], None]] = None,
        role_resolver: Optional[Callable[[str], str]] = None,
        login_prompt: Optional[str] = "select_account",
    ):
        self.public_url = public_url.rstrip("/")
        self.cookie_name = cookie_name
        self.session_max_age = session_max_age
        self.login_path = login_path
        self.callback_path = callback_path
        self.logout_path = logout_path
        self.healthz = healthz
        self.healthz_path = healthz_path
        self.on_login = on_login
        self.role_resolver = role_resolver
        self.login_prompt = login_prompt

        self.redirect_uri = self.public_url + callback_path
        self.secure = self.public_url.startswith("https://")
        self.enabled = get_config().is_configured

        # The signing helpers read the secret from the config, so an
        # explicitly passed one has to be installed there to take effect.
        if session_secret:
            get_config().session_secret = session_secret
        if not get_config().session_secret:
            raise RuntimeError(
                "SESSION_SECRET is not set, so the session cookie cannot be "
                "signed. Generate one with: "
                "python -c \"import secrets; print(secrets.token_urlsafe(48))\""
            )

        self._skip = tuple(public_prefixes) + tuple(
            FRAMEWORK_PREFIXES.get(framework or "", ())
        )

    # -- introspection -----------------------------------------------------

    def user_from_cookies(self, cookie_header: str) -> Optional[Dict[str, Any]]:
        """The signed-in user, or None. Never raises on a hostile cookie."""
        raw = _parse_cookies(cookie_header).get(self.cookie_name)
        if not raw:
            return None
        data = deserialize_session(raw, max_age=self.session_max_age)
        if not data or not data.get("email"):
            return None
        data = dict(data)
        # Resolved per request rather than stored in the cookie, so promoting
        # someone takes effect without them signing out.
        data["role"] = self.resolve_role(data["email"])
        return data

    def resolve_role(self, email: str) -> str:
        if self.role_resolver is not None:
            try:
                return self.role_resolver(email)
            except Exception as exc:
                # A database hiccup must not escalate or crash the request.
                print(f"[auth] role_resolver failed for {email}: {exc!r}")
                return "member"
        return role_for(email)

    def announce(self) -> None:
        """Print the startup banner. Neither value printed is a secret."""
        warn_if_unrestricted()
        if self.public_url.startswith("http://") and not any(
            h in self.public_url for h in ("localhost", "127.0.0.1", "0.0.0.0")
        ):
            print(f"[auth] WARNING: {self.public_url} is http:// on a public "
                  "host. Google will reject the redirect URI and the session "
                  "cookie will not be marked secure. Use https://.")
        print(f"[auth] client id:    {get_config().client_id or '(not set)'}")
        print(f"[auth] redirect URI: {self.redirect_uri}")
        print("[auth]   ^ must be registered byte-identically on THAT client "
              "in the Google console, or sign-in fails with "
              "redirect_uri_mismatch")

    # -- the request hook --------------------------------------------------

    def handle(
        self,
        method: str,
        path: str,
        query_string: str,
        cookie_header: str,
    ) -> Optional[Response]:
        """None means "let the app serve it". A Response means "send this"."""
        if self.healthz and path == self.healthz_path:
            # Deliberately ungated -- a platform probe cannot sign in, and a
            # gated probe path fails every deploy.
            return Response(200, [("Content-Type", "application/json")], b'{"ok":true}')

        if not self.enabled:
            return None  # no client id/secret: the app runs unprotected

        query = urllib.parse.parse_qs(query_string or "")
        cookies = _parse_cookies(cookie_header)

        if path == self.login_path:
            return self._login(query)
        if path == self.callback_path:
            return self._callback(query, cookies)
        if path == self.logout_path:
            return self._logout()

        if self._skip and path.startswith(self._skip):
            return None

        # The guard.
        raw = cookies.get(self.cookie_name)
        if raw and deserialize_session(raw, max_age=self.session_max_age):
            return None

        full = path + ("?" + query_string if query_string else "")
        login_url = self.login_path + "?" + urllib.parse.urlencode({"next": full})
        headers = [("Location", login_url)]
        if raw:
            # Expired or tampered -- clear it, or the browser re-sends it
            # forever and the user ping-pongs between here and Google.
            headers.append(_clear_cookie(self.cookie_name, secure=self.secure))
        return Response(302, headers)

    # -- the three endpoints -----------------------------------------------

    def _login(self, query: Dict[str, List[str]]) -> Response:
        state = new_state()
        next_path = _safe_next((query.get("next") or [""])[0])
        token = serialize_session({"_k": "state", "state": state, "next": next_path})
        return Response(
            302,
            [
                ("Location", authorization_url(
                    self.redirect_uri, state, prompt=self.login_prompt)),
                _set_cookie(_STATE_COOKIE, token,
                            max_age=_STATE_MAX_AGE, secure=self.secure),
            ],
        )

    def _callback(
        self,
        query: Dict[str, List[str]],
        cookies: Dict[str, str],
    ) -> Response:
        error = (query.get("error") or [""])[0]
        if error:
            return _html(400, f"Authentication error: {error}")

        # The state must round-trip through our own cookie, or a third party
        # could hand the user a callback URL of their choosing.
        state_token = cookies.get(_STATE_COOKIE, "")
        state_param = (query.get("state") or [""])[0]
        state_data = deserialize_session(state_token, max_age=_STATE_MAX_AGE) or {}
        if (
            not state_param
            or state_data.get("_k") != "state"
            or state_param != state_data.get("state")
        ):
            return _html(400, "Invalid or expired login request. Please try again.")
        next_path = _safe_next(state_data.get("next"))

        # exchange_code rather than the one-call sign_in, because apps that log
        # usage want given_name / family_name and only the raw profile has
        # them. The checks below are the ones sign_in makes.
        try:
            info = exchange_code((query.get("code") or [""])[0], self.redirect_uri)
        except AuthError as exc:
            print(f"[auth] sign-in failed: {exc}")
            return _html(400, "Sign-in failed. Please try again.")

        email = (info.get("email") or "").strip()
        if not email or not info.get("email_verified", True):
            return _html(403, "That Google account has no verified email address.")
        if not is_email_allowed(email):
            print(f"[auth] denied: {email}")
            return _html(
                403,
                f"Access denied: only {get_config().describe()} accounts "
                "are allowed to use this app.",
            )

        user = {
            "email":      email,
            "name":       info.get("name") or email.split("@")[0],
            "first_name": info.get("given_name", ""),
            "last_name":  info.get("family_name", ""),
            "picture":    info.get("picture", ""),
            "session_id": str(uuid.uuid4()),
        }
        print(f"[auth] login: {email}  session={user['session_id']}")

        if self.on_login is not None:
            try:
                self.on_login(user)
            except Exception as exc:  # never let analytics block a login
                print(f"[auth] on_login hook failed: {exc}")

        return Response(
            302,
            [
                ("Location", next_path),
                _set_cookie(self.cookie_name, serialize_session(user),
                            max_age=self.session_max_age, secure=self.secure),
                _clear_cookie(_STATE_COOKIE, secure=self.secure),
            ],
        )

    def _logout(self) -> Response:
        # The guard bounces the next request to login, which forces Google's
        # account picker so a different account can sign in.
        return Response(
            302,
            [("Location", "/"), _clear_cookie(self.cookie_name, secure=self.secure)],
        )
