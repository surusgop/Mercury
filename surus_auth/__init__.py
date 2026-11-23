"""Google sign-in with a multi-domain allowlist.

The core of surus-auth. It has **no dependencies** beyond the
standard library and knows nothing about Flask, FastAPI, Gradio or Django --
you wire the three calls into whatever framework the app already uses.

    ALLOWED_EMAIL_DOMAIN=surusenterprises.com,americanstewardship.org

Adding a domain is an environment-variable change, not a code change.

--------------------------------------------------------------------------
The flow, in three calls
--------------------------------------------------------------------------

1. Your /auth/login route:

       state = new_state()                  # store it in the user's session
       return redirect(authorization_url(redirect_uri, state))

2. Your /auth/callback route:

       # compare the returned state against the one you stored, THEN:
       user = sign_in(code, redirect_uri)   # raises AccessDenied
       # user == {"email", "name", "picture", "role"}

3. Every other route: check the session you set in step 2.

`serialize_session` / `deserialize_session` are here if you want a signed
cookie and don't already have one.

--------------------------------------------------------------------------
Environment variables
--------------------------------------------------------------------------

    GOOGLE_CLIENT_ID        required
    GOOGLE_CLIENT_SECRET    required
    ALLOWED_EMAIL_DOMAIN    comma-separated domains, e.g. "a.com,b.org"
    ALLOWED_EMAILS          comma-separated individual addresses (optional)
    ADMIN_BOOTSTRAP_EMAILS  comma-separated addresses given role="admin"
    SESSION_SECRET          required only if you use the cookie helpers

A user is admitted if they match ALLOWED_EMAIL_DOMAIN **or** ALLOWED_EMAILS.
Leaving both empty admits anyone with a Google account -- call
`warn_if_unrestricted()` at startup so that is never a silent accident.

--------------------------------------------------------------------------
Two gotchas this file exists to encode
--------------------------------------------------------------------------

* The `hd` parameter is sent only when exactly ONE domain is allowed. With
  two, `hd` makes Google turn the second one away at its own screen, before
  your callback ever runs.

* `Error 403: org_internal` is not this code. It means the OAuth consent
  screen's User type is *Internal*, which Google enforces as "same Workspace
  org as the Cloud project". Set it to External and publish the app. See the
  README.

--------------------------------------------------------------------------
Check the configuration without deploying
--------------------------------------------------------------------------

    python -m surus_auth                       # show resolved config
    python -m surus_auth someone@example.com   # would they be admitted?
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

__all__ = [
    "AccessDenied",
    "AuthError",
    "Config",
    "get_config",
    "reload_config",
    "warn_if_unrestricted",
    "is_email_allowed",
    "role_for",
    "new_state",
    "authorization_url",
    "exchange_code",
    "sign_in",
    "serialize_session",
    "deserialize_session",
]

AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
USERINFO_ENDPOINT = "https://openidconnect.googleapis.com/v1/userinfo"

HTTP_TIMEOUT = 10  # seconds


class AuthError(Exception):
    """Sign-in failed for a reason that is not the user's fault.

    Misconfiguration, or Google returned an error. Show a retry page; a 500 or
    503 is usually the right status.
    """


class AccessDenied(AuthError):
    """The account authenticated fine but is not on the allowlist.

    Carries `.email` so you can log who was turned away. Render `str(exc)` --
    it already names the domains that would have worked. Status 403.
    """

    def __init__(self, email: str, message: str):
        super().__init__(message)
        self.email = email


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def _csv_env(name: str, strip_at: bool = False) -> List[str]:
    """Parse a comma-separated variable. Tolerates spaces, empties and "@".

    "  a.com , @b.org ,, " -> ["a.com", "b.org"]
    """
    vals = [v.strip().lower() for v in os.getenv(name, "").split(",") if v.strip()]
    return [v.lstrip("@") for v in vals] if strip_at else vals


def _first_set(*names: str) -> tuple:
    """(name, values) for the first of these variables that is non-empty.

    Returns ("", []) when none are set. Deliberately first-wins rather than
    a union: a forgotten variable should lock people out, which is loud,
    never admit a domain someone thought they had removed.
    """
    for name in names:
        vals = _csv_env(name, strip_at=True)
        if vals:
            return name, vals
    return "", []


class Config:
    """Resolved auth settings. Build with `Config.from_env()`."""

    def __init__(
        self,
        client_id: str = "",
        client_secret: str = "",
        allowed_domains: Optional[List[str]] = None,
        allowed_emails: Optional[List[str]] = None,
        admin_emails: Optional[List[str]] = None,
        session_secret: str = "",
        domain_source: str = "",
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.allowed_domains = allowed_domains or []
        self.allowed_emails = allowed_emails or []
        self.admin_emails = admin_emails or []
        self.session_secret = session_secret
        self.domain_source = domain_source

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            client_id=os.getenv("GOOGLE_CLIENT_ID", "").strip(),
            client_secret=os.getenv("GOOGLE_CLIENT_SECRET", "").strip(),
            # Apps in this org deploy under three different names for the
            # same list. Honour all of them, or installing this package
            # silently finds no allowlist and the app falls OPEN on its next
            # deploy. First one that is set wins -- they are not merged, so a
            # stale variable can never widen the list behind your back.
            # domain_source records which one was used; it is logged at
            # startup so the choice is never silent.
            allowed_domains=_first_set(
                "ALLOWED_EMAIL_DOMAIN",      # the documented name
                "ALLOWED_EMAIL_DOMAINS",     # plural variant, some apps use it
                "GOOGLE_WORKSPACE_DOMAIN",   # older single-domain name
            )[1],
            domain_source=_first_set(
                "ALLOWED_EMAIL_DOMAIN", "ALLOWED_EMAIL_DOMAINS", "GOOGLE_WORKSPACE_DOMAIN"
            )[0],
            allowed_emails=_csv_env("ALLOWED_EMAILS"),
            admin_emails=_csv_env("ADMIN_BOOTSTRAP_EMAILS"),
            session_secret=os.getenv("SESSION_SECRET", ""),
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    @property
    def is_unrestricted(self) -> bool:
        return not self.allowed_domains and not self.allowed_emails

    def describe(self) -> str:
        if self.allowed_domains and self.allowed_emails:
            return (
                ", ".join("@" + d for d in self.allowed_domains)
                + f" (plus {len(self.allowed_emails)} individual address"
                + ("es)" if len(self.allowed_emails) != 1 else ")")
            )
        if self.allowed_domains:
            return " or ".join("@" + d for d in self.allowed_domains)
        if self.allowed_emails:
            return f"{len(self.allowed_emails)} individually allowed address(es)"
        return "ANY Google account (no restriction configured)"


_config: Optional[Config] = None


def get_config() -> Config:
    """The active config, read from the environment on first use.

    Lazy on purpose: apps that call `load_dotenv()` after importing this
    module would otherwise get a config built before the .env was loaded.
    """
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def reload_config(config: Optional[Config] = None) -> Config:
    """Re-read the environment, or install an explicit Config (handy in tests)."""
    global _config
    _config = config if config is not None else Config.from_env()
    return _config


def warn_if_unrestricted(logger: Any = None) -> bool:
    """Call at startup. Returns True if the app currently admits anyone.

    Pass a logger, or leave it out to print. An app that is open to the whole
    internet should say so out loud in its boot log.
    """
    cfg = get_config()
    if not cfg.is_configured:
        msg = "[auth] Google sign-in DISABLED - GOOGLE_CLIENT_ID/SECRET not set"
    elif cfg.is_unrestricted:
        msg = ("[auth] WARNING: no ALLOWED_EMAIL_DOMAIN set - "
               "ANY Google account can sign in")
    else:
        via = f" (via {cfg.domain_source})" if cfg.domain_source else ""
        msg = f"[auth] Sign-in restricted to {cfg.describe()}{via}"
    (logger.warning if logger else print)(msg)
    return cfg.is_configured and cfg.is_unrestricted


# --------------------------------------------------------------------------
# The allowlist
# --------------------------------------------------------------------------

def is_email_allowed(email: str, config: Optional[Config] = None) -> bool:
    """True if this address may sign in.

    Admitted when it is on an allowed domain OR on the individual allowlist.
    Both lists empty means no restriction, so everyone is admitted.

    The domain test matches on "@" + domain, never a bare suffix -- otherwise
    evil-surusenterprises.com would pass as surusenterprises.com.
    """
    cfg = config or get_config()
    e = (email or "").strip().lower()
    if not e or "@" not in e:
        return False
    if cfg.is_unrestricted:
        return True
    if any(e.endswith("@" + d) for d in cfg.allowed_domains):
        return True
    return e in cfg.allowed_emails


def role_for(email: str, config: Optional[Config] = None) -> str:
    """"admin" if the address is in ADMIN_BOOTSTRAP_EMAILS, else "member".

    Resolve this per request rather than storing it in the session, so
    promoting someone takes effect without them logging out.
    """
    cfg = config or get_config()
    return "admin" if (email or "").strip().lower() in cfg.admin_emails else "member"


# --------------------------------------------------------------------------
# The OAuth flow
# --------------------------------------------------------------------------

def new_state() -> str:
    """A CSRF token for the round trip. Store it, then compare on callback."""
    return secrets.token_urlsafe(24)


def authorization_url(
    redirect_uri: str,
    state: str,
    config: Optional[Config] = None,
    prompt: Optional[str] = None,
) -> str:
    """The Google URL to redirect the user to.

    `redirect_uri` must exactly match one registered in the Google console,
    trailing slash and http/https included, or Google returns
    Error 400: redirect_uri_mismatch.
    """
    cfg = config or get_config()
    if not cfg.is_configured:
        raise AuthError("Google OAuth is not configured "
                        "(GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET).")
    params = {
        "client_id": cfg.client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "online",
    }
    # Pre-select the Workspace domain ONLY when exactly one is allowed. With
    # two, "hd" would make Google block the second domain at its own screen,
    # before our callback could admit them.
    if len(cfg.allowed_domains) == 1 and not cfg.allowed_emails:
        params["hd"] = cfg.allowed_domains[0]
    if prompt:
        params["prompt"] = prompt
    return AUTH_ENDPOINT + "?" + urllib.parse.urlencode(params)


def _post_form(url: str, data: Dict[str, str]) -> Dict[str, Any]:
    body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        raise AuthError(f"Google token exchange failed ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise AuthError(f"Could not reach Google: {exc.reason}") from exc


def _get_json(url: str, bearer: str) -> Dict[str, Any]:
    req = urllib.request.Request(url)
    req.add_header("Authorization", "Bearer " + bearer)
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        raise AuthError(f"Could not fetch Google profile ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise AuthError(f"Could not reach Google: {exc.reason}") from exc


def exchange_code(
    code: str,
    redirect_uri: str,
    config: Optional[Config] = None,
) -> Dict[str, Any]:
    """Trade the callback's `code` for the user's Google profile.

    Returns Google's raw userinfo. No allowlist check -- use `sign_in` unless
    you specifically need the unfiltered profile.
    """
    cfg = config or get_config()
    if not cfg.is_configured:
        raise AuthError("Google OAuth is not configured "
                        "(GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET).")
    if not code:
        raise AuthError("Google did not return an authorization code.")
    token = _post_form(TOKEN_ENDPOINT, {
        "code": code,
        "client_id": cfg.client_id,
        "client_secret": cfg.client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    })
    access_token = token.get("access_token")
    if not access_token:
        raise AuthError("Google's token response contained no access_token.")
    return _get_json(USERINFO_ENDPOINT, access_token)


def sign_in(
    code: str,
    redirect_uri: str,
    config: Optional[Config] = None,
) -> Dict[str, str]:
    """The whole callback in one call: exchange, then apply the allowlist.

    Returns {"email", "name", "picture", "role"}.
    Raises AccessDenied if the account is not allowed, AuthError otherwise.
    """
    cfg = config or get_config()
    info = exchange_code(code, redirect_uri, cfg)
    email = (info.get("email") or "").strip()
    if not email:
        raise AuthError("Google returned no email address for this account.")
    if not info.get("email_verified", True):
        raise AccessDenied(email, "That Google account has an unverified email address.")
    if not is_email_allowed(email, cfg):
        raise AccessDenied(
            email,
            f"Access denied: only {cfg.describe()} accounts are allowed.",
        )
    return {
        "email": email,
        "name": info.get("name") or email.split("@")[0],
        "picture": info.get("picture") or "",
        "role": role_for(email, cfg),
    }


# --------------------------------------------------------------------------
# Signed session cookie (optional)
# --------------------------------------------------------------------------
# Use these only if the app has no session mechanism of its own. The value is
# signed, NOT encrypted -- anyone can read it, nobody can alter it. Keep
# SESSION_SECRET stable across deploys or everyone is signed out.

def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _secret(config: Optional[Config] = None) -> bytes:
    cfg = config or get_config()
    if not cfg.session_secret:
        raise AuthError(
            "SESSION_SECRET is not set, so sessions cannot be signed. Generate "
            'one with: python -c "import secrets; print(secrets.token_urlsafe(48))"'
        )
    return cfg.session_secret.encode()


def serialize_session(data: Dict[str, Any], config: Optional[Config] = None) -> str:
    """Sign a dict into a cookie-safe string, stamped with the current time."""
    payload = dict(data)
    payload["_iat"] = int(time.time())
    body = _b64e(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
    sig = hmac.new(_secret(config), body.encode(), hashlib.sha256).digest()
    return body + "." + _b64e(sig)


def deserialize_session(
    token: str,
    max_age: int = 14 * 24 * 3600,
    config: Optional[Config] = None,
) -> Optional[Dict[str, Any]]:
    """Verify and unpack a cookie. Returns None if invalid, tampered or expired.

    None always means "treat them as signed out" -- never raises on bad input,
    because the value comes straight from the browser.
    """
    if not token or "." not in token:
        return None
    body, _, sig = token.rpartition(".")
    try:
        expected = hmac.new(_secret(config), body.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _b64d(sig)):
            return None
        payload = json.loads(_b64d(body).decode())
    except AuthError:
        raise
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    issued = payload.get("_iat")
    if not isinstance(issued, int) or time.time() - issued > max_age:
        return None
    payload.pop("_iat", None)
    return payload
