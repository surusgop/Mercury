"""Google sign-in (surus-auth): staff reach /ask, admins keep Setup and the
Hermes dashboard, and the username/password login stays as an emergency way in."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # the image runs from /app, where surus_auth/ lives

import surus_auth  # noqa: E402  (vendored)

HTML = {"accept": "text/html"}
JSON = {"accept": "application/json"}
ADMIN_EMAIL = "admin@surusenterprises.com"
GOOGLE_ENV = {"GOOGLE_CLIENT_ID": "client-id.apps.googleusercontent.com",
              "GOOGLE_CLIENT_SECRET": "client-secret",
              "APP_URL": "https://mercury.example",
              "ADMIN_BOOTSTRAP_EMAILS": ADMIN_EMAIL}


class GoogleSigninTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="google-signin-")
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name) / ".hermes"
        self.home.mkdir()

    def load(self, extra_env: dict | None = None, remove: tuple = ()):
        env = {"HERMES_HOME": str(self.home), "ADMIN_PASSWORD": "test-password", **(extra_env or {})}
        patcher = patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "SESSION_SECRET",
                     "ALLOWED_EMAIL_DOMAIN", "ALLOWED_EMAIL_DOMAINS", "GOOGLE_WORKSPACE_DOMAIN",
                     "ALLOWED_EMAILS", "ADMIN_BOOTSTRAP_EMAILS", *remove):
            if name not in env:
                os.environ.pop(name, None)
        spec = importlib.util.spec_from_file_location(f"google_signin_{uuid.uuid4().hex}", ROOT / "server.py")
        assert spec and spec.loader
        server = importlib.util.module_from_spec(spec)
        if not hasattr(os, "fchmod"):  # Windows test hosts only; Linux has it
            no_fchmod = patch.object(os, "fchmod", lambda fd, mode: None, create=True)
            no_fchmod.start()
            self.addCleanup(no_fchmod.stop)
        spec.loader.exec_module(server)
        return server

    @staticmethod
    def client(server):
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=server.asgi_app),
                                 base_url="https://mercury.example")

    @staticmethod
    def google_cookie(email: str) -> dict:
        token = surus_auth.serialize_session({"email": email, "name": email.split("@")[0],
                                              "first_name": "Pat", "session_id": "s1"})
        return {"cookie": f"mercury_session={token}"}

    async def test_without_google_settings_the_password_login_is_unchanged(self):
        server = self.load()
        self.assertIsNone(server._google_engine)
        async with self.client(server) as c:
            r = await c.get("/setup/nope", headers=HTML)
            self.assertEqual(r.status_code, 302)
            self.assertTrue(r.headers["location"].startswith("/setup/login"))
            admin = {"cookie": f"hermes_auth={server._make_auth_token()}", **HTML}
            self.assertEqual((await c.get("/setup/nope", headers=admin)).status_code, 404)
            login = await c.get("/setup/login", headers=HTML)
            self.assertNotIn("Sign in with Google", login.text)

    async def test_signed_out_visitors_are_sent_to_google(self):
        server = self.load(GOOGLE_ENV)
        async with self.client(server) as c:
            r = await c.get("/setup", headers=HTML)
            self.assertEqual(r.status_code, 302)
            self.assertEqual(r.headers["location"], "/auth/login?next=%2Fsetup")
            r = await c.get("/auth/login?next=/ask")
            self.assertTrue(r.headers["location"].startswith("https://accounts.google.com/"))
            self.assertIn("redirect_uri=https%3A%2F%2Fmercury.example%2Fauth%2Fcallback", r.headers["location"])

    async def test_health_check_and_emergency_login_stay_public(self):
        server = self.load(GOOGLE_ENV)
        async with self.client(server) as c:
            health = await c.get("/health")
            self.assertFalse(health.headers.get("location", "").startswith("/auth/login"))
            login = await c.get("/setup/login", headers=HTML)
            self.assertEqual(login.status_code, 200)
            self.assertIn("Sign in with Google instead", login.text)
            brand = await c.get("/setup/brand/surus-icon.png")
            self.assertEqual(brand.status_code, 200)

    async def test_staff_reach_their_page_but_not_admin_pages(self):
        server = self.load(GOOGLE_ENV)
        member = self.google_cookie("analyst@americanstewardship.org")
        async with self.client(server) as c:
            r = await c.get("/setup", headers={**member, **HTML})
            self.assertEqual((r.status_code, r.headers["location"]), (302, "/ask"))
            r = await c.get("/setup/api/config", headers={**member, **JSON})
            self.assertEqual(r.status_code, 403)
            r = await c.get("/sessions", headers={**member, **HTML})  # native Hermes dashboard
            self.assertEqual((r.status_code, r.headers["location"]), (302, "/ask"))
            page = await c.get("/ask", headers={**member, **HTML})
            self.assertEqual(page.status_code, 200)
            self.assertIn("analyst@americanstewardship.org", page.text)
            self.assertNotIn("Open Setup", page.text)

    async def test_google_admin_reaches_admin_pages(self):
        server = self.load(GOOGLE_ENV)
        admin = self.google_cookie(ADMIN_EMAIL)
        async with self.client(server) as c:
            self.assertEqual((await c.get("/setup/nope", headers={**admin, **HTML})).status_code, 404)
            page = await c.get("/ask", headers={**admin, **HTML})
            self.assertIn("Open Setup", page.text)

    async def test_emergency_login_cookie_skips_google(self):
        server = self.load(GOOGLE_ENV)
        async with self.client(server) as c:
            admin = {"cookie": f"hermes_auth={server._make_auth_token()}", **HTML}
            self.assertEqual((await c.get("/setup/nope", headers=admin)).status_code, 404)
            forged = {"cookie": "hermes_auth=9999999999.deadbeef", **HTML}
            r = await c.get("/setup/nope", headers=forged)
            self.assertTrue(r.headers["location"].startswith("/auth/login"))

    async def test_logout_ends_both_sessions(self):
        server = self.load(GOOGLE_ENV)
        async with self.client(server) as c:
            r = await c.get("/logout")
            self.assertEqual(r.headers["location"], "/auth/logout")
            self.assertIn("hermes_auth=", r.headers.get("set-cookie", ""))

    async def test_websocket_admin_check_reads_the_google_cookie(self):
        server = self.load(GOOGLE_ENV)

        def ws(headers):
            return types.SimpleNamespace(scope={"type": "websocket"}, headers=headers, cookies={})

        self.assertTrue(server._is_authenticated(ws(self.google_cookie(ADMIN_EMAIL))))
        self.assertFalse(server._is_authenticated(ws(self.google_cookie("analyst@surusenterprises.com"))))
        self.assertFalse(server._is_authenticated(ws({})))

    def test_blank_allowlist_falls_back_to_the_surus_domains(self):
        self.load({**GOOGLE_ENV, "ALLOWED_EMAIL_DOMAIN": "  "})
        cfg = surus_auth.get_config()
        self.assertEqual(cfg.allowed_domains, ["surusenterprises.com", "americanstewardship.org"])
        self.assertFalse(surus_auth.is_email_allowed("someone@gmail.com"))

    async def test_without_admin_list_no_google_account_is_admin(self):
        env = {k: v for k, v in GOOGLE_ENV.items() if k != "ADMIN_BOOTSTRAP_EMAILS"}
        server = self.load(env)
        self.assertEqual(surus_auth.get_config().admin_emails, [])
        async with self.client(server) as c:
            r = await c.get("/setup", headers={**self.google_cookie(ADMIN_EMAIL), **HTML})
            self.assertEqual((r.status_code, r.headers["location"]), (302, "/ask"))

    def test_explicit_allowlist_and_admins_win(self):
        self.load({**GOOGLE_ENV, "ALLOWED_EMAIL_DOMAIN": "surusenterprises.com",
                   "ADMIN_BOOTSTRAP_EMAILS": "boss@surusenterprises.com"})
        cfg = surus_auth.get_config()
        self.assertEqual(cfg.allowed_domains, ["surusenterprises.com"])
        self.assertEqual(cfg.admin_emails, ["boss@surusenterprises.com"])

    def test_session_secret_is_generated_once_and_kept(self):
        server = self.load(GOOGLE_ENV)
        path = server.MERCURY_DIR / "session_secret"
        first = path.read_text(encoding="utf-8").strip()
        self.assertGreaterEqual(len(first), 40)
        again = self.load(GOOGLE_ENV)
        self.assertEqual(again._mercury_session_secret(), first)

    def test_google_secrets_are_kept_from_hermes_processes(self):
        server = self.load({**GOOGLE_ENV, "SESSION_SECRET": "explicit-session-secret"})
        # build_hermes_env first validates Hermes profile settings, which needs the
        # Hermes package (present only in the image); stub just that helper here.
        helpers = types.ModuleType("template_helpers.migrate_hermes_configs")
        helpers.IMAGE_EXECUTION_ENV_KEYS = ()
        helpers.validate_profile_execution_environment = lambda home: None
        stub = patch.dict(sys.modules, {"template_helpers.migrate_hermes_configs": helpers})
        stub.start()
        self.addCleanup(stub.stop)
        env = server.build_hermes_env()
        self.assertNotIn("GOOGLE_CLIENT_SECRET", env)
        self.assertNotIn("SESSION_SECRET", env)
        self.assertEqual(env["HERMES_HOME"], str(self.home))


if __name__ == "__main__":
    unittest.main()
