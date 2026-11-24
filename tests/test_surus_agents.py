"""Mercury provisions the staff and optimizer Hermes profiles once Databricks is
configured: persona, plugin, keys, and a locked-down config."""

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

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATABRICKS_ENV = {"DATABRICKS_HOST": "https://adb-test.azuredatabricks.net", "DATABRICKS_TOKEN": "dapi-test"}
_ENV_NAMES = ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_CLIENT_ID", "DATABRICKS_CLIENT_SECRET",
              "SURUS_AGENT_MODEL", "GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "OPENROUTER_API_KEY")


class SurusAgentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="surus-agents-")
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name) / ".hermes"
        self.home.mkdir()
        (self.home / ".env").write_text("OPENROUTER_API_KEY=sk-or-v1-test\n", encoding="utf-8")
        self.created = []

    def load(self, env: dict | None = None):
        env = {"HERMES_HOME": str(self.home), "ADMIN_PASSWORD": "test-password", **(env or {})}
        patcher = patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in _ENV_NAMES:
            if name not in env:
                os.environ.pop(name, None)
        if not hasattr(os, "fchmod"):  # Windows test hosts only
            p = patch.object(os, "fchmod", lambda fd, mode: None, create=True)
            p.start()
            self.addCleanup(p.stop)
        if "hermes_yaml" not in sys.modules and importlib.util.find_spec("hermes_yaml") is None:  # in the image
            p = patch.dict(sys.modules, {"hermes_yaml": types.SimpleNamespace(
                safe_load=yaml.safe_load, YAMLError=yaml.YAMLError)})
            p.start()
            self.addCleanup(p.stop)
        spec = importlib.util.spec_from_file_location(f"surus_agents_{uuid.uuid4().hex}", ROOT / "server.py")
        server = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(server)
        self.writes = []

        def write_config(path, data):
            self.writes.append(Path(path).parent.name)
            Path(path).write_text(yaml.safe_dump(data), encoding="utf-8")

        async def fake_cli(*args, timeout=None):
            if args[:2] == ("profile", "create"):
                name = args[2]
                self.created.append((name, args[3:]))
                if name in getattr(self, "fail_create", ()):
                    return 1, "boom"
                home = self.home / "profiles" / name
                home.mkdir(parents=True)
                (home / "config.yaml").write_text("model:\n  provider: auto\nlanguage: en\n", encoding="utf-8")
                return 0, ""
            return 1, "unexpected"

        server._write_config_atomic = write_config
        server._run_hermes_cli = fake_cli
        return server

    def config(self, name: str) -> dict:
        return yaml.safe_load((self.home / "profiles" / name / "config.yaml").read_text(encoding="utf-8"))

    def env(self, server, name: str) -> dict:
        return server.read_env(self.home / "profiles" / name / ".env")

    async def test_nothing_happens_until_databricks_is_configured(self):
        server = self.load()
        await server.provision_surus_agents()
        self.assertFalse((self.home / "profiles").exists())
        self.assertFalse(server.surus_agents_status["enabled"])

    async def test_both_agents_are_created_and_locked_down(self):
        server = self.load(DATABRICKS_ENV)
        await server.provision_surus_agents()
        self.assertEqual([c[0] for c in self.created], ["staff", "optimizer"])
        self.assertEqual(self.created[0][1], ("--no-alias", "--no-skills"))
        for name, toolsets in (("staff", ["surus_databricks"]),
                               ("optimizer", ["surus_databricks", "surus_optimizer"])):
            home = self.home / "profiles" / name
            soul = (home / "SOUL.md").read_text(encoding="utf-8")
            prompt = (ROOT / "templates" / "agents" / name / "prompt.txt").read_text(encoding="utf-8").rstrip()
            self.assertTrue(soul.startswith(prompt))
            self.assertIn("## How you work in Mercury", soul)
            self.assertTrue((home / "plugins" / "surus_databricks" / "plugin.yaml").is_file())
            self.assertTrue((home / "plugins" / "surus_databricks" / "prelude" / "surus_tools.py").is_file())
            env = self.env(server, name)
            self.assertGreaterEqual(len(env["API_SERVER_KEY"]), 32)
            self.assertEqual(env["OPENROUTER_API_KEY"], "sk-or-v1-test")
            cfg = self.config(name)
            self.assertEqual(cfg["model"], {"provider": "openrouter", "default": "x-ai/grok-4.7"})
            self.assertEqual(cfg["platform_toolsets"],
                             {"api_server": toolsets, "signal": toolsets, "cli": toolsets})
            for blocked in ("terminal", "file", "code_execution", "memory", "session_search", "web"):
                self.assertIn(blocked, cfg["agent"]["disabled_toolsets"])
            self.assertEqual(cfg["memory"], {"memory_enabled": False, "user_profile_enabled": False})
            self.assertEqual(cfg["plugins"]["enabled"], ["surus_databricks"])
            self.assertEqual(cfg["terminal"]["cwd"], str(server.MERCURY_DIR / "agents" / name))
            self.assertEqual(cfg["language"], "en")  # unrelated settings are kept
            self.assertTrue(server.surus_agents_status["agents"][name]["ready"])
        self.assertNotEqual(self.env(server, "staff")["API_SERVER_KEY"],
                            self.env(server, "optimizer")["API_SERVER_KEY"])

    async def test_a_second_start_changes_nothing_and_keeps_keys(self):
        server = self.load(DATABRICKS_ENV)
        await server.provision_surus_agents()
        key = server.surus_agent_api_key("staff")
        stale = self.home / "profiles" / "staff" / "plugins" / "surus_databricks" / "old_module.py"
        stale.write_text("x = 1", encoding="utf-8")
        self.writes.clear()
        await server.provision_surus_agents()
        self.assertEqual(self.writes, [])
        self.assertEqual(server.surus_agent_api_key("staff"), key)
        self.assertFalse(stale.exists(), "files no longer in the plugin are removed")
        self.assertEqual(len(self.created), 2, "existing profiles are not re-created")

    async def test_one_agent_failing_does_not_stop_the_other(self):
        server = self.load(DATABRICKS_ENV)
        self.fail_create = ("staff",)
        await server.provision_surus_agents()
        status = server.surus_agents_status["agents"]
        self.assertFalse(status["staff"]["ready"])
        self.assertIn("boom", status["staff"]["error"])
        self.assertTrue(status["optimizer"]["ready"])

    async def test_model_can_be_overridden(self):
        server = self.load({**DATABRICKS_ENV, "SURUS_AGENT_MODEL": "x-ai/grok-4.6"})
        await server.provision_surus_agents()
        self.assertEqual(self.config("staff")["model"]["default"], "x-ai/grok-4.6")

    def _hermes_env(self, server):
        helpers = types.ModuleType("template_helpers.migrate_hermes_configs")
        helpers.IMAGE_EXECUTION_ENV_KEYS = ()
        helpers.validate_profile_execution_environment = lambda home: None
        with patch.dict(sys.modules, {"template_helpers.migrate_hermes_configs": helpers}):
            return server.build_hermes_env()

    def test_api_server_listener_is_on_only_with_the_agents(self):
        env = self._hermes_env(self.load(DATABRICKS_ENV))
        self.assertEqual((env["API_SERVER_ENABLED"], env["API_SERVER_HOST"], env["API_SERVER_PORT"]),
                         ("true", "127.0.0.1", "8642"))
        self.assertGreaterEqual(len(env["API_SERVER_KEY"]), 32)
        self.assertTrue(env["MERCURY_DIR"].endswith(".mercury"))
        off = self._hermes_env(self.load())
        self.assertNotIn("API_SERVER_ENABLED", off)
        self.assertIn("MERCURY_DIR", off)


if __name__ == "__main__":
    unittest.main()
