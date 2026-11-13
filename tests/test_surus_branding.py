"""Surus Enterprises styles: the brand route, and the Hermes dashboard theme +
terminal skin installed into HERMES_HOME and activated in config.yaml."""

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
BRAND = ROOT / "templates" / "brand"

# Present in the deployed image; outside it, these tests read config with PyYAML.
# Installed per test (patch.dict) so the stand-in never leaks into other test modules.
_HERMES_YAML_SHIM = {} if importlib.util.find_spec("hermes_yaml") else {
    "hermes_yaml": types.SimpleNamespace(safe_load=yaml.safe_load, YAMLError=yaml.YAMLError)}


def _hermes_importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


class SurusHermesStylesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="surus-brand-")
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name) / ".hermes"
        self.home.mkdir()
        env = patch.dict(os.environ, {"HERMES_HOME": str(self.home), "ADMIN_PASSWORD": "test-password"})
        env.start()
        self.addCleanup(env.stop)
        shim = patch.dict(sys.modules, _HERMES_YAML_SHIM)
        shim.start()
        self.addCleanup(shim.stop)
        spec = importlib.util.spec_from_file_location(f"surus_brand_{uuid.uuid4().hex}", ROOT / "server.py")
        assert spec and spec.loader
        self.server = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.server)
        self.writes = []

        def fake_write(path, data):
            self.writes.append(data)
            Path(path).write_text(yaml.safe_dump(data), encoding="utf-8")

        self.server._write_config_atomic = fake_write
        self.config = self.home / "config.yaml"

    def test_installs_theme_and_skin_and_activates_them_keeping_other_settings(self):
        self.config.write_text("model:\n  default: claude\ndashboard:\n  theme: nous-blue\n", encoding="utf-8")

        self.server.apply_surus_hermes_styles()

        self.assertEqual((self.home / "dashboard-themes" / "surus.yaml").read_text(encoding="utf-8"),
                         (BRAND / "hermes-dashboard-theme.yaml").read_text(encoding="utf-8"))
        self.assertEqual((self.home / "skins" / "surus.yaml").read_text(encoding="utf-8"),
                         (BRAND / "hermes-skin.yaml").read_text(encoding="utf-8"))
        saved = yaml.safe_load(self.config.read_text(encoding="utf-8"))
        self.assertEqual(saved["dashboard"]["theme"], "surus")
        self.assertEqual(saved["display"]["skin"], "surus")
        self.assertEqual(saved["model"], {"default": "claude"})

    def test_second_start_does_not_rewrite_config(self):
        self.config.write_text("dashboard:\n  theme: default\n", encoding="utf-8")
        self.server.apply_surus_hermes_styles()
        self.server.apply_surus_hermes_styles()
        self.assertEqual(len(self.writes), 1)

    def test_malformed_section_is_left_alone(self):
        self.config.write_text("dashboard: compact\n", encoding="utf-8")
        self.server.apply_surus_hermes_styles()
        saved = yaml.safe_load(self.config.read_text(encoding="utf-8"))
        self.assertEqual(saved["dashboard"], "compact")
        self.assertEqual(saved["display"]["skin"], "surus")

    def test_missing_config_installs_files_without_creating_config(self):
        self.server.apply_surus_hermes_styles()
        self.assertTrue((self.home / "dashboard-themes" / "surus.yaml").is_file())
        self.assertFalse(self.config.exists())
        self.assertEqual(self.writes, [])

    def test_brand_route_serves_only_allowlisted_files(self):
        self.assertIn("surus.css", self.server.BRAND_ASSETS)
        for name in self.server.BRAND_ASSETS:
            self.assertTrue((BRAND / name).is_file(), name)
        self.assertNotIn("hermes-dashboard-theme.yaml", self.server.BRAND_ASSETS)


@unittest.skipUnless(_hermes_importable("hermes_cli"), "needs the pinned Hermes package")
class SurusStylesMatchHermesSchemaTests(unittest.TestCase):
    def test_dashboard_theme_normalises_with_brand_values(self):
        from hermes_cli.web_server_dashboard import _normalise_theme_definition

        theme = _normalise_theme_definition(
            yaml.safe_load((BRAND / "hermes-dashboard-theme.yaml").read_text(encoding="utf-8")))
        self.assertIsNotNone(theme)
        self.assertEqual(theme["name"], "surus")
        self.assertEqual(theme["palette"]["background"]["hex"], "#F5F7F9")
        self.assertEqual(theme["colorOverrides"]["primary"], "#C3262F")
        self.assertIn("#app-sidebar", theme["customCSS"])

    def test_skin_builds_with_brand_accent(self):
        from hermes_cli.skin_engine import _build_skin_config

        skin = _build_skin_config(yaml.safe_load((BRAND / "hermes-skin.yaml").read_text(encoding="utf-8")))
        self.assertEqual(skin.name, "surus")
        self.assertEqual(skin.get_color("ui_accent"), "#D25E64")
        self.assertIn("[bold #C3262F]", skin.banner_logo)
        self.assertIn("E N T E R P R I S E S", skin.banner_logo)


if __name__ == "__main__":
    unittest.main()
