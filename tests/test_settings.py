"""Tests for the settings panel widget."""

import pytest

from textual.app import App, ComposeResult
from textual.widgets import Select

from unittest.mock import patch

from app.config import TranscribeConfig, MAX_CPU_THREADS
from app.widgets.settings import SettingsPanel


class SettingsApp(App):
    """Test app that hosts a SettingsPanel widget."""

    def compose(self) -> ComposeResult:
        yield SettingsPanel()


class TestSettingsPanel:
    async def test_mounts(self):
        app = SettingsApp()
        async with app.run_test() as pilot:
            panel = app.query_one(SettingsPanel)
            assert panel is not None

    async def test_has_model_select(self):
        app = SettingsApp()
        async with app.run_test() as pilot:
            panel = app.query_one(SettingsPanel)
            select = panel.query_one("#model-select")
            assert select is not None

    async def test_has_compute_type_select(self):
        app = SettingsApp()
        async with app.run_test() as pilot:
            panel = app.query_one(SettingsPanel)
            select = panel.query_one("#compute-select")
            assert select is not None

    async def test_has_device_select(self):
        app = SettingsApp()
        async with app.run_test() as pilot:
            panel = app.query_one(SettingsPanel)
            select = panel.query_one("#device-select")
            assert select is not None

    async def test_has_language_select(self):
        app = SettingsApp()
        async with app.run_test() as pilot:
            panel = app.query_one(SettingsPanel)
            select = panel.query_one("#language-select")
            assert select is not None

    async def test_has_cores_select(self):
        app = SettingsApp()
        async with app.run_test() as pilot:
            panel = app.query_one(SettingsPanel)
            select = panel.query_one("#cores-select")
            assert select is not None

    async def test_get_config_returns_transcribe_config(self):
        app = SettingsApp()
        async with app.run_test() as pilot:
            panel = app.query_one(SettingsPanel)
            config = panel.get_config()
            assert isinstance(config, TranscribeConfig)

    async def test_default_config_values(self):
        app = SettingsApp()
        async with app.run_test() as pilot:
            panel = app.query_one(SettingsPanel)
            config = panel.get_config()
            assert config.model == "large-v3-turbo"
            assert config.compute_type == "int8"
            assert config.language == "hu"
            assert config.cpu_threads == 4

    async def test_cores_select_has_correct_range(self):
        app = SettingsApp()
        async with app.run_test() as pilot:
            panel = app.query_one(SettingsPanel)
            cores_select = panel.query_one("#cores-select", Select)
            assert cores_select.value == 4

    @patch("app.widgets.settings.available_devices", return_value=("cuda", "cpu"))
    @patch("app.widgets.settings.detect_device", return_value="cuda")
    async def test_cores_hidden_when_device_is_not_cpu(self, mock_detect, mock_devices):
        """CPU cores selector should be hidden when device is not CPU."""
        app = SettingsApp()
        async with app.run_test() as pilot:
            panel = app.query_one(SettingsPanel)
            cores_label = panel.query_one("#cores-label")
            cores_select = panel.query_one("#cores-select")
            assert cores_label.display is False
            assert cores_select.display is False

    @patch("app.widgets.settings.detect_device", return_value="cpu")
    async def test_cores_visible_when_device_is_cpu(self, mock_detect):
        """CPU cores selector should be visible when device is CPU."""
        app = SettingsApp()
        async with app.run_test() as pilot:
            panel = app.query_one(SettingsPanel)
            cores_label = panel.query_one("#cores-label")
            cores_select = panel.query_one("#cores-select")
            assert cores_label.display is True
            assert cores_select.display is True

    @patch("app.widgets.settings.available_devices", return_value=("cuda", "cpu"))
    @patch("app.widgets.settings.detect_device", return_value="cuda")
    async def test_cores_toggles_on_device_change(self, mock_detect, mock_devices):
        """CPU cores selector should toggle when device changes."""
        app = SettingsApp()
        async with app.run_test() as pilot:
            panel = app.query_one(SettingsPanel)
            device_select = panel.query_one("#device-select", Select)

            # Switch to CPU
            device_select.value = "cpu"
            await pilot.pause()
            assert panel.query_one("#cores-select").display is True

            # Switch back to cuda
            device_select.value = "cuda"
            await pilot.pause()
            assert panel.query_one("#cores-select").display is False

    async def test_settings_panel_does_not_overflow(self):
        """All Select widgets must fit within the settings panel width."""
        app = SettingsApp()
        async with app.run_test(size=(120, 40)) as pilot:
            panel = app.query_one(SettingsPanel)
            panel_width = panel.size.width
            for select in panel.query(Select):
                assert select.size.width <= panel_width, (
                    f"Select {select.id} width {select.size.width} > panel {panel_width}"
                )
