"""Integration tests for the main app."""

from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from textual.widgets import Button

from app.main import WhisperApp
from app.widgets.file_browser import FileBrowser
from app.widgets.settings import SettingsPanel
from app.widgets.progress import TranscribeProgress


@pytest.fixture
def app_with_files(tmp_audio_dir):
    """Create a WhisperApp pointed at a directory with audio files."""
    return WhisperApp(directory=tmp_audio_dir)


@pytest.fixture
def app_empty(empty_dir):
    """Create a WhisperApp pointed at an empty directory."""
    return WhisperApp(directory=empty_dir)


class TestAppComposition:
    async def test_mounts_file_browser(self, app_with_files):
        async with app_with_files.run_test() as pilot:
            assert app_with_files.query_one(FileBrowser) is not None

    async def test_mounts_settings_panel(self, app_with_files):
        async with app_with_files.run_test() as pilot:
            assert app_with_files.query_one(SettingsPanel) is not None

    async def test_mounts_progress_widget(self, app_with_files):
        async with app_with_files.run_test() as pilot:
            assert app_with_files.query_one(TranscribeProgress) is not None

    async def test_has_transcribe_button(self, app_with_files):
        async with app_with_files.run_test() as pilot:
            btn = app_with_files.query_one("#transcribe-btn")
            assert btn is not None

    async def test_has_transcribe_all_button(self, app_with_files):
        async with app_with_files.run_test() as pilot:
            btn = app_with_files.query_one("#transcribe-all-btn")
            assert btn is not None

    async def test_has_quit_button(self, app_with_files):
        async with app_with_files.run_test() as pilot:
            btn = app_with_files.query_one("#quit-btn")
            assert btn is not None

    async def test_has_title(self, app_with_files):
        async with app_with_files.run_test() as pilot:
            assert app_with_files.title == "Whisper Transcriber"


class TestAppBehavior:
    async def test_quit_button_exits(self, app_with_files):
        async with app_with_files.run_test(size=(100, 40)) as pilot:
            btn = app_with_files.query_one("#quit-btn", Button)
            btn.press()
            await pilot.pause()
            assert app_with_files._exit

    async def test_q_key_exits(self, app_with_files):
        async with app_with_files.run_test(size=(100, 40)) as pilot:
            await pilot.press("q")
            assert app_with_files._exit

    @patch("app.main.Transcriber")
    async def test_transcribe_button_triggers_transcription(
        self, MockTranscriber, tmp_audio_dir
    ):
        mock_result = MagicMock()
        mock_result.text = "Hello world"
        mock_result.output_path = tmp_audio_dir / "speech.txt"
        mock_result.speed_factor = 1.5
        mock_instance = MockTranscriber.return_value
        mock_instance.transcribe.return_value = mock_result

        app = WhisperApp(directory=tmp_audio_dir)
        async with app.run_test(size=(100, 40)) as pilot:
            browser = app.query_one(FileBrowser)
            browser.highlighted = 0
            await pilot.pause()

            btn = app.query_one("#transcribe-btn", Button)
            btn.press()
            await pilot.pause(delay=0.5)

            mock_instance.transcribe.assert_called_once()

    @patch("app.main.Transcriber")
    async def test_transcribe_all_processes_all_files(
        self, MockTranscriber, tmp_audio_dir
    ):
        mock_result = MagicMock()
        mock_result.text = "text"
        mock_result.output_path = tmp_audio_dir / "out.txt"
        mock_result.speed_factor = 2.0
        mock_instance = MockTranscriber.return_value
        mock_instance.transcribe.return_value = mock_result

        app = WhisperApp(directory=tmp_audio_dir)
        async with app.run_test(size=(100, 40)) as pilot:
            btn = app.query_one("#transcribe-all-btn", Button)
            btn.press()
            await pilot.pause(delay=0.5)

            assert mock_instance.transcribe.call_count == 5


class TestLayoutVisibility:
    async def test_buttons_visible_on_screen(self, app_with_files):
        """All buttons must be within the visible screen area."""
        async with app_with_files.run_test(size=(120, 40)) as pilot:
            screen_height = 40
            for btn_id in ("transcribe-btn", "transcribe-all-btn", "quit-btn"):
                btn = app_with_files.query_one(f"#{btn_id}", Button)
                r = btn.region
                assert r.height > 0, f"{btn_id} has zero height"
                assert r.y + r.height <= screen_height, (
                    f"{btn_id} at y={r.y} h={r.height} is off-screen (screen_h={screen_height})"
                )

    async def test_progress_panel_visible(self, app_with_files):
        """Progress panel must be visible on screen."""
        async with app_with_files.run_test(size=(120, 40)) as pilot:
            progress = app_with_files.query_one(TranscribeProgress)
            r = progress.region
            assert r.height > 0, "Progress panel has zero height"
            assert r.y + r.height <= 40, "Progress panel is off-screen"

    async def test_file_panel_visible(self, app_with_files):
        """File panel must have visible area."""
        async with app_with_files.run_test(size=(120, 40)) as pilot:
            panel = app_with_files.query_one("#file-panel")
            assert panel.size.height > 0, "File panel has zero height"

    async def test_settings_panel_visible(self, app_with_files):
        """Settings panel must have visible area."""
        async with app_with_files.run_test(size=(120, 40)) as pilot:
            panel = app_with_files.query_one(SettingsPanel)
            assert panel.size.height > 0, "Settings panel has zero height"


class TestModelLoadingStatus:
    @patch("app.main.Transcriber")
    async def test_transcriber_receives_on_status_callback(
        self, MockTranscriber, tmp_audio_dir
    ):
        """Verify Transcriber is created with an on_status callback."""
        mock_result = MagicMock()
        mock_result.text = "text"
        mock_result.output_path = tmp_audio_dir / "out.txt"
        mock_result.speed_factor = 1.0
        mock_instance = MockTranscriber.return_value
        mock_instance.transcribe.return_value = mock_result

        app = WhisperApp(directory=tmp_audio_dir)
        async with app.run_test(size=(100, 40)) as pilot:
            btn = app.query_one("#transcribe-all-btn", Button)
            btn.press()
            await pilot.pause(delay=0.5)

            MockTranscriber.assert_called_once()
            call_kwargs = MockTranscriber.call_args
            assert "on_status" in call_kwargs.kwargs
            assert callable(call_kwargs.kwargs["on_status"])

    @patch("app.main.Transcriber")
    async def test_on_status_callback_updates_progress_widget(
        self, MockTranscriber, tmp_audio_dir
    ):
        """Verify the on_status callback updates the progress panel status."""
        captured_on_status = None

        def capture_init(config, on_status=None):
            nonlocal captured_on_status
            captured_on_status = on_status
            if on_status:
                on_status("Loading model 'tiny'...")
            return MagicMock(
                transcribe=MagicMock(
                    return_value=MagicMock(
                        text="text",
                        output_path=tmp_audio_dir / "out.txt",
                        speed_factor=1.0,
                    )
                )
            )

        MockTranscriber.side_effect = capture_init

        app = WhisperApp(directory=tmp_audio_dir)
        async with app.run_test(size=(100, 40)) as pilot:
            btn = app.query_one("#transcribe-all-btn", Button)
            btn.press()
            await pilot.pause(delay=0.5)

            assert captured_on_status is not None


class TestModelLoadingError:
    @patch("app.main.Transcriber")
    async def test_model_loading_failure_shows_error_status(
        self, MockTranscriber, tmp_audio_dir, tmp_path, monkeypatch
    ):
        """When model loading raises, status should show the error."""
        monkeypatch.chdir(tmp_path)
        MockTranscriber.side_effect = RuntimeError("OpenVINO not installed")

        app = WhisperApp(directory=tmp_audio_dir)
        async with app.run_test(size=(100, 40)) as pilot:
            btn = app.query_one("#transcribe-all-btn", Button)
            btn.press()
            await pilot.pause(delay=0.5)

            progress = app.query_one(TranscribeProgress)
            assert "OpenVINO not installed" in progress.status_text

    @patch("app.main.Transcriber")
    async def test_model_loading_failure_writes_errorlog(
        self, MockTranscriber, tmp_audio_dir, tmp_path, monkeypatch
    ):
        """When model loading fails, error should be logged to errorlog.txt."""
        import os

        monkeypatch.chdir(tmp_path)
        MockTranscriber.side_effect = RuntimeError("GPU not available")

        app = WhisperApp(directory=tmp_audio_dir)
        async with app.run_test(size=(100, 40)) as pilot:
            btn = app.query_one("#transcribe-all-btn", Button)
            btn.press()
            await pilot.pause(delay=0.5)

        log_file = tmp_path / "errorlog.txt"
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "Model loading failed" in content
        assert "GPU not available" in content
        assert "RuntimeError" in content

    @patch("app.main.Transcriber")
    async def test_model_loading_failure_reenables_buttons(
        self, MockTranscriber, tmp_audio_dir, tmp_path, monkeypatch
    ):
        """Buttons should be re-enabled after model loading failure."""
        monkeypatch.chdir(tmp_path)
        MockTranscriber.side_effect = RuntimeError("OOM")

        app = WhisperApp(directory=tmp_audio_dir)
        async with app.run_test(size=(100, 40)) as pilot:
            btn = app.query_one("#transcribe-btn", Button)
            btn_all = app.query_one("#transcribe-all-btn", Button)
            btn_all.press()
            await pilot.pause(delay=0.5)

            assert not btn.disabled
            assert not btn_all.disabled


class TestButtonDisabling:
    @patch("app.main.Transcriber")
    async def test_buttons_disabled_during_transcription(
        self, MockTranscriber, tmp_audio_dir
    ):
        """Buttons should be disabled while worker is running."""
        import threading

        barrier = threading.Event()

        def slow_init(config, on_status=None):
            barrier.wait(timeout=5)
            return MagicMock(
                transcribe=MagicMock(
                    return_value=MagicMock(
                        text="text",
                        output_path=tmp_audio_dir / "out.txt",
                        speed_factor=1.0,
                    )
                )
            )

        MockTranscriber.side_effect = slow_init

        app = WhisperApp(directory=tmp_audio_dir)
        async with app.run_test(size=(100, 40)) as pilot:
            btn = app.query_one("#transcribe-btn", Button)
            btn_all = app.query_one("#transcribe-all-btn", Button)

            btn_all.press()
            await pilot.pause(delay=0.1)

            assert btn.disabled
            assert btn_all.disabled

            barrier.set()
            await pilot.pause(delay=0.5)

            assert not btn.disabled
            assert not btn_all.disabled
