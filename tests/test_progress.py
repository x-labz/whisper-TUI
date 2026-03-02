"""Tests for the progress widget."""

import pytest

from textual.app import App, ComposeResult

from app.widgets.progress import TranscribeProgress


class ProgressApp(App):
    """Test app that hosts a TranscribeProgress widget."""

    def compose(self) -> ComposeResult:
        yield TranscribeProgress()


class TestTranscribeProgress:
    async def test_mounts(self):
        app = ProgressApp()
        async with app.run_test() as pilot:
            widget = app.query_one(TranscribeProgress)
            assert widget is not None

    async def test_initial_progress_is_zero(self):
        app = ProgressApp()
        async with app.run_test() as pilot:
            widget = app.query_one(TranscribeProgress)
            assert widget.progress_value == 0.0

    async def test_update_progress(self):
        app = ProgressApp()
        async with app.run_test() as pilot:
            widget = app.query_one(TranscribeProgress)
            widget.update_progress(0.5, "Half done")
            assert widget.progress_value == pytest.approx(0.5)

    async def test_update_status_text(self):
        app = ProgressApp()
        async with app.run_test() as pilot:
            widget = app.query_one(TranscribeProgress)
            widget.update_status("Transcribing file1.mp3...")
            assert widget.status_text == "Transcribing file1.mp3..."

    async def test_update_segment_text(self):
        app = ProgressApp()
        async with app.run_test() as pilot:
            widget = app.query_one(TranscribeProgress)
            widget.update_progress(0.3, "Hello world")
            assert widget.segment_text == "Hello world"

    async def test_update_progress_with_speed(self):
        app = ProgressApp()
        async with app.run_test() as pilot:
            widget = app.query_one(TranscribeProgress)
            widget.update_progress(0.5, "Half done", audio_duration=60.0, elapsed=20.0)
            assert widget.speed_text == "3.0x RT"

    async def test_speed_not_shown_without_timing(self):
        app = ProgressApp()
        async with app.run_test() as pilot:
            widget = app.query_one(TranscribeProgress)
            widget.update_progress(0.5, "Half done")
            assert widget.speed_text == ""

    async def test_reset(self):
        app = ProgressApp()
        async with app.run_test() as pilot:
            widget = app.query_one(TranscribeProgress)
            widget.update_progress(0.8, "Some text", audio_duration=60.0, elapsed=10.0)
            widget.update_status("Working...")
            widget.reset()
            assert widget.progress_value == 0.0
            assert widget.status_text == ""
            assert widget.segment_text == ""
            assert widget.speed_text == ""

    async def test_set_indeterminate(self):
        app = ProgressApp()
        async with app.run_test() as pilot:
            widget = app.query_one(TranscribeProgress)
            widget.set_indeterminate()
            from textual.widgets import ProgressBar
            bar = widget.query_one("#progress-bar", ProgressBar)
            assert bar.total is None

    async def test_set_determinate(self):
        app = ProgressApp()
        async with app.run_test() as pilot:
            widget = app.query_one(TranscribeProgress)
            widget.update_progress(0.5, "halfway")
            widget.set_determinate()
            from textual.widgets import ProgressBar
            bar = widget.query_one("#progress-bar", ProgressBar)
            assert bar.total == 100
            assert bar.progress == 0
            assert widget.progress_value == 0.0

    async def test_indeterminate_then_determinate_roundtrip(self):
        app = ProgressApp()
        async with app.run_test() as pilot:
            widget = app.query_one(TranscribeProgress)
            from textual.widgets import ProgressBar
            bar = widget.query_one("#progress-bar", ProgressBar)
            assert bar.total == 100
            widget.set_indeterminate()
            assert bar.total is None
            widget.set_determinate()
            assert bar.total == 100
            assert bar.progress == 0
