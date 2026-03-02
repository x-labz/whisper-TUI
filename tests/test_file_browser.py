"""Tests for the file browser widget."""

from pathlib import Path

import pytest

from app.widgets.file_browser import AUDIO_EXTENSIONS, list_audio_files, FileBrowser, truncate_filename

from textual.app import App, ComposeResult


class FileBrowserApp(App):
    """Test app that hosts a FileBrowser widget."""

    def __init__(self, directory: Path):
        super().__init__()
        self.directory = directory

    def compose(self) -> ComposeResult:
        yield FileBrowser(self.directory)


class TestListAudioFiles:
    def test_lists_audio_files(self, tmp_audio_dir):
        files = list_audio_files(tmp_audio_dir)
        names = {f.name for f in files}
        assert names == {"speech.mp3", "interview.wav", "podcast.flac", "music.ogg", "voice.m4a"}

    def test_ignores_non_audio_files(self, tmp_audio_dir):
        files = list_audio_files(tmp_audio_dir)
        names = {f.name for f in files}
        assert "readme.txt" not in names
        assert "image.png" not in names
        assert "data.csv" not in names

    def test_empty_directory(self, empty_dir):
        files = list_audio_files(empty_dir)
        assert files == []

    def test_returns_sorted_by_name(self, tmp_audio_dir):
        files = list_audio_files(tmp_audio_dir)
        names = [f.name for f in files]
        assert names == sorted(names)

    def test_audio_extensions_constant(self):
        assert ".mp3" in AUDIO_EXTENSIONS
        assert ".wav" in AUDIO_EXTENSIONS
        assert ".flac" in AUDIO_EXTENSIONS
        assert ".ogg" in AUDIO_EXTENSIONS
        assert ".m4a" in AUDIO_EXTENSIONS


class TestTruncateFilename:
    def test_short_name_unchanged(self):
        assert truncate_filename("short.mp3", 50) == "short.mp3"

    def test_long_name_truncated_with_ellipsis(self):
        long_name = "this_is_a_very_long_filename_that_exceeds_the_max_width_limit.mp3"
        result = truncate_filename(long_name, 40)
        assert len(result) <= 40
        assert "..." in result
        assert result.endswith(".mp3")

    def test_preserves_extension(self):
        result = truncate_filename("a" * 60 + ".flac", 30)
        assert result.endswith(".flac")

    def test_exact_width_unchanged(self):
        name = "x" * 46 + ".mp3"  # exactly 50 chars
        assert truncate_filename(name, 50) == name

    def test_contains_start_and_end_of_stem(self):
        name = "recording_from_zoom_meeting_2024_january_15_participants.wav"
        result = truncate_filename(name, 35)
        assert result.startswith("recor")
        assert "..." in result
        assert result.endswith(".wav")


class TestFileBrowserWidget:
    async def test_mounts_with_files(self, tmp_audio_dir):
        app = FileBrowserApp(tmp_audio_dir)
        async with app.run_test() as pilot:
            browser = app.query_one(FileBrowser)
            assert browser is not None

    async def test_shows_audio_files(self, tmp_audio_dir):
        app = FileBrowserApp(tmp_audio_dir)
        async with app.run_test() as pilot:
            browser = app.query_one(FileBrowser)
            assert browser.option_count == 5

    async def test_empty_dir_shows_no_files(self, empty_dir):
        app = FileBrowserApp(empty_dir)
        async with app.run_test() as pilot:
            browser = app.query_one(FileBrowser)
            assert browser.option_count == 0

    async def test_selected_file_property(self, tmp_audio_dir):
        app = FileBrowserApp(tmp_audio_dir)
        async with app.run_test() as pilot:
            browser = app.query_one(FileBrowser)
            # Initially nothing selected
            assert browser.selected_files == []
