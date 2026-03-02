"""Shared test fixtures for whisper-tui tests."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def tmp_audio_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with fake audio files."""
    audio_files = ["speech.mp3", "interview.wav", "podcast.flac", "music.ogg", "voice.m4a"]
    non_audio_files = ["readme.txt", "image.png", "data.csv", "script.py"]

    for f in audio_files:
        (tmp_path / f).write_bytes(b"\x00" * 1024)

    for f in non_audio_files:
        (tmp_path / f).write_bytes(b"\x00" * 512)

    return tmp_path


@pytest.fixture
def empty_dir(tmp_path: Path) -> Path:
    """An empty temporary directory."""
    return tmp_path


@pytest.fixture
def mock_whisper_segment():
    """Create a mock whisper segment."""
    def _make_segment(start: float, end: float, text: str):
        seg = MagicMock()
        seg.start = start
        seg.end = end
        seg.text = text
        return seg
    return _make_segment


@pytest.fixture
def mock_whisper_info():
    """Create a mock transcription info object."""
    info = MagicMock()
    info.language = "hu"
    info.language_probability = 0.95
    info.duration = 60.0
    return info
