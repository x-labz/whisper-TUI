"""File browser widget for listing audio files."""

from pathlib import Path
from typing import List

from textual.widgets import OptionList
from textual.widgets.option_list import Option

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


def list_audio_files(directory: Path) -> List[Path]:
    """List audio files in a directory, sorted by name."""
    files = [
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    ]
    return sorted(files, key=lambda f: f.name)


def _format_size(size_bytes: int) -> str:
    """Format file size for display."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def truncate_filename(name: str, max_width: int) -> str:
    """Truncate a filename with ellipsis in the middle, preserving extension.

    Args:
        name: The filename to truncate.
        max_width: Maximum character width for the result.

    Returns:
        Truncated filename or original if it fits.
    """
    if len(name) <= max_width:
        return name
    stem = Path(name).stem
    ext = Path(name).suffix
    # Reserve space for "..." and the extension
    available = max_width - 3 - len(ext)
    if available < 4:
        # Too narrow even for truncation, just hard-cut
        return name[:max_width - 3] + "..."
    left = available // 2
    right = available - left
    return f"{stem[:left]}...{stem[-right:]}{ext}"


class FileBrowser(OptionList):
    """Widget that lists audio files from a directory."""

    FILENAME_MAX_WIDTH = 50

    def __init__(self, directory: Path, **kwargs):
        self.directory = directory
        self._files: List[Path] = []
        super().__init__(**kwargs)

    def on_mount(self) -> None:
        self.refresh_files()

    def refresh_files(self) -> None:
        """Reload audio files from directory."""
        self.clear_options()
        self._files = list_audio_files(self.directory)
        for f in self._files:
            size = _format_size(f.stat().st_size)
            display_name = truncate_filename(f.name, self.FILENAME_MAX_WIDTH)
            self.add_option(Option(f"{display_name}  ({size})", id=str(f)))

    @property
    def selected_files(self) -> List[Path]:
        """Return currently highlighted file as a list (empty if none)."""
        if self.highlighted is not None and self._files:
            return [self._files[self.highlighted]]
        return []
