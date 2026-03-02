"""Progress display widget for transcription."""

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import ProgressBar, Label, Static


class TranscribeProgress(Static):
    """Widget showing transcription progress."""

    DEFAULT_CSS = """
    TranscribeProgress {
        width: 100%;
        height: 5;
        padding: 0 1;
    }
    #progress-row {
        height: 1;
    }
    #speed-label {
        width: auto;
        min-width: 16;
        text-align: right;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.progress_value = 0.0
        self.status_text = ""
        self.segment_text = ""
        self.speed_text = ""

    def compose(self) -> ComposeResult:
        with Vertical():
            with Horizontal(id="progress-row"):
                yield ProgressBar(total=100, show_eta=False, id="progress-bar")
                yield Label("", id="speed-label")
            yield Label("", id="status-label")
            yield Label("", id="segment-label")

    def update_progress(
        self,
        ratio: float,
        segment_text: str = "",
        audio_duration: float = 0.0,
        elapsed: float = 0.0,
    ) -> None:
        """Update progress bar, segment text, and speed display.

        Args:
            ratio: Progress as 0.0 to 1.0.
            segment_text: Current segment text to display.
            audio_duration: Total audio duration in seconds.
            elapsed: Wall time elapsed so far in seconds.
        """
        self.progress_value = ratio
        self.segment_text = segment_text
        bar = self.query_one("#progress-bar", ProgressBar)
        bar.update(progress=ratio * 100)
        self.query_one("#segment-label", Label).update(segment_text)

        if elapsed > 0 and audio_duration > 0:
            speed = audio_duration / elapsed
            self.speed_text = f"{speed:.1f}x RT"
            self.query_one("#speed-label", Label).update(self.speed_text)

    def update_status(self, text: str) -> None:
        """Update the status label."""
        self.status_text = text
        self.query_one("#status-label", Label).update(text)

    def reset(self) -> None:
        """Reset progress to initial state."""
        self.progress_value = 0.0
        self.status_text = ""
        self.segment_text = ""
        self.speed_text = ""
        bar = self.query_one("#progress-bar", ProgressBar)
        bar.update(progress=0)
        self.query_one("#status-label", Label).update("")
        self.query_one("#segment-label", Label).update("")
        self.query_one("#speed-label", Label).update("")

    def set_indeterminate(self) -> None:
        """Switch progress bar to indeterminate mode (spinner animation)."""
        bar = self.query_one("#progress-bar", ProgressBar)
        bar.update(total=None)

    def set_determinate(self) -> None:
        """Switch progress bar back to determinate mode (0-100%)."""
        bar = self.query_one("#progress-bar", ProgressBar)
        bar.update(total=100, progress=0)
        self.progress_value = 0.0
