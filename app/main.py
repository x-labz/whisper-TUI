"""Main TUI application for whisper-tui."""

import traceback
from datetime import datetime
from pathlib import Path
from threading import Thread

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Header, Footer

from app.config import TranscribeConfig
from app.transcriber import Transcriber, TranscribeResult
from app.widgets.file_browser import FileBrowser, list_audio_files
from app.widgets.settings import SettingsPanel
from app.widgets.progress import TranscribeProgress


class WhisperApp(App):
    """Whisper audio transcription TUI application."""

    TITLE = "Whisper Transcriber"

    CSS = """
    #main-container {
        width: 100%;
        height: 1fr;
        border: round #888888;
        background: $surface;
    }
    #file-panel {
        width: 1fr;
        height: 1fr;
        border-right: solid #888888;
        padding: 1;
    }
    #file-panel:focus-within {
        border-right: solid $accent;
    }
    #settings-panel {
        width: 44;
        height: 1fr;
    }
    #progress-panel {
        height: 5;
        border: round #888888;
        background: $surface;
        padding: 0 1;
    }
    #button-bar {
        height: 3;
        align: center middle;
        padding: 0 1;
    }
    #button-bar Button {
        margin: 0 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, directory: Path | None = None, **kwargs):
        super().__init__(**kwargs)
        self.directory = directory or Path.cwd() / "audio"
        self._exit = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-container"):
            with Vertical(id="file-panel"):
                yield FileBrowser(self.directory)
            yield SettingsPanel(id="settings-panel")
        yield TranscribeProgress(id="progress-panel")
        with Horizontal(id="button-bar"):
            yield Button("Transcribe", id="transcribe-btn", variant="primary")
            yield Button("Transcribe All", id="transcribe-all-btn", variant="default")
            yield Button("Quit", id="quit-btn", variant="error")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit-btn":
            self._exit = True
            self.exit()
        elif event.button.id == "transcribe-btn":
            self._transcribe_selected()
        elif event.button.id == "transcribe-all-btn":
            self._transcribe_all()

    def action_quit(self) -> None:
        self._exit = True
        self.exit()

    def _get_config(self) -> TranscribeConfig:
        panel = self.query_one(SettingsPanel)
        return panel.get_config()

    def _transcribe_selected(self) -> None:
        browser = self.query_one(FileBrowser)
        files = browser.selected_files
        if not files:
            self.notify("No file selected", severity="warning")
            return
        self._run_transcription(files)

    def _transcribe_all(self) -> None:
        files = list_audio_files(self.directory)
        if not files:
            self.notify("No audio files found", severity="warning")
            return
        self._run_transcription(files)

    def _set_buttons_disabled(self, disabled: bool) -> None:
        """Enable or disable transcription buttons."""
        self.query_one("#transcribe-btn", Button).disabled = disabled
        self.query_one("#transcribe-all-btn", Button).disabled = disabled

    @staticmethod
    def _log_error(context: str, error: Exception) -> None:
        """Append error details to errorlog.txt."""
        timestamp = datetime.now().isoformat()
        tb = traceback.format_exception(type(error), error, error.__traceback__)
        with open("errorlog.txt", "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {context}\n")
            f.write(f"{''.join(tb)}\n")

    def _run_transcription(self, files: list[Path]) -> None:
        config = self._get_config()
        progress = self.query_one(TranscribeProgress)
        self._set_buttons_disabled(True)

        def worker():
            try:
                def on_status(msg: str):
                    self.call_from_thread(progress.update_status, msg)

                self.call_from_thread(progress.reset)
                self.call_from_thread(progress.set_indeterminate)

                try:
                    transcriber = Transcriber(config, on_status=on_status)
                except Exception as e:
                    self._log_error("Model loading failed", e)
                    self.call_from_thread(
                        self.notify,
                        f"Model loading failed: {e}",
                        severity="error",
                    )
                    self.call_from_thread(progress.update_status, f"Error: {e}")
                    return

                self.call_from_thread(progress.set_determinate)

                for i, audio_path in enumerate(files):
                    self.call_from_thread(
                        progress.update_status,
                        f"Transcribing {audio_path.name} ({i + 1}/{len(files)})...",
                    )
                    self.call_from_thread(progress.reset)

                    def on_progress(ratio: float, text: str, audio_dur: float, elapsed: float):
                        self.call_from_thread(
                            progress.update_progress, ratio, text, audio_dur, elapsed,
                        )

                    def on_transcribe_status(phase: str):
                        if phase == "loading_audio":
                            self.call_from_thread(
                                progress.update_status,
                                f"Loading audio: {audio_path.name}...",
                            )
                        elif phase == "processing":
                            self.call_from_thread(
                                progress.update_status,
                                f"Transcribing on Intel GPU: {audio_path.name} ({i + 1}/{len(files)})...",
                            )

                    try:
                        result = transcriber.transcribe(
                            audio_path,
                            on_progress=on_progress,
                            on_status=on_transcribe_status,
                        )
                        speed_info = ""
                        if result.speed_factor > 0:
                            speed_info = f" ({result.speed_factor:.1f}x RT)"
                        self.call_from_thread(
                            self.notify,
                            f"Done: {result.output_path.name}{speed_info}",
                        )
                    except Exception as e:
                        self._log_error(f"Transcription failed: {audio_path.name}", e)
                        self.call_from_thread(
                            self.notify,
                            f"Error: {e}",
                            severity="error",
                        )

                self.call_from_thread(progress.update_status, "Complete!")
            except Exception as e:
                self._log_error("Unexpected error in worker", e)
                self.call_from_thread(
                    self.notify,
                    f"Unexpected error: {e}",
                    severity="error",
                )
                self.call_from_thread(progress.update_status, f"Error: {e}")
            finally:
                self.call_from_thread(self._set_buttons_disabled, False)

        thread = Thread(target=worker, daemon=True)
        thread.start()


def run():
    """Entry point for the application."""
    app = WhisperApp()
    app.run()


if __name__ == "__main__":
    run()
