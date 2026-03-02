"""Settings panel widget for transcription configuration."""

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Select, Label, Static

from app.config import (
    TranscribeConfig,
    VALID_MODELS,
    VALID_COMPUTE_TYPES,
    VALID_LANGUAGES,
    DEVICE_LABELS,
    MAX_CPU_THREADS,
    available_devices,
    detect_device,
)


class SettingsPanel(VerticalScroll):
    """Widget for configuring transcription settings."""

    DEFAULT_CSS = """
    SettingsPanel {
        width: 1fr;
        height: 1fr;
        padding: 0 1;
    }
    SettingsPanel Label {
        margin-top: 1;
    }
    SettingsPanel Select {
        width: 100%;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("Model")
        yield Select(
            [(m, m) for m in VALID_MODELS],
            value="large-v3-turbo",
            id="model-select",
        )
        yield Label("Quantization")
        yield Select(
            [(ct, ct) for ct in VALID_COMPUTE_TYPES],
            value="int8",
            id="compute-select",
        )
        yield Label("Device")
        yield Select(
            [(DEVICE_LABELS.get(d, d.upper()), d) for d in available_devices()],
            value=detect_device(),
            id="device-select",
        )
        yield Label("CPU Cores", id="cores-label")
        yield Select(
            [(str(n), n) for n in range(1, MAX_CPU_THREADS + 1)],
            value=4,
            id="cores-select",
        )
        yield Label("Language")
        yield Select(
            [("Magyar (HU)", "hu"), ("English (EN)", "en")],
            value="hu",
            id="language-select",
        )

    def on_mount(self) -> None:
        """Hide cores selector if device is not CPU on initial mount."""
        self._update_cores_visibility()

    def on_select_changed(self, event: Select.Changed) -> None:
        """React to device selection changes."""
        if event.select.id == "device-select":
            self._update_cores_visibility()

    def _update_cores_visibility(self) -> None:
        """Show or hide CPU cores selector based on selected device."""
        device = self.query_one("#device-select", Select).value
        show = device == "cpu"
        self.query_one("#cores-label").display = show
        self.query_one("#cores-select").display = show

    def get_config(self) -> TranscribeConfig:
        """Build a TranscribeConfig from current widget values."""
        model = self.query_one("#model-select", Select).value
        compute_type = self.query_one("#compute-select", Select).value
        device = self.query_one("#device-select", Select).value
        language = self.query_one("#language-select", Select).value
        cpu_threads = self.query_one("#cores-select", Select).value

        return TranscribeConfig(
            model=model,
            compute_type=compute_type,
            device=device,
            cpu_threads=cpu_threads,
            language=language,
        )
