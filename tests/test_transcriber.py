"""Tests for the transcription engine."""

import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from app.config import TranscribeConfig
from app.transcriber import Transcriber, _load_audio


def _make_wav(path: Path, duration_s: float = 1.0, sr: int = 16000) -> Path:
    """Create a valid WAV file with silence."""
    n_samples = int(sr * duration_s)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"\x00\x00" * n_samples)
    return path


class TestLoadAudio:
    def test_returns_float32_array(self, tmp_path):
        wav = _make_wav(tmp_path / "test.wav", duration_s=0.5)
        audio = _load_audio(wav)
        assert audio.dtype == np.float32

    def test_returns_correct_length(self, tmp_path):
        wav = _make_wav(tmp_path / "test.wav", duration_s=1.0, sr=16000)
        audio = _load_audio(wav)
        assert len(audio) == pytest.approx(16000, abs=100)

    def test_resamples_to_16khz(self, tmp_path):
        wav = _make_wav(tmp_path / "test.wav", duration_s=1.0, sr=44100)
        audio = _load_audio(wav, sr=16000)
        assert len(audio) == pytest.approx(16000, abs=100)

    def test_invalid_file_raises(self, tmp_path):
        bad_file = tmp_path / "bad.wav"
        bad_file.write_bytes(b"not audio data")
        with pytest.raises(Exception):
            _load_audio(bad_file)


@pytest.fixture
def config():
    return TranscribeConfig(device="cpu", model="tiny")


@pytest.fixture
def openvino_config():
    return TranscribeConfig(device="openvino", model="tiny")


@pytest.fixture
def mock_segments(mock_whisper_segment):
    """Three segments spanning a 60-second audio."""
    return [
        mock_whisper_segment(0.0, 20.0, " Hello world."),
        mock_whisper_segment(20.0, 40.0, " This is a test."),
        mock_whisper_segment(40.0, 60.0, " Goodbye."),
    ]


class TestTranscriberInit:
    @patch("app.transcriber.WhisperModel")
    def test_creates_model_with_config(self, MockModel, config):
        t = Transcriber(config)
        MockModel.assert_called_once_with(
            "tiny",
            device="cpu",
            compute_type="int8",
            cpu_threads=4,
        )

    @patch("app.transcriber.WhisperModel")
    def test_gpu_config(self, MockModel):
        config = TranscribeConfig(device="cuda", model="large-v3", compute_type="float16")
        t = Transcriber(config)
        MockModel.assert_called_once_with(
            "large-v3",
            device="cuda",
            compute_type="float16",
            cpu_threads=4,
        )

    @patch("app.transcriber.WhisperModel")
    def test_on_status_called_before_and_after_model_load(self, MockModel, config):
        status_messages = []
        Transcriber(config, on_status=lambda msg: status_messages.append(msg))

        assert len(status_messages) == 2
        assert "Loading model" in status_messages[0]
        assert "tiny" in status_messages[0]
        assert "ready" in status_messages[1]
        assert "tiny" in status_messages[1]

    @patch("app.transcriber.WhisperModel")
    def test_on_status_called_before_model_init(self, MockModel, config):
        """Verify status callback fires before WhisperModel is constructed."""
        call_order = []
        MockModel.side_effect = lambda *a, **kw: call_order.append("model_init")

        def on_status(msg):
            call_order.append(f"status:{msg}")

        Transcriber(config, on_status=on_status)

        assert call_order[0].startswith("status:")
        assert "Loading" in call_order[0]
        assert call_order[1] == "model_init"
        assert call_order[2].startswith("status:")
        assert "ready" in call_order[2]

    @patch("app.transcriber.WhisperModel")
    def test_no_on_status_does_not_raise(self, MockModel, config):
        t = Transcriber(config)
        assert t.model is not None


class TestTranscribe:
    @patch("app.transcriber.WhisperModel")
    def test_returns_full_text(self, MockModel, config, mock_segments, mock_whisper_info):
        instance = MockModel.return_value
        instance.transcribe.return_value = (iter(mock_segments), mock_whisper_info)

        t = Transcriber(config)
        result = t.transcribe(Path("test.mp3"))

        assert result.text == "Hello world. This is a test. Goodbye."

    @patch("app.transcriber.WhisperModel")
    def test_calls_whisper_with_language(self, MockModel, config, mock_segments, mock_whisper_info):
        instance = MockModel.return_value
        instance.transcribe.return_value = (iter(mock_segments), mock_whisper_info)

        t = Transcriber(config)
        t.transcribe(Path("test.mp3"))

        instance.transcribe.assert_called_once()
        call_kwargs = instance.transcribe.call_args
        assert call_kwargs[1]["language"] == "hu"

    @patch("app.transcriber.WhisperModel")
    def test_progress_callback_called(self, MockModel, config, mock_segments, mock_whisper_info):
        instance = MockModel.return_value
        instance.transcribe.return_value = (iter(mock_segments), mock_whisper_info)

        t = Transcriber(config)
        progress_values = []
        t.transcribe(Path("test.mp3"), on_progress=lambda p, txt, dur, el: progress_values.append(p))

        assert len(progress_values) == 3
        assert progress_values[0] == pytest.approx(20.0 / 60.0, abs=0.01)
        assert progress_values[1] == pytest.approx(40.0 / 60.0, abs=0.01)
        assert progress_values[2] == pytest.approx(60.0 / 60.0, abs=0.01)

    @patch("app.transcriber.WhisperModel")
    def test_progress_callback_receives_segment_text(self, MockModel, config, mock_segments, mock_whisper_info):
        instance = MockModel.return_value
        instance.transcribe.return_value = (iter(mock_segments), mock_whisper_info)

        t = Transcriber(config)
        texts = []
        t.transcribe(Path("test.mp3"), on_progress=lambda p, txt, dur, el: texts.append(txt))

        assert texts == [" Hello world.", " This is a test.", " Goodbye."]

    @patch("app.transcriber.WhisperModel")
    def test_saves_txt_file(self, MockModel, config, mock_segments, mock_whisper_info, tmp_path):
        instance = MockModel.return_value
        instance.transcribe.return_value = (iter(mock_segments), mock_whisper_info)

        audio_path = tmp_path / "speech.mp3"
        audio_path.write_bytes(b"\x00" * 100)

        t = Transcriber(config)
        result = t.transcribe(audio_path)

        output_path = tmp_path / "speech.txt"
        assert output_path.exists()
        assert output_path.read_text(encoding="utf-8") == "Hello world. This is a test. Goodbye."

    @patch("app.transcriber.WhisperModel")
    def test_returns_output_path(self, MockModel, config, mock_segments, mock_whisper_info, tmp_path):
        instance = MockModel.return_value
        instance.transcribe.return_value = (iter(mock_segments), mock_whisper_info)

        audio_path = tmp_path / "speech.mp3"
        audio_path.write_bytes(b"\x00" * 100)

        t = Transcriber(config)
        result = t.transcribe(audio_path)

        assert result.output_path == tmp_path / "speech.txt"

    @patch("app.transcriber.WhisperModel")
    def test_transcribe_error_raises(self, MockModel, config):
        instance = MockModel.return_value
        instance.transcribe.side_effect = RuntimeError("Model failed")

        t = Transcriber(config)
        with pytest.raises(RuntimeError, match="Model failed"):
            t.transcribe(Path("test.mp3"))

    @patch("app.transcriber.WhisperModel")
    def test_progress_callback_receives_duration_and_elapsed(self, MockModel, config, mock_segments, mock_whisper_info):
        instance = MockModel.return_value
        instance.transcribe.return_value = (iter(mock_segments), mock_whisper_info)

        t = Transcriber(config)
        calls = []
        t.transcribe(Path("test.mp3"), on_progress=lambda p, txt, dur, el: calls.append((dur, el)))

        assert len(calls) == 3
        for dur, el in calls:
            assert dur == 60.0
            assert el >= 0

    @patch("app.transcriber.WhisperModel")
    def test_result_contains_speed_factor(self, MockModel, config, mock_segments, mock_whisper_info, tmp_path):
        instance = MockModel.return_value
        instance.transcribe.return_value = (iter(mock_segments), mock_whisper_info)

        audio_path = tmp_path / "speed.mp3"
        audio_path.write_bytes(b"\x00" * 100)

        t = Transcriber(config)
        result = t.transcribe(audio_path)

        assert result.audio_duration == 60.0
        assert result.elapsed_seconds >= 0

    @patch("app.transcriber.WhisperModel")
    def test_empty_audio_returns_empty_text(self, MockModel, config, mock_whisper_info, tmp_path):
        mock_whisper_info.duration = 0.0
        instance = MockModel.return_value
        instance.transcribe.return_value = (iter([]), mock_whisper_info)

        audio_path = tmp_path / "empty.mp3"
        audio_path.write_bytes(b"\x00" * 10)

        t = Transcriber(config)
        result = t.transcribe(audio_path)

        assert result.text == ""


class TestResolveOpenvinoRepoId:
    def test_large_v3_turbo_falls_back_to_large_v3(self):
        from app.transcriber import _resolve_openvino_repo_id
        config = TranscribeConfig(device="openvino", model="large-v3-turbo", compute_type="int8")
        assert _resolve_openvino_repo_id(config) == "OpenVINO/whisper-large-v3-int8-ov"

    def test_float16_maps_to_fp16(self):
        from app.transcriber import _resolve_openvino_repo_id
        config = TranscribeConfig(device="openvino", model="tiny", compute_type="float16")
        assert _resolve_openvino_repo_id(config) == "OpenVINO/whisper-tiny-fp16-ov"

    def test_float32_maps_to_fp16(self):
        from app.transcriber import _resolve_openvino_repo_id
        config = TranscribeConfig(device="openvino", model="base", compute_type="float32")
        assert _resolve_openvino_repo_id(config) == "OpenVINO/whisper-base-fp16-ov"

    def test_medium_int8(self):
        from app.transcriber import _resolve_openvino_repo_id
        config = TranscribeConfig(device="openvino", model="medium", compute_type="int8")
        assert _resolve_openvino_repo_id(config) == "OpenVINO/whisper-medium-int8-ov"


class TestOpenVinoTranscriberInit:
    @patch("app.transcriber._create_openvino_pipeline")
    def test_creates_openvino_pipeline(self, mock_create, openvino_config):
        mock_create.return_value = MagicMock()
        t = Transcriber(openvino_config)
        mock_create.assert_called_once_with(openvino_config)
        assert t.pipeline is not None
        assert t.model is None

    @patch("app.transcriber._create_openvino_pipeline")
    def test_on_status_called(self, mock_create, openvino_config):
        mock_create.return_value = MagicMock()
        status_messages = []
        Transcriber(openvino_config, on_status=lambda msg: status_messages.append(msg))
        assert len(status_messages) == 2
        assert "Loading model" in status_messages[0]
        assert "OpenVINO/whisper-tiny-int8-ov" in status_messages[0]
        assert "ready" in status_messages[1]
        assert "OpenVINO/whisper-tiny-int8-ov" in status_messages[1]


class TestOpenVinoTranscribe:
    """Tests for chunked OpenVINO transcription (30s windows)."""

    @pytest.fixture
    def mock_load_audio(self):
        """Mock _load_audio to return 60s of silence at 16kHz."""
        audio_data = np.zeros(960000, dtype=np.float32)  # 60s at 16kHz
        with patch("app.transcriber._load_audio", return_value=audio_data) as mock_fn:
            yield mock_fn

    @staticmethod
    def _make_generate_result(text: str):
        """Create a mock generate result with texts attribute."""
        result = MagicMock()
        result.texts = [text]
        result.chunks = []
        return result

    @patch("app.transcriber._create_openvino_pipeline")
    def test_returns_full_text(self, mock_create, openvino_config, mock_load_audio, tmp_path):
        mock_pipeline = MagicMock()
        # 60s audio = 2 x 30s windows
        mock_pipeline.generate.side_effect = [
            self._make_generate_result("Hello world."),
            self._make_generate_result("Goodbye."),
        ]
        mock_create.return_value = mock_pipeline

        audio_path = tmp_path / "test.mp3"
        audio_path.write_bytes(b"\x00" * 100)

        t = Transcriber(openvino_config)
        result = t.transcribe(audio_path)

        assert result.text == "Hello world. Goodbye."

    @patch("app.transcriber._create_openvino_pipeline")
    def test_progress_callback_per_window(self, mock_create, openvino_config, mock_load_audio, tmp_path):
        mock_pipeline = MagicMock()
        mock_pipeline.generate.side_effect = [
            self._make_generate_result(" Part one."),
            self._make_generate_result(" Part two."),
        ]
        mock_create.return_value = mock_pipeline

        audio_path = tmp_path / "test.mp3"
        audio_path.write_bytes(b"\x00" * 100)

        t = Transcriber(openvino_config)
        progress_calls = []
        t.transcribe(audio_path, on_progress=lambda p, txt, dur, el: progress_calls.append((p, txt, dur, el)))

        assert len(progress_calls) == 2
        # First window: 30s / 60s = 0.5
        assert progress_calls[0][0] == pytest.approx(0.5, abs=0.01)
        assert progress_calls[0][1] == " Part one."
        assert progress_calls[0][2] == pytest.approx(60.0, abs=0.1)
        assert progress_calls[0][3] >= 0  # elapsed
        # Second window: 60s / 60s = 1.0
        assert progress_calls[1][0] == pytest.approx(1.0, abs=0.01)
        assert progress_calls[1][1] == " Part two."

    @patch("app.transcriber._create_openvino_pipeline")
    def test_saves_txt_file(self, mock_create, openvino_config, mock_load_audio, tmp_path):
        mock_pipeline = MagicMock()
        mock_pipeline.generate.side_effect = [
            self._make_generate_result("Hello."),
            self._make_generate_result("World."),
        ]
        mock_create.return_value = mock_pipeline

        audio_path = tmp_path / "speech.mp3"
        audio_path.write_bytes(b"\x00" * 100)

        t = Transcriber(openvino_config)
        result = t.transcribe(audio_path)

        output_path = tmp_path / "speech.txt"
        assert output_path.exists()
        assert output_path.read_text(encoding="utf-8") == "Hello. World."
        assert result.output_path == output_path

    @patch("app.transcriber._create_openvino_pipeline")
    def test_result_contains_duration_and_speed(self, mock_create, openvino_config, mock_load_audio, tmp_path):
        mock_pipeline = MagicMock()
        mock_pipeline.generate.side_effect = [
            self._make_generate_result(" A."),
            self._make_generate_result(" B."),
        ]
        mock_create.return_value = mock_pipeline

        audio_path = tmp_path / "test.mp3"
        audio_path.write_bytes(b"\x00" * 100)

        t = Transcriber(openvino_config)
        result = t.transcribe(audio_path)

        assert result.audio_duration == pytest.approx(60.0, abs=0.1)
        assert result.elapsed_seconds >= 0
        assert result.speed_factor >= 0

    @patch("app.transcriber._create_openvino_pipeline")
    def test_on_status_reports_phases(self, mock_create, openvino_config, mock_load_audio, tmp_path):
        mock_pipeline = MagicMock()
        mock_pipeline.generate.side_effect = [
            self._make_generate_result(" A."),
            self._make_generate_result(" B."),
        ]
        mock_create.return_value = mock_pipeline

        audio_path = tmp_path / "test.mp3"
        audio_path.write_bytes(b"\x00" * 100)

        t = Transcriber(openvino_config)
        phases = []
        t.transcribe(audio_path, on_status=lambda phase: phases.append(phase))

        assert phases == ["loading_audio", "processing", "finalizing"]

    @patch("app.transcriber._create_openvino_pipeline")
    def test_short_audio_single_window(self, mock_create, openvino_config, tmp_path):
        """Audio shorter than 30s should process in a single window."""
        audio_data = np.zeros(160000, dtype=np.float32)  # 10s
        with patch("app.transcriber._load_audio", return_value=audio_data):
            mock_pipeline = MagicMock()
            mock_pipeline.generate.side_effect = [
                self._make_generate_result(" Short clip."),
            ]
            mock_create.return_value = mock_pipeline

            audio_path = tmp_path / "short.mp3"
            audio_path.write_bytes(b"\x00" * 100)

            t = Transcriber(openvino_config)
            result = t.transcribe(audio_path)

            assert result.text == "Short clip."
            assert mock_pipeline.generate.call_count == 1
