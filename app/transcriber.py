"""Transcription engine wrapping faster-whisper and OpenVINO GenAI."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

from app.config import TranscribeConfig


@dataclass
class TranscribeResult:
    """Result of a transcription."""
    text: str
    output_path: Optional[Path] = None
    audio_duration: float = 0.0
    elapsed_seconds: float = 0.0

    @property
    def speed_factor(self) -> float:
        """Real-time factor: audio_duration / elapsed. Higher = faster."""
        if self.elapsed_seconds > 0:
            return self.audio_duration / self.elapsed_seconds
        return 0.0


def _load_audio(audio_path: Path, sr: int = 16000) -> np.ndarray:
    """Load audio file to float32 numpy array using PyAV (bundled with faster-whisper).

    Works with all formats ffmpeg/PyAV supports (mp3, wav, m4a, flac, ogg, etc.).
    No system ffmpeg installation required.
    """
    return decode_audio(str(audio_path), sampling_rate=sr)


def _create_faster_whisper_model(config: TranscribeConfig) -> WhisperModel:
    """Create a faster-whisper WhisperModel for CUDA/CPU backends."""
    return WhisperModel(
        config.model,
        device=config.device,
        compute_type=config.compute_type,
        cpu_threads=config.cpu_threads,
    )


_OPENVINO_COMPUTE_MAP = {
    "int8": "int8",
    "float16": "fp16",
    "float32": "fp16",
}

_OPENVINO_MODEL_MAP = {
    "large-v3-turbo": "large-v3",
    "large-v3": "large-v3",
    "medium": "medium",
    "small": "small",
    "base": "base",
    "tiny": "tiny",
}


def _resolve_openvino_repo_id(config: TranscribeConfig) -> str:
    """Resolve the HuggingFace repo ID for an OpenVINO whisper model."""
    model_name = _OPENVINO_MODEL_MAP.get(config.model, config.model)
    precision = _OPENVINO_COMPUTE_MAP.get(config.compute_type, "fp16")
    return f"OpenVINO/whisper-{model_name}-{precision}-ov"


def _create_openvino_pipeline(config: TranscribeConfig):
    """Create an OpenVINO GenAI WhisperPipeline for Intel GPU backend.

    Auto-downloads the pre-converted model from HuggingFace if not cached.
    """
    import openvino_genai
    from huggingface_hub import snapshot_download

    repo_id = _resolve_openvino_repo_id(config)
    model_path = snapshot_download(repo_id)
    return openvino_genai.WhisperPipeline(model_path, device="GPU")


class Transcriber:
    """Wraps faster-whisper / OpenVINO GenAI for audio transcription."""

    def __init__(
        self,
        config: TranscribeConfig,
        on_status: Optional[Callable[[str], None]] = None,
    ):
        self.config = config
        if config.device == "openvino":
            repo_id = _resolve_openvino_repo_id(config)
            if on_status:
                on_status(f"Loading model '{repo_id}'...")
            self.pipeline = _create_openvino_pipeline(config)
            self.model = None
            if on_status:
                on_status(f"Model '{repo_id}' ready.")
        else:
            if on_status:
                on_status(f"Loading model '{config.model}'...")
            self.model = _create_faster_whisper_model(config)
            self.pipeline = None
            if on_status:
                on_status(f"Model '{config.model}' ready.")

    def transcribe(
        self,
        audio_path: Path,
        on_progress: Optional[Callable[[float, str, float, float], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> TranscribeResult:
        """Transcribe an audio file.

        Args:
            audio_path: Path to the audio file.
            on_progress: Callback(ratio, segment_text, audio_duration, elapsed_seconds).
            on_status: Callback(status_message) for phase updates.

        Returns:
            TranscribeResult with full text and output file path.
        """
        if self.config.device == "openvino":
            return self._transcribe_openvino(audio_path, on_progress, on_status)
        return self._transcribe_faster_whisper(audio_path, on_progress)

    def _transcribe_faster_whisper(
        self,
        audio_path: Path,
        on_progress: Optional[Callable[[float, str, float, float], None]] = None,
    ) -> TranscribeResult:
        """Transcribe using faster-whisper backend."""
        start_time = time.monotonic()

        segments, info = self.model.transcribe(
            str(audio_path),
            language=self.config.language,
        )

        duration = info.duration
        all_text_parts: list[str] = []

        for segment in segments:
            all_text_parts.append(segment.text)

            if on_progress and duration > 0:
                progress = segment.end / duration
                elapsed = time.monotonic() - start_time
                on_progress(min(progress, 1.0), segment.text, duration, elapsed)

        elapsed_seconds = time.monotonic() - start_time
        full_text = "".join(all_text_parts).strip()

        output_path = audio_path.with_suffix(".txt")
        output_path.write_text(full_text, encoding="utf-8")

        return TranscribeResult(
            text=full_text,
            output_path=output_path,
            audio_duration=duration,
            elapsed_seconds=elapsed_seconds,
        )

    def _transcribe_openvino(
        self,
        audio_path: Path,
        on_progress: Optional[Callable[[float, str, float, float], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> TranscribeResult:
        """Transcribe using OpenVINO GenAI WhisperPipeline backend.

        Splits audio into 30-second windows and processes each separately
        to provide real-time progress, speed info, and incremental text.
        """
        start_time = time.monotonic()

        if on_status:
            on_status("loading_audio")
        audio = _load_audio(audio_path)
        duration = len(audio) / 16000

        if on_status:
            on_status("processing")

        chunk_samples = 30 * 16000  # 30s windows at 16kHz
        all_text_parts: list[str] = []
        samples_done = 0

        while samples_done < len(audio):
            chunk_audio = audio[samples_done : samples_done + chunk_samples]
            chunk_duration = len(chunk_audio) / 16000

            gen_config = self.pipeline.get_generation_config()
            gen_config.language = f"<|{self.config.language}|>"
            gen_config.return_timestamps = False

            result = self.pipeline.generate(chunk_audio, gen_config)

            text = ""
            if hasattr(result, "texts") and result.texts:
                text = result.texts[0]
            elif hasattr(result, "chunks") and result.chunks:
                text = "".join(c.text for c in result.chunks)

            if text:
                all_text_parts.append(text)

            samples_done += chunk_samples

            if on_progress and duration > 0:
                progress = min(samples_done / len(audio), 1.0)
                elapsed = time.monotonic() - start_time
                on_progress(progress, text, duration, elapsed)

        if on_status:
            on_status("finalizing")

        elapsed_seconds = time.monotonic() - start_time
        full_text = " ".join(all_text_parts).strip()

        output_path = audio_path.with_suffix(".txt")
        output_path.write_text(full_text, encoding="utf-8")

        return TranscribeResult(
            text=full_text,
            output_path=output_path,
            audio_duration=duration,
            elapsed_seconds=elapsed_seconds,
        )
