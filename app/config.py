"""Configuration module for whisper-tui."""

import os
from dataclasses import dataclass, field
from functools import lru_cache

import ctranslate2

VALID_MODELS = ("large-v3-turbo", "large-v3", "medium", "small", "base", "tiny")
VALID_COMPUTE_TYPES = ("int8", "float16", "float32")
VALID_DEVICES = ("cuda", "openvino", "cpu")
VALID_LANGUAGES = ("hu", "en")
MAX_CPU_THREADS = os.cpu_count() or 4

DEVICE_LABELS = {
    "cuda": "CUDA (NVIDIA)",
    "openvino": "Intel GPU",
    "cpu": "CPU",
}


def has_cuda() -> bool:
    """Check whether an NVIDIA CUDA GPU is available."""
    try:
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def has_openvino_gpu() -> bool:
    """Check whether an Intel GPU is available via OpenVINO."""
    try:
        import openvino as ov
        core = ov.Core()
        return any(d == "GPU" or d.startswith("GPU.") for d in core.available_devices)
    except Exception:
        return False


@lru_cache(maxsize=1)
def available_devices() -> tuple[str, ...]:
    """Return device options based on detected hardware.

    Always includes CPU. Adds CUDA if NVIDIA GPU detected.
    Adds openvino if Intel GPU detected.
    """
    devices: list[str] = []
    if has_cuda():
        devices.append("cuda")
    if has_openvino_gpu():
        devices.append("openvino")
    devices.append("cpu")
    return tuple(devices)


def detect_device() -> str:
    """Detect the best available device from detected hardware."""
    devices = available_devices()
    # Prefer CUDA > openvino > cpu (order in available_devices)
    return devices[0]


@dataclass
class TranscribeConfig:
    """Transcription configuration with validated defaults."""

    model: str = "large-v3-turbo"
    compute_type: str = "int8"
    device: str = field(default_factory=detect_device)
    cpu_threads: int = 4
    language: str = "hu"

    def __post_init__(self):
        if self.model not in VALID_MODELS:
            raise ValueError(f"Invalid model: {self.model!r}. Must be one of {VALID_MODELS}")
        if self.compute_type not in VALID_COMPUTE_TYPES:
            raise ValueError(f"Invalid compute_type: {self.compute_type!r}. Must be one of {VALID_COMPUTE_TYPES}")
        if self.device not in VALID_DEVICES:
            raise ValueError(f"Invalid device: {self.device!r}. Must be one of {VALID_DEVICES}")
        if self.language not in VALID_LANGUAGES:
            raise ValueError(f"Invalid language: {self.language!r}. Must be one of {VALID_LANGUAGES}")
        if not (1 <= self.cpu_threads <= MAX_CPU_THREADS):
            raise ValueError(f"Invalid cpu_threads: {self.cpu_threads}. Must be between 1 and {MAX_CPU_THREADS}")
