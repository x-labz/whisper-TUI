"""Tests for the configuration module."""

from unittest.mock import patch, MagicMock

import pytest

from app.config import TranscribeConfig, detect_device, has_cuda, has_openvino_gpu, available_devices


class TestTranscribeConfigDefaults:
    def test_default_model(self):
        config = TranscribeConfig()
        assert config.model == "large-v3-turbo"

    def test_default_quantization(self):
        config = TranscribeConfig()
        assert config.compute_type == "int8"

    def test_default_device(self):
        config = TranscribeConfig()
        assert config.device in ("cuda", "openvino", "cpu")

    def test_default_threads(self):
        config = TranscribeConfig()
        assert config.cpu_threads == 4

    def test_default_language(self):
        config = TranscribeConfig()
        assert config.language == "hu"


class TestTranscribeConfigValidation:
    def test_valid_models(self):
        for model in ("large-v3", "large-v3-turbo", "medium", "small", "base", "tiny"):
            config = TranscribeConfig(model=model)
            assert config.model == model

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError, match="model"):
            TranscribeConfig(model="nonexistent-model")

    def test_valid_compute_types(self):
        for ct in ("int8", "float16", "float32"):
            config = TranscribeConfig(compute_type=ct)
            assert config.compute_type == ct

    def test_invalid_compute_type_raises(self):
        with pytest.raises(ValueError, match="compute_type"):
            TranscribeConfig(compute_type="int4")

    def test_valid_devices(self):
        for device in ("cuda", "openvino", "cpu"):
            config = TranscribeConfig(device=device)
            assert config.device == device

    def test_invalid_device_raises(self):
        with pytest.raises(ValueError, match="device"):
            TranscribeConfig(device="tpu")

    def test_openvino_is_valid_device(self):
        config = TranscribeConfig(device="openvino")
        assert config.device == "openvino"

    def test_valid_languages(self):
        for lang in ("hu", "en"):
            config = TranscribeConfig(language=lang)
            assert config.language == lang

    def test_invalid_language_raises(self):
        with pytest.raises(ValueError, match="language"):
            TranscribeConfig(language="xx")

    def test_threads_min(self):
        with pytest.raises(ValueError, match="cpu_threads"):
            TranscribeConfig(cpu_threads=0)

    def test_threads_max(self):
        from app.config import MAX_CPU_THREADS
        with pytest.raises(ValueError, match="cpu_threads"):
            TranscribeConfig(cpu_threads=MAX_CPU_THREADS + 1)

    def test_threads_valid_range(self):
        from app.config import MAX_CPU_THREADS
        for t in (1, 4, min(8, MAX_CPU_THREADS)):
            config = TranscribeConfig(cpu_threads=t)
            assert config.cpu_threads == t


class TestHasCuda:
    @patch("app.config.ctranslate2.get_cuda_device_count", return_value=1)
    def test_cuda_available(self, mock_cuda):
        assert has_cuda() is True

    @patch("app.config.ctranslate2.get_cuda_device_count", return_value=0)
    def test_cuda_not_available(self, mock_cuda):
        assert has_cuda() is False

    @patch("app.config.ctranslate2.get_cuda_device_count", side_effect=Exception("no CUDA"))
    def test_cuda_error_returns_false(self, mock_cuda):
        assert has_cuda() is False


class TestHasOpenvinoGpu:
    def test_openvino_gpu_available(self):
        mock_ov = MagicMock()
        mock_ov.Core.return_value.available_devices = ["CPU", "GPU"]
        with patch.dict("sys.modules", {"openvino": mock_ov}):
            assert has_openvino_gpu() is True

    def test_openvino_gpu_with_suffix(self):
        mock_ov = MagicMock()
        mock_ov.Core.return_value.available_devices = ["CPU", "GPU.0", "GPU.1"]
        with patch.dict("sys.modules", {"openvino": mock_ov}):
            assert has_openvino_gpu() is True

    def test_openvino_no_gpu(self):
        mock_ov = MagicMock()
        mock_ov.Core.return_value.available_devices = ["CPU"]
        with patch.dict("sys.modules", {"openvino": mock_ov}):
            assert has_openvino_gpu() is False

    def test_openvino_import_error_returns_false(self):
        with patch.dict("sys.modules", {"openvino": None}):
            assert has_openvino_gpu() is False


class TestAvailableDevices:
    @patch("app.config.has_openvino_gpu", return_value=False)
    @patch("app.config.has_cuda", return_value=True)
    def test_cuda_and_cpu(self, mock_cuda, mock_ov):
        available_devices.cache_clear()
        devices = available_devices()
        assert devices == ("cuda", "cpu")

    @patch("app.config.has_openvino_gpu", return_value=False)
    @patch("app.config.has_cuda", return_value=False)
    def test_cpu_only(self, mock_cuda, mock_ov):
        available_devices.cache_clear()
        devices = available_devices()
        assert devices == ("cpu",)

    @patch("app.config.has_openvino_gpu", return_value=False)
    @patch("app.config.has_cuda", return_value=False)
    def test_cpu_always_included(self, mock_cuda, mock_ov):
        available_devices.cache_clear()
        devices = available_devices()
        assert "cpu" in devices

    @patch("app.config.has_openvino_gpu", return_value=True)
    @patch("app.config.has_cuda", return_value=False)
    def test_openvino_and_cpu(self, mock_cuda, mock_ov):
        available_devices.cache_clear()
        devices = available_devices()
        assert devices == ("openvino", "cpu")

    @patch("app.config.has_openvino_gpu", return_value=True)
    @patch("app.config.has_cuda", return_value=True)
    def test_all_devices(self, mock_cuda, mock_ov):
        available_devices.cache_clear()
        devices = available_devices()
        assert devices == ("cuda", "openvino", "cpu")


class TestDetectDevice:
    @patch("app.config.available_devices", return_value=("cuda", "cpu"))
    def test_prefers_cuda(self, mock_devices):
        assert detect_device() == "cuda"

    @patch("app.config.available_devices", return_value=("cpu",))
    def test_falls_back_to_cpu(self, mock_devices):
        assert detect_device() == "cpu"

    @patch("app.config.available_devices", return_value=("openvino", "cpu"))
    def test_prefers_openvino_over_cpu(self, mock_devices):
        assert detect_device() == "openvino"

    @patch("app.config.available_devices", return_value=("cuda", "openvino", "cpu"))
    def test_prefers_cuda_over_openvino(self, mock_devices):
        assert detect_device() == "cuda"
