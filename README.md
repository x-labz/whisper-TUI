# Whisper TUI

A terminal-based audio transcription app powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai). Transcribe audio files using your NVIDIA GPU, Intel GPU, or CPU — all from a clean TUI interface.

## Features

- **Multiple hardware backends** — NVIDIA CUDA, Intel GPU (via OpenVINO), or CPU
- **6 Whisper models** — tiny, base, small, medium, large-v3, large-v3-turbo
- **Batch transcription** — transcribe a single file or all files in a directory
- **Real-time progress** — progress bar, current segment text, and speed factor (e.g. 2.5x real-time)
- **Configurable settings** — model, quantization (int8/float16/float32), device, CPU threads, language
- **Automatic device detection** — the app detects available GPUs and picks the best one
- **Supported audio formats** — MP3, WAV, FLAC, OGG, M4A
- **Output** — saves transcriptions as `.txt` files next to the source audio

## Requirements

- Python 3.10 or higher
- Windows / Linux / macOS

### Hardware-specific requirements

| Device | What you need |
|--------|---------------|
| **NVIDIA GPU** | CUDA-compatible GPU + [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) + [cuDNN](https://developer.nvidia.com/cudnn) |
| **Intel GPU** | Intel HD/Iris/Arc GPU + [Intel GPU drivers](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/software/drivers.html) |
| **CPU** | No extra requirements |

## Installation

### Option A: Using uv (recommended)

[Install uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it, then:

```bash
git clone <repo-url>
cd _whisper
uv sync
```

To include dev dependencies (pytest, etc.):

```bash
uv sync --dev
```

### Option B: Using pip

```bash
git clone <repo-url>
cd _whisper
pip install -e .
```

Or install dependencies manually:

```bash
pip install faster-whisper textual openvino openvino-genai numpy huggingface-hub
```

## Usage

### With uv

```bash
uv run whisper-tui
```

### With pip install

```bash
whisper-tui
```

### With Python directly

```bash
python -m app.main
```

### How it works

1. Place your audio files in the `./audio` directory (or browse to them in the app)
2. Select a file from the file browser on the left
3. Adjust settings on the right panel (model, device, quantization, language)
4. Click **Transcribe** for the selected file, or **Transcribe All** for every file in the directory
5. Watch the progress bar and live segment text as transcription runs
6. Find the resulting `.txt` file next to your original audio file

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `Q` | Quit the app |

## Settings

| Setting | Options | Default |
|---------|---------|---------|
| **Model** | tiny, base, small, medium, large-v3, large-v3-turbo | large-v3-turbo |
| **Quantization** | int8, float16, float32 | int8 |
| **Device** | CUDA (NVIDIA), Intel GPU, CPU | Auto-detected best |
| **CPU Threads** | 1 to max cores (shown only when CPU selected) | 4 |
| **Language** | Magyar (HU), English (EN) | HU |

## Running tests

```bash
uv run pytest
```

With coverage:

```bash
uv run pytest --cov=app
```

## Troubleshooting

### "Intel GPU" not showing up in device list

- Make sure Intel GPU drivers are installed
- Verify OpenVINO can see your GPU:
  ```bash
  python -c "import openvino as ov; print(ov.Core().available_devices)"
  ```
  You should see `GPU` or `GPU.0` in the output.

### CUDA device not detected

- Ensure CUDA Toolkit and cuDNN are installed
- Verify with:
  ```bash
  python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"
  ```

### ModuleNotFoundError on startup

All dependencies must be installed in the Python environment you're running from. If you installed with `uv`, run the app with `uv run`. If using system Python, install dependencies with `pip install -e .` first.
