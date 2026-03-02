# Whisper Audio Transcription TUI App

## Context
Build a Python console application with a Textual TUI that transcribes audio files using the faster-whisper library (CTranslate2-based Whisper implementation). The app supports NVIDIA GPU and multi-threaded CPU processing, with configurable model size, quantization, and language selection (HU/EN).

## Development Approach: TDD (Test-Driven Development)
Every component, feature, and integration must be verified with tests **before** the implementation is considered complete. The workflow for each task is:
1. **Write tests first** (unit tests for the component)
2. **Implement** until tests pass
3. **Refactor** if needed, keeping tests green

### Test Stack
- `pytest` — test runner
- `pytest-asyncio` — async test support (for Textual widgets)
- `pytest-mock` / `unittest.mock` — mocking faster-whisper, GPU detection, filesystem
- `textual.testing` — Textual's built-in pilot testing for widget/app tests

## Architecture

```
whisper-tui/
├── pyproject.toml          # Project config, dependencies
├── requirements.txt        # Pip dependencies
├── app/
│   ├── __init__.py
│   ├── main.py             # Entry point, Textual App class
│   ├── transcriber.py      # Whisper transcription logic (faster-whisper)
│   ├── config.py           # Configuration dataclass + defaults
│   └── widgets/
│       ├── __init__.py
│       ├── file_browser.py # Audio file list widget
│       ├── settings.py     # Settings panel (model, quant, device, language)
│       └── progress.py     # Progress display widget
├── tests/
│   ├── __init__.py
│   ├── conftest.py         # Shared fixtures (mock audio files, mock whisper model)
│   ├── test_config.py      # Unit tests for config module
│   ├── test_transcriber.py # Unit tests for transcription engine (mocked whisper)
│   ├── test_file_browser.py# Widget tests for file browser
│   ├── test_settings.py    # Widget tests for settings panel
│   ├── test_progress.py    # Widget tests for progress display
│   └── test_app.py         # Integration tests for full app flow
```

## Dependencies
- `faster-whisper` — CTranslate2-based Whisper (supports GPU + CPU, int8/float16 quantization)
- `textual` — TUI framework
- `torch` / `nvidia-cublas` — GPU support (optional, via faster-whisper's CUDA deps)
- `pytest` — test runner
- `pytest-asyncio` — async test support for Textual
- `pytest-mock` — mocking utilities

## Key Design Decisions

### Transcription Engine: `faster-whisper`
- Uses CTranslate2 under the hood (4x faster than openai-whisper)
- Native support for: GPU (CUDA), CPU multi-threading, int8/float16 quantization
- Supports all requested models: `large-v3`, `large-v3-turbo`

### Configuration Defaults
| Setting       | Default         | Options                              |
|---------------|-----------------|--------------------------------------|
| Model size    | `large-v3`      | `large-v3`, `large-v3-turbo`, `medium`, `small`, `base`, `tiny` |
| Quantization  | `int8`          | `int8`, `float16`, `float32`         |
| Device        | `cuda` (GPU)    | `cuda`, `cpu`                        |
| CPU threads   | `4`             | `1-16`                               |
| Language      | `hu`            | `hu`, `en`                           |

### TUI Layout (Textual)
```
┌─────────────────────────────────────────────┐
│  Whisper Transcriber                  [Settings] │
├──────────────────────┬──────────────────────┤
│  Audio Files         │  Settings Panel      │
│  ─────────────       │  Model: [large-v3 ▼] │
│  > file1.mp3         │  Quant: [int8 ▼]     │
│    file2.wav         │  Device: [GPU ▼]     │
│    file3.flac        │  Lang: [HU ▼]        │
│                      │  Threads: [4]        │
│                      │                      │
├──────────────────────┴──────────────────────┤
│  Progress: ████████░░░░░░ 58%  02:34        │
│  Status: Transcribing file1.mp3...          │
├─────────────────────────────────────────────┤
│  [Transcribe]  [Transcribe All]  [Quit]     │
└─────────────────────────────────────────────┘
```

### Progress Tracking
- `faster-whisper` provides segment-by-segment output; we estimate progress from audio duration vs. processed segments' timestamps
- Display: progress bar + percentage + elapsed time + current segment text preview

### Output
- Plain text `.txt` file saved alongside the source audio file (e.g., `file1.mp3` → `file1.txt`)

## Implementation Tasks

### Task 1: Project Setup
- Create `pyproject.toml` with dependencies (including test deps)
- Create `requirements.txt`
- Create package structure (`app/`, `app/widgets/`, `tests/`)
- Create `__init__.py` files
- Create `tests/conftest.py` with shared fixtures
- Verify: `pytest` runs and discovers test directory

### Task 2: Configuration Module (`app/config.py`) — TDD
- **Tests first** (`tests/test_config.py`):
  - Default values are correct (large-v3, int8, cuda, 4 threads, hu)
  - GPU detection returns `cpu` when CUDA unavailable (mocked)
  - GPU detection returns `cuda` when CUDA available (mocked)
  - Config validates invalid values (unknown model, bad thread count)
  - Config serialization/deserialization works
- **Implement**: Dataclass `TranscribeConfig` with defaults and validation
- Verify: `pytest tests/test_config.py` — all green

### Task 3: Transcription Engine (`app/transcriber.py`) — TDD
- **Tests first** (`tests/test_transcriber.py`):
  - `Transcriber` initializes `WhisperModel` with correct params from config
  - `transcribe()` calls faster-whisper with correct language and returns text
  - Progress callback is called with increasing percentages
  - Output `.txt` file is written with correct content
  - Handles transcription errors gracefully
  - All tests mock `faster-whisper` (no real model needed)
- **Implement**: `Transcriber` class with mocked whisper dependency
- Verify: `pytest tests/test_transcriber.py` — all green

### Task 4: File Browser Widget (`app/widgets/file_browser.py`) — TDD
- **Tests first** (`tests/test_file_browser.py`):
  - Lists only audio files (.mp3, .wav, .flac, .ogg, .m4a) from a directory
  - Ignores non-audio files
  - Handles empty directory
  - File selection works (Textual pilot test)
  - Shows file size
- **Implement**: Textual widget with selectable file list
- Verify: `pytest tests/test_file_browser.py` — all green

### Task 5: Settings Panel Widget (`app/widgets/settings.py`) — TDD
- **Tests first** (`tests/test_settings.py`):
  - All dropdowns render with correct default values
  - Changing model selection updates config
  - Changing device to CPU shows thread count input
  - Changing device to GPU hides thread count input
  - Language toggle between HU/EN works
  - Returns correct `TranscribeConfig` from current selections
- **Implement**: Textual widget with Select/RadioSet components
- Verify: `pytest tests/test_settings.py` — all green

### Task 6: Progress Widget (`app/widgets/progress.py`) — TDD
- **Tests first** (`tests/test_progress.py`):
  - Progress bar renders at 0% initially
  - `update_progress(50)` updates bar to 50%
  - Status text updates correctly
  - Elapsed time displays and increments
  - Reset clears all state
- **Implement**: Textual widget with ProgressBar
- Verify: `pytest tests/test_progress.py` — all green

### Task 7: Main App (`app/main.py`) — TDD
- **Tests first** (`tests/test_app.py`):
  - App mounts with all widgets (file browser, settings, progress, buttons)
  - Transcribe button is disabled when no file selected
  - Transcribe button triggers transcription with current settings
  - Transcribe All processes all listed files
  - Quit button / `q` key exits app
  - Full flow integration: select file → configure → transcribe → verify output
- **Implement**: Textual `App` class composing all widgets
- Verify: `pytest tests/test_app.py` — all green

### Task 8: Final Verification
- `pytest` — all tests pass (unit + integration)
- `pytest --cov=app` — check coverage report
- Manual test with real audio file on GPU and CPU
- Handle edge cases: no audio files, model download progress, CUDA not available
- Error handling and user-friendly messages

## Verification
1. `pip install -e .` or `pip install -r requirements.txt`
2. `pytest` — all unit and integration tests pass
3. `pytest --cov=app` — review coverage (target: >80%)
4. Place audio files in working directory
5. Run `python -m app.main` (or entry point)
6. Select audio file, verify settings, click Transcribe
7. Confirm progress bar updates and `.txt` output is created
8. Test with both GPU and CPU device settings
