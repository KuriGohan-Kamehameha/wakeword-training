# wakeword-training

Docker-first wakeword training for openWakeWord, with an end-to-end workflow that runs from one command.

## Platform support

- Linux: supported (`bash`, Docker Engine + Compose plugin)
- macOS: supported (Docker Desktop)
- Windows: supported via `docker-train.ps1` (PowerShell + Docker Desktop; Git Bash or WSL fallback also works)

## Prerequisites

- Docker Desktop (Windows/macOS) or Docker Engine + Compose plugin (Linux)
- Running Docker daemon
- `ffmpeg` on host if using `--generate-samples`
- Local TTS backend on host for spoken positives when using `--generate-samples`:
  - Linux: `espeak-ng` or `espeak`
  - macOS: `say` (built in)
  - Windows: PowerShell speech synthesis (`System.Speech`)

## Quick start

### Linux/macOS

```bash
./docker-train.sh --list-devices
./docker-train.sh --wake-phrase "Theodora" --device esphome_generic --threads 4 --format tflite
```

### Windows PowerShell

```powershell
./docker-train.ps1 --wake-phrase "Theodora" --device esphome_generic --threads 4 --format tflite
```

Artifacts are written to:

- `wakeword_lab/custom_models/`

## Sample diversity

If you use `--generate-samples`, the generator now creates:

- positives in a wakeword-specific folder: `wakeword_lab/data/positives/<wakeword_slug>/`
  - spoken wake phrase with highest-quality available local TTS voices (novelty voices excluded) plus speech variants (`clean`, `fast`, `slow`, `telephone`, `quiet`, `loud`, `bright`)
- negatives in a pooled folder: `wakeword_lab/data/negatives/`
  - appends only new uniquely indexed clips (no overwriting existing negatives)
  - adds at least 50 new negatives per generation run
  - includes both non-speech (silence/noise/tones/chirps) and speech negatives (diverse voices saying diverse non-wake phrases)

To regenerate local sample data:

```bash
./docker-train.sh --wake-phrase "Theodora" --generate-samples --positives 240 --negatives 240
```

Each generation run replaces prior positives only for the current wakeword folder, while `negative_*.wav` files are append-only and indexed so each run contributes unique negatives without duplicating existing filenames.

## Recommended device settings

| Device / Target | `--device` | Format | Profile | Threads | Size Target |
|---|---|---|---|---:|---:|
| Anki Vector (wire-pod) | `anki_vector_wirepod` | `tflite` | `tiny` | 2 | <=100 KB |
| ReSpeaker XVF3800 | `respeaker_xvf3800` | `tflite` | `tiny` | 2 | <=50 KB |
| ReSpeaker 2-Mics Pi HAT | `respeaker_2mic_pi_hat` | `tflite` | `tiny` | 2 | <=100 KB |
| ReSpeaker 4-Mics Pi HAT | `respeaker_4mic_pi_hat` | `tflite` | `tiny` | 2 | <=150 KB |
| Atom Echo | `atom_echo` | `tflite` | `tiny` | 2 | <=80 KB |
| Home Assistant Voice (ESPHome) | `esphome_generic` | `tflite` | `tiny` | 2-4 | <=50 KB |
| Home Assistant server | `custom_manual` | `tflite` | `medium` | 2-4 | <=200 KB |

## Reliability/performance notes

- Health checks fail fast if `piper`/`openwakeword` do not become healthy.
- Generated negatives are normalized to mono 16k PCM before augmentation to avoid sample-rate crashes.
- `tiny` profile now uses lower default manifest caps to reduce memory pressure on constrained hosts.
- If clip generation is OOM-killed (`exit 137`), training auto-retries once with reduced clip counts.
- ONNX -> TFLite conversion runs via `onnx2tf` and reuses a cached converter venv to reduce repeat-run time.
- Training auto-patches upstream `openwakeword/train.py` defaults that otherwise trigger unintended conversion behavior.

## Data layout

- `wakeword_lab/data/positives/<wakeword_slug>/`: wakeword-specific positive audio clips
- `wakeword_lab/data/negatives/`: negative audio clips
- `wakeword_lab/custom_models/`: exported model artifacts

For more operational detail, see `README-docker.md`.
