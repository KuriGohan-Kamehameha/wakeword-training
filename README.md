# wakeword-training

Docker-first wake word training for openWakeWord.

## Prerequisites

- Docker Desktop (or Docker Engine + Compose plugin)
- Running Docker daemon

## Quick Start

```bash
# Optional: inspect available device presets
./docker-train.sh --list-devices

# Train Argus for Atom Echo (tiny+tflite defaults from device profile)
./docker-train.sh \
  --wake-phrase "Argus" \
  --device atom_echo \
  --generate-samples
```

Artifacts are written to:

- `./wakeword_lab/custom_models/`

## Recommended Device Settings

Use these as practical starting points. Adjust after testing.

| Device / Target | Recommended `--device` | Format | Profile | Threads | Size Target | Example |
|---|---|---|---|---:|---:|---|
| Anki Vector (wire-pod) | `anki_vector_wirepod` | `tflite` | `tiny` | 2 | <=100 KB | `./docker-train.sh --wake-phrase "Vector" --device anki_vector_wirepod --generate-samples` |
| ReSpeaker XVF3800 | `respeaker_xvf3800` | `tflite` | `tiny` | 2 | <=50 KB | `./docker-train.sh --wake-phrase "Computer" --device respeaker_xvf3800` |
| ReSpeaker 2-Mics Pi HAT | `respeaker_2mic_pi_hat` | `tflite` | `tiny` | 2 | <=100 KB | `./docker-train.sh --wake-phrase "Computer" --device respeaker_2mic_pi_hat` |
| ReSpeaker 4-Mics Pi HAT | `respeaker_4mic_pi_hat` | `tflite` | `tiny` | 2 | <=150 KB | `./docker-train.sh --wake-phrase "Computer" --device respeaker_4mic_pi_hat` |
| Atom Echo (M5Stack) | `atom_echo` | `tflite` | `tiny` | 2 | <=80 KB | `./docker-train.sh --wake-phrase "Argus" --device atom_echo --generate-samples` |
| Home Assistant Voice (ESPHome-based) | `esphome_generic` | `tflite` | `tiny` | 2 | <=50 KB | `./docker-train.sh --wake-phrase "Hey Home" --device esphome_generic --generate-samples` |
| Home Assistant server (Wyoming openWakeWord on host) | `custom_manual` | `tflite` | `medium` | 2-4 | <=200 KB | `./docker-train.sh --wake-phrase "Hey Home" --device custom_manual --profile medium --threads 4 --format tflite` |

## Common Commands

```bash
# Rebuild images then train
./docker-train.sh --build --wake-phrase "Computer" --device esphome_generic

# Train without synthetic sample generation (use your own data in wakeword_lab/data)
./docker-train.sh --wake-phrase "Jarvis" --profile medium --format tflite

# Open shell in trainer container
./docker-train.sh --shell

# Stop services
docker compose down
```

## Data Layout

- `wakeword_lab/data/positives/` - positive audio clips
- `wakeword_lab/data/negatives/` - negative audio clips
- `wakeword_lab/custom_models/` - exported model artifacts

For deeper Docker stack details, see `README-docker.md`.
