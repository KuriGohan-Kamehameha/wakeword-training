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
