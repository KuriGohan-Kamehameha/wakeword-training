# Wakeword Training - Docker Workflow

This project is Docker-first. Local host bootstrap scripts are intentionally removed from the primary workflow.

## Services

`docker compose` defines:

- `piper` (Wyoming Piper TTS)
- `openwakeword` (Wyoming openWakeWord server)
- `trainer` (training runtime)
- `web-wizard` (optional Flask UI profile)

## Primary CLI

Use `docker-train.sh` as the main training entrypoint.

### Show available devices

```bash
./docker-train.sh --list-devices
```

### Train for Atom Echo

```bash
./docker-train.sh \
  --wake-phrase "Argus" \
  --device atom_echo \
  --generate-samples
```

### Train with explicit settings

```bash
./docker-train.sh \
  --wake-phrase "Jarvis" \
  --profile medium \
  --threads 2 \
  --format tflite
```

## Common Device Settings

Recommended starting points for common targets:

| Device / Target | Recommended `--device` | Format | Profile | Threads | Size Target | Example |
|---|---|---|---|---:|---:|---|
| Anki Vector (wire-pod) | `anki_vector_wirepod` | `tflite` | `tiny` | 2 | <=100 KB | `./docker-train.sh --wake-phrase "Vector" --device anki_vector_wirepod` |
| ReSpeaker XVF3800 | `respeaker_xvf3800` | `tflite` | `tiny` | 2 | <=50 KB | `./docker-train.sh --wake-phrase "Computer" --device respeaker_xvf3800` |
| ReSpeaker 2-Mics Pi HAT | `respeaker_2mic_pi_hat` | `tflite` | `tiny` | 2 | <=100 KB | `./docker-train.sh --wake-phrase "Computer" --device respeaker_2mic_pi_hat` |
| ReSpeaker 4-Mics Pi HAT | `respeaker_4mic_pi_hat` | `tflite` | `tiny` | 2 | <=150 KB | `./docker-train.sh --wake-phrase "Computer" --device respeaker_4mic_pi_hat` |
| Atom Echo (M5Stack) | `atom_echo` | `tflite` | `tiny` | 2 | <=80 KB | `./docker-train.sh --wake-phrase "Argus" --device atom_echo --generate-samples` |
| Home Assistant Voice (ESPHome-based) | `esphome_generic` | `tflite` | `tiny` | 2 | <=50 KB | `./docker-train.sh --wake-phrase "Hey Home" --device esphome_generic` |
| Home Assistant server (Wyoming openWakeWord on host) | `custom_manual` | `tflite` | `medium` | 2-4 | <=200 KB | `./docker-train.sh --wake-phrase "Hey Home" --device custom_manual --profile medium --threads 4 --format tflite` |

### Rebuild image first

```bash
./docker-train.sh --build --wake-phrase "Computer" --device esphome_generic
```

### Open trainer shell

```bash
./docker-train.sh --shell
```

## Web Wizard (optional)

```bash
docker compose --profile web up -d web-wizard
```

Open [http://localhost:5000](http://localhost:5000)

## Outputs and Data

- Models: `./wakeword_lab/custom_models/`
- Input data: `./wakeword_lab/data/positives/` and `./wakeword_lab/data/negatives/`
- Training logs/runs: Docker volume `training-workspace`

## Stop Stack

```bash
docker compose down
```

## Troubleshooting

### Docker daemon not running

Start Docker Desktop and rerun:

```bash
docker info
```

### Services not healthy

```bash
docker compose ps
docker compose logs piper
docker compose logs openwakeword
```

### Clean stale resources

```bash
docker compose down -v
docker system prune -f
```
