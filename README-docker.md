# Wakeword Training - Docker Workflow

This project is Docker-first and optimized for repeatable, end-to-end training runs.

## Services

`docker compose` defines:

- `piper` (Wyoming Piper TTS)
- `openwakeword` (Wyoming openWakeWord server)
- `trainer` (training runtime)
- `web-wizard` (optional Flask UI)

## Entry points

- Linux/macOS: `./docker-train.sh`
- Windows PowerShell: `./docker-train.ps1`

## End-to-end run (Theodora, 4 threads)

```bash
./docker-train.sh \
  --wake-phrase "Theodora" \
  --device esphome_generic \
  --threads 4 \
  --format tflite
```

PowerShell equivalent:

```powershell
./docker-train.ps1 --wake-phrase "Theodora" --device esphome_generic --threads 4 --format tflite
```

## Optional sample generation

```bash
./docker-train.sh \
  --wake-phrase "Theodora" \
  --generate-samples \
  --positives 240 \
  --negatives 240
```

Generated positives are stored per wakeword in `wakeword_lab/data/positives/<wakeword_slug>/` and use highest-quality available local TTS voices (novelty voices excluded) plus speech variants.
Generated negatives are pooled in `wakeword_lab/data/negatives/`, append-only, and include both non-speech and speech (diverse voices + diverse non-wake phrases).

Host prerequisites for `--generate-samples`:

- `ffmpeg`
- local TTS backend (`espeak-ng`/`espeak` on Linux, `say` on macOS, or Windows PowerShell speech synthesis)

The generator replaces previous positives only for the current wakeword folder, and appends only new uniquely indexed negatives.
Each run adds at least 50 new negatives, even if `--negatives` is set lower.

## Stability/performance guards

- Service health checks fail fast with logs if stack readiness times out.
- Input negatives are normalized to mono 16k PCM for augmentation stability.
- `tiny` profile defaults now cap manifest size lower to avoid memory spikes on smaller machines.
- Clip generation auto-retries once with reduced counts when the first attempt is OOM-killed.
- TFLite conversion uses `onnx2tf` with a cached converter venv to reduce repeat-run time.
- Runtime patching fixes known upstream `openwakeword/train.py` default-flag behavior.

## Utility commands

```bash
# list device presets
./docker-train.sh --list-devices

# rebuild image
./docker-train.sh --build --wake-phrase "Theodora" --device esphome_generic

# trainer shell
./docker-train.sh --shell

# stop stack
docker compose down
```

## Outputs

- Models: `wakeword_lab/custom_models/`
- Input data: `wakeword_lab/data/positives/` and `wakeword_lab/data/negatives/`
- Training runs/logs: Docker volume `training-workspace` (`/workspace/training_runs` in container)

## Troubleshooting

### Docker daemon unavailable

```bash
docker info
```

### Health checks fail

```bash
docker compose ps
docker compose logs --tail=120 piper openwakeword
```

### Reset stack state

```bash
docker compose down -v
docker system prune -f
```
