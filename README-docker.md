# Wakeword Training - Docker Stack

Complete Docker Compose stack for training custom wake words with Wyoming Piper and openWakeWord services.

## Quick Start

### 1. Start the Stack

```bash
# Build and start all services
docker compose up -d

# Or rebuild images first
docker compose up -d --build
```

### 2. Train a Wake Word

Using the wrapper script (recommended):

```bash
# Simple training
./docker-train.sh --wake-phrase "Geronimo"

# With specific profile and format
./docker-train.sh \
  --wake-phrase "Jarvis" \
  --profile large \
  --threads 4 \
  --format tflite

# For a specific device (e.g., Anki Vector)
./docker-train.sh \
  --wake-phrase "Geronimo" \
  --device anki_vector_wirepod
```

Or directly in the container:

```bash
# Open shell in trainer container
docker compose exec trainer /bin/bash

# Run training
bash trainer.sh \
  --non-interactive \
  --no-tmux \
  --wake-phrase "Computer" \
  --train-profile medium \
  --train-threads 2 \
  --model-format tflite
```

### 3. Access Services

- **Web Wizard**: http://localhost:5000
- **Piper TTS**: tcp://localhost:10200
- **openWakeWord**: tcp://localhost:10400

## Services

### piper
Wyoming Piper text-to-speech service for generating training audio.
- Port: 10200
- Voice: en_US-lessac-medium (configurable)

### openwakeword
Wyoming openWakeWord wake word detection service.
- Port: 10400
- Custom models directory: `./wakeword_lab/custom_models`

### trainer
Training environment with all dependencies pre-installed.
- Python 3.11
- PyTorch, ScienceBrain, datasets, etc.
- openWakeWord training scripts
- Access to all training tools

### web-wizard (optional)
Web-based training wizard.
- Port: 5000
- Pre-configured to use containerized Piper and openWakeWord

## Volume Mounts

- `./wakeword_lab/custom_models` → Trained models output
- `./wakeword_lab/data` → Training datasets
- `training-workspace` → Workspace for training runs

## Environment Variables

Available in all containers:

```bash
WYOMING_PIPER_HOST=piper
WYOMING_PIPER_PORT=10200
WYOMING_OPENWAKEWORD_HOST=openwakeword
WYOMING_OPENWAKEWORD_PORT=10400
BASE_DIR=/workspace
ALLOW_LOW_DISK=1
```

## Training Workflow

1. **Start services**: `docker compose up -d`
2. **Train model**: `./docker-train.sh --wake-phrase "YourPhrase"`
3. **Find model**: Check `./wakeword_lab/custom_models/`
4. **Deploy model**: Copy `.tflite` to your target device

## Advanced Usage

### Rebuild Images

```bash
docker compose build
```

### View Logs

```bash
# All services
docker compose logs

# Specific service
docker compose logs trainer
docker compose logs -f piper  # Follow logs
```

### Stop Services

```bash
# Stop all
docker compose down

# Stop and remove volumes
docker compose down -v
```

### Custom Configuration

Edit `docker-compose.yml` to:
- Change Piper voice model
- Add preload models to openWakeWord
- Adjust resource limits
- Configure additional volumes

### Interactive Development

```bash
# Open shell in trainer
./docker-train.sh --shell

# Or directly
docker compose exec trainer /bin/bash

# Access Python
docker compose exec trainer python

# Run custom scripts
docker compose exec trainer python generate_dataset.py --help
```

## Troubleshooting

### Services not healthy

```bash
# Check service status
docker compose ps

# View logs
docker compose logs piper
docker compose logs openwakeword

# Restart services
docker compose restart piper openwakeword
```

### Training fails

```bash
# Check trainer logs
docker compose logs trainer

# Open shell and debug
docker compose exec trainer /bin/bash
cd /workspace
ls -la
```

### Out of disk space

```bash
# Clean up Docker
docker system prune -a

# Remove training workspace
docker volume rm wakeword-training_training-workspace
```

## Examples

### Train for ReSpeaker XVF3800

```bash
./docker-train.sh \
  --wake-phrase "Hey Assistant" \
  --device respeaker_xvf3800
```

### Train for ESPHome

```bash
./docker-train.sh \
  --wake-phrase "Computer" \
  --device esphome_generic \
  --profile tiny
```

### Train Multiple Models

```bash
for phrase in "Alexa" "Computer" "Jarvis"; do
  ./docker-train.sh --wake-phrase "$phrase" --profile medium
done
```

### Custom Dataset

```bash
# Copy your audio files
cp -r my_audio_files ./wakeword_lab/data/

# Train with custom dataset
docker compose exec trainer bash -c '
  python generate_dataset.py \
    --wake-phrase "CustomWord" \
    --positive-sources /workspace/data/my_audio_files \
    --output-dir /workspace/custom_dataset
'
```

## Production Deployment

For production use of the Wyoming services:

```bash
# Create production compose file
cp docker-compose.yml docker-compose.prod.yml

# Remove trainer and web-wizard services
# Add resource limits and restart policies
# Configure networks and security

# Start with production config
docker compose -f docker-compose.prod.yml up -d
```

## Integration

### Home Assistant

Add Wyoming integration pointing to:
- Piper: `localhost:10200`
- openWakeWord: `localhost:10400`

### Wire-pod (Anki Vector)

Copy trained `.tflite` models to wire-pod's custom model directory.

### ESPHome

Copy `.tflite` model and add to ESPHome configuration:

```yaml
micro_wake_word:
  model: your_model.tflite
```

## Support

For issues or questions:
- Check logs: `docker compose logs`
- Open shell: `./docker-train.sh --shell`
- Review training logs in `./wakeword_lab/training_runs/`
