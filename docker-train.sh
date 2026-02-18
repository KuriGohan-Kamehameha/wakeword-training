#!/usr/bin/env bash
# Wrapper script to train a wake word using the Docker Compose stack

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
WAKE_PHRASE="${WAKE_PHRASE:-hey assistant}"
TRAIN_PROFILE="${TRAIN_PROFILE:-medium}"
TRAIN_THREADS="${TRAIN_THREADS:-2}"
MODEL_FORMAT="${MODEL_FORMAT:-tflite}"
DEVICE_ID="${DEVICE_ID:-}"
GENERATE_SAMPLES="${GENERATE_SAMPLES:-0}"
NUM_POSITIVES="${NUM_POSITIVES:-100}"
NUM_NEGATIVES="${NUM_NEGATIVES:-100}"

# Show usage
usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Train a wake word using Docker Compose stack with automatic sample generation.

Options:
    --wake-phrase TEXT      Wake phrase to train (default: "hey assistant")
    --profile NAME          Training profile: tiny|medium|large (default: medium)
    --threads NUMBER        CPU threads to use (default: 2)
    --format NAME           Model format: tflite|onnx|both (default: tflite)
    --device ID             Device workflow ID from device_workflows.json
    --generate-samples      Auto-generate positive/negative training samples
    --positives NUMBER      Number of positive samples to generate (default: 100)
    --negatives NUMBER      Number of negative samples to generate (default: 100)
    --build                 Rebuild Docker images before training
    --logs                  Show trainer service logs after starting
    --shell                 Open shell in trainer container instead of training
    --help, -h              Show this help

Environment variables:
    WAKE_PHRASE, TRAIN_PROFILE, TRAIN_THREADS, MODEL_FORMAT, DEVICE_ID
    GENERATE_SAMPLES, NUM_POSITIVES, NUM_NEGATIVES

Examples:
    # Train "Geronimo" for Anki Vector with auto-generated samples
    $0 --wake-phrase "Geronimo" --device anki_vector_wirepod --generate-samples

    # Train with custom profile and sample count
    $0 --wake-phrase "Jarvis" --profile large --threads 4 --generate-samples --positives 200 --negatives 200

    # Rebuild and train
    $0 --build --wake-phrase "Computer" --generate-samples

    # Open shell in trainer container
    $0 --shell
EOF
    exit 0
}

# Parse arguments
BUILD=0
SHOW_LOGS=0
SHELL_MODE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wake-phrase)
            WAKE_PHRASE="$2"
            shift 2
            ;;
        --profile)
            TRAIN_PROFILE="$2"
            shift 2
            ;;
        --threads)
            TRAIN_THREADS="$2"
            shift 2
            ;;
        --format)
            MODEL_FORMAT="$2"
            shift 2
            ;;
        --device)
            DEVICE_ID="$2"
            shift 2
            ;;
        --generate-samples)
            GENERATE_SAMPLES=1
            shift
            ;;
        --positives)
            NUM_POSITIVES="$2"
            shift 2
            ;;
        --negatives)
            NUM_NEGATIVES="$2"
            shift 2
            ;;
        --build)
            BUILD=1
            shift
            ;;
        --logs)
            SHOW_LOGS=1
            shift
            ;;
        --shell)
            SHELL_MODE=1
            shift
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

cd "$SCRIPT_DIR"

echo "=== Wakeword Training Docker Stack ==="
echo ""

# Start services
if [[ $BUILD -eq 1 ]]; then
    echo "Building Docker images..."
    docker compose build
    echo ""
fi

echo "Starting services..."
docker compose up -d

echo "Waiting for services to be healthy..."
sleep 5

# Wait for services
MAX_WAIT=30
WAITED=0
while [[ $WAITED -lt $MAX_WAIT ]]; do
    if docker compose ps | grep -q "healthy.*piper" && \
       docker compose ps | grep -q "healthy.*openwakeword"; then
        echo "✓ All services are healthy"
        break
    fi
    echo "  Waiting for services... ($WAITED/$MAX_WAIT)"
    sleep 2
    WAITED=$((WAITED + 2))
done

if [[ $WAITED -ge $MAX_WAIT ]]; then
    echo "⚠ Warning: Services may not be fully ready"
fi

echo ""

# Generate training samples if requested
if [[ $GENERATE_SAMPLES -eq 1 ]]; then
    echo "=== Generating Training Samples ==="
    echo "Wake phrase:    $WAKE_PHRASE"
    echo "Positives:      $NUM_POSITIVES"
    echo "Negatives:      $NUM_NEGATIVES"
    echo ""
    
    if command -v python3 >/dev/null 2>&1; then
        python3 generate_training_samples.py \
            --wake-phrase "$WAKE_PHRASE" \
            --positives "$NUM_POSITIVES" \
            --negatives "$NUM_NEGATIVES"
        
        echo ""
        echo "Creating dataset manifest..."
        python3 create_dataset_json.py
        echo ""
    else
        echo "⚠ Warning: python3 not found, skipping sample generation"
        echo ""
    fi
fi

# Shell mode
if [[ $SHELL_MODE -eq 1 ]]; then
    echo "Opening interactive shell in trainer container..."
    docker compose run --rm \
        -e WYOMING_PIPER_HOST=piper \
        -e WYOMING_PIPER_PORT=10200 \
        -e WYOMING_OPENWAKEWORD_HOST=openwakeword \
        -e WYOMING_OPENWAKEWORD_PORT=10400 \
        trainer /bin/bash
    exit $?
fi

# Training mode
echo "=== Training Configuration ==="
echo "Wake phrase:    $WAKE_PHRASE"
echo "Profile:        $TRAIN_PROFILE"
echo "Threads:        $TRAIN_THREADS"
echo "Format:         $MODEL_FORMAT"
[[ -n "$DEVICE_ID" ]] && echo "Device:         $DEVICE_ID"
echo ""

# Build training command
CMD="bash trainer.sh \
    --non-interactive \
    --no-tmux \
    --allow-low-disk \
    --wake-phrase \"$WAKE_PHRASE\" \
    --train-profile \"$TRAIN_PROFILE\" \
    --train-threads \"$TRAIN_THREADS\" \
    --model-format \"$MODEL_FORMAT\""

echo "Starting training in container..."
echo ""

# Use docker compose run instead of exec
# This creates a temporary container that runs the command
docker compose run --rm \
    -e WYOMING_PIPER_HOST=piper \
    -e WYOMING_PIPER_PORT=10200 \
    -e WYOMING_OPENWAKEWORD_HOST=openwakeword \
    -e WYOMING_OPENWAKEWORD_PORT=10400 \
    trainer bash -c "$CMD"

EXIT_CODE=$?

echo ""
echo "=== Training Complete ==="
echo "Models saved to: ./wakeword_lab/custom_models/"
echo ""
echo "To view logs:    docker compose logs trainer"
echo "To stop stack:   docker compose down"
echo "To open shell:   $0 --shell"

exit $EXIT_CODE
