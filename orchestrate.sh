#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# Orchestrator for wakeword-training
# Ensures all dependencies, PATH, venv, and user prompts are handled

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
TRAINER_SH="$SCRIPT_DIR/trainer.sh"

# Ensure coreutils (for gtimeout) is installed
if ! command -v gtimeout >/dev/null 2>&1; then
  echo "Installing coreutils for gtimeout..."
  if command -v brew >/dev/null 2>&1; then
    brew install coreutils || { echo "Failed to install coreutils."; exit 1; }
  else
    echo "Homebrew not found. Please install coreutils manually."; exit 1
  fi
fi

export PATH="/opt/homebrew/opt/coreutils/libexec/gnubin:$PATH"

# Default behavior: launch web wizard (create venv + start Flask app). Use --cli to run the old shell wizard.
if [[ "${1:-}" != "--cli" ]]; then
  VENV_DIR="$SCRIPT_DIR/.venv"
  if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating repo venv: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install -U pip setuptools wheel >/dev/null 2>&1 || true
    "$VENV_DIR/bin/pip" install flask >/dev/null 2>&1 || true
  fi
  echo "Starting web wizard (http://127.0.0.1:5000)"
  "$VENV_DIR/bin/python" "$SCRIPT_DIR/wakeword_web.py" &
  # try to open a browser where available
  (sleep 1; python3 -m webbrowser http://127.0.0.1:5000) >/dev/null 2>&1 || true
  exit 0
fi

# Prompt for wake word (allow env override)
WAKE_WORD=${WAKE_WORD:-}
if [[ -z "$WAKE_WORD" ]]; then
  read -r -p "Enter the wake word to train: " WAKE_WORD
  WAKE_WORD=${WAKE_WORD:-hey assistant}
fi

# Prompt for model format (allow env override)
FORMAT=${MODEL_FORMAT:-}
if [[ -z "$FORMAT" ]]; then
  PS3="Select model format: "
  select FORMAT in "tflite" "onnx" "both"; do
    case $FORMAT in
      tflite|onnx|both) break;;
      *) echo "Invalid selection.";;
    esac
  done
fi

# Prompt for training profile (allow env override)
PROFILE=${PROFILE:-}
if [[ -z "$PROFILE" ]]; then
  PS3="Select training profile: "
  select PROFILE in "tiny" "medium" "large"; do
    case $PROFILE in
      tiny|medium|large) break;;
      *) echo "Invalid selection.";;
    esac
  done
fi

# Prompt for sample size preset (allow env override)
SAMPLE_PRESET=${SAMPLE_PRESET:-}
if [[ -z "$SAMPLE_PRESET" ]]; then
  echo "Sample size presets (diverse selection):"
  echo "  1) XS (quick)     ~5-10 min   50 pos / 200 neg"
  echo "  2) S  (fast)      ~10-20 min  100 pos / 400 neg"
  echo "  3) M  (balanced)  ~25-45 min  250 pos / 1000 neg"
  echo "  4) L  (thorough)  ~45-90 min  500 pos / 2000 neg"
  echo "  5) XL (max)       ~90-150 min 1000 pos / 4000 neg"
  read -r -p "Choose sample size preset [default: 3]: " SAMPLE_PRESET
  SAMPLE_PRESET=${SAMPLE_PRESET:-3}
fi

case "$SAMPLE_PRESET" in
  1|xs|XS)
    MAX_POSITIVE_SAMPLES=50
    MAX_NEGATIVE_SAMPLES=200
    MIN_PER_SOURCE=1
    ;;
  2|s|S)
    MAX_POSITIVE_SAMPLES=100
    MAX_NEGATIVE_SAMPLES=400
    MIN_PER_SOURCE=2
    ;;
  3|m|M)
    MAX_POSITIVE_SAMPLES=250
    MAX_NEGATIVE_SAMPLES=1000
    MIN_PER_SOURCE=3
    ;;
  4|l|L)
    MAX_POSITIVE_SAMPLES=500
    MAX_NEGATIVE_SAMPLES=2000
    MIN_PER_SOURCE=4
    ;;
  5|xl|XL)
    MAX_POSITIVE_SAMPLES=1000
    MAX_NEGATIVE_SAMPLES=4000
    MIN_PER_SOURCE=5
    ;;
  *)
    echo "Invalid sample preset. Using balanced defaults."
    MAX_POSITIVE_SAMPLES=250
    MAX_NEGATIVE_SAMPLES=1000
    MIN_PER_SOURCE=3
    ;;
esac

# Prompt for threads (allow env override)
THREADS=${THREADS:-}
if [[ -z "$THREADS" ]]; then
  read -r -p "Enter number of CPU threads to use [default: 1]: " THREADS
  THREADS=${THREADS:-1}
fi

# Run trainer.sh with all options
cmd=(
  env
  MAX_POSITIVE_SAMPLES="$MAX_POSITIVE_SAMPLES"
  MAX_NEGATIVE_SAMPLES="$MAX_NEGATIVE_SAMPLES"
  MIN_PER_SOURCE="$MIN_PER_SOURCE"
  bash "$TRAINER_SH"
  --allow-low-disk
  --wake-phrase "$WAKE_WORD"
  --train-profile "$PROFILE"
  --train-threads "$THREADS"
  --model-format "$FORMAT"
)

if [[ -n "${WYOMING_PIPER_HOST:-}" ]]; then
  cmd+=(--wyoming-piper-host "${WYOMING_PIPER_HOST}")
fi
if [[ -n "${WYOMING_PIPER_PORT:-}" ]]; then
  cmd+=(--wyoming-piper-port "${WYOMING_PIPER_PORT}")
fi
if [[ -n "${WYOMING_OWW_HOST:-}" ]]; then
  cmd+=(--wyoming-oww-host "${WYOMING_OWW_HOST}")
fi
if [[ -n "${WYOMING_OWW_PORT:-}" ]]; then
  cmd+=(--wyoming-oww-port "${WYOMING_OWW_PORT}")
fi

"${cmd[@]}"

# Find and display the model artifact
ARTIFACT_DIR="$HOME/wakeword_lab/custom_models"
if [[ "$FORMAT" == "tflite" ]]; then
  find "$ARTIFACT_DIR" -name "*.tflite" -print
elif [[ "$FORMAT" == "onnx" ]]; then
  find "$ARTIFACT_DIR" -name "*.onnx" -print
else
  find "$ARTIFACT_DIR" -name "*.tflite" -print
  find "$ARTIFACT_DIR" -name "*.onnx" -print
fi

echo "Training complete. Model(s) saved in $ARTIFACT_DIR."
