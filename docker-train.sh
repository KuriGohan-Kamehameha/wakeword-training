#!/usr/bin/env bash
# Docker-first wrapper to train a wake word.

set -Eeuo pipefail
IFS=$'\n\t'

# Bug-hunt iter 321: NASA P10 readonly on constants — SCRIPT_DIR + WORKFLOWS_JSON
# + ORIG_ARGC set once at script entry, never reassigned.  Same iter-185/281
# scope-expansion class.  Wakeword trainer is launched from cron/launchd as
# well as interactively; pinning prevents env drift from rebinding paths.
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly WORKFLOWS_JSON="$SCRIPT_DIR/device_workflows.json"
readonly ORIG_ARGC=$#

# Defaults
WAKE_PHRASE="${WAKE_PHRASE:-hey assistant}"
TRAIN_PROFILE="${TRAIN_PROFILE:-medium}"
TRAIN_THREADS="${TRAIN_THREADS:-2}"
MODEL_FORMAT="${MODEL_FORMAT:-tflite}"
DEVICE_ID="${DEVICE_ID:-}"
DEVICE_LABEL=""
GENERATE_SAMPLES="${GENERATE_SAMPLES:-0}"
EMIT_PIRANESI_ENTRY="${EMIT_PIRANESI_ENTRY:-0}"
NUM_POSITIVES="${NUM_POSITIVES:-100}"
NUM_NEGATIVES="${NUM_NEGATIVES:-100}"

BUILD=0
SHELL_MODE=0
LIST_DEVICES=0

PROFILE_SET=0
THREADS_SET=0
FORMAT_SET=0

usage() {
  cat <<USAGE
Usage: $0 [OPTIONS]

Train a wake word using the Docker Compose stack.
Run with no options to use interactive mode.

Options:
  --wake-phrase TEXT      Wake phrase to train (default: "hey assistant")
  --profile NAME          Training profile: tiny|medium|large
  --threads NUMBER        CPU threads to use
  --format NAME           Model format: tflite|onnx|both
  --device ID             Device workflow ID from device_workflows.json
  --list-devices          Print available device IDs and exit
  --generate-samples      Auto-generate positive/negative training samples
  --emit-piranesi-entry   Also emit a <slug>.phrases-entry.json sidecar
                          shaped for Piranesi's vector-override receiver
  --positives NUMBER      Positive samples to generate (default: 100)
  --negatives NUMBER      Negative samples to generate (default: 100)
  --build                 Rebuild Docker images before training
  --shell                 Open shell in trainer container instead of training
  --help, -h              Show this help

Examples:
  $0 --wake-phrase "Argus" --device atom_echo --generate-samples
  $0 --wake-phrase "Geronimo" --profile tiny --format tflite
  $0 --shell
USAGE
  exit 0
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

trim() {
  local s="${1:-}"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

require_docker_daemon() {
  require_cmd docker
  docker info >/dev/null 2>&1 || die "Cannot connect to Docker daemon. Start Docker Desktop first."
}

list_devices() {
  require_cmd python3
  [[ -f "$WORKFLOWS_JSON" ]] || die "Missing $WORKFLOWS_JSON"
  python3 - "$WORKFLOWS_JSON" <<'PY'
import json,sys,os
path=sys.argv[1]
# Bug-hunt iter 499: json.load without file size cap — iter-330 class.
if os.path.getsize(path) > 1048576: sys.exit("workflows.json exceeds 1 MB; refusing")
with open(path,'r',encoding='utf-8') as f:
    data=json.load(f)
# Bug-hunt iter 500: data.get('devices',[]) without count cap — iter-338 class; NASA P10 Rule 2.
_MAX_DEVICES = 256
for d in data.get('devices',[])[:_MAX_DEVICES]:
    did=d.get('id','')
    label=d.get('label',did)
    if did:
        print(f"{did}\t{label}")
PY
}

apply_device_defaults() {
  local device_id="${1:?}"
  [[ -f "$WORKFLOWS_JSON" ]] || die "Missing $WORKFLOWS_JSON"

  local output
  output="$(python3 - "$WORKFLOWS_JSON" "$device_id" <<'PY'
import json,sys,os
path,device_id=sys.argv[1],sys.argv[2]
# Bug-hunt iter 499: json.load without file size cap — iter-330 class.
if os.path.getsize(path) > 1048576: sys.exit("workflows.json exceeds 1 MB; refusing")
with open(path,'r',encoding='utf-8') as f:
    data=json.load(f)
defaults=data.get('default',{})
# Bug-hunt iter 500: data.get('devices',[]) without count cap — iter-338 class; NASA P10 Rule 2.
_MAX_DEVICES = 256
devices={d.get('id'):d for d in data.get('devices',[])[:_MAX_DEVICES] if d.get('id')}
if device_id not in devices:
    print(f"Unknown device id: {device_id}", file=sys.stderr)
    sys.exit(2)
d=devices[device_id]
profile=d.get('profile', defaults.get('profile', 'medium'))
threads=d.get('threads', defaults.get('threads', 2))
fmt=d.get('default_format', defaults.get('default_format', 'tflite'))
label=d.get('label', device_id)
print(f"profile={profile}")
print(f"threads={threads}")
print(f"format={fmt}")
print(f"label={label}")
PY
)" || die "Failed resolving --device '${device_id}'"

  # Bug-hunt iter 681: while-read without count cap — iter-338 class; output has exactly 4 lines but cap defensively.
  local k v _n681=0
  while IFS='=' read -r k v; do
    (( _n681 < 16 )) || break; _n681=$(( _n681 + 1 ))
    case "$k" in
      profile)
        if [[ "$PROFILE_SET" -eq 0 ]]; then
          TRAIN_PROFILE="$v"
        fi
        ;;
      threads)
        if [[ "$THREADS_SET" -eq 0 ]]; then
          TRAIN_THREADS="$v"
        fi
        ;;
      format)
        if [[ "$FORMAT_SET" -eq 0 ]]; then
          MODEL_FORMAT="$v"
        fi
        ;;
      label)
        DEVICE_LABEL="$v"
        ;;
    esac
  done <<< "$output"
}

device_exists() {
  local device_id="${1:?}"
  [[ -f "$WORKFLOWS_JSON" ]] || return 1
  command -v python3 >/dev/null 2>&1 || return 1

  python3 - "$WORKFLOWS_JSON" "$device_id" <<'PY' >/dev/null 2>&1
import json,sys,os
path,device_id=sys.argv[1],sys.argv[2]
# Bug-hunt iter 499: json.load without file size cap — iter-330 class.
if os.path.getsize(path) > 1048576: sys.exit("workflows.json exceeds 1 MB; refusing")
with open(path,'r',encoding='utf-8') as f:
    data=json.load(f)
# Bug-hunt iter 500: data.get('devices',[]) without count cap — iter-338 class; NASA P10 Rule 2.
_MAX_DEVICES = 256
ids={d.get('id') for d in data.get('devices',[])[:_MAX_DEVICES] if d.get('id')}
sys.exit(0 if device_id in ids else 1)
PY
}

prompt_default() {
  local __var="${1:?}" prompt="${2:?}" default="${3-}" input
  # Bug-hunt iter 686: read without size cap — iter-386 class.
  read -r -n 4096 -p "$prompt [$default]: " input || input=""
  input="$(trim "$input")"
  if [[ -z "$input" ]]; then
    input="$default"
  fi
  printf -v "$__var" '%s' "$input"
}

prompt_int() {
  local __var="${1:?}" prompt="${2:?}" default="${3:?}" input
  # Bug-hunt iter 322: NASA P10 Rule 2 — `while true` is unbounded.
  # Operator typing garbage forever (or stdin piped with non-integer
  # lines) loops indefinitely.  Cap at 10 attempts; on exhaustion fall
  # back to the default rather than die — this is a prompt helper, not
  # a security gate.  Same iter-278 cryo::ask_yn bounded-loop class.
  local attempts=0
  local MAX_PROMPT_ATTEMPTS=10
  while (( attempts < MAX_PROMPT_ATTEMPTS )); do
    attempts=$(( attempts + 1 ))
    # Bug-hunt iter 687: read without size cap — iter-386 class.
    read -r -n 4096 -p "$prompt [$default]: " input || input=""
    input="$(trim "$input")"
    if [[ -z "$input" ]]; then
      input="$default"
    fi
    if [[ "$input" =~ ^[0-9]+$ ]]; then
      printf -v "$__var" '%s' "$input"
      return 0
    fi
    echo "Please enter a non-negative integer."
  done
  echo "Too many invalid inputs; using default: $default" >&2
  printf -v "$__var" '%s' "$default"
}

prompt_choice() {
  local __var="${1:?}" prompt="${2:?}" default="${3:?}" input options_display valid opt
  shift 3
  local options=("$@")
  options_display="$(IFS='/'; echo "${options[*]}")"

  # Bug-hunt iter 323: NASA P10 Rule 2 — same iter-322/278 class.  Cap at 10
  # attempts; fall back to default on exhaustion.
  local attempts=0
  local MAX_PROMPT_ATTEMPTS=10
  while (( attempts < MAX_PROMPT_ATTEMPTS )); do
    attempts=$(( attempts + 1 ))
    # Bug-hunt iter 688: read without size cap — iter-386 class.
    read -r -n 4096 -p "$prompt [$options_display] (default: $default): " input || input=""
    input="$(trim "$input")"
    if [[ -z "$input" ]]; then
      input="$default"
    fi

    valid=0
    for opt in "${options[@]}"; do
      if [[ "$input" == "$opt" ]]; then
        valid=1
        break
      fi
    done

    if [[ "$valid" -eq 1 ]]; then
      printf -v "$__var" '%s' "$input"
      return 0
    fi

    echo "Invalid choice: $input"
  done
  echo "Too many invalid inputs; using default: $default" >&2
  printf -v "$__var" '%s' "$default"
}

prompt_yes_no() {
  local __var="${1:?}" prompt="${2:?}" default="${3:?}" default_hint input

  case "$default" in
    y|Y|yes|YES|Yes)
      default="y"
      default_hint="Y/n"
      ;;
    n|N|no|NO|No)
      default="n"
      default_hint="y/N"
      ;;
    *)
      die "Internal error: prompt_yes_no default must be y or n"
      ;;
  esac

  # Bug-hunt iter 324: NASA P10 Rule 2 — same iter-322/323/278 class.
  local attempts=0
  local MAX_PROMPT_ATTEMPTS=10
  while (( attempts < MAX_PROMPT_ATTEMPTS )); do
    attempts=$(( attempts + 1 ))
    # Bug-hunt iter 689: read without size cap — iter-386 class.
    read -r -n 4096 -p "$prompt [$default_hint]: " input || input=""
    input="$(trim "$input")"
    if [[ -z "$input" ]]; then
      input="$default"
    fi

    case "$input" in
      y|Y|yes|YES|Yes)
        printf -v "$__var" '1'
        return 0
        ;;
      n|N|no|NO|No)
        printf -v "$__var" '0'
        return 0
        ;;
      *)
        echo "Please answer y or n."
        ;;
    esac
  done
  echo "Too many invalid inputs; using default: $default" >&2
  if [[ "$default" == "y" ]]; then
    printf -v "$__var" '1'
  else
    printf -v "$__var" '0'
  fi
}

interactive_wizard() {
  local device_input generate_default build_default shell_default

  echo "No options supplied; entering interactive mode."
  echo

  prompt_default WAKE_PHRASE "Wake phrase" "$WAKE_PHRASE"

  if [[ -f "$WORKFLOWS_JSON" ]] && command -v python3 >/dev/null 2>&1; then
    echo
    echo "Available devices:"
    list_devices
    echo

    # Bug-hunt iter 325: NASA P10 Rule 2 — same iter-322/323/324 class.
    # Operator typing garbage device IDs forever locks the wizard.
    local dev_attempts=0
    local DEV_MAX_ATTEMPTS=10
    while (( dev_attempts < DEV_MAX_ATTEMPTS )); do
      dev_attempts=$(( dev_attempts + 1 ))
      # Bug-hunt iter 690: read without size cap — iter-386 class.
      read -r -n 4096 -p "Device ID (leave empty for none): " device_input || device_input=""
      device_input="$(trim "$device_input")"

      if [[ -z "$device_input" ]]; then
        DEVICE_ID=""
        break
      fi

      if device_exists "$device_input"; then
        DEVICE_ID="$device_input"
        break
      fi

      echo "Unknown device ID: $device_input"
    done
    if (( dev_attempts >= DEV_MAX_ATTEMPTS )); then
      echo "Too many invalid device IDs; defaulting to none" >&2
      DEVICE_ID=""
    fi
  fi

  echo
  prompt_choice TRAIN_PROFILE "Training profile" "$TRAIN_PROFILE" tiny medium large
  prompt_int TRAIN_THREADS "CPU threads" "$TRAIN_THREADS"
  prompt_choice MODEL_FORMAT "Model format" "$MODEL_FORMAT" tflite onnx both

  echo
  generate_default="n"
  [[ "$GENERATE_SAMPLES" -eq 1 ]] && generate_default="y"
  prompt_yes_no GENERATE_SAMPLES "Generate synthetic training samples?" "$generate_default"
  if [[ "$GENERATE_SAMPLES" -eq 1 ]]; then
    prompt_int NUM_POSITIVES "Positive samples to generate" "$NUM_POSITIVES"
    prompt_int NUM_NEGATIVES "Negative samples to generate" "$NUM_NEGATIVES"
  fi

  echo
  build_default="n"
  [[ "$BUILD" -eq 1 ]] && build_default="y"
  prompt_yes_no BUILD "Rebuild Docker images first?" "$build_default"

  shell_default="n"
  [[ "$SHELL_MODE" -eq 1 ]] && shell_default="y"
  prompt_yes_no SHELL_MODE "Open shell in trainer container instead of training?" "$shell_default"
  echo
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wake-phrase)
      # Bug-hunt iter 496: $2 access without $# guard — iter-323 class; set -u aborts if $2 unbound.
      [[ $# -ge 2 ]] || die "--wake-phrase requires a value"
      WAKE_PHRASE="$2"
      shift 2
      ;;
    --profile)
      [[ $# -ge 2 ]] || die "--profile requires a value"
      TRAIN_PROFILE="$2"
      PROFILE_SET=1
      shift 2
      ;;
    --threads)
      [[ $# -ge 2 ]] || die "--threads requires a value"
      TRAIN_THREADS="$2"
      THREADS_SET=1
      shift 2
      ;;
    --format)
      [[ $# -ge 2 ]] || die "--format requires a value"
      MODEL_FORMAT="$2"
      FORMAT_SET=1
      shift 2
      ;;
    --device)
      [[ $# -ge 2 ]] || die "--device requires a value"
      DEVICE_ID="$2"
      shift 2
      ;;
    --list-devices)
      LIST_DEVICES=1
      shift
      ;;
    --generate-samples)
      GENERATE_SAMPLES=1
      shift
      ;;
    --emit-piranesi-entry)
      EMIT_PIRANESI_ENTRY=1
      shift
      ;;
    --positives)
      [[ $# -ge 2 ]] || die "--positives requires a value"
      NUM_POSITIVES="$2"
      shift 2
      ;;
    --negatives)
      [[ $# -ge 2 ]] || die "--negatives requires a value"
      NUM_NEGATIVES="$2"
      shift 2
      ;;
    --build)
      BUILD=1
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
      die "Unknown option: $1"
      ;;
  esac
done

cd "$SCRIPT_DIR"

if [[ "$ORIG_ARGC" -eq 0 && -t 0 ]]; then
  interactive_wizard
elif [[ "$ORIG_ARGC" -eq 0 ]]; then
  echo "No options supplied and no interactive TTY detected; using defaults."
fi

if [[ "$LIST_DEVICES" -eq 1 ]]; then
  list_devices
  exit 0
fi

if [[ -n "$DEVICE_ID" ]]; then
  apply_device_defaults "$DEVICE_ID"
fi

[[ "$TRAIN_THREADS" =~ ^[0-9]+$ ]] || die "--threads must be an integer"
[[ "$NUM_POSITIVES" =~ ^[0-9]+$ ]] || die "--positives must be an integer"
[[ "$NUM_NEGATIVES" =~ ^[0-9]+$ ]] || die "--negatives must be an integer"

case "$TRAIN_PROFILE" in tiny|medium|large) ;; *) die "--profile must be tiny|medium|large" ;; esac
case "$MODEL_FORMAT" in tflite|onnx|both) ;; *) die "--format must be tflite|onnx|both" ;; esac

require_docker_daemon

echo "=== Wakeword Training Docker Stack ==="

echo "Pulling latest runtime images before this run..."
docker compose pull piper openwakeword
echo

if [[ "$BUILD" -eq 1 ]]; then
  echo "Rebuilding trainer image with fresh base layers (no cache)..."
  docker compose build --pull --no-cache trainer
else
  echo "Refreshing trainer image base layers..."
  docker compose build --pull trainer
fi
echo

echo "Starting services..."
docker compose up -d piper openwakeword

echo "Waiting for services to become healthy..."
MAX_WAIT=40
WAITED=0
SERVICES_HEALTHY=0
while [[ "$WAITED" -lt "$MAX_WAIT" ]]; do
  if docker compose ps piper | grep -qi "healthy" && docker compose ps openwakeword | grep -qi "healthy"; then
    echo "Services are healthy."
    SERVICES_HEALTHY=1
    break
  fi
  sleep 2
  WAITED=$((WAITED + 2))
  echo "  waiting... (${WAITED}s/${MAX_WAIT}s)"
done

if [[ "$SERVICES_HEALTHY" -ne 1 ]]; then
  echo "ERROR: Services did not become healthy within ${MAX_WAIT}s." >&2
  docker compose ps >&2 || true
  docker compose logs --tail=80 piper openwakeword >&2 || true
  exit 1
fi

if [[ "$SHELL_MODE" -eq 1 ]]; then
  echo "Opening shell in trainer container..."
  exec docker compose run --rm \
    -e WYOMING_PIPER_HOST=piper \
    -e WYOMING_PIPER_PORT=10200 \
    -e WYOMING_OPENWAKEWORD_HOST=openwakeword \
    -e WYOMING_OPENWAKEWORD_PORT=10400 \
    trainer /bin/bash
fi

if [[ "$GENERATE_SAMPLES" -eq 1 ]]; then
  echo "Generating synthetic samples (Piper voices only)..."
  docker compose run --rm \
    -e WYOMING_PIPER_HOST=piper \
    -e WYOMING_PIPER_PORT=10200 \
    -e WYOMING_OPENWAKEWORD_HOST=openwakeword \
    -e WYOMING_OPENWAKEWORD_PORT=10400 \
    trainer python3 generate_training_samples.py \
    --wake-phrase "$WAKE_PHRASE" \
    --data-dir /workspace/data \
    --positives "$NUM_POSITIVES" \
    --negatives "$NUM_NEGATIVES"
  echo
fi

echo "=== Training Configuration ==="
echo "Wake phrase:    $WAKE_PHRASE"
echo "Profile:        $TRAIN_PROFILE"
echo "Threads:        $TRAIN_THREADS"
echo "Format:         $MODEL_FORMAT"
if [[ -n "$DEVICE_ID" ]]; then
  echo "Device:         $DEVICE_ID${DEVICE_LABEL:+ ($DEVICE_LABEL)}"
fi
echo

CMD=(
  bash trainer.sh
  --non-interactive
  --no-tmux
  --allow-low-disk
  --base-dir /workspace/data
  --data-dir /workspace/data
  --wake-phrase "$WAKE_PHRASE"
  --train-profile "$TRAIN_PROFILE"
  --train-threads "$TRAIN_THREADS"
  --model-format "$MODEL_FORMAT"
)
if [[ "$EMIT_PIRANESI_ENTRY" -eq 1 ]]; then
  CMD+=( --emit-piranesi-entry )
fi

docker compose run --rm \
  -e WYOMING_PIPER_HOST=piper \
  -e WYOMING_PIPER_PORT=10200 \
  -e WYOMING_OPENWAKEWORD_HOST=openwakeword \
  -e WYOMING_OPENWAKEWORD_PORT=10400 \
  -e DEVICE_ID="$DEVICE_ID" \
  trainer "${CMD[@]}"

EXIT_CODE=$?

echo
echo "=== Training Complete ==="
echo "Training data + runs + logs + models: ./wakeword_lab/data/"
echo "Stop stack: docker compose down"

exit "$EXIT_CODE"
