#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_VERSION="2.0.0-docker"

usage() {
  cat <<USAGE
${SCRIPT_NAME} v${SCRIPT_VERSION}

Docker-first wakeword trainer.

Usage:
  ${SCRIPT_NAME} [options]

Options:
  --destination PATH         Base workspace directory (alias: --base-dir).
  --base-dir PATH            Base workspace directory.
  --runs-dir PATH            Overrides RUNS_DIR.
  --logs-dir PATH            Overrides LOGS_DIR.
  --oww-repo-dir PATH        Overrides OWW_REPO_DIR.
  --custom-models-dir PATH   Overrides CUSTOM_MODELS_DIR.
  --data-dir PATH            Overrides DATA_DIR.
  --min-free-disk-gb NUMBER  Minimum free disk in GB (default: 2).
  --allow-low-disk           Continue even when free disk is below minimum.
  --wake-phrase TEXT         Wake phrase to train.
  --train-profile NAME       tiny|medium|large.
  --train-threads NUMBER     CPU threads to use.
  --model-format NAME        tflite|onnx|both.
  --wyoming-piper-host HOST  Optional connectivity probe target.
  --wyoming-piper-port PORT  Optional connectivity probe target.
  --wyoming-oww-host HOST    Optional connectivity probe target.
  --wyoming-oww-port PORT    Optional connectivity probe target.
  --non-interactive          Skip prompts and use defaults.
  --no-tmux                  Accepted for compatibility (ignored).
  --help, -h                 Show this help.

Environment overrides:
  BASE_DIR, RUNS_DIR, LOGS_DIR, OWW_REPO_DIR, CUSTOM_MODELS_DIR, DATA_DIR,
  WAKE_PHRASE, TRAIN_PROFILE, TRAIN_THREADS, MODEL_FORMAT,
  WYOMING_PIPER_HOST, WYOMING_PIPER_PORT,
  WYOMING_OPENWAKEWORD_HOST, WYOMING_OPENWAKEWORD_PORT,
  MAX_POSITIVE_SAMPLES, MAX_NEGATIVE_SAMPLES, MIN_PER_SOURCE, DATASET_SEED,
  ALLOW_LOW_DISK, MIN_FREE_DISK_GB, NON_INTERACTIVE.
USAGE
}

timestamp_utc() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(timestamp_utc)] [$SCRIPT_NAME] $*" >&2; }
die() { echo "[$(timestamp_utc)] [$SCRIPT_NAME] FATAL: $*" >&2; exit 1; }

on_err() {
  local exit_code=$?
  local line_no=${1:-"?"}
  die "Unhandled error at line ${line_no} (exit=${exit_code})."
}
trap 'on_err $LINENO' ERR

require_cmd() {
  local c="${1:?}"
  command -v "$c" >/dev/null 2>&1 || die "Missing required command: ${c}"
}

expand_tilde() {
  local path="${1:?}"
  if [[ "$path" == "~" ]]; then
    echo "$HOME"
  elif [[ "$path" == "~/"* ]]; then
    echo "${HOME}${path:1}"
  else
    echo "$path"
  fi
}

slugify() {
  echo -n "${1:?}" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//; s/_+/_/g'
}

require_free_disk_gb() {
  local path="${1:?}"
  local min_gb="${2:?}"
  local avail_kb
  avail_kb="$(df -Pk "$path" | awk 'NR==2 {print $4}')"
  [[ "$avail_kb" =~ ^[0-9]+$ ]] || die "Could not determine free disk space at ${path}"

  local avail_gb=$(( avail_kb / 1024 / 1024 ))
  if (( avail_gb < min_gb )); then
    if [[ "${ALLOW_LOW_DISK:-0}" == "1" ]]; then
      log "WARNING: Free disk at ${path} is ${avail_gb}GB (<${min_gb}GB). Continuing due to ALLOW_LOW_DISK=1."
    else
      die "Insufficient free disk at ${path}: ${avail_gb}GB available, need >= ${min_gb}GB."
    fi
  fi
}

prompt_nonempty() {
  local var_name="${1:?}"
  local prompt_text="${2:?}"
  local default_value="${3:?}"

  local value="${!var_name:-}"
  if [[ -z "$value" ]]; then
    if [[ -t 0 && "${NON_INTERACTIVE:-0}" -ne 1 ]]; then
      read -r -p "${prompt_text} [${default_value}]: " value || true
      value="${value:-$default_value}"
    else
      value="$default_value"
    fi
  fi

  value="$(echo -n "$value" | sed 's/^[[:space:]]\+//; s/[[:space:]]\+$//')"
  [[ -n "$value" ]] || die "Input for ${var_name} must not be empty."
  printf -v "$var_name" '%s' "$value"
}

prompt_choice() {
  local var_name="${1:?}"
  local prompt_text="${2:?}"
  local default_value="${3:?}"
  shift 3
  local -a choices=("$@")

  local value="${!var_name:-}"
  if [[ -z "$value" ]]; then
    if [[ -t 0 && "${NON_INTERACTIVE:-0}" -ne 1 ]]; then
      read -r -p "${prompt_text} [${default_value}] (choices: ${choices[*]}): " value || true
      value="${value:-$default_value}"
    else
      value="$default_value"
    fi
  fi

  local ok=0
  for c in "${choices[@]}"; do
    if [[ "$value" == "$c" ]]; then
      ok=1
      break
    fi
  done
  [[ "$ok" -eq 1 ]] || die "Invalid choice for ${var_name}: '${value}'. Allowed: ${choices[*]}"
  printf -v "$var_name" '%s' "$value"
}

port_open() {
  local host="${1:?}"
  local port="${2:?}"
  local timeout_s="${3:-1}"
  python3 - <<PY >/dev/null 2>&1
import socket, sys
s = socket.socket()
s.settimeout(${timeout_s})
try:
    s.connect(("${host}", int(${port})))
    s.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
}

# CLI placeholders
CLI_BASE_DIR=""
CLI_RUNS_DIR=""
CLI_LOGS_DIR=""
CLI_OWW_REPO_DIR=""
CLI_CUSTOM_MODELS_DIR=""
CLI_DATA_DIR=""
CLI_MIN_FREE_DISK_GB=""
CLI_ALLOW_LOW_DISK=0
CLI_WAKE_PHRASE=""
CLI_TRAIN_PROFILE=""
CLI_TRAIN_THREADS=""
CLI_MODEL_FORMAT=""
CLI_WYOMING_PIPER_HOST=""
CLI_WYOMING_PIPER_PORT=""
CLI_WYOMING_OWW_HOST=""
CLI_WYOMING_OWW_PORT=""
CLI_NON_INTERACTIVE=0
CLI_NO_TMUX=0

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --help|-h)
        usage
        exit 0
        ;;
      --allow-low-disk)
        CLI_ALLOW_LOW_DISK=1
        shift
        ;;
      --base-dir|--destination)
        [[ -n "${2:-}" ]] || die "$1 requires a path."
        CLI_BASE_DIR="$2"
        shift 2
        ;;
      --base-dir=*|--destination=*)
        CLI_BASE_DIR="${1#*=}"
        shift
        ;;
      --runs-dir)
        [[ -n "${2:-}" ]] || die "--runs-dir requires a path."
        CLI_RUNS_DIR="$2"
        shift 2
        ;;
      --runs-dir=*)
        CLI_RUNS_DIR="${1#*=}"
        shift
        ;;
      --logs-dir)
        [[ -n "${2:-}" ]] || die "--logs-dir requires a path."
        CLI_LOGS_DIR="$2"
        shift 2
        ;;
      --logs-dir=*)
        CLI_LOGS_DIR="${1#*=}"
        shift
        ;;
      --oww-repo-dir)
        [[ -n "${2:-}" ]] || die "--oww-repo-dir requires a path."
        CLI_OWW_REPO_DIR="$2"
        shift 2
        ;;
      --oww-repo-dir=*)
        CLI_OWW_REPO_DIR="${1#*=}"
        shift
        ;;
      --custom-models-dir)
        [[ -n "${2:-}" ]] || die "--custom-models-dir requires a path."
        CLI_CUSTOM_MODELS_DIR="$2"
        shift 2
        ;;
      --custom-models-dir=*)
        CLI_CUSTOM_MODELS_DIR="${1#*=}"
        shift
        ;;
      --data-dir)
        [[ -n "${2:-}" ]] || die "--data-dir requires a path."
        CLI_DATA_DIR="$2"
        shift 2
        ;;
      --data-dir=*)
        CLI_DATA_DIR="${1#*=}"
        shift
        ;;
      --min-free-disk-gb)
        [[ -n "${2:-}" ]] || die "--min-free-disk-gb requires a number."
        CLI_MIN_FREE_DISK_GB="$2"
        shift 2
        ;;
      --min-free-disk-gb=*)
        CLI_MIN_FREE_DISK_GB="${1#*=}"
        shift
        ;;
      --wake-phrase)
        [[ -n "${2:-}" ]] || die "--wake-phrase requires text."
        CLI_WAKE_PHRASE="$2"
        shift 2
        ;;
      --wake-phrase=*)
        CLI_WAKE_PHRASE="${1#*=}"
        shift
        ;;
      --train-profile)
        [[ -n "${2:-}" ]] || die "--train-profile requires a value."
        CLI_TRAIN_PROFILE="$2"
        shift 2
        ;;
      --train-profile=*)
        CLI_TRAIN_PROFILE="${1#*=}"
        shift
        ;;
      --train-threads)
        [[ -n "${2:-}" ]] || die "--train-threads requires a number."
        CLI_TRAIN_THREADS="$2"
        shift 2
        ;;
      --train-threads=*)
        CLI_TRAIN_THREADS="${1#*=}"
        shift
        ;;
      --model-format)
        [[ -n "${2:-}" ]] || die "--model-format requires a value."
        CLI_MODEL_FORMAT="$2"
        shift 2
        ;;
      --model-format=*)
        CLI_MODEL_FORMAT="${1#*=}"
        shift
        ;;
      --wyoming-piper-host)
        [[ -n "${2:-}" ]] || die "--wyoming-piper-host requires a host."
        CLI_WYOMING_PIPER_HOST="$2"
        shift 2
        ;;
      --wyoming-piper-host=*)
        CLI_WYOMING_PIPER_HOST="${1#*=}"
        shift
        ;;
      --wyoming-piper-port)
        [[ -n "${2:-}" ]] || die "--wyoming-piper-port requires a port."
        CLI_WYOMING_PIPER_PORT="$2"
        shift 2
        ;;
      --wyoming-piper-port=*)
        CLI_WYOMING_PIPER_PORT="${1#*=}"
        shift
        ;;
      --wyoming-oww-host)
        [[ -n "${2:-}" ]] || die "--wyoming-oww-host requires a host."
        CLI_WYOMING_OWW_HOST="$2"
        shift 2
        ;;
      --wyoming-oww-host=*)
        CLI_WYOMING_OWW_HOST="${1#*=}"
        shift
        ;;
      --wyoming-oww-port)
        [[ -n "${2:-}" ]] || die "--wyoming-oww-port requires a port."
        CLI_WYOMING_OWW_PORT="$2"
        shift 2
        ;;
      --wyoming-oww-port=*)
        CLI_WYOMING_OWW_PORT="${1#*=}"
        shift
        ;;
      --non-interactive)
        CLI_NON_INTERACTIVE=1
        shift
        ;;
      --no-tmux)
        CLI_NO_TMUX=1
        shift
        ;;
      --)
        shift
        break
        ;;
      *)
        die "Unknown argument: $1"
        ;;
    esac
  done
}

main() {
  parse_args "$@"

  # Apply CLI overrides to env-backed values.
  [[ -n "$CLI_BASE_DIR" ]] && BASE_DIR="$CLI_BASE_DIR"
  [[ -n "$CLI_RUNS_DIR" ]] && RUNS_DIR="$CLI_RUNS_DIR"
  [[ -n "$CLI_LOGS_DIR" ]] && LOGS_DIR="$CLI_LOGS_DIR"
  [[ -n "$CLI_OWW_REPO_DIR" ]] && OWW_REPO_DIR="$CLI_OWW_REPO_DIR"
  [[ -n "$CLI_CUSTOM_MODELS_DIR" ]] && CUSTOM_MODELS_DIR="$CLI_CUSTOM_MODELS_DIR"
  [[ -n "$CLI_DATA_DIR" ]] && DATA_DIR="$CLI_DATA_DIR"
  [[ -n "$CLI_MIN_FREE_DISK_GB" ]] && MIN_FREE_DISK_GB="$CLI_MIN_FREE_DISK_GB"
  [[ "$CLI_ALLOW_LOW_DISK" -eq 1 ]] && ALLOW_LOW_DISK=1
  [[ -n "$CLI_WAKE_PHRASE" ]] && WAKE_PHRASE="$CLI_WAKE_PHRASE"
  [[ -n "$CLI_TRAIN_PROFILE" ]] && TRAIN_PROFILE="$CLI_TRAIN_PROFILE"
  [[ -n "$CLI_TRAIN_THREADS" ]] && TRAIN_THREADS="$CLI_TRAIN_THREADS"
  [[ -n "$CLI_MODEL_FORMAT" ]] && MODEL_FORMAT="$CLI_MODEL_FORMAT"
  [[ -n "$CLI_WYOMING_PIPER_HOST" ]] && WYOMING_PIPER_HOST="$CLI_WYOMING_PIPER_HOST"
  [[ -n "$CLI_WYOMING_PIPER_PORT" ]] && WYOMING_PIPER_PORT="$CLI_WYOMING_PIPER_PORT"
  [[ -n "$CLI_WYOMING_OWW_HOST" ]] && WYOMING_OPENWAKEWORD_HOST="$CLI_WYOMING_OWW_HOST"
  [[ -n "$CLI_WYOMING_OWW_PORT" ]] && WYOMING_OPENWAKEWORD_PORT="$CLI_WYOMING_OWW_PORT"
  [[ "$CLI_NON_INTERACTIVE" -eq 1 ]] && NON_INTERACTIVE=1

  if [[ "$CLI_NO_TMUX" -ne 1 ]]; then
    log "NOTE: tmux mode is no longer used in Docker-first workflow; running inline."
  fi

  require_cmd bash
  require_cmd python3
  require_cmd tee
  require_cmd find
  require_cmd cp
  require_cmd df

  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

  local base_dir="${BASE_DIR:-/workspace}"
  base_dir="$(expand_tilde "$base_dir")"
  [[ "$base_dir" != "/" ]] || die "BASE_DIR must not be '/'."

  local runs_dir="${RUNS_DIR:-$base_dir/training_runs}"
  local logs_dir="${LOGS_DIR:-$base_dir/logs}"
  local repo_dir="${OWW_REPO_DIR:-$base_dir/openWakeWord_upstream}"
  local custom_models_dir="${CUSTOM_MODELS_DIR:-$base_dir/custom_models}"
  local data_dir="${DATA_DIR:-$base_dir/data}"

  runs_dir="$(expand_tilde "$runs_dir")"
  logs_dir="$(expand_tilde "$logs_dir")"
  repo_dir="$(expand_tilde "$repo_dir")"
  custom_models_dir="$(expand_tilde "$custom_models_dir")"
  data_dir="$(expand_tilde "$data_dir")"

  mkdir -p "$base_dir" "$runs_dir" "$logs_dir" "$custom_models_dir" "$data_dir"
  require_free_disk_gb "$base_dir" "${MIN_FREE_DISK_GB:-2}"

  local train_py="$repo_dir/openwakeword/train.py"
  [[ -f "$train_py" ]] || die "Missing openWakeWord trainer at $train_py. Ensure Docker image includes /workspace/openWakeWord_upstream."

  local dataset_generator="$script_dir/generate_dataset.py"
  [[ -f "$dataset_generator" ]] || die "Missing dataset generator: $dataset_generator"

  local host_piper="${WYOMING_PIPER_HOST:-127.0.0.1}"
  local port_piper="${WYOMING_PIPER_PORT:-10200}"
  local host_oww="${WYOMING_OPENWAKEWORD_HOST:-127.0.0.1}"
  local port_oww="${WYOMING_OPENWAKEWORD_PORT:-10400}"

  if port_open "$host_piper" "$port_piper" 1; then
    log "Detected Wyoming piper at ${host_piper}:${port_piper}"
  else
    log "WARNING: Wyoming piper not reachable at ${host_piper}:${port_piper}"
  fi
  if port_open "$host_oww" "$port_oww" 1; then
    log "Detected Wyoming openwakeword at ${host_oww}:${port_oww}"
  else
    log "WARNING: Wyoming openwakeword not reachable at ${host_oww}:${port_oww}"
  fi

  local wake_phrase="${WAKE_PHRASE:-}"
  prompt_nonempty wake_phrase "Wake phrase to train" "hey assistant"

  local train_profile="${TRAIN_PROFILE:-}"
  prompt_choice train_profile "Training profile" "medium" tiny medium large

  local default_threads
  default_threads="$(python3 - <<'PY'
import os
print(max(1, os.cpu_count() or 1))
PY
)"
  local train_threads="${TRAIN_THREADS:-}"
  prompt_nonempty train_threads "CPU threads to use" "$default_threads"
  [[ "$train_threads" =~ ^[0-9]+$ ]] || die "TRAIN_THREADS must be an integer."

  local model_format="${MODEL_FORMAT:-}"
  prompt_choice model_format "Model format" "tflite" tflite onnx both

  local model_slug
  model_slug="$(slugify "$wake_phrase")"
  [[ -n "$model_slug" ]] || die "Derived model slug is empty."

  local run_id
  run_id="$(date -u +%Y%m%dT%H%M%SZ)"
  local run_dir="$runs_dir/${model_slug}_${run_id}"
  local dataset_dir="$run_dir/dataset"
  local dataset_json="$dataset_dir/dataset.json"
  mkdir -p "$run_dir" "$dataset_dir"

  local epochs=25
  case "$train_profile" in
    tiny) epochs=10 ;;
    medium) epochs=25 ;;
    large) epochs=50 ;;
  esac

  local cfg_in="$repo_dir/examples/custom_model.yml"
  local cfg_out="$run_dir/training_config.yml"
  [[ -f "$cfg_in" ]] || die "Expected training template missing: $cfg_in"
  cp -f "$cfg_in" "$cfg_out"

  RUN_DIR="$run_dir" DATASET_JSON="$dataset_json" WAKE_PHRASE="$wake_phrase" MODEL_SLUG="$model_slug" EPOCHS="$epochs" python3 - <<'PY'
import os
import yaml

cfg_path = os.environ["RUN_DIR"] + "/training_config.yml"
wake_phrase = os.environ["WAKE_PHRASE"]
model_slug = os.environ["MODEL_SLUG"]
epochs = int(os.environ["EPOCHS"])
run_dir = os.environ["RUN_DIR"]
dataset_json = os.environ["DATASET_JSON"]

with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

updated = []

def set_key_recursive(obj, key, value):
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            if k == key:
                obj[k] = value
                updated.append(key)
            else:
                set_key_recursive(obj[k], key, value)
    elif isinstance(obj, list):
        for it in obj:
            set_key_recursive(it, key, value)

for k in ("target_phrase", "target_phrases", "wake_phrase", "wake_phrases"):
    set_key_recursive(cfg, k, [wake_phrase] if k.endswith("s") or k.startswith("target_") else wake_phrase)

for k in ("model_name", "wakeword_name", "wake_word_name"):
    set_key_recursive(cfg, k, model_slug)

for k in ("output_dir", "model_output_dir", "export_dir"):
    set_key_recursive(cfg, k, run_dir)

for k in ("dataset_path", "dataset_json", "custom_dataset_path", "custom_dataset"):
    set_key_recursive(cfg, k, dataset_json)

for k in ("epochs", "n_epochs", "num_epochs", "max_epochs"):
    set_key_recursive(cfg, k, epochs)

with open(cfg_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

print("Updated YAML keys:", sorted(set(updated)))
PY

  local positive_sources="${POSITIVE_SOURCES:-$data_dir/positives}"
  local negative_sources="${NEGATIVE_SOURCES:-$data_dir/negatives}"
  local max_positive="${MAX_POSITIVE_SAMPLES:-250}"
  local max_negative="${MAX_NEGATIVE_SAMPLES:-1000}"
  local min_per_source="${MIN_PER_SOURCE:-3}"
  local dataset_seed="${DATASET_SEED:-42}"

  log "Generating dataset manifest from positives='${positive_sources}' negatives='${negative_sources}'"
  python3 "$dataset_generator" \
    --output-dir "$dataset_dir" \
    --wake-phrase "$wake_phrase" \
    --positive-sources "$positive_sources" \
    --negative-sources "$negative_sources" \
    --max-positives "$max_positive" \
    --max-negatives "$max_negative" \
    --min-per-source "$min_per_source" \
    --seed "$dataset_seed"

  mapfile -t dataset_counts < <(python3 - <<PY
import json
with open("$dataset_json", "r", encoding="utf-8") as f:
    d = json.load(f)
s = d.get("summary", {})
print(int(s.get("selected_positives", 0)))
print(int(s.get("selected_negatives", 0)))
PY
)
  local selected_pos="${dataset_counts[0]:-0}"
  local selected_neg="${dataset_counts[1]:-0}"

  (( selected_pos > 0 )) || die "Dataset manifest has zero positive samples."
  (( selected_neg > 0 )) || die "Dataset manifest has zero negative samples."

  export OMP_NUM_THREADS="$train_threads"
  export OPENBLAS_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1

  local log_file="$run_dir/training.log"
  touch "$run_dir/.start_time"

  log "Training start"
  log "Wake phrase: $wake_phrase"
  log "Run dir: $run_dir"
  log "Config: $cfg_out"
  log "Threads: $train_threads"

  (
    cd "$repo_dir"
    python3 openwakeword/train.py --training_config "$cfg_out" --generate_clips
    python3 openwakeword/train.py --training_config "$cfg_out" --augment_clips
    python3 openwakeword/train.py --training_config "$cfg_out" --train_model
  ) 2>&1 | tee -a "$log_file"

  mapfile -t tflites < <(find "$run_dir" "$repo_dir" -type f -name "*.tflite" -newer "$run_dir/.start_time" 2>/dev/null | sort || true)
  mapfile -t onnxes  < <(find "$run_dir" "$repo_dir" -type f -name "*.onnx"  -newer "$run_dir/.start_time" 2>/dev/null | sort || true)

  case "$model_format" in
    tflite) onnxes=() ;;
    onnx) tflites=() ;;
    both) ;;
  esac

  if [[ ${#tflites[@]} -eq 0 && ${#onnxes[@]} -eq 0 ]]; then
    die "No trained model artifacts (.tflite/.onnx) were found. Check $log_file"
  fi

  for f in "${tflites[@]}" "${onnxes[@]}"; do
    [[ -n "$f" && -f "$f" ]] || continue
    cp -f "$f" "$custom_models_dir/"
    log "Copied artifact: $f -> $custom_models_dir/"
  done

  echo
  echo "=== COMPLETE ==="
  echo "Wake phrase      : $wake_phrase"
  echo "Model slug       : $model_slug"
  echo "Run dir          : $run_dir"
  echo "Training log     : $log_file"
  echo "Artifacts dir    : $custom_models_dir"
  echo "Model format     : $model_format"
  echo "================"
}

main "$@"
