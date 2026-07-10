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
  --emit-piranesi-entry      Emit a <slug>.phrases-entry.json sidecar
                             shaped for Piranesi's vector-override
                             receiver (state/wakeword/phrases.json).
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

# Structured exit codes for machine-consumable failure parsing.
# 0 = success
# 1 = generic / inherited fatal (legacy die() callers)
# 2 = eval gate failed (FPR / recall thresholds in device_workflows.json)
# 3 = dataset insufficient (zero positives or negatives)
# 4 = docker / environment failure
# 5 = config error (bad flags, missing device profile, etc.)
#
# The last line on stderr for any non-zero exit is a JSON blob
# {"exit_code","reason","details"} so downstream tools (CI, deploy
# automation, vector-override install scripts) can parse without
# scraping free-text logs.
emit_structured_exit() {
  local rc="$1" reason="$2" details="${3:-}"
  python3 -c 'import json, sys; print(json.dumps({"exit_code": int(sys.argv[1]), "reason": sys.argv[2], "details": sys.argv[3]}))' \
    "$rc" "$reason" "$details" >&2 || true
}

die() {
  echo "[$(timestamp_utc)] [$SCRIPT_NAME] FATAL: $*" >&2
  emit_structured_exit 1 "fatal" "$*"
  exit 1
}

# die_with_code <message> <exit_code> <reason>
# Use for new exit paths with semantic exit codes (2..5).
die_with_code() {
  local msg="$1" rc="${2:-1}" reason="${3:-fatal}"
  echo "[$(timestamp_utc)] [$SCRIPT_NAME] FATAL: $msg" >&2
  emit_structured_exit "$rc" "$reason" "$msg"
  exit "$rc"
}

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

convert_onnx_with_onnx2tf() {
  local onnx_path="${1:?}"
  local work_root="${2:-}"
  local cleanup_work_root=0

  [[ -f "$onnx_path" ]] || die "ONNX path does not exist: $onnx_path"

  if [[ -z "$work_root" ]]; then
    work_root="$(mktemp -d)"
    cleanup_work_root=1
  fi

  local model_base
  model_base="$(basename "${onnx_path%.onnx}")"
  local out_dir="$work_root/onnx2tf_${model_base}"
  local cache_root="${ONNX2TF_CACHE_DIR:-${BASE_DIR:-/workspace}/.cache/onnx2tf}"
  local venv_dir="$cache_root/venv_tf219_onnx2tf1263"
  local deps_stamp="$venv_dir/.deps_stamp"
  local deps_key="tensorflow==2.19.0|tf-keras==2.19.0|onnx==1.19.0|onnx2tf==1.26.3|onnxruntime|onnx-graphsurgeon|sng4onnx|psutil"
  local output_tflite="${onnx_path%.onnx}.tflite"
  mkdir -p "$out_dir" "$cache_root"

  if [[ ! -x "$venv_dir/bin/python3" ]]; then
    python3 -m venv "$venv_dir"
  fi

  # Bug-hunt iter 478: source without size cap — venv_dir path is influenced
  # by ONNX2TF_CACHE_DIR env var; a replaced activate script executes unbounded
  # shell code.  Same iter-382 class.  Cap at 1 MB; virtualenv activate is always < 10 KB.
  _activate_sz=$(wc -c < "$venv_dir/bin/activate" 2>/dev/null || echo 0)
  if (( _activate_sz > 1048576 )); then
    die "venv activate script too large (${_activate_sz}B > 1 MB): $venv_dir/bin/activate"
  fi
  # shellcheck disable=SC1091
  source "$venv_dir/bin/activate"

  local install_deps=1
  # Bug-hunt iter 702: cat deps_stamp without size cap — iter-330 class.
  _ds_sz=$(wc -c < "$deps_stamp" 2>/dev/null || echo 0)
  _ds_val=""; (( _ds_sz <= 4096 )) && _ds_val="$(cat "$deps_stamp" 2>/dev/null || echo '')"
  if [[ -f "$deps_stamp" ]] && [[ "${_ds_val}" == "$deps_key" ]]; then
    install_deps=0
  fi

  if [[ "$install_deps" -eq 1 ]]; then
    python3 -m pip install --upgrade pip
    python3 -m pip install \
      tensorflow==2.19.0 \
      tf-keras==2.19.0 \
      onnx==1.19.0 \
      onnx2tf==1.26.3 \
      onnxruntime \
      onnx-graphsurgeon \
      sng4onnx \
      psutil
    echo "$deps_key" > "$deps_stamp"
  fi

  local -a onnx2tf_args
  onnx2tf_args=(-i "$onnx_path" -o "$out_dir")
  if ! command -v onnxsim >/dev/null 2>&1; then
    # Avoid noisy optimizer traceback when onnxsim is intentionally absent.
    onnx2tf_args+=(--not_use_onnxsim)
  fi
  # Preserve input layout EXACTLY: onnx2tf's channel-order heuristics transpose
  # openwakeword's (1, frames, 96) input to (1, 96, frames), producing tflites
  # every openwakeword runtime rejects with 'Cannot set tensor: Dimension
  # mismatch'. -kat pins each input to its ONNX shape.
  # onnx2tf matches -kat against its SANITIZED op names (e.g. 'onnx::Flatten_0'
  # becomes 'onnx____Flatten_0') and silently ignores non-matches — emit the raw
  # name plus common sanitisations so one of them lands.
  local _onnx_inputs
  _onnx_inputs=$(python3 - "$onnx_path" <<'PY'
import sys
import onnx
m = onnx.load(sys.argv[1])
names = []
for i in m.graph.input:
    n = i.name
    names.extend({n, n.replace(":", "_"), n.replace(":", "__")})
print(" ".join(names))
PY
)
  local _in
  for _in in $_onnx_inputs; do
    onnx2tf_args+=(-kat "$_in")
  done
  python3 -m onnx2tf "${onnx2tf_args[@]}"

  local candidate="$out_dir/${model_base}_float32.tflite"
  if [[ ! -f "$candidate" ]]; then
    candidate="$(find "$out_dir" -maxdepth 1 -type f -name "*.tflite" | sort | head -n 1 || true)"
  fi
  [[ -n "$candidate" && -f "$candidate" ]] || die "onnx2tf fallback did not produce .tflite for: $onnx_path"

  cp -f "$candidate" "$output_tflite"
  deactivate >/dev/null 2>&1 || true

  rm -rf "$out_dir" 2>/dev/null || true
  if [[ "$cleanup_work_root" -eq 1 ]]; then
    rm -rf "$work_root" 2>/dev/null || true
  fi

  log "Fallback conversion complete: $onnx_path -> $output_tflite"
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
      # Bug-hunt iter 684: read without size cap — iter-386 class.
      read -r -n 4096 -p "${prompt_text} [${default_value}]: " value || true
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
      # Bug-hunt iter 685: read without size cap — iter-386 class.
      read -r -n 4096 -p "${prompt_text} [${default_value}] (choices: ${choices[*]}): " value || true
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
  # Bug-hunt iter 479: $host and $port interpolated into Python heredoc without
  # validation — iter-227 class (shell variables → Python string injection).
  # Validate host matches hostname/IP pattern; validate port is numeric 1-65535.
  if [[ ! "$host" =~ ^[A-Za-z0-9._-]+$ ]]; then
    log "WARN: port_open: invalid host '$host' — skipping probe"; return 1
  fi
  if ! [[ "$port" =~ ^[0-9]+$ ]] || (( port < 1 || port > 65535 )); then
    log "WARN: port_open: invalid port '$port' — skipping probe"; return 1
  fi
  _PORT_OPEN_HOST="$host" _PORT_OPEN_PORT="$port" _PORT_OPEN_TIMEOUT="$timeout_s" \
  python3 - <<'PY' >/dev/null 2>&1
import socket, sys, os
h = os.environ["_PORT_OPEN_HOST"]
p = int(os.environ["_PORT_OPEN_PORT"])
t = float(os.environ["_PORT_OPEN_TIMEOUT"])
s = socket.socket()
s.settimeout(t)
try:
    s.connect((h, p))
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
CLI_EMIT_PIRANESI_ENTRY=0
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
      --emit-piranesi-entry)
        CLI_EMIT_PIRANESI_ENTRY=1
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

  # Training scale presets (higher values for better model quality).
  local train_steps=1600
  local n_samples=700
  local n_samples_val=140
  local default_max_positive=250
  case "$train_profile" in
    tiny)
      train_steps=800
      n_samples=300
      n_samples_val=60
      default_max_positive=200
      ;;
    medium)
      train_steps=1600
      n_samples=700
      n_samples_val=140
      default_max_positive=250
      ;;
    large)
      train_steps=2500
      n_samples=1000
      n_samples_val=200
      default_max_positive=300
      ;;
  esac

  local cfg_in="$repo_dir/examples/custom_model.yml"
  local cfg_out="$run_dir/training_config.yml"
  [[ -f "$cfg_in" ]] || die "Expected training template missing: $cfg_in"
  cp -f "$cfg_in" "$cfg_out"

  local piper_generator_dir_src="${PIPER_SAMPLE_GENERATOR_DIR:-/app/piper-sample-generator}"
  local piper_generator_dir="$run_dir/piper-sample-generator-runtime"
  mkdir -p "$piper_generator_dir"
  if [[ -d "$piper_generator_dir_src" ]]; then
    cp -a "$piper_generator_dir_src/." "$piper_generator_dir/"
  else
    log "WARNING: piper sample generator source directory not found at $piper_generator_dir_src"
  fi

  local piper_gen_py="$piper_generator_dir/generate_samples.py"
  if [[ -f "$piper_gen_py" ]]; then
    PIPER_GEN_PY="$piper_gen_py" python3 - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["PIPER_GEN_PY"])
# Bug-hunt iter 533: read_text() without size cap — iter-330 class.
if path.stat().st_size > 10 * 1024 * 1024:
    raise SystemExit(f"piper generate_samples.py too large: {path.stat().st_size}B")
text = path.read_text(encoding="utf-8")
needle = "model = torch.load(model_path)"
patched = "model = torch.load(model_path, weights_only=False)"

if patched in text:
    print("piper generate_samples.py patch already present")
elif needle in text:
    path.write_text(text.replace(needle, patched, 1), encoding="utf-8")
    print("Patched piper generate_samples.py for PyTorch>=2.6 compatibility")
else:
    print("WARNING: Could not locate expected torch.load() call in piper generate_samples.py")
PY
  else
    log "WARNING: piper sample generator entrypoint not found at $piper_gen_py"
  fi

  # Same PyTorch>=2.6 weights_only treatment for deep-phonemizer: its baked
  # en_us_cmudict_forward.pt checkpoint pickles dp.preprocessing.text.Preprocessor,
  # which torch.load's new weights_only=True default refuses — adversarial-text
  # generation for OOV wake words then dies in Phonemizer.from_checkpoint.
  # The checkpoint is a build-time artifact from the DeepPhonemizer release
  # (trusted, baked into the image), so weights_only=False is appropriate.
  python3 - <<'PY'
from pathlib import Path

try:
    import dp.model.model as dp_model
except ImportError:
    raise SystemExit("deep-phonemizer not installed; skipping dp torch.load patch")
path = Path(dp_model.__file__)
if path.stat().st_size > 10 * 1024 * 1024:
    raise SystemExit(f"dp model.py too large: {path.stat().st_size}B")
text = path.read_text(encoding="utf-8")
needle = "checkpoint = torch.load(checkpoint_path, map_location=device)"
patched = "checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)"
if patched in text:
    print("dp load_checkpoint patch already present")
elif needle in text:
    path.write_text(text.replace(needle, patched, 1), encoding="utf-8")
    print("Patched deep-phonemizer load_checkpoint for PyTorch>=2.6 compatibility")
else:
    print("WARNING: Could not locate expected torch.load() call in dp model.py")
PY

  local piper_model_file="$piper_generator_dir/models/en-us-libritts-high.pt"
  # The submodule ships only the .pt.json sidecar — the ~243 MB checkpoint
  # comes from the rhasspy release (see piper-sample-generator/README.md).
  # Self-heal: download once into the persistent data dir, reuse per run.
  if [[ ! -f "$piper_model_file" ]]; then
    local piper_model_cache="$data_dir/piper_generator_models/en-us-libritts-high.pt"
    if [[ ! -f "$piper_model_cache" ]]; then
      log "Piper generator checkpoint missing; downloading once (~243 MB) to $piper_model_cache"
      mkdir -p "$(dirname "$piper_model_cache")"
      if curl -fL --retry 2 -o "$piper_model_cache.part" \
        "https://github.com/rhasspy/piper-sample-generator/releases/download/v1.0.0/en-us-libritts-high.pt"; then
        mv "$piper_model_cache.part" "$piper_model_cache"
      else
        rm -f "$piper_model_cache.part"
      fi
    fi
    [[ -f "$piper_model_cache" ]] && cp "$piper_model_cache" "$piper_model_file"
  fi
  [[ -f "$piper_model_file" ]] || die "Missing Piper generator model file: $piper_model_file"

  local source_negative_dir="${DATA_NEGATIVE_DIR:-$data_dir/negatives}"
  [[ -d "$source_negative_dir" ]] || die "Negative audio directory missing: $source_negative_dir"

  local normalized_negative_dir="$run_dir/negative_16k"
  mkdir -p "$normalized_negative_dir"

  SOURCE_NEGATIVE_DIR="$source_negative_dir" NORMALIZED_NEGATIVE_DIR="$normalized_negative_dir" python3 - <<'PY'
import os
import numpy as np
import resampy
import soundfile as sf

src = os.environ["SOURCE_NEGATIVE_DIR"]
dst = os.environ["NORMALIZED_NEGATIVE_DIR"]

# Bug-hunt iter 497: os.listdir without count cap — iter-338 class; NASA P10 Rule 2.
_MAX_NEG_FILES = 10_000
wav_files = sorted(
    f for f in os.listdir(src)
    if f.lower().endswith(".wav") and os.path.isfile(os.path.join(src, f))
)[:_MAX_NEG_FILES]
if not wav_files:
    raise SystemExit(f"No .wav files found in negative source directory: {src}")

processed = 0
for name in wav_files:
    in_path = os.path.join(src, name)
    out_path = os.path.join(dst, name)

    audio, sr = sf.read(in_path, dtype="float32", always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != 16000:
        audio = resampy.resample(audio, sr, 16000)

    sf.write(out_path, audio, 16000, subtype="PCM_16")
    processed += 1

print(f"Prepared normalized negative clips: {processed} -> {dst}")
PY

  RUN_DIR="$run_dir" DATASET_JSON="$dataset_json" WAKE_PHRASE="$wake_phrase" MODEL_SLUG="$model_slug" EPOCHS="$epochs" TRAIN_STEPS="$train_steps" N_SAMPLES="$n_samples" N_SAMPLES_VAL="$n_samples_val" PIPER_SAMPLE_GENERATOR_DIR="$piper_generator_dir" DATA_NEGATIVE_DIR="$normalized_negative_dir" python3 - <<'PY'
import os
import yaml

cfg_path = os.environ["RUN_DIR"] + "/training_config.yml"
wake_phrase = os.environ["WAKE_PHRASE"]
model_slug = os.environ["MODEL_SLUG"]
epochs = int(os.environ["EPOCHS"])
train_steps = int(os.environ["TRAIN_STEPS"])
n_samples = int(os.environ["N_SAMPLES"])
n_samples_val = int(os.environ["N_SAMPLES_VAL"])
run_dir = os.environ["RUN_DIR"]
dataset_json = os.environ["DATASET_JSON"]
piper_generator_dir = os.environ.get("PIPER_SAMPLE_GENERATOR_DIR", "").strip()
negative_dir = os.environ.get("DATA_NEGATIVE_DIR", "").strip() or os.path.join(run_dir, model_slug, "negative_train")

# Bug-hunt iter 534: yaml.safe_load without size cap — iter-330 class.
_cfg_sz = os.path.getsize(cfg_path) if os.path.exists(cfg_path) else 0
if _cfg_sz > 1 * 1024 * 1024:
    raise SystemExit(f"training_config.yml too large: {_cfg_sz}B")
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

updated = []

# Bug-hunt iter 708: previously `set_key_recursive` recursed on dict values
# and list elements — NASA P10 Rule 1 (no recursion) violation.  Replace with
# an explicit-stack iterative walk bounded by _MAX_NODES (any realistic
# training_config has well under this).
_MAX_NODES = 100000
def set_key_iterative(root, key, value):
    stack = [root]
    visited = 0
    while stack and visited < _MAX_NODES:
        visited += 1
        node = stack.pop()
        if isinstance(node, dict):
            for k in list(node.keys()):
                if k == key:
                    node[k] = value
                    updated.append(key)
                else:
                    stack.append(node[k])
        elif isinstance(node, list):
            stack.extend(node)
def set_key_recursive(obj, key, value):
    set_key_iterative(obj, key, value)

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

for k in ("steps", "max_steps"):
    set_key_recursive(cfg, k, train_steps)

for k in ("n_samples",):
    set_key_recursive(cfg, k, n_samples)

for k in ("n_samples_val",):
    set_key_recursive(cfg, k, n_samples_val)

if piper_generator_dir:
    for k in ("piper_sample_generator_path", "sample_generator_path"):
        set_key_recursive(cfg, k, piper_generator_dir)

for k in ("rir_paths",):
    set_key_recursive(cfg, k, [negative_dir])

for k in ("background_paths",):
    set_key_recursive(cfg, k, [negative_dir])

for k in ("background_paths_duplication_rate",):
    set_key_recursive(cfg, k, [1])

# Train only from generated positive/adversarial features for portability.
cfg["feature_data_files"] = {}
cfg["batch_n_per_class"] = {"positive": 64, "adversarial_negative": 64}
cfg["false_positive_validation_data_path"] = os.path.join(run_dir, model_slug, "false_positive_validation.npy")
updated.extend(["feature_data_files", "batch_n_per_class", "false_positive_validation_data_path"])

with open(cfg_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

print("Updated YAML keys:", sorted(set(updated)))
PY

  local default_positive_sources="$data_dir/positives"
  local scoped_positive_dir="$data_dir/positives/$model_slug"
  if [[ -d "$scoped_positive_dir" ]] && find "$scoped_positive_dir" -maxdepth 1 -type f | grep -q .; then
    default_positive_sources="$scoped_positive_dir"
  fi
  local default_hard_negative_sources="$data_dir/hard_negatives"
  local positive_sources="${POSITIVE_SOURCES:-$default_positive_sources}"
  local negative_sources="${NEGATIVE_SOURCES:-$normalized_negative_dir,$default_hard_negative_sources}"
  local max_positive="${MAX_POSITIVE_SAMPLES:-$default_max_positive}"
  local max_negative="${MAX_NEGATIVE_SAMPLES:-}"
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

  # Bug-hunt iter 480: $dataset_json interpolated into unquoted Python heredoc —
  # iter-227 class; a path with embedded quotes would break the Python string
  # literal.  Pass via environment variable instead.
  # Bug-hunt iter 691: mapfile without -n count cap — iter-338 class; Python prints exactly 2 lines.
  mapfile -t -n 8 dataset_counts < <(DATASET_JSON="$dataset_json" python3 - <<'PY'
import json, os
# Bug-hunt iter 535: json.load without size cap — iter-330 class.
_dsj = os.environ["DATASET_JSON"]
_dsj_sz = os.path.getsize(_dsj) if os.path.exists(_dsj) else 0
if _dsj_sz > 10 * 1024 * 1024:
    raise SystemExit(f"dataset.json too large: {_dsj_sz}B")
with open(_dsj, "r", encoding="utf-8") as f:
    d = json.load(f)
s = d.get("summary", {})
print(int(s.get("selected_positives", 0)))
print(int(s.get("selected_negatives", 0)))
PY
)
  local selected_pos="${dataset_counts[0]:-0}"
  local selected_neg="${dataset_counts[1]:-0}"

  (( selected_pos > 0 )) || die_with_code "Dataset manifest has zero positive samples." 3 "dataset_insufficient"
  (( selected_neg > 0 )) || die_with_code "Dataset manifest has zero negative samples." 3 "dataset_insufficient"

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
    python3 - <<'PY'
from pathlib import Path
import re

path = Path("openwakeword/train.py")
text = path.read_text(encoding="utf-8")
updated = re.sub(r"num_workers=n_cpus,\s*prefetch_factor=16", "num_workers=0", text, count=1)
updated = re.sub(r'default="False"', "default=False", updated)

if updated != text:
    path.write_text(updated, encoding="utf-8")
    print("Patched train.py defaults and DataLoader settings for container training")
else:
    print("train.py patches already applied")
PY
    run_generate_clips() {
      local attempts=0
      local max_attempts=2
      local rc=0
      while (( attempts < max_attempts )); do
        attempts=$((attempts + 1))
        if python3 openwakeword/train.py --training_config "$cfg_out" --generate_clips; then
          return 0
        fi
        rc=$?
        if [[ "$rc" -eq 137 && "$attempts" -lt "$max_attempts" ]]; then
          log "generate_clips was killed (exit 137). Reducing clip counts and retrying once."
          CFG_PATH="$cfg_out" python3 - <<'PY'
import os
import yaml

cfg_path = os.environ["CFG_PATH"]
# Bug-hunt iter 536: yaml.safe_load without size cap — iter-330 class.
_cfg536_sz = os.path.getsize(cfg_path) if os.path.exists(cfg_path) else 0
if _cfg536_sz > 1 * 1024 * 1024:
    raise SystemExit(f"training_config.yml too large: {_cfg536_sz}B")
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

changes = []

# Bug-hunt iter 709: previously `reduce_recursive` recursed on dict/list values —
# NASA P10 Rule 1 (no recursion) violation.  Replace with an explicit-stack
# iterative walk bounded by _MAX_NODES.
_MAX_NODES_REDUCE = 100000
def reduce_iterative(root, key, factor, minimum):
    stack = [root]
    visited = 0
    while stack and visited < _MAX_NODES_REDUCE:
        visited += 1
        node = stack.pop()
        if isinstance(node, dict):
            for k, v in node.items():
                if k == key and isinstance(v, (int, float)):
                    old = int(v)
                    new = max(minimum, int(round(old * factor)))
                    if new < old:
                        node[k] = new
                        changes.append((key, old, new))
                else:
                    stack.append(v)
        elif isinstance(node, list):
            stack.extend(node)
def reduce_recursive(obj, key, factor, minimum):
    reduce_iterative(obj, key, factor, minimum)

reduce_recursive(cfg, "n_samples", 0.6, 120)
reduce_recursive(cfg, "n_samples_val", 0.6, 24)
reduce_recursive(cfg, "steps", 0.7, 600)

with open(cfg_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

if changes:
    for key, old, new in changes:
        print(f"Reduced {key}: {old} -> {new}")
else:
    print("No reducible keys found in config; retrying with unchanged values")
PY
          continue
        fi
        return "$rc"
      done
      return 1
    }
    run_generate_clips
    # Bug-hunt iter 481: $cfg_out interpolated into unquoted Python heredoc —
    # iter-227 class; path with embedded quotes would break the string literal.
    # Pass via environment variable instead.
    CFG_OUT="$cfg_out" python3 - <<'PY'
import glob
import os
import numpy as np
import resampy
import soundfile as sf
import yaml

cfg_path = os.environ["CFG_OUT"]
# Bug-hunt iter 537: yaml.safe_load without size cap — iter-330 class.
_cfg537_sz = os.path.getsize(cfg_path) if os.path.exists(cfg_path) else 0
if _cfg537_sz > 1 * 1024 * 1024:
    raise SystemExit(f"training_config.yml too large: {_cfg537_sz}B")
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

feature_dir = os.path.join(cfg["output_dir"], cfg["model_name"])
clip_dirs = [
    os.path.join(feature_dir, "positive_train"),
    os.path.join(feature_dir, "positive_test"),
    os.path.join(feature_dir, "negative_train"),
    os.path.join(feature_dir, "negative_test"),
]

processed = 0
_MAX_CLIP_FILES = 10_000  # Bug-hunt iter 498: glob.glob without count cap — iter-338 class; NASA P10 Rule 2.
for clip_dir in clip_dirs:
    for wav_path in glob.glob(os.path.join(clip_dir, "*.wav"))[:_MAX_CLIP_FILES]:
        audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if isinstance(audio, np.ndarray) and audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != 16000:
            audio = resampy.resample(audio, sr, 16000)
        sf.write(wav_path, audio, 16000, subtype="PCM_16")
        processed += 1

print(f"Normalized generated clips to mono 16k PCM: {processed}")
PY
    python3 - <<'PY'
import os
import openwakeword.utils as oww_utils

target = os.path.join(os.getcwd(), "openwakeword", "resources", "models")
oww_utils.download_models(model_names=["_none_"], target_directory=target)
print(f"Ensured openWakeWord feature/VAD resources in {target}")
PY
    python3 openwakeword/train.py --training_config "$cfg_out" --augment_clips
    # Bug-hunt iter 493: $cfg_out interpolated into second unquoted Python heredoc —
    # iter-227 class (same as iter-481).  Pass via environment variable instead.
    CFG_OUT="$cfg_out" python3 - <<'PY'
import os
import numpy as np
import yaml

cfg_path = os.environ["CFG_OUT"]
# Bug-hunt iter 542: yaml.safe_load without size cap — iter-330 class (fp_validation block).
_cfg542_sz = os.path.getsize(cfg_path) if os.path.exists(cfg_path) else 0
if _cfg542_sz > 1 * 1024 * 1024:
    raise SystemExit(f"training_config.yml too large: {_cfg542_sz}B")
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

feature_dir = os.path.join(cfg["output_dir"], cfg["model_name"])
neg_features = os.path.join(feature_dir, "negative_features_test.npy")
fp_validation = os.path.join(feature_dir, "false_positive_validation.npy")

if not os.path.exists(neg_features):
    raise FileNotFoundError(f"Missing generated features: {neg_features}")

arr = np.load(neg_features)
if arr.ndim == 3:
    arr = arr.reshape(-1, arr.shape[-1])
elif arr.ndim != 2:
    raise ValueError(f"Unexpected feature shape for false-positive validation: {arr.shape}")

np.save(fp_validation, arr.astype(np.float32))
cfg["false_positive_validation_data_path"] = fp_validation

with open(cfg_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

print(f"Prepared false-positive validation features: {fp_validation} shape={arr.shape}")
PY
    python3 openwakeword/train.py --training_config "$cfg_out" --train_model
    if [[ "$model_format" == "tflite" || "$model_format" == "both" ]]; then
      # Bug-hunt iter 494: mapfile without count cap — iter-338 class; NASA P10 Rule 2.
      # Sweep ONLY the run dir: including $repo_dir here also caught openWakeWord's
      # per-run-downloaded RESOURCE models (melspectrogram.onnx has dynamic dims
      # onnx2tf can't convert -> hard fail AFTER the trained model converted fine),
      # and their .tflite twins are downloaded directly anyway.
      mapfile -t -n 256 fallback_onnxes < <(find "$run_dir" -type f -name "*.onnx" -newer "$run_dir/.start_time" 2>/dev/null | sort || true)
      [[ ${#fallback_onnxes[@]} -gt 0 ]] || die "No ONNX artifacts found for conversion."
      for onnx_path in "${fallback_onnxes[@]}"; do
        [[ -f "$onnx_path" ]] || continue
        convert_onnx_with_onnx2tf "$onnx_path" "$run_dir"
      done
    fi
  ) 2>&1 | tee -a "$log_file"

  # Bug-hunt iter 495: mapfile without count cap — iter-338 class; NASA P10 Rule 2.
  mapfile -t -n 256 tflites < <(find "$run_dir" "$repo_dir" -type f -name "*.tflite" -newer "$run_dir/.start_time" 2>/dev/null | sort || true)
  mapfile -t -n 256 onnxes  < <(find "$run_dir" "$repo_dir" -type f -name "*.onnx"  -newer "$run_dir/.start_time" 2>/dev/null | sort || true)

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

  local eval_model=""
  if [[ ${#tflites[@]} -gt 0 ]]; then
    eval_model="${tflites[0]}"
  elif [[ ${#onnxes[@]} -gt 0 ]]; then
    eval_model="${onnxes[0]}"
  fi

  # Sidecar manifest — every model artifact gets a <slug>.tflite.json (or
  # <slug>.onnx.json) next to it with training params, voices used, repo
  # SHA, openWakeWord version, device target. Universal substrate for
  # every downstream integration (Piranesi-shaped emit, --deploy-to,
  # future wakeword-install skill). One source of truth per model.
  local manifest_repo_sha="unknown"
  if command -v git >/dev/null 2>&1 && [[ -d "$script_dir/.git" || -f "$script_dir/.git" ]]; then
    manifest_repo_sha="$(git -C "$script_dir" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
  fi
  local manifest_oww_version="unknown"
  manifest_oww_version="$(python3 -c 'import openwakeword; print(getattr(openwakeword, "__version__", "unknown"))' 2>/dev/null || echo unknown)"
  local manifest_built_at; manifest_built_at="$(timestamp_utc)"
  local manifest_voices_csv="${PIPER_VOICES_USED:-}"

  # Read selected counts from the dataset_counts array (set after
  # generate_dataset.py at line 788-789). Passing $num_positives /
  # $num_negatives by name would silently give us empty strings — those
  # bash vars are never defined. Hot-fix W-1 (audit 2026-04-27).
  local manifest_pos="${dataset_counts[0]:-}"
  local manifest_neg="${dataset_counts[1]:-}"

  for f in "${tflites[@]}" "${onnxes[@]}"; do
    [[ -n "$f" && -f "$f" ]] || continue
    local artifact_name; artifact_name="$(basename -- "$f")"
    local artifact_dest="$custom_models_dir/$artifact_name"
    local manifest_path="${artifact_dest}.json"
    python3 - "$manifest_path" "$artifact_name" "$wake_phrase" "$model_slug" \
                "$manifest_pos" "$manifest_neg" "$train_profile" "$train_threads" \
                "$manifest_built_at" "$manifest_repo_sha" "$manifest_oww_version" \
                "${DEVICE_ID:-}" "$manifest_voices_csv" <<'PY' || die_with_code "manifest emit failed for $artifact_name" 4 "manifest_io"
import json, sys, os
(path, artifact, phrase, slug, n_pos, n_neg, profile, threads,
 built_at, repo_sha, oww_ver, device_id, voices_csv) = sys.argv[1:14]
# Hot-fix W-1: fail loud on null counts. A manifest with
# training_params.positives=null is structurally valid but semantically
# empty and will silently corrupt every downstream consumer that reads
# it. P10: assert at boundary.
positives = int(n_pos) if str(n_pos).isdigit() else None
negatives = int(n_neg) if str(n_neg).isdigit() else None
assert positives is not None, f"manifest emit got null positives count (raw={n_pos!r}); dataset_counts unset?"
assert negatives is not None, f"manifest emit got null negatives count (raw={n_neg!r}); dataset_counts unset?"
voices = [v.strip() for v in voices_csv.split(",") if v.strip()] if voices_csv else []
manifest = {
    "schema": "wakeword-training/manifest@v1",
    "phrase": phrase,
    "slug": slug,
    "artifact": artifact,
    "device_target": device_id or None,
    "training_params": {
        "positives": positives,
        "negatives": negatives,
        "profile": profile,
        "threads": int(threads) if str(threads).isdigit() else None,
    },
    "piper_voices_used": voices,
    "built_at": built_at,
    "repo_sha": repo_sha,
    "openwakeword_version": oww_ver,
    "threshold_suggestion": None,
    "eval_report": None,
}
os.makedirs(os.path.dirname(path), exist_ok=True)
tmp = path + ".tmp"
with open(tmp, "w") as fh:
    json.dump(manifest, fh, indent=2, sort_keys=True)
    fh.write("\n")
os.replace(tmp, path)
print(f"manifest: {path}", file=sys.stderr)
PY
    log "Manifest emitted: $manifest_path"

    # Piranesi-shaped emit (--emit-piranesi-entry): write a sibling
    # <artifact>.phrases-entry.json with exactly the per-phrase object
    # shape vector-override's state/wakeword/phrases.json expects. User
    # pastes one block — closes the integration fault line.
    # Threshold sourced from the manifest's threshold_suggestion when the
    # eval gate (ship #3) populates it; falls back to 0.5 when eval did
    # not run or did not produce a recommendation. enabled defaults to
    # false — Sat flips per-phrase after dropping the model on Piranesi.
    if [[ "${CLI_EMIT_PIRANESI_ENTRY:-0}" -eq 1 ]]; then
      local phrases_entry_path="${artifact_dest}.phrases-entry.json"
      python3 - "$phrases_entry_path" "$manifest_path" "$wake_phrase" "$model_slug" \
                "$artifact_name" "$manifest_built_at" <<'PY' || die_with_code "phrases-entry emit failed for $artifact_name" 4 "phrases_entry_io"
import json, sys, os
(out_path, manifest_path, phrase, slug, artifact, built_at) = sys.argv[1:7]
threshold = 0.5
try:
    with open(manifest_path) as fh:
        m = json.load(fh)
    sug = m.get("threshold_suggestion")
    if isinstance(sug, (int, float)) and 0.0 < sug < 1.0:
        threshold = float(sug)
except Exception:
    pass
entry = {
    "schema": "vector-override/phrases-entry@v1",
    "slug": slug,
    "phrase": phrase,
    "model": artifact,
    "trained_at": built_at,
    "threshold": threshold,
    "route": {
        "action": "takeover",
        "default_esn": None,
        "speak_template": None,
    },
    "enabled": False,
}
os.makedirs(os.path.dirname(out_path), exist_ok=True)
tmp = out_path + ".tmp"
with open(tmp, "w") as fh:
    json.dump(entry, fh, indent=2, sort_keys=True)
    fh.write("\n")
os.replace(tmp, out_path)
print(f"phrases-entry: {out_path}", file=sys.stderr)
PY
      log "Piranesi phrases-entry emitted: $phrases_entry_path"
    fi
  done

  local closed_loop_eval="$script_dir/closed_loop_eval.py"
  local eval_report=""
  local eval_ran=0
  local eval_recall=""
  local eval_far_per_hour=""
  local eval_threshold=""
  if [[ -n "$eval_model" && -f "$closed_loop_eval" ]]; then
    local feature_dir="$run_dir/$model_slug"
    local eval_pos_dir="$feature_dir/positive_test"
    local eval_neg_dir="$feature_dir/negative_test"
    eval_report="$run_dir/evaluation/closed_loop_report.json"
    local hard_neg_dir="$data_dir/hard_negatives/$model_slug"
    local target_far_per_hour="${TARGET_FALSE_ALARMS_PER_HOUR:-0.1}"
    local closed_loop_max_clips="${CLOSED_LOOP_MAX_CLIPS:-600}"
    local max_mined_hard_negatives="${MAX_MINED_HARD_NEGATIVES:-200}"
    mkdir -p "$(dirname "$eval_report")" "$hard_neg_dir"
    if [[ -d "$eval_pos_dir" && -d "$eval_neg_dir" ]]; then
      log "Running closed-loop evaluation + hard-negative mining"
      # Eval is no longer a soft-fail. A trainer that produces a model
      # which can't be evaluated is producing a model nobody can trust;
      # propagate the failure so CI / deploy automation see it.
      python3 "$closed_loop_eval" \
        --model-path "$eval_model" \
        --positives-dir "$eval_pos_dir" \
        --negatives-dir "$eval_neg_dir" \
        --target-far-per-hour "$target_far_per_hour" \
        --max-clips "$closed_loop_max_clips" \
        --hard-negatives-dir "$hard_neg_dir" \
        --max-mined "$max_mined_hard_negatives" \
        --report-path "$eval_report"
      eval_ran=1
      # Pull metrics for gate + manifest backfill.
      # Bug-hunt iter 692: mapfile without -n count cap — iter-338 class; Python prints exactly 3 lines.
      mapfile -t -n 8 _eval_metrics < <(python3 - "$eval_report" <<'PY'
import json, os, sys
try:
    # Bug-hunt iter 538: json.load without size cap — iter-330 class.
    _er_sz = os.path.getsize(sys.argv[1]) if os.path.exists(sys.argv[1]) else 0
    if _er_sz > 10 * 1024 * 1024:
        raise SystemExit(f"eval_report too large: {_er_sz}B")
    with open(sys.argv[1]) as fh:
        r = json.load(fh)
except Exception:
    print(""); print(""); print("")
    sys.exit(0)
print(r.get("positive_recall", ""))
print(r.get("observed_far_per_hour", ""))
print(r.get("recommended_threshold", ""))
PY
)
      eval_recall="${_eval_metrics[0]:-}"
      eval_far_per_hour="${_eval_metrics[1]:-}"
      eval_threshold="${_eval_metrics[2]:-}"
      log "Eval: recall=${eval_recall} far_per_hour=${eval_far_per_hour} threshold=${eval_threshold}"
    else
      log "WARNING: Skipping closed-loop eval (missing clip dirs: $eval_pos_dir / $eval_neg_dir)"
    fi
  fi

  # Backfill the manifest sidecar with eval results so downstream consumers
  # (Piranesi-shaped phrases-entry, future --deploy-to) see threshold +
  # observed metrics in the same JSON they already read.
  if (( eval_ran == 1 )) && [[ -n "$eval_report" && -f "$eval_report" ]]; then
    for f in "${tflites[@]}" "${onnxes[@]}"; do
      [[ -n "$f" && -f "$f" ]] || continue
      local artifact_name; artifact_name="$(basename -- "$f")"
      local manifest_path="$custom_models_dir/${artifact_name}.json"
      [[ -f "$manifest_path" ]] || continue
      python3 - "$manifest_path" "$eval_report" "$eval_threshold" <<'PY' || log "WARNING: manifest backfill failed for $manifest_path"
import json, os, sys
manifest_path, eval_path, threshold = sys.argv[1:4]
try:
    # Bug-hunt iter 540: json.load without size cap on manifest and eval — iter-330 class.
    for _p540, _cap540 in ((manifest_path, 1*1024*1024), (eval_path, 10*1024*1024)):
        _sz540 = os.path.getsize(_p540) if os.path.exists(_p540) else 0
        if _sz540 > _cap540:
            sys.exit(f"backfill: {_p540} too large: {_sz540}B")
    with open(manifest_path) as fh: m = json.load(fh)
    with open(eval_path) as fh: r = json.load(fh)
except Exception as e:
    sys.exit(f"backfill: {e}")
m["eval_report"] = {
    "positive_recall": r.get("positive_recall"),
    "observed_far_per_hour": r.get("observed_far_per_hour"),
    "false_alarms": r.get("false_alarms"),
    "negative_hours_evaluated": r.get("negative_hours_evaluated"),
    "positives_evaluated": r.get("positives_evaluated"),
    "negatives_evaluated": r.get("negatives_evaluated"),
    "hard_negatives_mined": r.get("hard_negatives_mined"),
    "report_path": eval_path,
}
try:
    t = float(threshold)
    if 0.0 < t < 1.0:
        m["threshold_suggestion"] = t
except (ValueError, TypeError):
    pass
tmp = manifest_path + ".tmp"
with open(tmp, "w") as fh:
    json.dump(m, fh, indent=2, sort_keys=True)
    fh.write("\n")
import os
os.replace(tmp, manifest_path)
PY
      # Backfill the phrases-entry sidecar's threshold too, if it exists.
      local phrases_entry_path="$custom_models_dir/${artifact_name}.phrases-entry.json"
      if [[ -f "$phrases_entry_path" && -n "$eval_threshold" ]]; then
        python3 - "$phrases_entry_path" "$eval_threshold" <<'PY' || true
import json, sys, os
path, threshold = sys.argv[1:3]
try:
    # Bug-hunt iter 541: json.load without size cap — iter-330 class.
    _pe_sz = os.path.getsize(path) if os.path.exists(path) else 0
    if _pe_sz > 1 * 1024 * 1024:
        raise Exception(f"phrases-entry too large: {_pe_sz}B")
    with open(path) as fh: e = json.load(fh)
    t = float(threshold)
    if 0.0 < t < 1.0:
        e["threshold"] = t
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(e, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
except Exception:
    pass
PY
      fi
    done
  fi

  # Eval gate (per Piranesi 2026-04-27 ship #3): if device profile
  # specifies an `eval_gate` block in device_workflows.json, fail the
  # build with exit code 2 when observed metrics fall outside the gate.
  # No gate configured ⇒ no gate enforced (preserves backwards compat).
  if (( eval_ran == 1 )) && [[ -n "${DEVICE_ID:-}" ]] && [[ -n "$eval_recall" && -n "$eval_far_per_hour" ]]; then
    local workflows_json="$script_dir/device_workflows.json"
    if [[ -r "$workflows_json" ]]; then
      # Bug-hunt iter 693: mapfile without -n count cap — iter-338 class; Python prints exactly 2 lines.
      mapfile -t -n 8 _gate_thresholds < <(python3 - "$workflows_json" "$DEVICE_ID" <<'PY'
import json, os, sys
try:
    # Bug-hunt iter 539: json.load without size cap — iter-330 class.
    _wj_sz = os.path.getsize(sys.argv[1]) if os.path.exists(sys.argv[1]) else 0
    if _wj_sz > 10 * 1024 * 1024:
        raise SystemExit(f"device_workflows.json too large: {_wj_sz}B")
    with open(sys.argv[1]) as fh:
        d = json.load(fh)
except Exception:
    print(""); print(""); sys.exit(0)
target = sys.argv[2]
gate = None
for dev in d.get("devices", []):
    if dev.get("id") == target:
        gate = dev.get("eval_gate")
        break
if not isinstance(gate, dict):
    print(""); print("")
    sys.exit(0)
print(gate.get("max_far_per_hour", ""))
print(gate.get("min_recall", ""))
PY
)
      local gate_max_far="${_gate_thresholds[0]:-}"
      local gate_min_recall="${_gate_thresholds[1]:-}"
      if [[ -n "$gate_max_far" && -n "$gate_min_recall" ]]; then
        log "Eval gate (device=$DEVICE_ID): max_far_per_hour=$gate_max_far min_recall=$gate_min_recall"
        local gate_pass="yes"
        if python3 -c "import sys; sys.exit(0 if float('$eval_far_per_hour') <= float('$gate_max_far') else 1)" 2>/dev/null; then
          :
        else
          log "Eval gate FAIL: observed_far_per_hour=$eval_far_per_hour > max_far_per_hour=$gate_max_far"
          gate_pass="no"
        fi
        if python3 -c "import sys; sys.exit(0 if float('$eval_recall') >= float('$gate_min_recall') else 1)" 2>/dev/null; then
          :
        else
          log "Eval gate FAIL: positive_recall=$eval_recall < min_recall=$gate_min_recall"
          gate_pass="no"
        fi
        if [[ "$gate_pass" == "no" ]]; then
          die_with_code "Eval gate failed for device=$DEVICE_ID (recall=$eval_recall far_per_hour=$eval_far_per_hour)" 2 "eval_gate"
        fi
        log "Eval gate PASS"
      else
        log "Eval gate not configured for device=$DEVICE_ID — skipping gate enforcement"
      fi
    fi
  fi

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
