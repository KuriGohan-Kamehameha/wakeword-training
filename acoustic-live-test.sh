#!/usr/bin/env bash
# acoustic-live-test.sh — cross-device live wake-word test over real airwaves.
#
# Plays labelled clips (pos_*.wav / neg_*.wav) through a REAL loudspeaker on
# one machine, records them through a REAL microphone on another, then scores
# the recordings with the trained model inside the trainer container.
#
# Usage:
#   ./acoustic-live-test.sh SLUG SPEAKER_SSH SPEAKER_DEV MIC_SSH MIC_DEV [MODEL]
# e.g.
#   ./acoustic-live-test.sh hey_piranesi root@100.111.40.21 plughw:AE5 \
#       internode-0 plughw:AE5
#
# Clips are read from  wakeword_lab/data/acoustic_clips/SLUG/
# Recordings + report land in  wakeword_lab/data/acoustic_runs/SLUG/
set -u

SLUG="${1:?slug}"; SPK="${2:?speaker ssh}"; SPKDEV="${3:?speaker dev}"
MIC="${4:?mic ssh}"; MICDEV="${5:?mic dev}"
MODEL="${6:-/workspace/data/custom_models/${SLUG}.tflite}"
SRC="$(cd "$(dirname "$0")" && pwd)"

CLIPS="$SRC/wakeword_lab/data/acoustic_clips/$SLUG"
RUNS="$SRC/wakeword_lab/data/acoustic_runs/$SLUG"

[ -d "$CLIPS" ] || { echo "no clips dir: $CLIPS" >&2; exit 2; }

python3 "$SRC/acoustic_live_test.py" record \
    --clips-dir "$CLIPS" --out-dir "$RUNS" \
    --speaker-ssh "$SPK" --speaker-device "$SPKDEV" \
    --mic-ssh "$MIC" --mic-device "$MICDEV" || exit $?

docker compose run --rm -T trainer python3 /app/acoustic_live_test.py score \
    --run-dir "/workspace/data/acoustic_runs/$SLUG" \
    --model-path "$MODEL"
