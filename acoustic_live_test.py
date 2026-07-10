#!/usr/bin/env python3
"""acoustic_live_test.py — cross-device live wake-word test over real airwaves.

Synthetic eval only proves the model on synthetic audio. This drives the loop
that actually matters: a REAL speaker on one machine plays test utterances into
the room, a REAL microphone on another machine records them, and the trained
model scores the recordings. Use it for go/no-go acceptance and for picking a
deployment threshold grounded in your room, your speaker, your mic.

Two phases (record needs only ssh + alsa-utils on the peers; score needs
openwakeword, i.e. run it inside the trainer container):

  # 1. play every clip through the remote speaker while recording the remote mic
  python3 acoustic_live_test.py record \
      --clips-dir wakeword_lab/data/acoustic_clips/hey_piranesi \
      --out-dir   wakeword_lab/data/acoustic_runs/hey_piranesi \
      --speaker-ssh root@100.111.40.21 --speaker-device plughw:AE5 \
      --mic-ssh internode-0 --mic-device plughw:AE5

  # 2. score the recordings with the trained model (inside the trainer container)
  docker compose run --rm trainer python3 acoustic_live_test.py score \
      --run-dir /workspace/data/acoustic_runs/hey_piranesi \
      --model-path /workspace/data/custom_models/hey_piranesi.tflite

Clip naming carries the label: pos_*.wav must fire, neg_*.wav must not.
`score` reports per-clip max scores and a suggested threshold (midpoint of the
worst positive and the best negative, when they separate).

The wrapper acoustic-live-test.sh runs both phases in order.
"""
import argparse
import json
import subprocess
import sys
import time
import wave
from pathlib import Path

SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
RECORD_PAD_S = 1.5           # room reverb tail + scheduling slack per clip
MAX_CLIPS = 200              # bounded (P10): a live run is minutes, not hours


def wav_duration(path):
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def record_phase(args):
    clips = sorted(Path(args.clips_dir).glob("*.wav"))[:MAX_CLIPS]
    if not clips:
        print(f"no clips in {args.clips_dir}", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    # stage all clips on the speaker host once
    run(["ssh", *SSH_OPTS, args.speaker_ssh, "mkdir -p /tmp/acoustic-live"])
    run(["scp", "-q", *[str(c) for c in clips],
         f"{args.speaker_ssh}:/tmp/acoustic-live/"])
    # clear any stale capture holding the device (a previous aborted run, or
    # the mic host's own listener mid-chunk after being switched off)
    # ([a]record: character-class so the pattern can't match this ssh
    # session's own command line — pkill -f 'arecord' kills its own shell)
    subprocess.run(["ssh", *SSH_OPTS, args.mic_ssh,
                    "pkill -f '[a]record -q -D' 2>/dev/null; sleep 1; true"],
                   check=False)
    for clip in clips:
        dur = wav_duration(clip) + RECORD_PAD_S
        rec_path = out_dir / f"rec_{clip.name}"
        ok = False
        for attempt in (1, 2):                   # bounded retry per clip
            # start the mic first (streams wav to our stdout), then play.
            # `timeout` bounds the REMOTE arecord's lifetime — killing the
            # local ssh alone leaves arecord holding the device and poisons
            # every subsequent clip.
            with open(rec_path, "wb") as f:
                rec = subprocess.Popen(
                    ["ssh", *SSH_OPTS, args.mic_ssh,
                     f"timeout {int(dur) + 4} arecord -q -D {args.mic_device} "
                     f"-f S16_LE -r 16000 -c 1 -d {int(dur + 0.999)} -t wav -"],
                    stdout=f)
                time.sleep(0.6)                   # let capture spin up
                subprocess.run(
                    ["ssh", *SSH_OPTS, args.speaker_ssh,
                     f"aplay -q -D {args.speaker_device} /tmp/acoustic-live/{clip.name}"],
                    check=False)
                try:
                    rc = rec.wait(timeout=dur + 20)
                except subprocess.TimeoutExpired:
                    rec.kill()
                    rc = -1
            if rc == 0 and rec_path.stat().st_size >= 8000:
                ok = True
                break
            print(f"  retry {clip.name} (attempt {attempt} rc={rc})",
                  file=sys.stderr)
            time.sleep(2.0)
        if not ok:
            print(f"  WARN: recording failed for {clip.name}", file=sys.stderr)
            continue
        label = "positive" if clip.name.startswith("pos_") else \
                "negative" if clip.name.startswith("neg_") else "unknown"
        manifest.append({"clip": clip.name, "recording": rec_path.name,
                         "label": label})
        print(f"  recorded {rec_path.name} ({label})")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"record phase done: {len(manifest)}/{len(clips)} clips -> {out_dir}")
    return 0 if manifest else 1


def score_phase(args):
    from openwakeword.model import Model     # trainer-container dependency
    run_dir = Path(args.run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())[:MAX_CLIPS]
    model = Model(wakeword_models=[args.model_path],
                  inference_framework=args.framework)
    name = list(model.models.keys())[0]
    results = []
    for entry in manifest:
        rec = run_dir / entry["recording"]
        if not rec.is_file():
            continue
        model.reset()
        preds = model.predict_clip(str(rec), padding=1, chunk_size=1280)
        score = max(float(p[name]) for p in preds)
        results.append({**entry, "max_score": round(score, 4)})
        print(f"  {entry['label']:<9} {entry['clip']:<40} score={score:.3f}")
    pos = [r["max_score"] for r in results if r["label"] == "positive"]
    neg = [r["max_score"] for r in results if r["label"] == "negative"]
    report = {"model_path": args.model_path, "results": results,
              "positives": len(pos), "negatives": len(neg),
              "min_positive_score": min(pos) if pos else None,
              "max_negative_score": max(neg) if neg else None,
              "suggested_threshold": None, "separates": None}
    if pos and neg:
        report["separates"] = min(pos) > max(neg)
        if report["separates"]:
            report["suggested_threshold"] = round((min(pos) + max(neg)) / 2, 3)
    elif pos:
        report["suggested_threshold"] = round(max(0.05, min(pos) * 0.6), 3)
    (run_dir / "acoustic_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "results"},
                     indent=1))
    return 0 if pos and (min(pos) > 0.1) else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="phase", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--clips-dir", required=True)
    rec.add_argument("--out-dir", required=True)
    rec.add_argument("--speaker-ssh", required=True,
                     help="ssh target with the loudspeaker (e.g. root@host)")
    rec.add_argument("--speaker-device", default="default",
                     help="ALSA playback device on the speaker host")
    rec.add_argument("--mic-ssh", required=True,
                     help="ssh target with the microphone")
    rec.add_argument("--mic-device", default="default",
                     help="ALSA capture device on the mic host")
    sco = sub.add_parser("score")
    sco.add_argument("--run-dir", required=True)
    sco.add_argument("--model-path", required=True)
    sco.add_argument("--framework", default="tflite", choices=["tflite", "onnx"])
    args = ap.parse_args()
    if args.phase == "record":
        return record_phase(args)
    return score_phase(args)


if __name__ == "__main__":
    sys.exit(main())
