#!/usr/bin/env python3
"""gibberlink_test.py — data-over-sound round-trip test across real airwaves.

GibberLink-style check of the same physical channel the wake word rides:
encode text payloads as ggwave chirps, play them through a REAL loudspeaker
on one machine, record them on another machine's REAL microphone, and decode.
A payload that survives the room proves the acoustic channel end-to-end
(speaker, air, mic, clock skew) independent of any speech model — and the
chirp recordings double as structured-audio negatives: a wake-word model must
NOT fire on machine-to-machine chirp traffic sharing its airspace.

Phases (encode/decode need the `ggwave` package — run them inside the trainer
container; record/play need only ssh + alsa-utils on the peers):

  encode  --out-dir DIR --payloads "msg1" "msg2" [--protocol 2]
  decode  --run-dir DIR                      # decodes rec_*.wav, writes report

The host-side transport reuses acoustic_live_test.py's record phase (chirp
clips are named neg_gibberlink_*.wav so the wake-model scorer treats them as
must-not-fire negatives). gibberlink-test.sh orchestrates all phases.
"""
import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

GGWAVE_RATE = 48000          # ggwave native sample rate
MAX_PAYLOADS = 32            # bounded


def _ggwave():
    import ggwave
    return ggwave


def encode_phase(args):
    gg = _ggwave()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = []
    for i, payload in enumerate(args.payloads[:MAX_PAYLOADS]):
        waveform = gg.encode(payload, protocolId=args.protocol, volume=60)
        x = np.frombuffer(waveform, dtype=np.float32)
        # 0.4 s of silence padding either side so room reverb doesn't clip it
        pad = np.zeros(int(0.4 * GGWAVE_RATE), dtype=np.float32)
        x = np.concatenate([pad, x, pad])
        pcm = np.clip(x * 32767.0, -32768, 32767).astype(np.int16)
        name = f"neg_gibberlink_{i:02d}.wav"
        with wave.open(str(out_dir / name), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(GGWAVE_RATE)
            w.writeframes(pcm.tobytes())
        meta.append({"clip": name, "payload": payload,
                     "protocol": args.protocol})
        print(f"  encoded {name}: {payload!r}")
    (out_dir / "gibberlink.json").write_text(json.dumps(meta, indent=1))
    return 0


def decode_phase(args):
    gg = _ggwave()
    run_dir = Path(args.run_dir)
    meta = json.loads((run_dir.parent / "gibberlink.json").read_text()) \
        if (run_dir.parent / "gibberlink.json").is_file() else \
        json.loads((run_dir / "gibberlink.json").read_text())
    inst = gg.init()
    results = []
    for entry in meta[:MAX_PAYLOADS]:
        rec = run_dir / f"rec_{entry['clip']}"
        if not rec.is_file():
            results.append({**entry, "decoded": None, "ok": False,
                            "note": "no recording"})
            continue
        with wave.open(str(rec), "rb") as w:
            rate = w.getframerate()
            x = np.frombuffer(w.readframes(w.getnframes()),
                              dtype=np.int16).astype(np.float32) / 32768.0
        if rate != GGWAVE_RATE:      # linear resample to ggwave's native rate
            t = np.arange(0, x.size, rate / float(GGWAVE_RATE))
            x = np.interp(t, np.arange(x.size), x).astype(np.float32)
        decoded = None
        # ggwave's decoder wants short frames (its samples_per_frame is 1024);
        # larger blocks silently decode nothing.
        step = 1024
        for off in range(0, x.size, step):
            chunk = x[off:off + step]
            res = gg.decode(inst, chunk.tobytes())
            if res:
                decoded = res.decode("utf-8", errors="replace")
                break
        ok = decoded == entry["payload"]
        results.append({**entry, "decoded": decoded, "ok": ok})
        print(f"  {entry['clip']}: sent={entry['payload']!r} "
              f"decoded={decoded!r} {'OK' if ok else 'FAIL'}")
    report = {"total": len(results), "ok": sum(1 for r in results if r["ok"]),
              "results": results}
    (run_dir / "gibberlink_report.json").write_text(json.dumps(report, indent=1))
    print(f"gibberlink round-trip: {report['ok']}/{report['total']} payloads survived the room")
    return 0 if report["ok"] == report["total"] and report["total"] else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="phase", required=True)
    enc = sub.add_parser("encode")
    enc.add_argument("--out-dir", required=True)
    enc.add_argument("--payloads", nargs="+", required=True)
    enc.add_argument("--protocol", type=int, default=2,
                     help="ggwave protocol id (2 = audible fast)")
    dec = sub.add_parser("decode")
    dec.add_argument("--run-dir", required=True,
                     help="dir holding rec_neg_gibberlink_*.wav")
    args = ap.parse_args()
    return encode_phase(args) if args.phase == "encode" else decode_phase(args)


if __name__ == "__main__":
    sys.exit(main())
