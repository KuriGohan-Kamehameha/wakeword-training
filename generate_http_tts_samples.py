#!/usr/bin/env python3
"""generate_http_tts_samples.py — corpus augmentation from an external HTTP TTS.

The bundled piper-sample-generator tops out at a handful of `-high` Piper
voices. Detector quality tracks corpus variety, so this script sweeps a
second, higher-quality TTS across a voice x speed grid and appends the
results to the same corpus layout the trainer consumes (the AVAAS seam
from the README: external voices feeding the positives pool).

Endpoint contract (kudzu-tts / Kokoro-style):
    POST <tts-url>  {"text": ..., "voice": ..., "speed": ...}  ->  WAV bytes

Positives land in  <data>/positives/<slug>/   as
    positive_9NNNNN_uid9NNNNN_<gen>_<voice>_s<speed>_ext.wav
(9NNNNN index space avoids collision with the bundled generator), speech
negatives in <data>/negatives/ with the same scheme. Output is resampled
to --rate (default 22050 to match the piper corpus) mono s16.

Voices that the endpoint rejects are skipped with a warning, so a broad
default grid degrades gracefully across endpoints.
"""
import argparse
import io
import json
import sys
import urllib.request
import wave
from pathlib import Path

import numpy as np

DEFAULT_VOICES = ["af_sarah", "af_bella", "af_nicole", "af_sky", "af_alloy",
                  "am_adam", "am_michael", "am_eric", "am_onyx",
                  "bf_emma", "bf_isabella", "bm_george", "bm_lewis"]
DEFAULT_SPEEDS = [0.85, 1.0, 1.15, 1.3]
NEG_PHRASES = [
    "please turn on the kitchen lights",
    "what is the weather like today",
    "set a timer for ten minutes",
    "the parcel arrived this morning",
    "remind me to water the plants",
    "play some quiet jazz in the study",
    "how far is the ferry terminal",
    "the printer is out of paper again",
    "paranoia is not a planning strategy",
    "the pyrenees are lovely in autumn",
    "her and his analysis were both wrong",
    "the parentheses are unbalanced in that expression",
]


def tts(url, text, voice, speed, timeout=60):
    body = json.dumps({"text": text, "voice": voice, "speed": speed}).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    if not data.startswith(b"RIFF"):
        raise RuntimeError("endpoint did not return WAV")
    return data


def to_rate_mono_s16(wav_bytes, rate_out):
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        rate_in = w.getframerate()
        ch = w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()),
                          dtype=np.int16).astype(np.float64) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if rate_in != rate_out:
        t = np.arange(0, x.size, rate_in / float(rate_out))
        x = np.interp(t, np.arange(x.size), x)
    return np.clip(x * 32767.0, -32768, 32767).astype(np.int16), rate_out


def write_wav(path, samples, rate):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())


def probe_voice(url, voice):
    try:
        tts(url, "test", voice, 1.0, timeout=30)
        return True
    except Exception as e:
        print("  skip voice %s: %s" % (voice, e), file=sys.stderr)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wake-phrase", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--tts-url", required=True,
                    help="e.g. http://host:5006/tts (kudzu-tts / Kokoro)")
    ap.add_argument("--generator-tag", default="kokoro")
    ap.add_argument("--data-dir", default="wakeword_lab/data")
    ap.add_argument("--voices", default=",".join(DEFAULT_VOICES))
    ap.add_argument("--speeds", default=",".join(str(s) for s in DEFAULT_SPEEDS))
    ap.add_argument("--rate", type=int, default=22050)
    ap.add_argument("--negatives-per-voice", type=int, default=3)
    args = ap.parse_args()

    pos_dir = Path(args.data_dir) / "positives" / args.slug
    neg_dir = Path(args.data_dir) / "negatives"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    speeds = [float(s) for s in args.speeds.split(",") if s.strip()]
    live = [v for v in voices if probe_voice(args.tts_url, v)]
    print("voices live: %d/%d  speeds: %s" % (len(live), len(voices), speeds))
    if not live:
        print("no usable voices; aborting")
        return 2

    idx = 900000
    made_pos = 0
    for voice in live:
        for speed in speeds:
            try:
                raw = tts(args.tts_url, args.wake_phrase, voice, speed)
            except Exception as e:
                print("  pos fail %s@%s: %s" % (voice, speed, e), file=sys.stderr)
                continue
            samples, rate = to_rate_mono_s16(raw, args.rate)
            name = "positive_%06d_uid%06d_%s_%s_s%s_ext.wav" % (
                idx, idx, args.generator_tag, voice,
                str(speed).replace(".", "p"))
            write_wav(pos_dir / name, samples, rate)
            idx += 1
            made_pos += 1

    made_neg = 0
    for vi, voice in enumerate(live):
        for k in range(args.negatives_per_voice):
            phrase = NEG_PHRASES[(vi * args.negatives_per_voice + k)
                                 % len(NEG_PHRASES)]
            speed = speeds[k % len(speeds)]
            try:
                raw = tts(args.tts_url, phrase, voice, speed)
            except Exception as e:
                print("  neg fail %s: %s" % (voice, e), file=sys.stderr)
                continue
            samples, rate = to_rate_mono_s16(raw, args.rate)
            name = "negative_%06d_uid%06d_%s_%s_s%s_ext.wav" % (
                idx, idx, args.generator_tag, voice,
                str(speed).replace(".", "p"))
            write_wav(neg_dir / name, samples, rate)
            idx += 1
            made_neg += 1

    print("wrote %d positives -> %s, %d negatives -> %s" % (
        made_pos, pos_dir, made_neg, neg_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
