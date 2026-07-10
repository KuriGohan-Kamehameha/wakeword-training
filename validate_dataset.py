#!/usr/bin/env python3
"""validate_dataset.py — malformed-training-data sweep for wakeword_lab data.

Two layers:
  1. Structural: unreadable/zero-length/too-short/silent/clipped WAVs.
  2. Label check via an external STT endpoint (kudzu-stt /stt contract):
     positives must transcribe to (a fuzzy match of) the wake phrase;
     negatives must NOT contain it. Mismatches are listed and optionally
     quarantined to a sibling `_quarantine/` dir (never deleted).

Usage:
  validate_dataset.py --phrase "Hey Piranesi" --slug hey_piranesi \
      [--stt-url http://100.94.168.14:5007/stt] [--neg-sample 120] \
      [--quarantine] [--base wakeword_lab/data]

Exit codes: 0 clean, 1 findings (report printed), 2 usage/IO error.
"""
import argparse
import difflib
import json
import random
import re
import sys
import urllib.request
import wave
from pathlib import Path

import numpy as np

MIN_DUR_S = 0.3
MAX_DUR_S = 12.0
SILENCE_RMS = 0.003          # full-scale fraction; below = effectively silent
CLIP_FRAC = 0.02             # >2% samples at rail = clipped
FUZZY_POS = 0.62             # SequenceRatio threshold for positive match
STT_TIMEOUT = 30


def norm(s):
    return re.sub(r"[^a-z ]+", "", s.lower()).strip()


def read_wav(path):
    """Return (float mono ndarray, rate) or (None, reason)."""
    try:
        with wave.open(str(path), "rb") as w:
            rate = w.getframerate()
            n = w.getnframes()
            ch = w.getnchannels()
            width = w.getsampwidth()
            raw = w.readframes(n)
    except (wave.Error, EOFError, OSError) as e:
        return None, "unreadable: %s" % e
    if n == 0 or not raw:
        return None, "zero frames"
    if width != 2:
        return None, "sampwidth %d != 2" % width
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return (x, rate), None


def structural_issue(path):
    r, reason = read_wav(path)
    if r is None:
        return reason
    x, rate = r
    dur = x.size / float(rate)
    if dur < MIN_DUR_S:
        return "too short (%.2fs)" % dur
    if dur > MAX_DUR_S:
        return "too long (%.2fs)" % dur
    rms = float(np.sqrt(np.mean(x * x)))
    if rms < SILENCE_RMS:
        return "silent (rms %.5f)" % rms
    clip = float(np.mean(np.abs(x) > 0.999))
    if clip > CLIP_FRAC:
        return "clipped (%.1f%% at rail)" % (clip * 100)
    return None


def stt(url, path):
    """Transcribe via kudzu-stt POST /stt. Returns text or None on error."""
    body = Path(path).read_bytes()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "audio/wav"})
    try:
        with urllib.request.urlopen(req, timeout=STT_TIMEOUT) as resp:
            out = json.loads(resp.read().decode())
    except Exception as e:
        print("  [stt-error] %s: %s" % (Path(path).name, e), file=sys.stderr)
        return None
    return out.get("text", "") if out.get("ok") else None


def fuzzy_contains(text, phrase):
    """Does `text` contain (a fuzzy version of) `phrase`?"""
    t, p = norm(text), norm(phrase)
    if not t:
        return False
    if p in t:
        return True
    words_t = t.split()
    words_p = p.split()
    n = len(words_p)
    best = 0.0
    for i in range(0, max(1, len(words_t) - n + 1)):
        win = " ".join(words_t[i:i + n])
        best = max(best, difflib.SequenceMatcher(None, win, p).ratio())
    return best >= FUZZY_POS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrase", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--base", default="wakeword_lab/data")
    ap.add_argument("--stt-url", default="http://100.94.168.14:5007/stt")
    ap.add_argument("--neg-sample", type=int, default=120)
    ap.add_argument("--quarantine", action="store_true")
    args = ap.parse_args()

    base = Path(args.base)
    pos_dir = base / "positives" / args.slug
    neg_dir = base / "negatives"
    if not pos_dir.is_dir() or not neg_dir.is_dir():
        print("missing dirs: %s / %s" % (pos_dir, neg_dir))
        return 2

    findings = []          # (path, kind, detail)
    pos = sorted(pos_dir.glob("*.wav"))
    neg = sorted(neg_dir.glob("*.wav"))
    print("positives=%d negatives=%d" % (len(pos), len(neg)))

    for f in pos + neg:
        issue = structural_issue(f)
        if issue:
            findings.append((f, "structural", issue))
    bad_struct = {f for f, _, _ in findings}

    checked = 0
    for f in pos:
        if f in bad_struct:
            continue
        text = stt(args.stt_url, f)
        checked += 1
        if text is None:
            continue                      # endpoint error — don't blame sample
        if not fuzzy_contains(text, args.phrase):
            findings.append((f, "pos-label", "heard: %r" % text[:80]))
    print("positives label-checked: %d" % checked)

    rng = random.Random(7)
    neg_ok = [f for f in neg if f not in bad_struct]
    sample = neg_ok if len(neg_ok) <= args.neg_sample else rng.sample(
        neg_ok, args.neg_sample)
    poison = 0
    for f in sample:
        text = stt(args.stt_url, f)
        if text and fuzzy_contains(text, args.phrase):
            findings.append((f, "neg-poison", "heard: %r" % text[:80]))
            poison += 1
    print("negatives sampled: %d, poison hits: %d" % (len(sample), poison))

    if not findings:
        print("CLEAN: no malformed or mislabeled samples found")
        return 0

    print("FINDINGS: %d" % len(findings))
    for f, kind, detail in findings:
        print("  [%s] %s — %s" % (kind, f, detail))
    if args.quarantine:
        for f, kind, _ in findings:
            qdir = f.parent / "_quarantine"
            qdir.mkdir(exist_ok=True)
            f.rename(qdir / f.name)
        print("quarantined %d files (moved to sibling _quarantine/)" %
              len(findings))
    return 1


if __name__ == "__main__":
    sys.exit(main())
