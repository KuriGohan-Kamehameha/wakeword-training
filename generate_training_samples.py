#!/usr/bin/env python3
"""Generate diverse training audio samples for wake word detection."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def log(msg):
    """Log with timestamp."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[{ts}] [generate_samples] {msg}", file=sys.stderr)


def gen_silence(filepath, duration=2.0):
    """Generate silent WAV file."""
    try:
        subprocess.run([
            "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono",
            "-t", str(duration), "-q:a", "9", "-acodec", "libmp3lame",
            "-y", filepath
        ], capture_output=True, timeout=15, check=True)
        return True
    except Exception as e:
        log(f"ERROR silence: {e}")
        return False


def gen_white_noise(filepath, duration=2.0, amplitude=0.05):
    """Generate white noise WAV file using sox."""
    try:
        subprocess.run([
            "sox", "-n", "-r", "22050", "-b", "16", "-c", "1",
            filepath, "synth", str(duration), "whitenoise", "vol", str(amplitude)
        ], capture_output=True, timeout=15, check=True)
        return True
    except Exception as e:
        log(f"ERROR white_noise: {e}")
        return False


def gen_pink_noise(filepath, duration=2.0, amplitude=0.05):
    """Generate pink noise WAV file using sox."""
    try:
        subprocess.run([
            "sox", "-n", "-r", "22050", "-b", "16", "-c", "1",
            filepath, "synth", str(duration), "pinknoise", "vol", str(amplitude)
        ], capture_output=True, timeout=15, check=True)
        return True
    except Exception as e:
        log(f"ERROR pink_noise: {e}")
        return False


def gen_positives(phrase, outdir, count=50):
    """Generate positive training samples (background variations)."""
    log(f"Generating {count} positive samples for '{phrase}'")
    Path(outdir).mkdir(parents=True, exist_ok=True)
    
    variations = [
        ("silence", lambda f: gen_silence(f, 2.0)),
        ("white_noise_low", lambda f: gen_white_noise(f, 2.0, 0.03)),
        ("pink_noise_low", lambda f: gen_pink_noise(f, 2.0, 0.03)),
    ]
    
    generated = 0
    for i in range(count):
        name, func = variations[i % len(variations)]
        outfile = os.path.join(outdir, f"positive_{i:04d}_{name}.wav")
        try:
            if (i + 1) % 10 == 0 or i == 0:
                log(f"  Positives: {i+1}/{count}")
            if func(outfile):
                generated += 1
        except Exception as e:
            log(f"ERROR positive {i}: {e}")
    
    log(f"✓ Positive: {generated}/{count} generated")
    return generated


def gen_negatives(outdir, count=50):
    """Generate negative training samples (background audio)."""
    log(f"Generating {count} negative samples")
    Path(outdir).mkdir(parents=True, exist_ok=True)
    
    variations = [
        ("silence", lambda f: gen_silence(f, 2.0)),
        ("white_noise", lambda f: gen_white_noise(f, 2.0, 0.05)),
        ("pink_noise", lambda f: gen_pink_noise(f, 2.0, 0.05)),
    ]
    
    generated = 0
    for i in range(count):
        name, func = variations[i % len(variations)]
        outfile = os.path.join(outdir, f"negative_{i:04d}_{name}.wav")
        try:
            if (i + 1) % 10 == 0 or i == 0:
                log(f"  Negatives: {i+1}/{count}")
            if func(outfile):
                generated += 1
        except Exception as e:
            log(f"ERROR negative {i}: {e}")
    
    log(f"✓ Negative: {generated}/{count} generated")
    return generated


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate training samples")
    parser.add_argument("--wake-phrase", default="hello world",
                        help="Wake word phrase")
    parser.add_argument("--data-dir", default="./wakeword_lab/data",
                        help="Output directory for samples")
    parser.add_argument("--positives", type=int, default=50,
                        help="Number of positive samples")
    parser.add_argument("--negatives", type=int, default=50,
                        help="Number of negative samples")
    args = parser.parse_args()
    
    log(f"Sample Generator: '{args.wake_phrase}'")
    log(f"  Positives: {args.positives}")
    log(f"  Negatives: {args.negatives}")
    
    pos_dir = os.path.join(args.data_dir, "positives")
    neg_dir = os.path.join(args.data_dir, "negatives")
    
    p_count = gen_positives(args.wake_phrase, pos_dir, args.positives)
    n_count = gen_negatives(neg_dir, args.negatives)
    
    total = p_count + n_count
    log(f"✓ Complete! {total} samples generated")
    
    if total == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
