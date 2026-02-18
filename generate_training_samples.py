#!/usr/bin/env python3
"""Generate diverse training audio samples for wake word detection."""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path


def log(msg):
    """Log with timestamp."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[{ts}] [generate_samples] {msg}", file=sys.stderr)


def require_cmd(cmd):
    if not shutil.which(cmd):
        raise RuntimeError(f"Missing required command: {cmd}")


def slugify_phrase(value):
    text = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return text or "wakeword"


NEGATIVE_SPOKEN_PHRASES = [
    "what time is it now",
    "set a timer for ten minutes",
    "play some relaxing music",
    "turn on the kitchen lights",
    "turn off the bedroom lights",
    "what is the weather forecast",
    "increase the volume slightly",
    "decrease the volume a bit",
    "open the calendar for today",
    "read me the latest headlines",
    "start a focus session now",
    "pause the current track",
    "resume the current track",
    "skip to the next song",
    "go to the previous song",
    "create a reminder for tonight",
    "how is traffic right now",
    "dim the living room lights",
    "brighten the office lights",
    "set thermostat to seventy",
    "turn on the hallway fan",
    "turn off the hallway fan",
    "open the garage door",
    "close the garage door",
    "lock the front door",
    "unlock the front door",
    "what events do i have today",
    "read my unread messages",
    "call my mobile phone",
    "start vacuum cleaning",
    "stop vacuum cleaning",
    "how long is my commute",
    "show me popular podcasts",
    "play white noise softly",
    "play rain sounds tonight",
    "mute the speaker output",
    "unmute the speaker output",
    "what is on my shopping list",
    "add milk to shopping list",
    "add eggs to shopping list",
    "remove bread from shopping list",
    "set an alarm for six thirty",
    "snooze the next alarm",
    "cancel all active alarms",
    "read the next calendar event",
    "start recording voice memo",
    "stop recording voice memo",
    "where did i park today",
    "what is the battery level",
    "check internet connection",
    "run a quick speed test",
    "what is cpu temperature",
    "show system memory usage",
    "enable do not disturb",
    "disable do not disturb",
    "open a map to downtown",
    "navigate to central station",
    "show nearby coffee shops",
    "book a table for two",
    "order lunch for noon",
]


def detect_tts_backend():
    """Pick the best available local TTS backend."""
    for cmd in ("espeak-ng", "espeak", "say"):
        if shutil.which(cmd):
            return cmd
    if os.name == "nt":
        for cmd in ("powershell", "powershell.exe", "pwsh", "pwsh.exe"):
            if shutil.which(cmd):
                return cmd
    return None


def list_tts_voices(backend):
    """List available voices for the selected backend."""
    voices = []
    try:
        if backend == "say":
            out = subprocess.check_output(
                ["say", "-v", "?"],
                stderr=subprocess.DEVNULL,
                timeout=20,
                text=True,
            )
            for line in out.splitlines():
                left = line.split("#", 1)[0].strip()
                if not left:
                    continue
                cols = re.split(r"\s{2,}", left)
                if cols:
                    voice = cols[0].strip()
                    if voice:
                        voices.append(voice)
        elif backend in ("espeak-ng", "espeak"):
            out = subprocess.check_output(
                [backend, "--voices"],
                stderr=subprocess.DEVNULL,
                timeout=20,
                text=True,
            )
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 4 and parts[0].isdigit():
                    voices.append(parts[3].strip())
        else:
            script = (
                "$ErrorActionPreference='Stop';"
                "Add-Type -AssemblyName System.Speech;"
                "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                "$s.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name };"
                "$s.Dispose();"
            )
            out = subprocess.check_output(
                [backend, "-NoProfile", "-Command", script],
                stderr=subprocess.DEVNULL,
                timeout=20,
                text=True,
            )
            for line in out.splitlines():
                voice = line.strip()
                if voice:
                    voices.append(voice)
    except Exception:
        pass

    unique = []
    seen = set()
    for v in voices:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique


def resolve_tts_voice_pool(backend):
    """Resolve a stable voice pool with platform-aware preferences."""
    available = list_tts_voices(backend)
    pool = []

    if backend == "say":
        novelty = {
            "bad news",
            "bahh",
            "bells",
            "boing",
            "bubbles",
            "cellos",
            "good news",
            "jester",
            "organ",
            "superstar",
            "trinoids",
            "whisper",
            "wobble",
            "zarvox",
            "fred",
        }
        candidates = []
        for v in available:
            key = v.strip().lower()
            if key in novelty:
                continue
            candidates.append(v)

        preferred = [
            "Flo (English (US))",
            "Flo (English (UK))",
            "Samantha",
            "Daniel",
            "Karen",
            "Moira",
            "Rishi",
            "Tessa",
            "Kathy",
            "Ralph",
            "Albert",
        ]
        pool = [v for v in preferred if v in candidates]
        if not pool:
            # Fallback to non-novelty voices in deterministic order.
            pool = sorted(candidates)
        # Keep a compact high-quality subset for consistency.
        pool = pool[:8]
    elif backend in ("espeak-ng", "espeak"):
        preferred = ["en-us", "en", "en-uk", "en-sc", "en+f3", "en+m3", "en+f4", "en+m5"]
        if available:
            pool = [v for v in preferred if v in available]
            if not pool:
                pool = available[:6]
        else:
            pool = preferred[:4]
    else:
        preferred = ["Microsoft David Desktop", "Microsoft Zira Desktop", "Microsoft Hazel Desktop"]
        pool = [v for v in preferred if v in available]
        if not pool:
            pool = available[:6]

    if not pool:
        pool = available[:6]
    if not pool:
        pool = [None]
    return pool


def purge_generated_wavs(outdir, prefix):
    """Delete prior generated WAVs for deterministic fresh runs."""
    removed = 0
    pattern = f"{prefix}_*.wav"
    for wav in Path(outdir).glob(pattern):
        try:
            wav.unlink()
            removed += 1
        except OSError:
            pass
    if removed:
        log(f"  Removed {removed} stale files matching {pattern}")


def purge_legacy_flat_positives(data_dir):
    """Remove old flat positive files so positives remain wakeword-scoped."""
    legacy_dir = Path(data_dir) / "positives"
    if not legacy_dir.exists():
        return 0
    removed = 0
    for wav in legacy_dir.glob("positive_*.wav"):
        try:
            wav.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def scan_existing_negatives(outdir):
    """Inspect existing generated negatives and return names, max index, and type counts."""
    existing_names = set()
    max_idx = -1
    type_counts = Counter()
    for wav in Path(outdir).glob("negative_*.wav"):
        existing_names.add(wav.name)
        stem = wav.stem
        parts = stem.split("_")
        if len(parts) >= 3 and parts[0] == "negative":
            if parts[1].isdigit():
                max_idx = max(max_idx, int(parts[1]))
            type_counts["_".join(parts[2:])] += 1
    return existing_names, max_idx, type_counts


def _ps_quote(value):
    return value.replace("'", "''")


def voice_label(value):
    if not value:
        return "default"
    text = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return text or "voice"


def synthesize_phrase_raw(filepath, phrase, backend, idx, voice=None):
    """Synthesize raw speech to an intermediate file."""
    try:
        if backend in ("espeak-ng", "espeak"):
            rates = [145, 165, 185]
            pitches = [40, 50, 60]
            rate = rates[idx % len(rates)]
            pitch = pitches[(idx // len(rates)) % len(pitches)]
            cmd = [backend, "-q", "-w", filepath, "-s", str(rate), "-p", str(pitch), phrase]
            if voice:
                cmd[1:1] = ["-v", voice]
        elif backend == "say":
            rates = [155, 185, 215]
            rate = rates[idx % len(rates)]
            cmd = ["say", "-o", filepath, "-r", str(rate), phrase]
            if voice:
                cmd[1:1] = ["-v", voice]
        else:
            # Windows PowerShell / pwsh with System.Speech.
            rates = [-2, 0, 2]
            rate = rates[idx % len(rates)]
            voice_stmt = ""
            if voice:
                voice_stmt = f"try {{$s.SelectVoice('{_ps_quote(voice)}')}} catch {{}};"
            script = (
                "$ErrorActionPreference='Stop';"
                "Add-Type -AssemblyName System.Speech;"
                "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                f"$s.Rate={rate};"
                "$s.Volume=100;"
                f"{voice_stmt}"
                f"$s.SetOutputToWaveFile('{_ps_quote(filepath)}');"
                f"$s.Speak('{_ps_quote(phrase)}');"
                "$s.Dispose();"
            )
            cmd = [backend, "-NoProfile", "-Command", script]

        subprocess.run(cmd, capture_output=True, timeout=40, check=True)
        return True
    except Exception as e:
        log(f"ERROR TTS synth via '{backend}': {e}")
        return False


def render_spoken_variant(raw_path, output_path, variant, duration=2.0):
    """Render a speech variant to canonical mono 22.05k PCM WAV."""
    variant_filters = {
        "clean": "volume=1.0",
        "fast": "atempo=1.08,volume=1.0",
        "slow": "atempo=0.92,volume=1.0",
        "telephone": "highpass=f=140,lowpass=f=3600,volume=1.1",
        "quiet": "volume=0.75",
        "loud": "volume=1.25",
        "bright": "highpass=f=120,volume=1.0",
    }
    af = variant_filters.get(variant, "volume=1.0")
    af = f"{af},apad=pad_dur={duration},atrim=0:{duration}"

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                raw_path,
                "-af",
                af,
                "-ar",
                "22050",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                "-y",
                output_path,
            ],
            capture_output=True,
            timeout=40,
            check=True,
        )
        return True
    except Exception as e:
        log(f"ERROR ffmpeg spoken render '{variant}': {e}")
        return False


def gen_spoken_positive(phrase, outfile, backend, variant, idx, duration=2.0, voice=None):
    suffix = ".aiff" if backend == "say" else ".wav"
    fd, raw_path = tempfile.mkstemp(prefix="wakeword_tts_", suffix=suffix)
    os.close(fd)
    try:
        if not synthesize_phrase_raw(raw_path, phrase, backend, idx, voice=voice):
            return False
        return render_spoken_variant(raw_path, outfile, variant, duration=duration)
    finally:
        try:
            os.unlink(raw_path)
        except OSError:
            pass


def ffmpeg_lavfi(filepath, lavfi_expr, duration=2.0):
    """Render a WAV file from a lavfi expression."""
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                lavfi_expr,
                "-t",
                str(duration),
                "-ar",
                "22050",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                "-y",
                filepath,
            ],
            capture_output=True,
            timeout=20,
            check=True,
        )
        return True
    except Exception as e:
        log(f"ERROR ffmpeg render '{lavfi_expr}': {e}")
        return False


def gen_silence(filepath, duration=2.0):
    return ffmpeg_lavfi(filepath, "anullsrc=r=22050:cl=mono", duration)


def gen_noise(filepath, color, duration=2.0, amplitude=0.05):
    return ffmpeg_lavfi(
        filepath,
        f"anoisesrc=color={color}:amplitude={amplitude}:r=22050",
        duration,
    )


def gen_tone(filepath, frequency=440, duration=2.0, amplitude=0.03):
    return ffmpeg_lavfi(
        filepath,
        f"sine=frequency={frequency}:sample_rate=22050,volume={amplitude}",
        duration,
    )


def gen_dual_tone(filepath, duration=2.0, frequency_a=220.0, frequency_b=440.0, amp_a=0.03, amp_b=0.02):
    return ffmpeg_lavfi(
        filepath,
        f"aevalsrc='{amp_a}*sin(2*PI*{frequency_a}*t)+{amp_b}*sin(2*PI*{frequency_b}*t)':s=22050",
        duration,
    )


def gen_chirp(filepath, duration=2.0, start_hz=180.0, sweep_hz_per_s=320.0, amplitude=0.02):
    return ffmpeg_lavfi(
        filepath,
        f"aevalsrc='{amplitude}*sin(2*PI*({start_hz}+{sweep_hz_per_s}*t)*t)':s=22050",
        duration,
    )


def pick_negative_phrase(global_index, wake_phrase):
    phrase = NEGATIVE_SPOKEN_PHRASES[(global_index * 7) % len(NEGATIVE_SPOKEN_PHRASES)]
    if wake_phrase and wake_phrase.lower() in phrase.lower():
        phrase = NEGATIVE_SPOKEN_PHRASES[(global_index * 11 + 3) % len(NEGATIVE_SPOKEN_PHRASES)]
    return phrase


def render_negative_variant(kind, outfile, global_index, tts_backend=None, voice_pool=None, wake_phrase=""):
    """Render a unique negative clip variant for the given running index."""
    tier = global_index // 10
    duration = 1.8 + (tier % 5) * 0.1
    amp_jitter = ((tier % 5) - 2) * 0.004

    if kind == "silence":
        return gen_silence(outfile, duration)
    if kind == "white_noise":
        return gen_noise(outfile, "white", duration, max(0.015, 0.05 + amp_jitter))
    if kind == "pink_noise":
        return gen_noise(outfile, "pink", duration, max(0.015, 0.05 + amp_jitter))
    if kind == "brown_noise":
        return gen_noise(outfile, "brown", duration, max(0.015, 0.04 + amp_jitter))
    if kind == "tone_220hz":
        return gen_tone(outfile, 220 + (tier % 7) * 3, duration, max(0.01, 0.03 + amp_jitter / 2))
    if kind == "tone_440hz":
        return gen_tone(outfile, 440 + (tier % 9) * 4, duration, max(0.01, 0.03 + amp_jitter / 2))
    if kind == "dual_tone":
        return gen_dual_tone(
            outfile,
            duration=duration,
            frequency_a=220 + (tier % 6) * 2,
            frequency_b=440 + (tier % 6) * 3,
            amp_a=max(0.01, 0.03 + amp_jitter / 2),
            amp_b=max(0.01, 0.02 + amp_jitter / 2),
        )
    if kind == "chirp":
        return gen_chirp(
            outfile,
            duration=duration,
            start_hz=170 + (tier % 6) * 5,
            sweep_hz_per_s=300 + (tier % 6) * 10,
            amplitude=max(0.01, 0.02 + amp_jitter / 2),
        )
    if kind.startswith("speech_"):
        if not tts_backend or not voice_pool:
            return False
        speech_variant = {
            "speech_clean": "clean",
            "speech_fast": "fast",
            "speech_slow": "slow",
            "speech_phone": "telephone",
            "speech_quiet": "quiet",
        }.get(kind, "clean")
        voice = voice_pool[global_index % len(voice_pool)]
        phrase = pick_negative_phrase(global_index, wake_phrase)
        return gen_spoken_positive(
            phrase,
            outfile,
            tts_backend,
            speech_variant,
            global_index,
            duration=duration,
            voice=voice,
        )
    return False


def gen_positives(phrase, outdir, count=50, tts_backend=None, voice_pool=None):
    """Generate positive training samples (background variations)."""
    log(f"Generating {count} positive samples for '{phrase}'")
    Path(outdir).mkdir(parents=True, exist_ok=True)

    if not tts_backend:
        raise RuntimeError("No TTS backend configured for positive sample generation")
    if not voice_pool:
        raise RuntimeError("No TTS voices available for positive sample generation")
    log(f"  Voice pool ({len(voice_pool)}) [highest-quality available]: {', '.join([v for v in voice_pool if v])}")
    if len([v for v in voice_pool if v]) < 2:
        log("WARNING: less than two voices available; positives may be low-voice-diversity.")
    purge_generated_wavs(outdir, "positive")

    variations = [
        "clean",
        "fast",
        "slow",
        "telephone",
        "quiet",
        "loud",
        "bright",
    ]

    generated = 0
    by_type = Counter()
    by_voice = Counter()
    for i in range(count):
        name = variations[i % len(variations)]
        voice = voice_pool[i % len(voice_pool)]
        voice_tag = voice_label(voice)
        outfile = os.path.join(outdir, f"positive_{i:04d}_{voice_tag}_{name}.wav")
        try:
            if (i + 1) % 10 == 0 or i == 0:
                log(f"  Positives: {i+1}/{count}")
            if gen_spoken_positive(phrase, outfile, tts_backend, name, i, duration=2.0, voice=voice):
                generated += 1
                by_type[name] += 1
                by_voice[voice_tag] += 1
        except Exception as e:
            log(f"ERROR positive {i}: {e}")

    log(f"✓ Positive: {generated}/{count} generated")
    for name in sorted(by_type):
        log(f"    - {name}: {by_type[name]}")
    for v in sorted(by_voice):
        log(f"    - voice:{v}: {by_voice[v]}")
    return generated


def gen_negatives(outdir, count=50, min_new=50, tts_backend=None, voice_pool=None, wake_phrase=""):
    """Generate negative training samples (background audio)."""
    target_new = max(count, min_new)
    if target_new != count:
        log(f"Generating at least {min_new} new negatives (requested {count} -> target {target_new})")
    else:
        log(f"Generating {target_new} negative samples")
    Path(outdir).mkdir(parents=True, exist_ok=True)
    existing_names, max_existing_idx, existing_type_counts = scan_existing_negatives(outdir)
    if existing_names:
        log(
            f"  Found {len(existing_names)} existing negatives (max index {max_existing_idx}); "
            "appending unique new samples."
        )
    if existing_type_counts:
        for name in sorted(existing_type_counts):
            log(f"    existing - {name}: {existing_type_counts[name]}")

    variations = [
        "silence",
        "white_noise",
        "pink_noise",
        "brown_noise",
        "tone_220hz",
        "tone_440hz",
        "dual_tone",
        "chirp",
        "speech_clean",
        "speech_fast",
        "speech_slow",
        "speech_phone",
        "speech_quiet",
    ]

    next_idx = max_existing_idx + 1
    if next_idx < 0:
        next_idx = 0
    generated = 0
    by_type = Counter()
    attempts = 0
    max_attempts = max(10, target_new * 30)
    while generated < target_new and attempts < max_attempts:
        attempts += 1
        name = variations[generated % len(variations)]
        outfile_name = f"negative_{next_idx:06d}_{name}.wav"
        outfile = os.path.join(outdir, outfile_name)
        if outfile_name in existing_names or os.path.exists(outfile):
            next_idx += 1
            continue
        try:
            if (generated + 1) % 10 == 0 or generated == 0:
                log(f"  Negatives: {generated+1}/{target_new}")
            if render_negative_variant(
                name,
                outfile,
                next_idx,
                tts_backend=tts_backend,
                voice_pool=voice_pool,
                wake_phrase=wake_phrase,
            ):
                generated += 1
                by_type[name] += 1
                existing_names.add(outfile_name)
                next_idx += 1
            else:
                next_idx += 1
        except Exception as e:
            log(f"ERROR negative idx={next_idx}: {e}")
            next_idx += 1

    if generated < target_new:
        log(f"WARNING: generated {generated}/{target_new} negatives before hitting attempt limit ({max_attempts}).")

    log(f"✓ Negative: {generated}/{target_new} generated")
    for name in sorted(by_type):
        log(f"    - {name}: {by_type[name]}")
    if generated > 0 and len(by_type) < 3:
        log("WARNING: low variation in generated negatives; check ffmpeg lavfi support.")
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
    parser.add_argument("--min-new-negatives", type=int, default=50,
                        help="Minimum number of new unique negatives to append each run")
    args = parser.parse_args()
    log(f"Sample Generator: '{args.wake_phrase}'")
    log(f"  Positives: {args.positives}")
    log(f"  Negatives: {args.negatives}")
    log(f"  Min new negatives: {args.min_new_negatives}")

    require_cmd("ffmpeg")
    tts_backend = None
    voice_pool = None
    if args.positives > 0 or args.negatives > 0:
        tts_backend = detect_tts_backend()
        if not tts_backend:
            raise RuntimeError(
                "No local TTS backend found. Install espeak-ng/espeak, use macOS say, "
                "or run on Windows with PowerShell speech synthesis."
            )
        voice_pool = resolve_tts_voice_pool(tts_backend)
        if not voice_pool:
            raise RuntimeError("No TTS voices detected for backend")
        log(f"  TTS backend: {tts_backend}")
        nonempty_voices = [v for v in voice_pool if v]
        if args.negatives > 0 and len(nonempty_voices) < 2:
            raise RuntimeError("Need at least two TTS voices for diverse spoken negatives.")

    phrase_slug = slugify_phrase(args.wake_phrase)
    removed_legacy = purge_legacy_flat_positives(args.data_dir)
    if removed_legacy:
        log(f"  Removed {removed_legacy} legacy flat positives from {os.path.join(args.data_dir, 'positives')}")
    pos_dir = os.path.join(args.data_dir, "positives", phrase_slug)
    neg_dir = os.path.join(args.data_dir, "negatives")
    log(f"  Positive dir: {pos_dir}")
    log(f"  Negative dir: {neg_dir}")

    p_count = 0
    if args.positives > 0:
        p_count = gen_positives(
            args.wake_phrase,
            pos_dir,
            args.positives,
            tts_backend=tts_backend,
            voice_pool=voice_pool,
        )
    n_count = 0
    if args.negatives > 0:
        n_count = gen_negatives(
            neg_dir,
            args.negatives,
            min_new=args.min_new_negatives,
            tts_backend=tts_backend,
            voice_pool=voice_pool,
            wake_phrase=args.wake_phrase,
        )

    total = p_count + n_count
    log(f"✓ Complete! {total} samples generated")

    if total == 0:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL: {e}")
        sys.exit(1)
