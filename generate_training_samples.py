#!/usr/bin/env python3
"""Generate diverse training audio samples for wake word detection using Piper voices only."""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

from piper.config import SynthesisConfig
from piper.download_voices import VOICES_JSON, download_voice
from piper.voice import PiperVoice


def log(msg):
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


PIPER_PREFERRED_HIGH_QUALITY_VOICES = [
    "en_US-lessac-high",
    "en_US-libritts-high",
    "en_US-ljspeech-high",
    "en_US-ryan-high",
    "en_GB-cori-high",
]


def fetch_piper_voices_index():
    with urlopen(VOICES_JSON, timeout=30) as response:
        return json.load(response)


def _cached_high_quality_voices(download_dir: Path):
    cached = []
    if not download_dir.exists():
        return cached
    for voice in PIPER_PREFERRED_HIGH_QUALITY_VOICES:
        model = download_dir / f"{voice}.onnx"
        config = download_dir / f"{voice}.onnx.json"
        if model.exists() and config.exists():
            cached.append(voice)
    return cached


def resolve_piper_voice_pool(max_voices=8, download_dir: Path | None = None):
    high_english = []
    try:
        voices = fetch_piper_voices_index()
        for name, meta in voices.items():
            quality = (meta or {}).get("quality", "")
            lang_code = ((meta or {}).get("language", {}) or {}).get("code", "")
            if quality == "high" and str(lang_code).startswith("en_"):
                high_english.append(name)
    except Exception as e:
        log(f"WARNING: failed to fetch Piper voices index; using cached voices only ({e})")
        if download_dir is not None:
            high_english = _cached_high_quality_voices(download_dir)
    if not high_english:
        return []

    ordered = [v for v in PIPER_PREFERRED_HIGH_QUALITY_VOICES if v in high_english]
    ordered.extend(v for v in sorted(high_english) if v not in ordered)
    return ordered[:max_voices]


class PiperSynthesizer:
    def __init__(self, voice_pool, download_dir):
        self.voice_pool = list(voice_pool)
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self._voices = {}

    def ensure_voices(self):
        for voice in self.voice_pool:
            download_voice(voice, self.download_dir, force_redownload=False)

    def _load_voice(self, voice_name):
        voice = self._voices.get(voice_name)
        if voice is not None:
            return voice

        model_path = self.download_dir / f"{voice_name}.onnx"
        config_path = self.download_dir / f"{voice_name}.onnx.json"
        if not model_path.exists() or not config_path.exists():
            download_voice(voice_name, self.download_dir, force_redownload=False)

        voice = PiperVoice.load(model_path=model_path, config_path=config_path)
        self._voices[voice_name] = voice
        return voice

    def synthesize_phrase_raw(self, filepath, phrase, idx, voice_name):
        try:
            voice = self._load_voice(voice_name)
            length_scales = [0.95, 1.0, 1.05]
            noise_scales = [0.55, 0.67, 0.78]
            noise_w_scales = [0.7, 0.8, 0.9]
            syn_cfg = SynthesisConfig(
                length_scale=length_scales[idx % len(length_scales)],
                noise_scale=noise_scales[(idx // len(length_scales)) % len(noise_scales)],
                noise_w_scale=noise_w_scales[(idx // (len(length_scales) * len(noise_scales))) % len(noise_w_scales)],
            )
            with wave.open(filepath, "wb") as wav_file:
                voice.synthesize_wav(phrase, wav_file, syn_config=syn_cfg, set_wav_format=True)
            return True
        except Exception as e:
            log(f"ERROR Piper synth voice='{voice_name}': {e}")
            return False


def _token_is_identifier(token):
    if not token:
        return False
    if token.isdigit():
        return True
    if re.fullmatch(r"[0-9a-f]{8,}", token):
        return True
    if re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", token):
        return True
    if re.fullmatch(r"(uid|uuid|id|clip|sample|run|gen)[0-9a-f-]*", token):
        return True
    if len(token) >= 6 and re.fullmatch(r"[a-z0-9-]+", token):
        has_alpha = any(c.isalpha() for c in token)
        has_digit = any(c.isdigit() for c in token)
        if has_alpha and has_digit:
            return True
    return False


def canonical_clip_key(filename, prefix):
    stem = Path(filename).stem.lower()
    head = f"{prefix}_"
    if not stem.startswith(head):
        return None
    raw_tokens = [t for t in stem[len(head):].split("_") if t]
    semantic_tokens = [t for t in raw_tokens if not _token_is_identifier(t)]
    if not semantic_tokens:
        return None
    return f"{prefix}:{'_'.join(semantic_tokens)}"


def scan_existing_clips(outdir, prefix):
    existing_names = set()
    existing_keys = set()
    max_idx = -1
    type_counts = Counter()
    for wav in Path(outdir).glob(f"{prefix}_*.wav"):
        existing_names.add(wav.name)
        parts = wav.stem.split("_")
        if len(parts) >= 3 and parts[0] == prefix and parts[1].isdigit():
            max_idx = max(max_idx, int(parts[1]))
        key = canonical_clip_key(wav.name, prefix)
        if key:
            existing_keys.add(key)
            type_counts[key.split(":", 1)[1]] += 1
    return existing_names, max_idx, existing_keys, type_counts


def voice_label(value):
    if not value:
        return "default"
    text = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return text or "voice"


def alpha_token(index):
    n = int(index)
    if n < 0:
        n = 0
    chars = []
    while True:
        n, rem = divmod(n, 26)
        chars.append(chr(ord("a") + rem))
        if n == 0:
            break
        n -= 1
    return "".join(reversed(chars))


def render_spoken_variant(raw_path, output_path, variant, duration=2.0):
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


def gen_spoken_positive(phrase, outfile, synthesizer, variant, idx, duration=2.0, voice=None):
    fd, raw_path = tempfile.mkstemp(prefix="wakeword_tts_", suffix=".wav")
    os.close(fd)
    try:
        if not synthesizer.synthesize_phrase_raw(raw_path, phrase, idx, voice_name=voice):
            return False
        return render_spoken_variant(raw_path, outfile, variant, duration=duration)
    finally:
        try:
            os.unlink(raw_path)
        except OSError:
            pass


def ffmpeg_lavfi(filepath, lavfi_expr, duration=2.0):
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
    return ffmpeg_lavfi(filepath, f"anoisesrc=color={color}:amplitude={amplitude}:r=22050", duration)


def gen_tone(filepath, frequency=440, duration=2.0, amplitude=0.03):
    return ffmpeg_lavfi(filepath, f"sine=frequency={frequency}:sample_rate=22050,volume={amplitude}", duration)


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


def render_negative_variant(kind, outfile, global_index, synthesizer=None, voice_pool=None, wake_phrase=""):
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
        if not synthesizer or not voice_pool:
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
            synthesizer,
            speech_variant,
            global_index,
            duration=duration,
            voice=voice,
        )
    return False


def gen_positives(phrase, outdir, count=50, synthesizer=None, voice_pool=None):
    log(f"Generating {count} positive samples for '{phrase}'")
    Path(outdir).mkdir(parents=True, exist_ok=True)

    if not synthesizer:
        raise RuntimeError("No Piper synthesizer configured for positive sample generation")
    if not voice_pool:
        raise RuntimeError("No Piper voices available for positive sample generation")
    log(f"  Voice pool ({len(voice_pool)}) [highest-quality Piper]: {', '.join([v for v in voice_pool if v])}")
    if len([v for v in voice_pool if v]) < 2:
        log("WARNING: less than two voices available; positives may be low-voice-diversity.")

    existing_names, max_existing_idx, existing_keys, existing_type_counts = scan_existing_clips(outdir, "positive")
    if existing_names:
        log(f"  Found {len(existing_names)} existing positives (max index {max_existing_idx}); appending unique new samples.")
    if existing_type_counts:
        for name in sorted(existing_type_counts):
            log(f"    existing - {name}: {existing_type_counts[name]}")

    variations = ["clean", "fast", "slow", "telephone", "quiet", "loud", "bright"]
    voice_count = len(voice_pool)
    variant_count = len(variations)

    generated = 0
    by_type = Counter()
    by_voice = Counter()
    next_idx = max(max_existing_idx + 1, 0)
    semantic_cursor = len(existing_keys)
    attempts = 0
    max_attempts = max(count * 40, 100)
    while generated < count and attempts < max_attempts:
        attempts += 1
        voice = voice_pool[semantic_cursor % voice_count]
        name = variations[(semantic_cursor // voice_count) % variant_count]
        style_tag = f"style{alpha_token((semantic_cursor // (voice_count * variant_count)))}"
        voice_tag = voice_label(voice)
        semantic_name = f"{voice_tag}_{name}_{style_tag}"
        semantic_key = f"positive:{semantic_name}"
        semantic_cursor += 1
        if semantic_key in existing_keys:
            continue
        uid_tag = f"uid{next_idx:06x}"
        outfile_name = f"positive_{next_idx:06d}_{uid_tag}_{semantic_name}.wav"
        outfile = os.path.join(outdir, outfile_name)
        if outfile_name in existing_names or os.path.exists(outfile):
            next_idx += 1
            continue
        try:
            if (generated + 1) % 10 == 0 or generated == 0:
                log(f"  Positives: {generated+1}/{count}")
            if gen_spoken_positive(phrase, outfile, synthesizer, name, next_idx, duration=2.0, voice=voice):
                generated += 1
                by_type[name] += 1
                by_voice[voice_tag] += 1
                existing_names.add(outfile_name)
                existing_keys.add(semantic_key)
                next_idx += 1
            else:
                next_idx += 1
        except Exception as e:
            log(f"ERROR positive idx={next_idx}: {e}")
            next_idx += 1

    if generated < count:
        log(f"WARNING: generated {generated}/{count} positives before hitting attempt limit ({max_attempts}).")

    log(f"✓ Positive: {generated}/{count} generated")
    for name in sorted(by_type):
        log(f"    - {name}: {by_type[name]}")
    for v in sorted(by_voice):
        log(f"    - voice:{v}: {by_voice[v]}")
    return generated


def gen_negatives(outdir, count=50, min_new=50, synthesizer=None, voice_pool=None, wake_phrase=""):
    target_new = max(count, min_new)
    if target_new != count:
        log(f"Generating at least {min_new} new negatives (requested {count} -> target {target_new})")
    else:
        log(f"Generating {target_new} negative samples")
    Path(outdir).mkdir(parents=True, exist_ok=True)
    existing_names, max_existing_idx, existing_keys, existing_type_counts = scan_existing_clips(outdir, "negative")
    if existing_names:
        log(f"  Found {len(existing_names)} existing negatives (max index {max_existing_idx}); appending unique new samples.")
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

    next_idx = max(max_existing_idx + 1, 0)
    generated = 0
    by_type = Counter()
    attempts = 0
    max_attempts = max(10, target_new * 30)
    semantic_cursor = len(existing_keys)
    while generated < target_new and attempts < max_attempts:
        attempts += 1
        name = variations[semantic_cursor % len(variations)]
        style_tag = f"style{alpha_token(semantic_cursor // len(variations))}"
        semantic_name = f"{name}_{style_tag}"
        semantic_key = f"negative:{semantic_name}"
        semantic_cursor += 1
        if semantic_key in existing_keys:
            continue
        uid_tag = f"uid{next_idx:06x}"
        outfile_name = f"negative_{next_idx:06d}_{uid_tag}_{semantic_name}.wav"
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
                synthesizer=synthesizer,
                voice_pool=voice_pool,
                wake_phrase=wake_phrase,
            ):
                generated += 1
                by_type[name] += 1
                existing_names.add(outfile_name)
                existing_keys.add(semantic_key)
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
    parser = argparse.ArgumentParser(description="Generate training samples")
    parser.add_argument("--wake-phrase", default="hello world", help="Wake word phrase")
    parser.add_argument("--data-dir", default="./wakeword_lab/data", help="Output directory for samples")
    parser.add_argument("--positives", type=int, default=50, help="Number of positive samples")
    parser.add_argument("--negatives", type=int, default=50, help="Number of negative samples")
    parser.add_argument("--min-new-negatives", type=int, default=50, help="Minimum new unique negatives to append each run")
    parser.add_argument("--piper-max-voices", type=int, default=8, help="Maximum number of high-quality Piper voices")
    args = parser.parse_args()

    log(f"Sample Generator: '{args.wake_phrase}'")
    log(f"  Positives: {args.positives}")
    log(f"  Negatives: {args.negatives}")
    log(f"  Min new negatives: {args.min_new_negatives}")
    log(f"  Max Piper voices: {args.piper_max_voices}")

    require_cmd("ffmpeg")

    synthesizer = None
    voice_pool = None
    if args.positives > 0 or args.negatives > 0:
        piper_download_dir = os.path.join(args.data_dir, "piper_voices")
        voice_pool = resolve_piper_voice_pool(
            max_voices=max(1, args.piper_max_voices),
            download_dir=Path(piper_download_dir),
        )
        if not voice_pool:
            raise RuntimeError("No high-quality English Piper voices found")
        synthesizer = PiperSynthesizer(voice_pool=voice_pool, download_dir=piper_download_dir)
        log(f"  Piper voice pool ({len(voice_pool)}): {', '.join(voice_pool)}")
        log(f"  Piper voice cache: {piper_download_dir}")
        synthesizer.ensure_voices()
        if args.negatives > 0 and len(voice_pool) < 2:
            raise RuntimeError("Need at least two Piper voices for diverse spoken negatives.")

    phrase_slug = slugify_phrase(args.wake_phrase)
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
            synthesizer=synthesizer,
            voice_pool=voice_pool,
        )

    n_count = 0
    if args.negatives > 0:
        n_count = gen_negatives(
            neg_dir,
            args.negatives,
            min_new=args.min_new_negatives,
            synthesizer=synthesizer,
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
