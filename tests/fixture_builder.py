from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import wave
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_tone(path: Path, *, frequency: float = 180.0, seconds: float = 0.6) -> dict:
    sample_rate = 16_000
    frames = int(sample_rate * seconds)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        samples = bytearray()
        for index in range(frames):
            value = int(3_000 * math.sin(2.0 * math.pi * frequency * index / sample_rate))
            samples.extend(struct.pack("<h", value))
        handle.writeframes(bytes(samples))
    return {
        "sha256": sha256(path),
        "frames": frames,
        "duration_ms": round(frames * 1_000 / sample_rate),
    }


def seal(bundle: Path) -> None:
    (bundle / "READY").unlink(missing_ok=True)
    (bundle / "checksums.sha256").unlink(missing_ok=True)
    files = sorted(
        item.relative_to(bundle).as_posix()
        for item in bundle.rglob("*")
        if item.is_file() and not item.is_symlink()
    )
    (bundle / "checksums.sha256").write_text(
        "".join(f"{sha256(bundle / relative)}  {relative}\n" for relative in files),
        encoding="ascii",
    )
    (bundle / "READY").write_bytes(b"")
    os.utime(bundle / "READY", None)


def build_export(
    root: Path,
    *,
    export_id: str = "hey-piranesi-fixture-v1",
    clip_count: int = 10,
    promotable: bool = False,
    fixture: bool = True,
) -> Path:
    bundle = root / export_id
    (bundle / "clips").mkdir(parents=True)
    (bundle / "provenance").mkdir()
    clips = []
    total_frames = 0
    for index in range(clip_count):
        temporary = bundle / "clips" / f"temporary-{index}.wav"
        observed = write_tone(temporary, frequency=180.0 + index * 7.0)
        relative = f"clips/{observed['sha256']}.wav"
        temporary.rename(bundle / relative)
        total_frames += observed["frames"]
        clips.append(
            {
                "path": relative,
                "sha256": observed["sha256"],
                "source_sha256": hashlib.sha256(f"source-{index}".encode()).hexdigest(),
                "source_sample_rate_hz": 24_000,
                "phrase": "Hey Piranesi",
                "style": "neutral" if index % 2 == 0 else "warm",
                "speed": 1.0,
                "pitch_semitones": 0.0,
                "volume": 1.0,
                "seed": 20_260_716 + index,
                "frames": observed["frames"],
                "duration_ms": observed["duration_ms"],
            }
        )
    source_hash = "a" * 64
    write_json(
        bundle / "clips/manifest.json",
        {
            "schema": "avaas/wakeword-clips@v1",
            "phrase": "Hey Piranesi",
            "alias": "piranesi",
            "source_artifact_manifest_sha256": source_hash,
            "clips": clips,
        },
    )
    write_json(
        bundle / "provenance/licenses.json",
        {
            "schema": "avaas/license-ledger@v1",
            "entries": [
                {
                    "component": "source-voice-artifact",
                    "name": "Fixture profile",
                    "spdx": "LicenseRef-Fixture",
                    "source_url": "https://example.invalid/profile",
                    "license_url": "https://example.invalid/profile/license",
                    "distribution_restrictions": "Fixture only.",
                },
                {
                    "component": "wakeword-corpus",
                    "name": "AVAAS wakeword export",
                    "spdx": "LicenseRef-AVAAS-Wakeword",
                    "source_url": "https://github.com/KuriGohan-Kamehameha/AVAAS",
                    "license_url": "https://github.com/KuriGohan-Kamehameha/AVAAS/blob/main/LICENSE",
                    "distribution_restrictions": "Use only with speaker consent.",
                },
            ],
        },
    )
    (bundle / "README.md").write_text("# Fixture Hey Piranesi export\n", encoding="utf-8")
    clips_manifest_hash = sha256(bundle / "clips/manifest.json")
    write_json(
        bundle / "manifest.json",
        {
            "schema": "avaas/wakeword-export@v1",
            "export_id": export_id,
            "version": "1.0.0",
            "phrase": {
                "text": "Hey Piranesi",
                "normalized": "hey piranesi",
                "language_tag": "en",
            },
            "source_artifact": {
                "schema": "avaas/voice-profile@v1",
                "artifact_id": "satraj-piranesi-profile-v1",
                "manifest_sha256": source_hash,
                "alias": "piranesi",
                "presentation_id": "piranesi.en-gb.neutral",
                "engine": "cosyvoice3",
                "promotable": promotable,
            },
            "synthesis": {
                "schema": "avaas/wakeword-synthesis@v1",
                "engine": "cosyvoice3",
                "presentation_id": "piranesi.en-gb.neutral",
                "normalizer": "avaas-wake-normalize/1",
                "source_sample_rates_hz": [24_000],
            },
            "audio": {"sample_rate_hz": 16_000, "channels": 1, "sample_format": "pcm_s16le"},
            "clips": {
                "manifest_path": "clips/manifest.json",
                "manifest_sha256": clips_manifest_hash,
                "count": clip_count,
                "total_frames": total_frames,
            },
            "license_ledger": {"path": "provenance/licenses.json"},
            "promotable": promotable,
            "fixture": fixture,
            "created_at": "2026-07-16T20:00:00Z",
            "author": "CPCS",
        },
    )
    seal(bundle)
    return bundle
