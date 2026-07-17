#!/usr/bin/env python3
"""Validate and append-only import ``avaas/wakeword-export@v1`` bundles."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import struct
import tempfile
import wave
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


SCHEMA = "avaas/wakeword-export@v1"
CLIP_SCHEMA = "avaas/wakeword-clips@v1"
SYNTHESIS_SCHEMA = "avaas/wakeword-synthesis@v1"
PHRASE = "Hey Piranesi"
PRESENTATION_ID = "piranesi.en-gb.neutral"
MAX_FILES = 1_024
MAX_DIRECTORIES = 64
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_DOCUMENT_BYTES = 256 * 1024
MAX_CHECKSUM_BYTES = 256 * 1024
MAX_CLIP_BYTES = 1024 * 1024
MAX_CLIPS = 512
MAX_JSON_NODES = 65_536
MAX_JSON_DEPTH = 24
MAX_TOTAL_FRAMES = 16_000 * 60 * 30
MIN_CLIP_FRAMES = 4_000
MAX_CLIP_FRAMES = 80_000
HASH_CHUNK_BYTES = 64 * 1024
ALLOWED_STYLES = {"neutral", "warm", "authoritative", "urgent", "whisper", "projected"}
ALLOWED_SOURCE_RATES = {16_000, 22_050, 24_000, 48_000}
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9.-]{0,126}[a-z0-9])?$")
_SEMVER_RE = re.compile(r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$")
_CREATED_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
_CLIP_RE = re.compile(r"^clips/([0-9a-f]{64})\.wav$")
_CHECKSUM_RE = re.compile(r"^([0-9a-f]{64})  ([A-Za-z0-9_.\-/]+)$")
_FORBIDDEN_SUFFIXES = {".pickle", ".pkl", ".pt", ".pth", ".py", ".pyc", ".sh", ".so", ".dll", ".exe"}


class ImportContractError(ValueError):
    """An AVAAS export or append-only import operation failed closed."""


class _DuplicateKey(ValueError):
    pass


@dataclass(frozen=True)
class ClipInfo:
    path: str
    sha256: str
    frames: int
    duration_ms: int
    style: str
    speed: float
    pitch_semitones: float
    volume: float
    seed: int


@dataclass(frozen=True)
class BundleInfo:
    path: Path
    export_id: str
    version: str
    phrase: str
    alias: str
    presentation_id: str
    source_artifact_id: str
    source_artifact_manifest_sha256: str
    manifest_sha256: str
    clip_count: int
    total_frames: int
    promotable: bool
    fixture: bool
    clips: tuple[ClipInfo, ...]


@dataclass(frozen=True)
class ImportReceipt:
    path: Path
    export_id: str
    manifest_sha256: str
    clip_count: int
    promotable: bool
    fixture: bool
    idempotent: bool


def _safe_relative(value: Any, label: str = "path") -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 1_024
        or "\\" in value
        or "\x00" in value
    ):
        raise ImportContractError(f"invalid {label}")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ImportContractError(f"invalid {label}")
    return candidate.as_posix()


def _root(path: Path) -> Path:
    path = Path(path)
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise ImportContractError("bundle root is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ImportContractError("bundle root must be a non-symlink directory")
    return path


def _bounded_entries(directory: Path) -> list[os.DirEntry[str]]:
    entries: list[os.DirEntry[str]] = []
    try:
        with os.scandir(directory) as iterator:
            for _entry_index in range(MAX_FILES + MAX_DIRECTORIES + 1):
                try:
                    entry = next(iterator)
                except StopIteration:
                    break
                entries.append(entry)
                if len(entries) > MAX_FILES + MAX_DIRECTORIES:
                    raise ImportContractError("bundle entry count exceeds bound")
    except ImportContractError:
        raise
    except OSError as exc:
        raise ImportContractError("cannot enumerate bundle") from exc
    return sorted(entries, key=lambda item: item.name)


def _inventory(root: Path) -> list[str]:
    root = _root(root)
    directories = [root]
    files: list[str] = []
    visited_directories = 0
    for _index in range(MAX_DIRECTORIES):
        if not directories:
            break
        directory = directories.pop()
        visited_directories += 1
        for entry in _bounded_entries(directory):
            relative = Path(entry.path).relative_to(root).as_posix()
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ImportContractError(f"cannot inspect bundle path: {relative}") from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise ImportContractError(f"bundle contains symlink: {relative}")
            if stat.S_ISDIR(metadata.st_mode):
                directories.append(Path(entry.path))
            elif stat.S_ISREG(metadata.st_mode):
                files.append(_safe_relative(relative))
                if len(files) > MAX_FILES:
                    raise ImportContractError("bundle file count exceeds bound")
            else:
                raise ImportContractError(f"bundle contains special file: {relative}")
    if directories or visited_directories > MAX_DIRECTORIES:
        raise ImportContractError("bundle directory count exceeds bound")
    return sorted(files)


def _file_limit(relative: str) -> int:
    if relative.endswith(".wav"):
        return MAX_CLIP_BYTES
    if relative == "checksums.sha256":
        return MAX_CHECKSUM_BYTES
    if relative == "README.md":
        return MAX_DOCUMENT_BYTES
    if relative == "READY":
        return 0
    return MAX_JSON_BYTES


def _regular_path(root: Path, relative: str, maximum: int) -> Path:
    cursor = root
    parts = PurePosixPath(relative).parts
    for index, part in enumerate(parts):
        cursor = cursor / part
        try:
            metadata = os.lstat(cursor)
        except OSError as exc:
            raise ImportContractError(f"missing bundle file: {relative}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ImportContractError(f"bundle path is a symlink: {relative}")
        if index < len(parts) - 1:
            if not stat.S_ISDIR(metadata.st_mode):
                raise ImportContractError(f"bundle parent is not a directory: {relative}")
        elif not stat.S_ISREG(metadata.st_mode) or not 0 <= metadata.st_size <= maximum:
            raise ImportContractError(f"bundle file size outside bounds: {relative}")
    return cursor


def _read_regular(root: Path, relative: str, maximum: int | None = None) -> bytes:
    root = _root(root)
    relative = _safe_relative(relative)
    maximum = _file_limit(relative) if maximum is None else maximum
    cursor = _regular_path(root, relative, maximum)
    descriptor = -1
    try:
        descriptor = os.open(cursor, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > maximum:
            raise ImportContractError(f"bundle file changed: {relative}")
        chunks: list[bytes] = []
        total = 0
        for _index in range((maximum // HASH_CHUNK_BYTES) + 2):
            chunk = os.read(descriptor, min(HASH_CHUNK_BYTES, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise ImportContractError(f"bundle file exceeds bound: {relative}")
        data = b"".join(chunks)
        if len(data) != metadata.st_size or os.read(descriptor, 1):
            raise ImportContractError(f"bundle file changed: {relative}")
        return data
    except ImportContractError:
        raise
    except OSError as exc:
        raise ImportContractError(f"cannot read bundle file: {relative}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _sha256(root: Path, relative: str) -> str:
    return hashlib.sha256(_read_regular(root, relative)).hexdigest()


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _bound_json(value: Any) -> None:
    stack: list[tuple[Any, int]] = [(value, 0)]
    for _index in range(MAX_JSON_NODES):
        if not stack:
            return
        current, depth = stack.pop()
        if depth > MAX_JSON_DEPTH:
            raise ImportContractError("JSON depth exceeds bound")
        if isinstance(current, dict):
            if len(current) > 1_024:
                raise ImportContractError("JSON object exceeds bound")
            for key, child in current.items():
                if not isinstance(key, str) or len(key) > 256:
                    raise ImportContractError("JSON key exceeds bound")
                stack.append((child, depth + 1))
        elif isinstance(current, list):
            if len(current) > MAX_JSON_NODES:
                raise ImportContractError("JSON array exceeds bound")
            for child in current:
                stack.append((child, depth + 1))
        elif isinstance(current, str) and len(current) > 100_000:
            raise ImportContractError("JSON string exceeds bound")
        elif isinstance(current, float) and not math.isfinite(current):
            raise ImportContractError("JSON contains non-finite number")
    raise ImportContractError("JSON node count exceeds bound")


def _json(root: Path, relative: str) -> dict[str, Any]:
    try:
        value = json.loads(
            _read_regular(root, relative).decode("utf-8"),
            object_pairs_hook=_object_no_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ImportContractError(f"invalid JSON file: {relative}") from exc
    _bound_json(value)
    if not isinstance(value, dict):
        raise ImportContractError(f"JSON root must be an object: {relative}")
    return value


def _keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ImportContractError(f"{label} keys mismatch")
    return value


def _string(value: Any, label: str, maximum: int = 4_096) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= maximum:
        raise ImportContractError(f"invalid {label}")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ImportContractError(f"invalid {label}")
    return value


def _created(value: Any) -> str:
    value = _string(value, "created_at", 32)
    if _CREATED_RE.fullmatch(value) is None:
        raise ImportContractError("created_at must be UTC RFC3339 seconds")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ImportContractError("created_at is not a real timestamp") from exc
    return value


def _checksums(root: Path, files: list[str]) -> dict[str, str]:
    try:
        text = _read_regular(root, "checksums.sha256").decode("ascii")
    except UnicodeDecodeError as exc:
        raise ImportContractError("checksums must be ASCII") from exc
    expected_paths = sorted(set(files) - {"checksums.sha256", "READY"})
    lines = text.splitlines()
    if not text.endswith("\n") or len(lines) != len(expected_paths):
        raise ImportContractError("checksum inventory mismatch")
    result: dict[str, str] = {}
    ordered: list[str] = []
    for line in lines:
        match = _CHECKSUM_RE.fullmatch(line)
        if match is None:
            raise ImportContractError("invalid checksum line")
        digest, relative = match.groups()
        relative = _safe_relative(relative, "checksum path")
        if relative in result or relative in {"checksums.sha256", "READY"}:
            raise ImportContractError("duplicate checksum path")
        result[relative] = digest
        ordered.append(relative)
    if ordered != expected_paths:
        raise ImportContractError("checksum inventory or order mismatch")
    for relative, expected in result.items():
        if _sha256(root, relative) != expected:
            raise ImportContractError(f"checksum mismatch: {relative}")
    return result


def _wav_info(root: Path, relative: str, expected_sha: str) -> tuple[int, int]:
    raw = _read_regular(root, relative, MAX_CLIP_BYTES)
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ImportContractError("wakeword audio checksum mismatch")
    if len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ImportContractError("wakeword audio is not a RIFF/WAVE file")
    if struct.unpack("<I", raw[4:8])[0] + 8 != len(raw):
        raise ImportContractError("wakeword WAV length mismatch")
    try:
        with wave.open(io.BytesIO(raw), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frames = handle.getnframes()
            compression = handle.getcomptype()
            samples = handle.readframes(frames + 1)
    except (wave.Error, EOFError) as exc:
        raise ImportContractError("wakeword audio cannot be decoded") from exc
    if (
        channels != 1
        or width != 2
        or sample_rate != 16_000
        or compression != "NONE"
        or not MIN_CLIP_FRAMES <= frames <= MAX_CLIP_FRAMES
        or len(samples) != frames * 2
        or not any(samples)
    ):
        raise ImportContractError("wakeword audio must be non-silent mono 16 kHz PCM16")
    return frames, round(frames * 1_000 / sample_rate)


def _validate_licenses(root: Path) -> None:
    value = _keys(
        _json(root, "provenance/licenses.json"), {"schema", "entries"}, "license ledger"
    )
    entries = value.get("entries")
    if value.get("schema") != "avaas/license-ledger@v1" or not isinstance(entries, list):
        raise ImportContractError("license ledger schema mismatch")
    expected_keys = {
        "component",
        "name",
        "spdx",
        "source_url",
        "license_url",
        "distribution_restrictions",
    }
    components: set[str] = set()
    for raw in entries[:64]:
        item = _keys(raw, expected_keys, "license entry")
        component = _string(item.get("component"), "license component", 128)
        if component in components:
            raise ImportContractError("duplicate license component")
        components.add(component)
        for key in ("name", "spdx", "distribution_restrictions"):
            _string(item.get(key), f"license {key}")
        for key in ("source_url", "license_url"):
            if not _string(item.get(key), f"license {key}").startswith("https://"):
                raise ImportContractError("license URLs must use HTTPS")
    if len(entries) > 64 or not {"source-voice-artifact", "wakeword-corpus"} <= components:
        raise ImportContractError("license coverage is incomplete")


def _validate_bundle(
    root: Path,
    *,
    require_promotable: bool,
    expected_export_id: str | None = None,
) -> BundleInfo:
    files = _inventory(root)
    if not {"manifest.json", "checksums.sha256", "READY"} <= set(files):
        raise ImportContractError("bundle inventory lacks required files")
    for relative in files:
        if Path(relative).suffix.lower() in _FORBIDDEN_SUFFIXES:
            raise ImportContractError(f"unexpected executable or serialized file: {relative}")
        mode = os.lstat(root / relative).st_mode
        if mode & 0o111:
            raise ImportContractError(f"unexpected executable file: {relative}")
    ready = os.lstat(root / "READY")
    if not stat.S_ISREG(ready.st_mode) or ready.st_size != 0:
        raise ImportContractError("READY marker must be empty")
    latest_payload = max(os.lstat(root / item).st_mtime_ns for item in files if item != "READY")
    if ready.st_mtime_ns < latest_payload:
        raise ImportContractError("READY marker was not written last")
    checksums = _checksums(root, files)
    clip_files = [item for item in files if _CLIP_RE.fullmatch(item)]
    expected_inventory = {
        "manifest.json",
        "clips/manifest.json",
        "provenance/licenses.json",
        "README.md",
        "checksums.sha256",
        "READY",
        *clip_files,
    }
    if not 1 <= len(clip_files) <= MAX_CLIPS or set(files) != expected_inventory:
        raise ImportContractError("unexpected wakeword export inventory")
    manifest = _keys(
        _json(root, "manifest.json"),
        {
            "schema",
            "export_id",
            "version",
            "phrase",
            "source_artifact",
            "synthesis",
            "audio",
            "clips",
            "license_ledger",
            "promotable",
            "fixture",
            "created_at",
            "author",
        },
        "export manifest",
    )
    if manifest.get("schema") != SCHEMA:
        raise ImportContractError("unknown AVAAS wakeword export schema")
    export_id = _string(manifest.get("export_id"), "export id", 128)
    version = _string(manifest.get("version"), "version", 32)
    expected_name = root.name if expected_export_id is None else expected_export_id
    if (
        _SLUG_RE.fullmatch(export_id) is None
        or export_id != expected_name
        or _SEMVER_RE.fullmatch(version) is None
    ):
        raise ImportContractError("export id or version mismatch")
    if manifest.get("author") != "CPCS":
        raise ImportContractError("export author mismatch")
    _created(manifest.get("created_at"))
    if manifest.get("phrase") != {
        "text": PHRASE,
        "normalized": "hey piranesi",
        "language_tag": "en",
    }:
        raise ImportContractError("wake phrase is not the Hey Piranesi allow-list entry")
    promotable = manifest.get("promotable")
    fixture = manifest.get("fixture")
    if (
        not isinstance(promotable, bool)
        or not isinstance(fixture, bool)
        or (fixture and promotable)
        or (require_promotable and not promotable)
    ):
        raise ImportContractError("bundle is a fixture or is not promotable")
    source = _keys(
        manifest.get("source_artifact"),
        {
            "schema",
            "artifact_id",
            "manifest_sha256",
            "alias",
            "presentation_id",
            "engine",
            "promotable",
        },
        "source artifact",
    )
    expected_engine = {
        "avaas/voice-profile@v1": "cosyvoice3",
        "avaas/voice-model@v1": "piper",
    }.get(source.get("schema"))
    source_artifact_id = _string(source.get("artifact_id"), "source artifact id", 128)
    source_hash = _sha(source.get("manifest_sha256"), "source artifact manifest hash")
    if (
        expected_engine is None
        or _SLUG_RE.fullmatch(source_artifact_id) is None
        or source.get("alias") != "piranesi"
        or source.get("presentation_id") != PRESENTATION_ID
        or source.get("engine") != expected_engine
        or not isinstance(source.get("promotable"), bool)
        or (promotable and not source["promotable"])
    ):
        raise ImportContractError("source artifact must be Piranesi's declared alias/presentation")
    synthesis = _keys(
        manifest.get("synthesis"),
        {"schema", "engine", "presentation_id", "normalizer", "source_sample_rates_hz"},
        "synthesis provenance",
    )
    source_rates = synthesis.get("source_sample_rates_hz")
    if (
        synthesis.get("schema") != SYNTHESIS_SCHEMA
        or synthesis.get("engine") != expected_engine
        or synthesis.get("presentation_id") != PRESENTATION_ID
        or synthesis.get("normalizer") != "avaas-wake-normalize/1"
        or not isinstance(source_rates, list)
        or source_rates != sorted(set(source_rates))
        or any(rate not in ALLOWED_SOURCE_RATES for rate in source_rates)
    ):
        raise ImportContractError("synthesis provenance mismatch")
    if manifest.get("audio") != {
        "sample_rate_hz": 16_000,
        "channels": 1,
        "sample_format": "pcm_s16le",
    }:
        raise ImportContractError("wakeword audio/rate contract mismatch")
    clip_declaration = _keys(
        manifest.get("clips"),
        {"manifest_path", "manifest_sha256", "count", "total_frames"},
        "clip declaration",
    )
    clip_manifest_hash = _sha(clip_declaration.get("manifest_sha256"), "clip manifest hash")
    if (
        clip_declaration.get("manifest_path") != "clips/manifest.json"
        or checksums.get("clips/manifest.json") != clip_manifest_hash
    ):
        raise ImportContractError("clip manifest checksum/path mismatch")
    clip_manifest = _keys(
        _json(root, "clips/manifest.json"),
        {"schema", "phrase", "alias", "source_artifact_manifest_sha256", "clips"},
        "clip manifest",
    )
    entries = clip_manifest.get("clips")
    if (
        clip_manifest.get("schema") != CLIP_SCHEMA
        or clip_manifest.get("phrase") != PHRASE
        or clip_manifest.get("alias") != "piranesi"
        or clip_manifest.get("source_artifact_manifest_sha256") != source_hash
        or not isinstance(entries, list)
        or not 1 <= len(entries) <= MAX_CLIPS
        or clip_declaration.get("count") != len(entries)
    ):
        raise ImportContractError("clip manifest identity/count mismatch")
    expected_entry_keys = {
        "path",
        "sha256",
        "source_sha256",
        "source_sample_rate_hz",
        "phrase",
        "style",
        "speed",
        "pitch_semitones",
        "volume",
        "seed",
        "frames",
        "duration_ms",
    }
    seen_paths: set[str] = set()
    observed_rates: set[int] = set()
    clips: list[ClipInfo] = []
    total_frames = 0
    for raw in entries:
        item = _keys(raw, expected_entry_keys, "clip entry")
        relative = _safe_relative(item.get("path"), "clip path")
        match = _CLIP_RE.fullmatch(relative)
        digest = _sha(item.get("sha256"), "clip hash")
        _sha(item.get("source_sha256"), "source clip hash")
        if (
            match is None
            or match.group(1) != digest
            or checksums.get(relative) != digest
            or relative in seen_paths
        ):
            raise ImportContractError("duplicate or mismatched clip identity")
        seen_paths.add(relative)
        source_rate = item.get("source_sample_rate_hz")
        speed, pitch, volume, seed = (
            item.get("speed"),
            item.get("pitch_semitones"),
            item.get("volume"),
            item.get("seed"),
        )
        if (
            item.get("phrase") != PHRASE
            or item.get("style") not in ALLOWED_STYLES
            or source_rate not in ALLOWED_SOURCE_RATES
            or not isinstance(speed, (int, float))
            or isinstance(speed, bool)
            or not 0.75 <= speed <= 1.25
            or not isinstance(pitch, (int, float))
            or isinstance(pitch, bool)
            or not -6.0 <= pitch <= 6.0
            or not isinstance(volume, (int, float))
            or isinstance(volume, bool)
            or not 0.5 <= volume <= 1.5
            or not isinstance(seed, int)
            or isinstance(seed, bool)
            or not 0 <= seed <= 2**63 - 1
        ):
            raise ImportContractError("clip phrase/prosody provenance is invalid")
        frames, duration_ms = _wav_info(root, relative, digest)
        if item.get("frames") != frames or item.get("duration_ms") != duration_ms:
            raise ImportContractError("clip audio metadata mismatch")
        total_frames += frames
        observed_rates.add(source_rate)
        clips.append(
            ClipInfo(
                path=relative,
                sha256=digest,
                frames=frames,
                duration_ms=duration_ms,
                style=item["style"],
                speed=float(speed),
                pitch_semitones=float(pitch),
                volume=float(volume),
                seed=seed,
            )
        )
    if (
        seen_paths != set(clip_files)
        or total_frames != clip_declaration.get("total_frames")
        or total_frames > MAX_TOTAL_FRAMES
        or sorted(observed_rates) != source_rates
    ):
        raise ImportContractError("clip inventory, frames, or source rates mismatch")
    if manifest.get("license_ledger") != {"path": "provenance/licenses.json"}:
        raise ImportContractError("license ledger path mismatch")
    _validate_licenses(root)
    try:
        readme = _read_regular(root, "README.md").decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ImportContractError("README must be UTF-8") from exc
    if not readme.strip():
        raise ImportContractError("README must not be empty")
    return BundleInfo(
        path=root,
        export_id=export_id,
        version=version,
        phrase=PHRASE,
        alias="piranesi",
        presentation_id=PRESENTATION_ID,
        source_artifact_id=source_artifact_id,
        source_artifact_manifest_sha256=source_hash,
        manifest_sha256=checksums["manifest.json"],
        clip_count=len(clips),
        total_frames=total_frames,
        promotable=promotable,
        fixture=fixture,
        clips=tuple(clips),
    )


def validate_bundle(root: Path | str, *, require_promotable: bool = False) -> BundleInfo:
    """Validate one export without executing or importing any model payload."""

    return _validate_bundle(
        Path(root), require_promotable=require_promotable, expected_export_id=None
    )


def _ensure_destination(path: Path) -> Path:
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True, mode=0o750)
        metadata = os.lstat(path)
    except OSError as exc:
        raise ImportContractError("cannot create import destination") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ImportContractError("import destination must be a non-symlink directory")
    return path


def _write_exclusive(path: Path, data: bytes, mode: int = 0o640) -> None:
    descriptor = -1
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            mode,
        )
        offset = 0
        for _index in range((len(data) // HASH_CHUNK_BYTES) + 2):
            if offset == len(data):
                break
            written = os.write(descriptor, data[offset : offset + HASH_CHUNK_BYTES])
            if written <= 0:
                raise ImportContractError("short import write")
            offset += written
        if offset != len(data):
            raise ImportContractError("short import write")
        os.fsync(descriptor)
    except ImportContractError:
        raise
    except OSError as exc:
        raise ImportContractError(f"cannot write imported file: {path.name}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY)
        os.fsync(descriptor)
    except OSError as exc:
        raise ImportContractError("cannot fsync import directory") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _receipt(info: BundleInfo, *, idempotent: bool) -> ImportReceipt:
    return ImportReceipt(
        path=info.path,
        export_id=info.export_id,
        manifest_sha256=info.manifest_sha256,
        clip_count=info.clip_count,
        promotable=info.promotable,
        fixture=info.fixture,
        idempotent=idempotent,
    )


def import_bundle(
    bundle: Path | str,
    destination: Path | str,
    *,
    allow_nonpromotable: bool = False,
) -> ImportReceipt:
    """Append one validated export, returning an idempotent receipt on replay."""

    source = Path(bundle)
    info = validate_bundle(source, require_promotable=not allow_nonpromotable)
    destination = _ensure_destination(Path(destination))
    final = destination / info.export_id
    if final.exists() or final.is_symlink():
        try:
            existing = validate_bundle(final, require_promotable=not allow_nonpromotable)
        except ImportContractError as exc:
            raise ImportContractError("append-only import collision contains an invalid bundle") from exc
        if existing.manifest_sha256 != info.manifest_sha256:
            raise ImportContractError("append-only import collision has a different manifest")
        return _receipt(existing, idempotent=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{info.export_id}.", dir=destination))
    try:
        files = _inventory(source)
        for relative in files:
            if relative == "READY":
                continue
            _write_exclusive(staging / relative, _read_regular(source, relative))
        _write_exclusive(staging / "READY", b"")
        staged = _validate_bundle(
            staging,
            require_promotable=not allow_nonpromotable,
            expected_export_id=info.export_id,
        )
        if staged.manifest_sha256 != info.manifest_sha256:
            raise ImportContractError("staged import manifest changed")
        try:
            os.rename(staging, final)
        except OSError as exc:
            if final.is_dir() and not final.is_symlink():
                try:
                    concurrent = validate_bundle(
                        final, require_promotable=not allow_nonpromotable
                    )
                except ImportContractError:
                    raise ImportContractError("cannot publish append-only import") from exc
                if concurrent.manifest_sha256 == info.manifest_sha256:
                    shutil.rmtree(staging)
                    return _receipt(concurrent, idempotent=True)
            raise ImportContractError("cannot publish append-only import") from exc
        _fsync_directory(destination)
        published = validate_bundle(final, require_promotable=not allow_nonpromotable)
        return _receipt(published, idempotent=False)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def main(argv: Sequence[str] | None = None) -> int:
    if argv is not None and isinstance(argv, (str, bytes)):
        raise TypeError("argv must be a sequence of arguments")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--allow-nonpromotable",
        action="store_true",
        help="explicitly permit synthetic fixtures for contract testing only",
    )
    args = parser.parse_args(argv)
    receipt = import_bundle(
        args.bundle,
        args.destination,
        allow_nonpromotable=args.allow_nonpromotable,
    )
    if not isinstance(receipt, ImportReceipt):
        raise AssertionError("import_bundle returned an invalid receipt")
    print(json.dumps(asdict(receipt), sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
