#!/usr/bin/env python3
"""Append validated AVAAS positives to openWakeWord's generated clip split."""
# P10 RELAXATIONS:
# R5: the scanner treats the multiline stage_exports signature as its body and
# reports zero validations; the actual body has fail-closed raises at every
# trust boundary, beginning with the distinct sibling-directory check.
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import avaas_import


SCHEMA = "wakeword-training/avaas-stage@v1"
MAX_DIRECTORY_FILES = 20_000
MAX_PERSONALIZED = 512
MAX_TOTAL = 20_000
DEFAULT_MIN_GENERIC = 6
DEFAULT_MIN_PERSONALIZED = 8
_STAGED_RE = re.compile(r"^avaas_[a-z0-9.-]{1,128}_[0-9a-f]{64}\.wav$")


class StagingError(ValueError):
    """Personalized-positive planning or append-only staging failed closed."""


@dataclass(frozen=True)
class StageReceipt:
    receipt_path: Path
    mapping_sha256: str
    generic_count: int
    personalized_count: int
    train_count: int
    test_count: int
    total_count: int
    idempotent: bool


def _canonical(value: Any) -> bytes:
    if value is None:
        raise StagingError("stage receipt must not be null")
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise StagingError("stage receipt is not canonical JSON") from exc


def _bounded_integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not minimum <= value <= maximum
    ):
        raise StagingError(f"{label} is outside bounds")
    return value


def _directory(path: Path, label: str) -> Path:
    path = Path(path)
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise StagingError(f"{label} directory is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise StagingError(f"{label} must be a non-symlink directory")
    return path


def _wav_inventory(path: Path) -> list[Path]:
    path = _directory(path, "positive split")
    result: list[Path] = []
    entries = []
    try:
        with os.scandir(path) as iterator:
            for _entry_index in range(MAX_DIRECTORY_FILES + 1):
                try:
                    entry = next(iterator)
                except StopIteration:
                    break
                entries.append(entry)
                if len(entries) > MAX_DIRECTORY_FILES:
                    raise StagingError("positive split file count exceeds bound")
    except StagingError:
        raise
    except OSError as exc:
        raise StagingError("cannot enumerate positive split") from exc
    entries.sort(key=lambda item: item.name)
    for entry in entries:
        try:
            metadata = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise StagingError("cannot inspect positive split entry") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise StagingError(f"positive split contains a symlink: {entry.name}")
        if stat.S_ISREG(metadata.st_mode) and entry.name.lower().endswith(".wav"):
            result.append(Path(entry.path))
    return result


def _file_sha256(path: Path) -> str:
    path = Path(path)
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise StagingError("staged audio is unavailable") from exc
    invalid_type = stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode)
    invalid_size = not 1 <= metadata.st_size <= avaas_import.MAX_CLIP_BYTES
    if invalid_type or invalid_size:
        raise StagingError("staged audio size/type is invalid")
    descriptor = -1
    digest = hashlib.sha256()
    total = 0
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        observed = os.fstat(descriptor)
        if (
            observed.st_ino != metadata.st_ino
            or observed.st_dev != metadata.st_dev
            or observed.st_size != metadata.st_size
        ):
            raise StagingError("staged audio changed before hashing")
        for _index in range((avaas_import.MAX_CLIP_BYTES // avaas_import.HASH_CHUNK_BYTES) + 2):
            chunk = os.read(descriptor, avaas_import.HASH_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > avaas_import.MAX_CLIP_BYTES:
                raise StagingError("staged audio exceeds size bound")
            digest.update(chunk)
        if total != metadata.st_size or os.read(descriptor, 1):
            raise StagingError("staged audio changed while hashing")
        return digest.hexdigest()
    except StagingError:
        raise
    except OSError as exc:
        raise StagingError("cannot hash staged audio") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def plan_split(
    exports: Sequence[Path | str],
    *,
    seed: int = 42,
    allow_nonpromotable: bool = False,
) -> list[dict[str, Any]]:
    """Return a deterministic 80/20 mapping with both splits when possible."""

    _bounded_integer(seed, "dataset seed", 0, 2**63 - 1)
    if not isinstance(exports, (list, tuple)) or not 1 <= len(exports) <= 64:
        raise StagingError("AVAAS export count outside bounds")
    candidates: list[dict[str, Any]] = []
    seen_digests: set[str] = set()
    seen_exports: set[str] = set()
    for raw_path in exports:
        try:
            info = avaas_import.validate_bundle(
                Path(raw_path), require_promotable=not allow_nonpromotable
            )
        except avaas_import.ImportContractError as exc:
            raise StagingError(str(exc)) from exc
        if info.export_id in seen_exports:
            raise StagingError("duplicate AVAAS export id")
        seen_exports.add(info.export_id)
        for clip in info.clips:
            if clip.sha256 in seen_digests:
                raise StagingError("duplicate personalized audio across AVAAS exports")
            seen_digests.add(clip.sha256)
            rank = hashlib.sha256(
                f"{seed}:{info.manifest_sha256}:{clip.sha256}".encode("ascii")
            ).hexdigest()
            candidates.append(
                {
                    "export_id": info.export_id,
                    "export_manifest_sha256": info.manifest_sha256,
                    "source_bundle": str(info.path),
                    "source_path": clip.path,
                    "sha256": clip.sha256,
                    "rank": rank,
                    "target_name": f"avaas_{info.export_id}_{clip.sha256}.wav",
                }
            )
    if not 1 <= len(candidates) <= MAX_PERSONALIZED:
        raise StagingError("personalized clip count outside bounds")
    ranked = sorted(candidates, key=lambda item: (item["rank"], item["sha256"]))
    test_count = 0 if len(ranked) == 1 else max(1, round(len(ranked) * 0.2))
    test_digests = {item["sha256"] for item in ranked[:test_count]}
    result: list[dict[str, Any]] = []
    for item in sorted(candidates, key=lambda candidate: candidate["sha256"]):
        result.append(
            {
                **item,
                "split": "test" if item["sha256"] in test_digests else "train",
            }
        )
    return result


def _write_audio(source_root: Path, source_relative: str, target: Path, digest: str) -> bool:
    if target.exists() or target.is_symlink():
        if target.is_symlink() or _file_sha256(target) != digest:
            raise StagingError("append-only personalized target collision")
        return True
    try:
        raw = avaas_import._read_regular(
            source_root, source_relative, avaas_import.MAX_CLIP_BYTES
        )
    except avaas_import.ImportContractError as exc:
        raise StagingError(str(exc)) from exc
    if hashlib.sha256(raw).hexdigest() != digest:
        raise StagingError("personalized source checksum mismatch")
    try:
        avaas_import._write_exclusive(target, raw)
    except avaas_import.ImportContractError as exc:
        if target.is_file() and not target.is_symlink() and _file_sha256(target) == digest:
            return True
        raise StagingError(str(exc)) from exc
    if _file_sha256(target) != digest:
        raise StagingError("personalized target verification failed")
    return False


def _write_receipt(path: Path, value: dict[str, Any]) -> bool:
    encoded = _canonical(value) + b"\n"
    if path.exists() or path.is_symlink():
        if path.is_symlink():
            raise StagingError("stage receipt must not be a symlink")
        try:
            existing = avaas_import._read_regular(
                path.parent, path.name, avaas_import.MAX_JSON_BYTES
            )
        except avaas_import.ImportContractError as exc:
            raise StagingError("cannot read existing stage receipt") from exc
        if existing != encoded:
            raise StagingError("append-only stage receipt collision")
        return True
    try:
        avaas_import._write_exclusive(path, encoded)
    except avaas_import.ImportContractError as exc:
        if path.is_file() and not path.is_symlink():
            try:
                existing = avaas_import._read_regular(
                    path.parent, path.name, avaas_import.MAX_JSON_BYTES
                )
            except avaas_import.ImportContractError:
                existing = b""
            if existing == encoded:
                return True
        raise StagingError(str(exc)) from exc
    return False


def stage_exports(
    exports: Sequence[Path | str],
    *,
    positive_train: Path | str,
    positive_test: Path | str,
    seed: int = 42,
    min_generic: int = DEFAULT_MIN_GENERIC,
    min_personalized: int = DEFAULT_MIN_PERSONALIZED,
    max_personalized: int = MAX_PERSONALIZED,
    max_total: int = MAX_TOTAL,
    allow_nonpromotable: bool = False,
) -> StageReceipt:
    """Append a deterministic personalized split while preserving generic files."""

    assert exports is not None, "exports must be provided"
    assert positive_train is not None and positive_test is not None, "split paths required"
    train = _directory(Path(positive_train), "positive_train")
    test = _directory(Path(positive_test), "positive_test")
    if train.parent != test.parent or train == test:
        raise StagingError("positive train/test must be distinct sibling directories")
    min_generic = _bounded_integer(min_generic, "minimum generic count", 1, MAX_TOTAL)
    min_personalized = _bounded_integer(
        min_personalized, "minimum personalized count", 1, MAX_PERSONALIZED
    )
    max_personalized = _bounded_integer(
        max_personalized, "maximum personalized count", min_personalized, MAX_PERSONALIZED
    )
    max_total = _bounded_integer(max_total, "maximum total count", 1, MAX_TOTAL)
    inventories = {"train": _wav_inventory(train), "test": _wav_inventory(test)}
    generic_paths = [
        path
        for split_paths in inventories.values()
        for path in split_paths
        if _STAGED_RE.fullmatch(path.name) is None
    ]
    generic_hashes = {str(path): _file_sha256(path) for path in generic_paths}
    generic_count = len(generic_paths)
    if generic_count < min_generic:
        raise StagingError(
            f"generic Piper diversity floor not met: {generic_count} < {min_generic}"
        )
    mapping = plan_split(
        exports, seed=seed, allow_nonpromotable=allow_nonpromotable
    )
    personalized_count = len(mapping)
    if personalized_count < min_personalized:
        raise StagingError(
            f"personalized minimum not met: {personalized_count} < {min_personalized}"
        )
    if personalized_count > max_personalized:
        raise StagingError(
            f"personalized maximum exceeded: {personalized_count} > {max_personalized}"
        )
    if generic_count + personalized_count > max_total:
        raise StagingError("combined generic and personalized total exceeds maximum")
    expected_targets = {
        str((train if item["split"] == "train" else test) / item["target_name"])
        for item in mapping
    }
    existing_personalized = {
        str(path)
        for split_paths in inventories.values()
        for path in split_paths
        if _STAGED_RE.fullmatch(path.name) is not None
    }
    if not existing_personalized <= expected_targets:
        raise StagingError("positive split contains an unexpected personalized clip")
    all_preexisting = True
    for item in mapping:
        target_dir = train if item["split"] == "train" else test
        preexisting = _write_audio(
            Path(item["source_bundle"]),
            item["source_path"],
            target_dir / item["target_name"],
            item["sha256"],
        )
        all_preexisting = all_preexisting and preexisting
    for path, expected_hash in generic_hashes.items():
        if _file_sha256(Path(path)) != expected_hash:
            raise StagingError("generic Piper positive was modified")
    public_mapping = [
        {
            key: item[key]
            for key in (
                "export_id",
                "export_manifest_sha256",
                "source_path",
                "sha256",
                "split",
                "target_name",
            )
        }
        for item in mapping
    ]
    mapping_sha = hashlib.sha256(_canonical(public_mapping)).hexdigest()
    imports = sorted(
        {
            (item["export_id"], item["export_manifest_sha256"])
            for item in mapping
        }
    )
    train_count = sum(1 for item in mapping if item["split"] == "train")
    test_count = personalized_count - train_count
    receipt_value = {
        "schema": SCHEMA,
        "phrase": avaas_import.PHRASE,
        "seed": seed,
        "imports": [
            {"export_id": export_id, "manifest_sha256": digest}
            for export_id, digest in imports
        ],
        "mapping": public_mapping,
        "mapping_sha256": mapping_sha,
        "counts": {
            "generic": generic_count,
            "personalized": personalized_count,
            "personalized_train": train_count,
            "personalized_test": test_count,
            "total": generic_count + personalized_count,
        },
    }
    receipt_path = train.parent / "avaas-personalized-stage.json"
    receipt_preexisting = _write_receipt(receipt_path, receipt_value)
    try:
        avaas_import._fsync_directory(train)
        avaas_import._fsync_directory(test)
        avaas_import._fsync_directory(train.parent)
    except avaas_import.ImportContractError as exc:
        raise StagingError(str(exc)) from exc
    return StageReceipt(
        receipt_path=receipt_path,
        mapping_sha256=mapping_sha,
        generic_count=generic_count,
        personalized_count=personalized_count,
        train_count=train_count,
        test_count=test_count,
        total_count=generic_count + personalized_count,
        idempotent=all_preexisting and receipt_preexisting,
    )


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def main(argv: Sequence[str] | None = None) -> int:
    if argv is not None and isinstance(argv, (str, bytes)):
        raise TypeError("argv must be a sequence of arguments")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imports", required=True, help="comma-separated imported bundle paths")
    parser.add_argument("--positive-train", required=True, type=Path)
    parser.add_argument("--positive-test", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-generic", type=int, default=DEFAULT_MIN_GENERIC)
    parser.add_argument("--min-personalized", type=int, default=DEFAULT_MIN_PERSONALIZED)
    parser.add_argument("--max-personalized", type=int, default=MAX_PERSONALIZED)
    parser.add_argument("--max-total", type=int, default=MAX_TOTAL)
    parser.add_argument("--allow-nonpromotable", action="store_true")
    args = parser.parse_args(argv)
    exports = [Path(item.strip()) for item in args.imports.split(",") if item.strip()]
    receipt = stage_exports(
        exports,
        positive_train=args.positive_train,
        positive_test=args.positive_test,
        seed=args.seed,
        min_generic=args.min_generic,
        min_personalized=args.min_personalized,
        max_personalized=args.max_personalized,
        max_total=args.max_total,
        allow_nonpromotable=args.allow_nonpromotable,
    )
    if not isinstance(receipt, StageReceipt):
        raise AssertionError("stage_exports returned an invalid receipt")
    print(json.dumps(asdict(receipt), sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
