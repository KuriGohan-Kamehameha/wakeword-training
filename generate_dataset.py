#!/usr/bin/env python3
import argparse
import json
import os
import random
from collections import deque
from pathlib import Path


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def parse_sources(raw_sources: str) -> list[str]:
    if not raw_sources:
        return []
    return [s.strip() for s in raw_sources.split(",") if s.strip()]


def collect_files(sources: list[str]) -> dict[str, list[str]]:
    # Bug-hunt iter 338: path.rglob("*") had no file count cap — a source
    # directory with millions of files iterated without bound.  NASA P10 Rule 2:
    # explicit upper bound.  Cap at 100 000 audio files per source.
    _MAX_FILES_PER_SOURCE = 100_000
    collected: dict[str, list[str]] = {}
    for source in sources:
        path = Path(source).expanduser()
        if path.is_file():
            collected[source] = [str(path)]
            continue
        if not path.exists():
            collected[source] = []
            continue
        files: list[str] = []
        for p in path.rglob("*"):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                files.append(str(p))
                if len(files) >= _MAX_FILES_PER_SOURCE:
                    print(f"WARNING: {source}: hit {_MAX_FILES_PER_SOURCE} file cap; truncating", flush=True)
                    break
        collected[source] = files
    return collected


def distribute_diverse(
    collected: dict[str, list[str]],
    max_total: int | None,
    min_per_source: int,
    rng: random.Random,
) -> list[str]:
    sources = [s for s, files in collected.items() if files]
    if not sources:
        return []

    if max_total is not None:
        max_per_source_min = max_total // len(sources)
        effective_min = min(min_per_source, max_per_source_min)
    else:
        effective_min = min_per_source

    per_source_files: dict[str, deque[str]] = {}
    for source in sources:
        files = list(collected[source])
        rng.shuffle(files)
        per_source_files[source] = deque(files)

    selection: list[str] = []
    for source in sources:
        if effective_min <= 0:
            continue
        files = per_source_files[source]
        take = min(effective_min, len(files))
        for _ in range(take):
            selection.append(files.popleft())

    if max_total is not None:
        remaining_slots = max(0, max_total - len(selection))
    else:
        remaining_slots = None

    # Bug-hunt iter 345: `while True:` had no explicit iteration cap — when
    # max_total is None the loop relied entirely on the deques draining, with no
    # upper bound visible to a static analyzer.  NASA P10 Rule 2: add an
    # explicit cap equal to the total files across all sources (a tighter bound
    # than `while True:` while preserving the existing termination logic).
    _total_files = sum(len(list(q)) for q in per_source_files.values())
    _max_iters = _total_files + 1  # +1 for the final made_progress=False cycle
    _iter = 0
    while _iter < _max_iters:
        _iter += 1
        if remaining_slots is not None and remaining_slots <= 0:
            break
        made_progress = False
        for source in sources:
            if remaining_slots is not None and remaining_slots <= 0:
                break
            files = per_source_files[source]
            if not files:
                continue
            selection.append(files.popleft())
            made_progress = True
            if remaining_slots is not None:
                remaining_slots -= 1
        if not made_progress:
            break

    return selection


def write_list(path: Path, entries: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in entries:
            handle.write(f"{item}\n")


def parse_int(value: str, label: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except ValueError:
        # Bug-hunt iter 347: raised argparse.ArgumentTypeError from a direct
        # call in main() — ArgumentTypeError is only handled by argparse
        # internally; from a manual call it becomes an uncaught exception
        # producing a Python traceback instead of a clean error message.
        # Use SystemExit to get a clean CLI error line.
        raise SystemExit(f"error: {label} must be an integer (got: {value!r})")
    if parsed < 0:
        raise SystemExit(f"error: {label} must be >= 0 (got: {parsed})")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a diversified dataset manifest for wakeword training."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wake-phrase", required=True)
    parser.add_argument("--positive-sources", required=True)
    parser.add_argument("--negative-sources", required=True)
    parser.add_argument("--max-positives", default="")
    parser.add_argument("--max-negatives", default="")
    parser.add_argument("--min-per-source", default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    positive_sources = parse_sources(args.positive_sources)
    negative_sources = parse_sources(args.negative_sources)

    if not positive_sources:
        raise SystemExit("No positive sources provided.")
    if not negative_sources:
        raise SystemExit("No negative sources provided.")

    max_positives = parse_int(args.max_positives, "max-positives")
    max_negatives = parse_int(args.max_negatives, "max-negatives")
    min_per_source = parse_int(args.min_per_source, "min-per-source") or 0

    rng = random.Random(args.seed)

    positive_collected = collect_files(positive_sources)
    negative_collected = collect_files(negative_sources)

    positives = distribute_diverse(
        positive_collected, max_positives, min_per_source, rng
    )
    negatives = distribute_diverse(
        negative_collected, max_negatives, min_per_source, rng
    )

    manifest = {
        "wake_phrase": args.wake_phrase,
        "positives": positives,
        "negatives": negatives,
        "summary": {
            "positive_sources": {
                source: len(files) for source, files in positive_collected.items()
            },
            "negative_sources": {
                source: len(files) for source, files in negative_collected.items()
            },
            "selected_positives": len(positives),
            "selected_negatives": len(negatives),
            "min_per_source": min_per_source,
            "max_positives": max_positives,
            "max_negatives": max_negatives,
        },
    }

    manifest_path = output_dir / "dataset.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    write_list(output_dir / "positives.txt", positives)
    write_list(output_dir / "negatives.txt", negatives)

    print(json.dumps(manifest["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
