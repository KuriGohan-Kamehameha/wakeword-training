#!/usr/bin/env python3
# P10 RELAXATIONS:
# R5: small orchestration helpers delegate their pre/postcondition checks to
# argparse, parse_int, select_positive_sets, and distribute_diverse. Keeping
# those shared validators centralized avoids divergent CLI and test behavior.
import argparse
import json
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


def select_positive_sets(
    generic_collected: dict[str, list[str]],
    personalized_collected: dict[str, list[str]],
    *,
    max_total: int | None,
    min_per_source: int,
    min_generic: int,
    min_personalized: int,
    max_personalized: int,
    rng: random.Random,
) -> tuple[list[str], list[str]]:
    """Reserve generic diversity before admitting bounded AVAAS positives."""
    values = (min_per_source, min_generic, min_personalized, max_personalized)
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
        raise SystemExit("generic/personalized positive bounds must be non-negative integers")
    if min_personalized > max_personalized:
        raise SystemExit("personalized minimum exceeds personalized maximum")
    if max_total is not None and max_total < min_generic:
        raise SystemExit("total positive maximum is below the generic diversity floor")

    personalized_available = any(personalized_collected.values())
    personalized_limit = max_personalized
    if max_total is not None:
        personalized_limit = min(personalized_limit, max(0, max_total - min_generic))
    personalized = (
        distribute_diverse(
            personalized_collected,
            personalized_limit,
            min_per_source,
            rng,
        )
        if personalized_available
        else []
    )
    if personalized_available and len(personalized) < min_personalized:
        raise SystemExit(
            f"personalized positive floor not met: {len(personalized)} < {min_personalized}"
        )
    generic_limit = None if max_total is None else max_total - len(personalized)
    generic = distribute_diverse(
        generic_collected,
        generic_limit,
        min_per_source,
        rng,
    )
    if len(generic) < min_generic:
        raise SystemExit(f"generic positive floor not met: {len(generic)} < {min_generic}")
    return generic, personalized


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a diversified dataset manifest for wakeword training."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wake-phrase", required=True)
    parser.add_argument("--positive-sources", required=True)
    parser.add_argument("--personalized-sources", default="")
    parser.add_argument("--negative-sources", required=True)
    parser.add_argument("--max-positives", default="")
    parser.add_argument("--max-negatives", default="")
    parser.add_argument("--min-per-source", default="")
    parser.add_argument("--min-generic-positive", default="1")
    parser.add_argument("--min-personalized", default="0")
    parser.add_argument("--max-personalized", default="512")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _collect_sources(args: argparse.Namespace):
    positive_sources = parse_sources(args.positive_sources)
    personalized_sources = parse_sources(args.personalized_sources)
    negative_sources = parse_sources(args.negative_sources)
    if not positive_sources:
        raise SystemExit("No positive sources provided.")
    if not negative_sources:
        raise SystemExit("No negative sources provided.")
    return (
        collect_files(positive_sources),
        collect_files(personalized_sources),
        collect_files(negative_sources),
    )


def _select_entries(args: argparse.Namespace, collections):
    positive_collected, personalized_collected, negative_collected = collections
    max_positives = parse_int(args.max_positives, "max-positives")
    max_negatives = parse_int(args.max_negatives, "max-negatives")
    min_per_source = parse_int(args.min_per_source, "min-per-source") or 0
    min_generic = parse_int(args.min_generic_positive, "min-generic-positive")
    min_personalized = parse_int(args.min_personalized, "min-personalized")
    max_personalized = parse_int(args.max_personalized, "max-personalized")
    if min_generic is None or min_personalized is None or max_personalized is None:
        raise SystemExit("generic/personalized positive bounds are required")
    rng = random.Random(args.seed)
    generic_positives, personalized_positives = select_positive_sets(
        positive_collected,
        personalized_collected,
        max_total=max_positives,
        min_per_source=min_per_source,
        min_generic=min_generic,
        min_personalized=min_personalized,
        max_personalized=max_personalized,
        rng=rng,
    )
    negatives = distribute_diverse(
        negative_collected, max_negatives, min_per_source, rng
    )
    bounds = (
        min_per_source,
        max_positives,
        min_generic,
        min_personalized,
        max_personalized,
        max_negatives,
    )
    return (generic_positives, personalized_positives, negatives), bounds


def _build_manifest(args: argparse.Namespace, collections, selected, bounds):
    positive_collected, personalized_collected, negative_collected = collections
    generic_positives, personalized_positives, negatives = selected
    positives = generic_positives + personalized_positives
    (
        min_per_source,
        max_positives,
        min_generic,
        min_personalized,
        max_personalized,
        max_negatives,
    ) = bounds
    manifest = {
        "wake_phrase": args.wake_phrase,
        "positives": positives,
        "negatives": negatives,
        "summary": {
            "positive_sources": {
                source: len(files) for source, files in positive_collected.items()
            },
            "personalized_sources": {
                source: len(files) for source, files in personalized_collected.items()
            },
            "negative_sources": {
                source: len(files) for source, files in negative_collected.items()
            },
            "selected_positives": len(positives),
            "selected_generic_positives": len(generic_positives),
            "selected_personalized_positives": len(personalized_positives),
            "selected_negatives": len(negatives),
            "min_per_source": min_per_source,
            "max_positives": max_positives,
            "min_generic_positive": min_generic,
            "min_personalized": min_personalized,
            "max_personalized": max_personalized,
            "max_negatives": max_negatives,
        },
    }
    return manifest


def _write_outputs(output_dir: Path, manifest: dict) -> None:
    positives = manifest["positives"]
    negatives = manifest["negatives"]
    manifest_path = output_dir / "dataset.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    write_list(output_dir / "positives.txt", positives)
    write_list(output_dir / "negatives.txt", negatives)


def main() -> int:
    args = _parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    collections = _collect_sources(args)
    selected, bounds = _select_entries(args, collections)
    manifest = _build_manifest(args, collections, selected, bounds)
    _write_outputs(output_dir, manifest)
    print(json.dumps(manifest["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
