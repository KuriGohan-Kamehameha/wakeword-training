#!/usr/bin/env python3
"""Closed-loop calibration and hard-negative mining for wakeword models."""
import argparse
import json
import re
import shutil
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from openwakeword.model import Model

AUDIO_EXTS = {".wav"}


def collect_wavs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


def sample_even(paths: List[Path], max_items: int) -> List[Path]:
    if max_items <= 0 or len(paths) <= max_items:
        return paths
    idx = np.linspace(0, len(paths) - 1, num=max_items, dtype=int)
    return [paths[i] for i in idx]


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        nframes = wf.getnframes()
        rate = wf.getframerate()
    return float(nframes) / float(rate) if rate > 0 else 0.0


def _token_is_identifier(token: str) -> bool:
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


def semantic_name(path: Path) -> str:
    stem = path.stem.lower()
    tokens = [t for t in stem.split("_") if t]
    semantic_tokens = [t for t in tokens if not _token_is_identifier(t)]
    if not semantic_tokens:
        semantic_tokens = tokens or [stem]
    return "_".join(semantic_tokens)


def infer_framework(model_path: Path) -> str:
    ext = model_path.suffix.lower()
    if ext == ".onnx":
        return "onnx"
    return "tflite"


@dataclass
class ClipEval:
    path: Path
    duration_s: float
    frame_scores: np.ndarray
    max_score: float


def evaluate_clip(model: Model, audio_path: Path) -> ClipEval:
    preds = model.predict_clip(str(audio_path), padding=1, chunk_size=1280)
    if not preds:
        return ClipEval(path=audio_path, duration_s=wav_duration_seconds(audio_path), frame_scores=np.zeros(0), max_score=0.0)
    model_key = next(iter(preds[0].keys()))
    scores = np.array([float(frame.get(model_key, 0.0)) for frame in preds], dtype=np.float32)
    max_score = float(np.max(scores)) if scores.size else 0.0
    return ClipEval(path=audio_path, duration_s=wav_duration_seconds(audio_path), frame_scores=scores, max_score=max_score)


def count_false_alarms(frame_scores: np.ndarray, threshold: float, cooldown_frames: int) -> int:
    alarms = 0
    cooldown = 0
    for score in frame_scores:
        if cooldown > 0:
            cooldown -= 1
            continue
        if score >= threshold:
            alarms += 1
            cooldown = cooldown_frames
    return alarms


def choose_threshold(negatives: List[ClipEval], target_far_per_hour: float, cooldown_sec: float = 1.0, frame_hz: float = 12.5):
    thresholds = [round(i / 100.0, 2) for i in range(5, 100)]
    total_neg_hours = sum(c.duration_s for c in negatives) / 3600.0
    cooldown_frames = max(1, int(round(cooldown_sec * frame_hz)))

    best = None
    for thr in thresholds:
        events = sum(count_false_alarms(c.frame_scores, thr, cooldown_frames) for c in negatives)
        far_h = (events / total_neg_hours) if total_neg_hours > 0 else 0.0
        if far_h <= target_far_per_hour:
            best = (thr, events, far_h)
            break
    if best is None:
        thr = 0.99
        events = sum(count_false_alarms(c.frame_scores, thr, cooldown_frames) for c in negatives)
        far_h = (events / total_neg_hours) if total_neg_hours > 0 else 0.0
        best = (thr, events, far_h)
    return best[0], best[1], best[2], total_neg_hours


def mine_hard_negatives(negatives: List[ClipEval], threshold: float, out_dir: Path, max_mined: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing_keys = set()
    for wav in out_dir.glob("*.wav"):
        existing_keys.add(semantic_name(wav))

    mined = 0
    candidates = sorted((c for c in negatives if c.max_score >= threshold), key=lambda x: x.max_score, reverse=True)
    for clip in candidates:
        if mined >= max_mined:
            break
        key = semantic_name(clip.path)
        if key in existing_keys:
            continue
        dst = out_dir / f"hardneg_{len(existing_keys)+1:06d}_{key}.wav"
        shutil.copy2(clip.path, dst)
        existing_keys.add(key)
        mined += 1
    return mined


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate threshold and mine hard negatives.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--positives-dir", required=True)
    parser.add_argument("--negatives-dir", required=True)
    parser.add_argument("--target-far-per-hour", type=float, default=0.1)
    parser.add_argument("--max-clips", type=int, default=600)
    parser.add_argument("--hard-negatives-dir", required=True)
    parser.add_argument("--max-mined", type=int, default=200)
    parser.add_argument("--report-path", required=True)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise SystemExit(f"Missing model file: {model_path}")

    pos_paths = sample_even(collect_wavs(Path(args.positives_dir)), args.max_clips)
    neg_paths = sample_even(collect_wavs(Path(args.negatives_dir)), args.max_clips)
    if not pos_paths or not neg_paths:
        raise SystemExit("Need both positive and negative clips for closed-loop evaluation.")

    framework = infer_framework(model_path)
    model = Model(wakeword_models=[str(model_path)], inference_framework=framework)

    pos_eval = [evaluate_clip(model, p) for p in pos_paths]
    model.reset()
    neg_eval = [evaluate_clip(model, n) for n in neg_paths]
    model.reset()

    threshold, false_alarms, far_per_hour, neg_hours = choose_threshold(
        neg_eval, target_far_per_hour=args.target_far_per_hour
    )
    pos_hits = sum(1 for c in pos_eval if c.max_score >= threshold)
    pos_recall = pos_hits / len(pos_eval) if pos_eval else 0.0

    mined = mine_hard_negatives(
        negatives=neg_eval,
        threshold=threshold,
        out_dir=Path(args.hard_negatives_dir),
        max_mined=max(0, args.max_mined),
    )

    report = {
        "model_path": str(model_path),
        "inference_framework": framework,
        "target_far_per_hour": float(args.target_far_per_hour),
        "recommended_threshold": float(threshold),
        "observed_far_per_hour": float(far_per_hour),
        "false_alarms": int(false_alarms),
        "negative_hours_evaluated": float(neg_hours),
        "positive_recall": float(pos_recall),
        "positives_evaluated": len(pos_eval),
        "negatives_evaluated": len(neg_eval),
        "hard_negatives_mined": int(mined),
        "hard_negatives_dir": str(Path(args.hard_negatives_dir)),
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
