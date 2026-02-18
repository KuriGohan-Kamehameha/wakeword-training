#!/usr/bin/env python3
"""Create dataset.json from generated audio samples."""
import json
import os
import sys
from pathlib import Path


def create_dataset_json(data_dir="./wakeword_lab/data", output_file=None):
    """Create dataset.json from audio files in data directory."""
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Data directory '{data_dir}' not found", file=sys.stderr)
        sys.exit(1)
    
    # If no output specified, use data_dir
    if output_file is None:
        output_file = data_path / "dataset.json"
    else:
        output_file = Path(output_file)
    
    samples = []
    
    # Collect positive samples
    pos_dir = data_path / "positives"
    if pos_dir.exists():
        for wav_file in sorted(pos_dir.glob("*.wav")):
            samples.append({
                "filename": str(wav_file.relative_to(data_path.parent)),
                "label": 1
            })
    
    # Collect negative samples
    neg_dir = data_path / "negatives"
    if neg_dir.exists():
        for wav_file in sorted(neg_dir.glob("*.wav")):
            samples.append({
                "filename": str(wav_file.relative_to(data_path.parent)),
                "label": 0
            })
    
    # Create dataset
    dataset = {
        "samples": samples
    }
    
    # Write JSON
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Created dataset.json with {len(samples)} samples")
    print(f"  Positives: {len([s for s in samples if s['label'] == 1])}")
    print(f"  Negatives: {len([s for s in samples if s['label'] == 0])}")
    print(f"  Output: {output_file}")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./wakeword_lab/data"
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    create_dataset_json(data_dir, output_file)
