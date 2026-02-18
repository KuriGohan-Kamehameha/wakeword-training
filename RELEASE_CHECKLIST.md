# Release Checklist

1. Verify local checks:
   - `python3 -m py_compile *.py`
   - `bash -n docker-train.sh trainer.sh`
   - `docker compose config`
2. Verify sample generation smoke test:
   - `docker compose run --rm trainer python3 generate_training_samples.py --wake-phrase "release check" --data-dir /workspace/data --positives 1 --negatives 0 --piper-max-voices 1`
3. Verify persistence paths on host:
   - `wakeword_lab/data/custom_models/`
   - `wakeword_lab/data/training_runs/`
   - `wakeword_lab/data/logs/`
   - `wakeword_lab/data/services/piper/`
   - `wakeword_lab/data/services/openwakeword/`
4. Confirm no generated artifacts are accidentally tracked:
   - `git status --short`
   - Ensure only source/config/docs changes are staged.
5. Tag and publish release from a clean branch after CI passes.
