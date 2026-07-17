# Release Checklist

1. Verify local checks:
   - `python3 -m py_compile *.py`
   - `bash -n docker-train.sh trainer.sh`
   - `docker compose config`
   - `git submodule status` (no dirty submodule state)
2. Verify sample generation smoke test:
   - `docker compose run --rm trainer python3 generate_training_samples.py --wake-phrase "release check" --data-dir /workspace/data --positives 1 --negatives 0 --piper-max-voices 1`
3. Verify the AVAAS contract and actual upstream staging path:
   - `python3 -m unittest discover -s tests -v`
   - Import the nonpromotable fixture only in a disposable directory with `--allow-nonpromotable`.
   - Run the stager twice and confirm the second receipt is idempotent.
   - Inspect the run's actual `positive_train` and `positive_test` directories; generic files must remain byte-identical and AVAAS files must be present in both.
   - For a release, omit all nonpromotable overrides and require a promoted source artifact.
4. Verify persistence paths on host:
   - `wakeword_lab/data/custom_models/`
   - `wakeword_lab/data/training_runs/`
   - `wakeword_lab/data/logs/`
   - `wakeword_lab/data/services/piper/`
   - `wakeword_lab/data/services/openwakeword/`
5. Confirm no generated artifacts are accidentally tracked:
   - `git status --short`
   - Ensure only source/config/docs changes are staged.
6. Tag and publish release from a clean branch after CI passes.
