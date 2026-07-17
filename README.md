# wakeword-training

Docker-first wakeword training for openWakeWord, with an end-to-end workflow that runs from one command.

## Part of the Piranesi voice stack

The **wake-word detector** half. Its sibling is **[AVAAS](https://github.com/KuriGohan-Kamehameha/AVAAS)** — *Automated Voice Assimilation and Application System*, a browser studio that records and standardizes a voice into a corpus for TTS fine-tuning.

They connect end to end: **AVAAS assimilates the voice → this trains the wake word for it.** Once AVAAS can synthesize a voice, it can produce *personalized* positive samples of the wake phrase **in that voice**. Those samples augment—not replace—the generic `piper-sample-generator` voices, preserving speaker diversity while tuning the detector for Piranesi's own presentation.

## Platform support

- Linux: supported (`bash`, Docker Engine + Compose plugin)
- macOS: supported (Docker Desktop)
- Windows: supported via `docker-train.ps1` (PowerShell + Docker Desktop; Git Bash or WSL fallback also works)

## Prerequisites

- Docker Desktop (Windows/macOS) or Docker Engine + Compose plugin (Linux)
- Running Docker daemon
- Initialize bundled submodule once after clone:
  - `git submodule update --init --recursive`

## Quick start

### Linux/macOS

```bash
./docker-train.sh --list-devices
./docker-train.sh --wake-phrase "Theodora" --device esphome_generic --threads 4 --format tflite
```

### Windows PowerShell

```powershell
./docker-train.ps1 --wake-phrase "Theodora" --device esphome_generic --threads 4 --format tflite
```

Artifacts are written to:

- `wakeword_lab/data/custom_models/`

## Sample diversity

If you use `--generate-samples`, the generator now creates:

- positives in a wakeword-specific folder: `wakeword_lab/data/positives/<wakeword_slug>/`
  - spoken wake phrase with highest-quality available Piper voices plus speech variants (`clean`, `fast`, `slow`, `telephone`, `quiet`, `loud`, `bright`)
- negatives in a pooled folder: `wakeword_lab/data/negatives/`
  - appends only new uniquely indexed clips (no overwriting existing negatives)
  - adds at least 50 new negatives per generation run
  - includes both non-speech (silence/noise/tones/chirps) and speech negatives (diverse voices saying diverse non-wake phrases)

To regenerate local sample data:

```bash
./docker-train.sh --wake-phrase "Theodora" --generate-samples --positives 240 --negatives 240
```

Each generation run appends only unique files and does not discard prior samples.

## Recommended device settings

| Device / Target | `--device` | Format | Profile | Threads | Size Target |
|---|---|---|---|---:|---:|
| Anki Vector (wire-pod) | `anki_vector_wirepod` | `tflite` | `tiny` | 2 | <=100 KB |
| ReSpeaker XVF3800 | `respeaker_xvf3800` | `tflite` | `tiny` | 2 | <=50 KB |
| ReSpeaker 2-Mics Pi HAT | `respeaker_2mic_pi_hat` | `tflite` | `tiny` | 2 | <=100 KB |
| ReSpeaker 4-Mics Pi HAT | `respeaker_4mic_pi_hat` | `tflite` | `tiny` | 2 | <=150 KB |
| Atom Echo | `atom_echo` | `tflite` | `tiny` | 2 | <=80 KB |
| Home Assistant Voice (ESPHome) | `esphome_generic` | `tflite` | `tiny` | 2-4 | <=50 KB |
| Home Assistant server | `custom_manual` | `tflite` | `medium` | 2-4 | <=200 KB |

## Trust boundary

`wakeword_web.py` is a privileged surface — it synthesizes audio, spawns Docker via the trainer, writes to disk under `BASE_DIR`, and exposes log contents to the caller. **It binds `127.0.0.1` by default.** To expose on the LAN (or any non-loopback interface), set `WAKEWORD_WEB_BIND_ALL=1` in the environment. Even then, there is no built-in authentication — the operator owns the trust decision.

User-supplied `piper_host` / `oww_host` / `piper_port` / `oww_port` are regex-validated before being passed to the trainer subprocess; rejected inputs return HTTP 400 rather than silently rewriting (silent rewrite hides probing).

## Reliability/performance notes

- Health checks fail fast if `piper`/`openwakeword` do not become healthy.
- Generated negatives are normalized to mono 16k PCM before augmentation to avoid sample-rate crashes.
- `tiny` profile now uses lower default manifest caps to reduce memory pressure on constrained hosts.
- If clip generation is OOM-killed (`exit 137`), training auto-retries once with reduced clip counts.
- ONNX -> TFLite conversion runs via `onnx2tf` and reuses a cached converter venv to reduce repeat-run time.
- Training auto-patches upstream `openwakeword/train.py` defaults that otherwise trigger unintended conversion behavior.

## Data layout

- `wakeword_lab/data/positives/<wakeword_slug>/`: wakeword-specific positive audio clips
- `wakeword_lab/data/negatives/`: negative audio clips
- `wakeword_lab/data/hard_negatives/`: mined hard negatives from false activations
- `wakeword_lab/data/custom_models/`: exported model artifacts
- `wakeword_lab/data/training_runs/`: run artifacts and logs

## Artifacts

Each model written to `custom_models/` ships with a sidecar manifest and (optionally) a Piranesi-shaped phrase entry:

- `<slug>.<format>` — the model itself (e.g., `hey-piranesi.tflite`)
- `<slug>.<format>.json` — manifest: phrase, training params, voices used, repo SHA, openwakeword version, device target, threshold suggestion, eval report
- `<slug>.<format>.phrases-entry.json` — opt-in via `--emit-piranesi-entry`. Drops directly into Piranesi's `state/wakeword/phrases.json` (`vector-override` skill consumes it). Threshold sourced from the manifest's eval recommendation; `enabled: false` by default

Trainer exit codes (machine-consumable; last line on stderr is a JSON `{exit_code, reason, details}` blob on failure):

| code | meaning |
|-----:|---------|
| 0 | success, all gates passed |
| 1 | generic / inherited fatal |
| 2 | eval gate failed (false-positive rate or recall below threshold in `device_workflows.json`) |
| 3 | dataset insufficient (zero positives or negatives) |
| 4 | docker / environment failure |
| 5 | config error (bad flags, missing device profile) |

## Training a Piranesi-shaped wake word

Piranesi's TTS path uses `en_US-lessac-low` for spoken output, which means a Vector running Piranesi's vector-override speaks in Lessac's voice. **Do not use `en_US-lessac-low` for positive sample generation when training a wake word that Piranesi himself might trigger** — it creates a feedback loop where the model learns to fire on Piranesi's own speech. Use the default diverse Piper pool (which prefers `*-high` voices, not `*-low`), or explicitly select 3+ voices that are not lessac-low.

### Importing AVAAS positives

AVAAS exports an immutable `avaas/wakeword-export@v1` directory containing
the exact phrase `Hey Piranesi`, source voice/presentation and prosody
provenance, mono 16 kHz PCM16 clips, a complete checksum inventory, and a
`READY` marker written last. Import refuses symlinks, traversal, unexpected or
executable files, corruption, incompatible audio, the wrong alias, and
nonpromotable fixtures by default.

```bash
python3 avaas_import.py \
  /path/to/hey-piranesi-export-v1 \
  wakeword_lab/data/avaas_imports
```

The command is append-only and idempotent. Point training at one or more
validated imported bundle directories (comma-separated):

```bash
export AVAAS_IMPORT_DIRS="$PWD/wakeword_lab/data/avaas_imports/hey-piranesi-export-v1"
./docker-train.sh --wake-phrase "Hey Piranesi"
```

After openWakeWord generates its generic Piper clips, the trainer validates
the AVAAS bundle again and deterministically copies its clips into the real
`positive_train` and `positive_test` directories before normalization and
augmentation. It enforces independent generic, personalized, and total count
gates and writes `avaas-personalized-stage.json` into the run directory.

`--allow-nonpromotable` and `AVAAS_ALLOW_NONPROMOTABLE=1` exist only for
synthetic contract smoke tests; never use them for a release model.

Quick start:

```bash
./docker-train.sh \
  --wake-phrase "Hey Piranesi" \
  --device anki_vector_wirepod \
  --threads 2 \
  --format tflite \
  --emit-piranesi-entry
```

Then on the Piranesi host:

```bash
cp wakeword_lab/data/custom_models/hey-piranesi.tflite \
   ~/Piranesi/openclaw/workspace/state/wakeword/models/
# Edit phrases.json: paste contents of hey-piranesi.tflite.phrases-entry.json
# into the phrases array, then flip enabled:true.
```

For more operational detail, see `README-docker.md`.
