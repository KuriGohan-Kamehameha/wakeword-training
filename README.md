# wakeword-training

Docker-first wakeword training for openWakeWord, with an end-to-end workflow that runs from one command.

## Part of the Piranesi voice stack

The **wake-word detector** half. Its sibling is **[AVAAS](https://github.com/P1R4N351/AVAAS)** — *Automated Voice Assimilation and Application System*, a browser studio that records and standardizes a voice into a corpus for TTS fine-tuning.

They connect end to end: **AVAAS assimilates the voice → this trains the wake word for it.** Once AVAAS can synthesize a voice, it can produce *personalized* positive samples of the wake phrase **in that voice** to feed the sample-generation step here (alongside or instead of the generic `piper-sample-generator` voices) — yielding a detector tuned to how the speaker actually says it, not a generic TTS approximation.

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

## Dataset validation (malformed / mislabeled data)

Synthetic corpora rot silently: TTS voices drift off-phrase, generators emit
silent or clipped WAVs, and a single mislabeled negative poisons recall. The
eval gate catches some of this *after* a full training run; `validate_dataset.py`
catches it *before*, in seconds.

```bash
python3 validate_dataset.py --phrase "Hey Piranesi" --slug hey_piranesi \
    --stt-url http://<your-stt-host>:5007/stt   # any endpoint speaking POST /stt -> {"text": ...}
    # add --quarantine to move findings into a sibling _quarantine/ dir (never deletes)
```

Checks:

- **structural** — unreadable/zero-frame WAVs, wrong sample width, <0.3 s or
  >12 s, effectively-silent (RMS), clipped (>2 % of samples at the rail)
- **pos-label** — every positive is transcribed by the external STT and
  fuzzy-matched against the wake phrase; misses are listed with what was heard
- **neg-poison** — a random sample of negatives (default 120) is transcribed;
  any negative containing the wake phrase is flagged as a poison candidate

Exit 0 = clean, 1 = findings (report printed), 2 = usage error. Run it after
`--generate-samples` and before training; retrain only on a clean corpus.

## Corpus augmentation from an external TTS (`generate_http_tts_samples.py`)

The bundled generator tops out at ~5 `-high` Piper voices. Detector quality
tracks corpus variety, so this sweeps any HTTP TTS (kudzu-tts/Kokoro-style
`POST /tts {"text","voice","speed"} -> WAV`) across a voice x speed grid and
appends matching-format samples (22050 Hz mono s16, collision-free 9NNNNN
index space) to the same corpus:

```bash
python3 generate_http_tts_samples.py --wake-phrase "Hey Piranesi" \
    --slug hey_piranesi --tts-url http://<host>:5006/tts
# then re-run training WITHOUT --generate-samples to train on the merged corpus
```

Unknown voices are probed and skipped, so the broad default grid degrades
gracefully. Prefer quality and variety over generation speed — the corpus is
a one-time cost, the detector is forever.

## Cross-device live acoustic testing (`acoustic_live_test.py`)

Synthetic eval proves the model on synthetic audio only. This drives the loop
that matters: a REAL loudspeaker on one machine plays labelled clips
(`pos_*.wav` / `neg_*.wav`) into the room, a REAL microphone on another
machine records them, and the trained model scores the recordings —
acceptance + threshold selection grounded in your room, speaker, and mic.

```bash
./acoustic-live-test.sh hey_piranesi root@speaker-host plughw:AE5 mic-host plughw:AE5
# or the two phases separately: `record` (host, needs ssh+alsa-utils on peers)
# and `score` (inside the trainer container). Recordings are model-independent:
# record once, score every candidate model against the same real-room audio.
```

The report (`acoustic_runs/<slug>/acoustic_report.json`) gives per-clip max
scores, whether positives separate from negatives, and a suggested threshold.
A dead or overfit model shows up immediately as `separates: false` — synthetic
eval alone will not tell you that.
