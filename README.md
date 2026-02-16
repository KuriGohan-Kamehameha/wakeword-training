# wakeword-training
Training project for wake word detection

## Usage

### Interactive Mode (Default)

Run the trainer script and it will prompt you for required parameters:

```bash
bash trainer.sh
```

### CLI Mode (Non-Interactive)

For automated testing or CI/CD pipelines, you can run the script in non-interactive mode using command-line arguments or environment variables:

#### Using Command-Line Arguments

```bash
bash trainer.sh \
  --wake-phrase "hey assistant" \
  --train-profile tiny \
  --train-threads 2 \
  --non-interactive
```

#### Using Environment Variables

```bash
WAKE_PHRASE="hey assistant" \
TRAIN_PROFILE="tiny" \
TRAIN_THREADS="2" \
bash trainer.sh --non-interactive
```

#### Running Without tmux

By default, training runs in a tmux session. To run training directly in the current terminal (useful for testing or CI):

```bash
bash trainer.sh \
  --wake-phrase "hey assistant" \
  --train-profile tiny \
  --train-threads 2 \
  --non-interactive \
  --no-tmux
```

### Available Options

Run `bash trainer.sh --help` to see all available options.

Key options for CLI mode:
- `--wake-phrase TEXT` - The wake phrase to train (e.g., "hey assistant")
- `--train-profile NAME` - Training profile: tiny, medium, or large
- `--train-threads NUMBER` - Number of CPU threads to use
- `--non-interactive` - Skip all interactive prompts; use defaults or provided values
- `--no-tmux` - Run training in current terminal instead of tmux session
