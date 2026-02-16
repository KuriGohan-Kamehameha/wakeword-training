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

## Testing

To test the training script without actually running a full training session, you can use the `--help` flag to verify the CLI arguments are recognized:

```bash
bash trainer.sh --help
```

For quick testing in non-interactive mode:

```bash
# Test with minimal parameters (will use defaults for everything else)
bash trainer.sh \
  --wake-phrase "test phrase" \
  --train-profile tiny \
  --train-threads 1 \
  --non-interactive \
  --no-tmux
```

**Note**: The script automatically detects when it's not running in a TTY (e.g., in CI/CD pipelines) and will use default values for any parameters not explicitly provided, even without the `--non-interactive` flag.
