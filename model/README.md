# RiffNet Training Pipeline

This directory contains the PyTorch training pipeline for the RiffNet models. We train separate models for guitar and bass, export them to ONNX, and embed them into the WASM binary at compile time.

Uses `uv` for dependency management (`uv sync`). Pre-trained checkpoints and ONNX exports are committed to `weights/` for reproducibility.

- **Guitar** (`riffnet.pt` / `riffnet.onnx`): MIDI 36–90, 55 pitch classes. Covers everything from Drop C to standard high range.
- **Bass** (`bass_riffnet.pt` / `bass_riffnet.onnx`): MIDI 23–65, 43 pitch classes.
  The MIDI 23 floor is a 5-string bass's open low B; 4-string basses simply
  never play the bottom five classes.

---

## Datasets

Datasets are not included in this repo. Bring your own copies and place them as sibling directories to `model/`.

**Guitar**

- **EGDB** (`../EGDB/`): 240 phrases across multiple amp timbres with MIDI labels. Download link on the [project page](https://ss12f32v.github.io/Guitar-Transcription) (paper: [arXiv 2202.09907](https://arxiv.org/abs/2202.09907)).
- **GOAT** ([Zenodo 15690894](https://zenodo.org/records/15690894)): Large dataset of polyphonic guitar chords.
- _Note:_ Because the raw datasets rarely drop below MIDI 40, `build_dataset.py` derives two pitch-shifted sources (`EGDB_Shifted` and `GOAT_Shifted`) down to MIDI 36 using only the training DI splits. These never leak into validation or test sets.

**Bass**

- **IDMT-SMT-Bass** (`../IDMTSMTBass/extracted`): Isolated bass notes and phrases across different playing techniques. [Zenodo 7188892](https://zenodo.org/records/7188892); unzip `IDMT-SMT-BASS.zip` into `extracted/`.
- **SlakhBass** (`../SlakhBass/`): Bass stems from the Slakh2100 dataset ([Zenodo 4599666](https://zenodo.org/records/4599666), the `slakh2100_flac_redux` release). The full download is ~105 GB but only the bass stems are used: for each track, take the first stem whose `metadata.yaml` entry has `inst_class: Bass` and copy its `stems/<id>.flac` and `MIDI/<id>.mid` to `SlakhBass/<split>/<Track>/bass.flac` and `bass.mid` (`<split>` is `train`, `validation` or `test`; skip `omitted`).
- _Note:_ Neither corpus has any audio below MIDI 28 (a 4-string's open low E) — IDMT floors there and SlakhBass at 47 — so the 5-string low-B register, MIDI 23–27, is derived by `build_dataset.py` as `IDMTBassShifted`, real IDMT notes shifted down 1–7 semitones. It is train-only and excludes the held-out `BS_3` instrument. Sources are capped at MIDI 30 because low bass notes lose their fundamental (measured ratio 0.22 at MIDI 28 vs 2.06 at 37) and a pitch shifter preserves whatever it is given; shifting from higher notes would synthesize low notes with a fundamental no real bass produces. **This means MIDI 23–27 has no held-out real audio of any kind** — nothing in this split can validate it; only recordings from a real 5-string can, and none have been made yet.

---

## Pipeline Overview

Always run scripts as modules from the `model/` root (e.g., `uv run python -m scripts.train`) so the internal imports resolve correctly.

### Core Modules

- `cqt.py`: A PyTorch bit-parity replica of the Rust CQT front-end.
- `dataset.py`: Train/test splits, quarantine lists for bad clips, and the data loader (handles random crops, pitch-shift, gain, and soft-onset augmentations).
- `model.py`: The core CNN architecture (depthwise-separable + dilated convs). Guitar and bass share the identical structure and differ only in bin/pitch-class count.
- `decode.py`: Shared decoding and scoring logic used for evaluation.

### CLI Scripts (`scripts/`)

- `build_dataset.py`: Converts raw audio datasets into a CQT cache (`.npz` files).
- `train.py`: Trains the model using focal loss and a harmonic octave-loss penalty.
- `eval.py`: Provides a deep breakdown of a checkpoint's performance by timbre, range, polyphony, and timing.
- `export.py`: Converts a `.pt` checkpoint to an ONNX graph with a dynamic time axis.

---

## Key Invariants (Do Not Change)

- **Hop Size = 256:** Must match across dataset building, training, and the Rust inference engine. The dilated convolutions rely on this exact hop timing.
- **TIME_CONTEXT = 16:** The model requires 16 frames of left context. The input shape is `[1, 6, 17, num_bins]`. The caller handles left-padding.
- **FEAT_SCALE = 256:** The scaling function (`log1p(mag*256)`) is baked directly into the ONNX graph. You must feed raw CQT magnitude from the DSP layer.

---

## Usage Examples

### Guitar

```bash
# 1. Build the CQT cache (if missing)
uv run python -m scripts.build_dataset --dataset all

# 2. Train the model
uv run python -m scripts.train --epochs 30 --out checkpoints/riffnet.pt

# 3. Evaluate (optional)
uv run python -m scripts.eval --ckpt checkpoints/riffnet.pt

# 4. Export to ONNX
uv run python -m scripts.export --ckpt checkpoints/riffnet.pt --onnx weights/riffnet.onnx

# 5. Rebuild WASM to embed the new weights (run from repo root)
python build.py --ckpt model/checkpoints/riffnet.pt

```

### Bass

```bash
# 1. Build the bass CQT cache
uv run python -m scripts.build_dataset --dataset bass

# 2. Train the bass model
uv run python -m scripts.train --instrument bass --data data/hop256_bass --epochs 30 --out checkpoints/bass_riffnet.pt

# 3. Export to ONNX
uv run python -m scripts.export --instrument bass --ckpt checkpoints/bass_riffnet.pt --onnx weights/bass_riffnet.onnx

# 4. Rebuild WASM (run from repo root)
python build.py

```

_Note: If you only need to reproduce the ONNX file from a committed checkpoint, run `uv run python -m scripts.export` (add `--instrument bass` for the bass model)._
