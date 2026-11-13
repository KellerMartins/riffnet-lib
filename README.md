# RiffNet

Real-time electric guitar and bass note and chord detection for the browser and mobile. Powered by a small CNN, with DSP, decoding, and inference running entirely in Rust/WASM (no JS-side ML runtime needed).

Includes the core library, a web demo, and the PyTorch model training pipeline.

**[Try the live demo](https://kellermartins.github.io/riffnet-lib/)**: open it on your phone or computer, allow microphone access, and play.

## Features

- **All-in-One WASM:** The CQT front-end, CNN inference, and decoding happen in a single WASM module. Zero JS-side ML dependencies and no separate model-loading step.
- **CQT Front-end:** Constant-Q transform covering MIDI 36–90 (guitar) and 23–65 (bass, down to a 5-string's low B).
- **Fast Inference:** A ~200-250K parameter CNN (depending on instrument) evaluates every audio hop (~5.8ms), typically converging on a confident read within ~35ms of a note onset.
- **Instant Instrument Switching:** Swap between guitar and bass models on the fly. Weights are embedded directly in the WASM binary, so switching has zero loading overhead.
- **Play-along Verification:** Pass an array of expected notes to check against live input for rhythm game or tablature apps.

## API Usage

```js
import init, { RiffNetDetector, Instrument } from "./pkg/riffnet.js";

await init();

// Initialize the detector
const det = new RiffNetDetector(sampleRate); // Defaults to Instrument.Guitar
det.instrument = Instrument.Bass; // Swap models on the fly

// Feed audio and poll for results
det.process_audio(samples);
const res = det.poll();

// `res` is null if no new hop is ready.
// Otherwise, returns { onset, notes, chord_name }
if (res) {
  console.log(`Detected: ${res.chord_name} | Notes: ${res.notes}`);
}

// Optional Configuration
det.onset_threshold = 0.6; // Onset-fire probability threshold
det.pitch_threshold = 0.5; // Pitch posterior threshold
det.refractory = 6; // Min frames between onsets
det.max_notes = 6; // Polyphony cap

// Verify against expected notes (returns Float32Array of confidences [0, 1])
const confidences = det.verify_expected(new Int32Array([36, 40, 47, 52]));

// Benchmarking
console.log(det.cqt_ms); // Time spent on CQT extraction for the last hop
```

## Run the Demo

The [live demo](https://kellermartins.github.io/riffnet-lib/) is this same `web/` folder, deployed to GitHub Pages on every push. To run it locally, no build is needed since the compiled artifacts are committed:

```bash
python web/serve.py

```

Open the localhost URL in your browser, grant microphone access, and play. The demo (`web/index.html`) includes live detection, WASM latency metrics, and raw mic recording.

## Project Layout

| Path                  | Description                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------ |
| `riffnet_dsp/`        | Rust DSP front-end: 4th-order LPF, constant-Q transform, and chord templates.                    |
| `riffnet/`            | Rust/WASM bindings and streaming CNN engine. Guitar and bass share the same core inference code. |
| `web/`                | Browser demo and pre-compiled WASM artifacts (`pkg/`).                                           |
| `model/`              | Python training pipeline, PyTorch models, and deployed ONNX weights.                             |
| `build.py`            | Build script for regenerating WASM and exporting models.                                         |

## Building from Source

If you modify the Rust code or retrain a model, you will need to regenerate the artifacts.

```bash
# Rebuild the WASM into web/pkg
python build.py

# Export a new PyTorch checkpoint to ONNX, then rebuild WASM
python build.py --ckpt path/to/model.pt

```

_Note: `build.py --ckpt` handles the guitar model by default. To retrain and re-export the bass model, run `model/scripts/export.py --instrument bass` directly, then run `python build.py` to compile the changes into WASM._

## License

[AGPL-3.0-or-later](LICENSE).
