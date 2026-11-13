mod burn_model;
mod model;
mod resample;
mod streaming;

use riffnet_dsp::DetectedNote;
use serde::Serialize;
use wasm_bindgen::prelude::*;

use burn_model::Backend;

// ----- Shared result shape: both detectors return this from `poll()` -----

#[derive(Serialize)]
pub struct JsDetectedNote {
    pub midi_note: f32,
    pub frequency: f32,
    pub amplitude: f32,
}

impl From<&DetectedNote> for JsDetectedNote {
    fn from(n: &DetectedNote) -> Self {
        Self { midi_note: n.midi_note, frequency: n.frequency, amplitude: n.amplitude }
    }
}

#[derive(Serialize)]
pub struct JsPollResult {
    /// True on a fresh onset, re-evaluated (and can re-fire) every hop, since
    /// `RiffNetDetector` is a continuously-running per-frame estimator.
    pub onset: bool,
    pub notes: Vec<JsDetectedNote>,
    pub chord_name: Option<String>,
}

// ----- RiffNetDetector: full CNN, self-contained (burn-flex, no ONNX runtime) -----

/// Which trained model `RiffNetDetector` runs. Guitar (MIDI 36-90) and bass
/// (MIDI 23-65, a 5-string's low B upward) are separately trained models,
/// not a shared backbone, so switching rebuilds the detector — cheap, since
/// the weights are already embedded in the wasm binary (no I/O, just
/// re-reading static bytes).
#[wasm_bindgen]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Instrument {
    Guitar,
    Bass,
}

fn build_backend(instrument: Instrument, sample_rate: f32) -> Box<dyn Backend> {
    match instrument {
        Instrument::Guitar => Box::new(burn_model::new_guitar(sample_rate)),
        Instrument::Bass => Box::new(burn_model::new_bass(sample_rate)),
    }
}

/// Real-time note detector — the full learned model (a ~200-250K-param CNN,
/// depending on instrument), running entirely in Rust/WASM via a hand-rolled
/// streaming forward pass
/// (see `streaming.rs`). No `onnxruntime` dependency, no JS-driven inference
/// call: `process_audio` does everything internally, `poll()` reads the
/// result. Continuously re-evaluates every hop (~5.8ms), so pitch estimates
/// are always current, converging to a confident read ~35ms after a note
/// starts (rolling mean over recent frames).
#[wasm_bindgen]
pub struct RiffNetDetector {
    inner: Box<dyn Backend>,
    sample_rate: f32,
    instrument: Instrument,
}

#[wasm_bindgen]
impl RiffNetDetector {
    /// * `sample_rate` — microphone sample rate (e.g. 44100.0)
    ///
    /// Defaults to `Instrument::Guitar`; set `.instrument` to switch.
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: f32) -> Self {
        Self {
            inner: build_backend(Instrument::Guitar, sample_rate),
            sample_rate,
            instrument: Instrument::Guitar,
        }
    }

    /// Feed a `Float32Array` of audio samples. Call `poll()` afterward.
    pub fn process_audio(&mut self, samples: &[f32]) {
        self.inner.process_block(samples);
    }

    /// Returns `{onset, notes, chord_name}` for the most recently completed
    /// hop, or `null` if no hop has landed since the last poll — callers must
    /// keep their previous display state on `null`. This distinction matters:
    /// an object with empty `notes` is a real per-hop verdict ("nothing
    /// playing", blank the display), while `null` just means "no new data
    /// this block" (hops are 256 canonical samples but audio arrives in
    /// 128-sample worklet blocks, so roughly every other poll has no hop —
    /// treating those as empty verdicts made the display blink).
    pub fn poll(&mut self) -> JsValue {
        match self.inner.poll() {
            Some(r) => {
                let js = JsPollResult {
                    onset: r.onset_detected,
                    notes: r.peaks.iter().map(JsDetectedNote::from).collect(),
                    chord_name: r.chord_name,
                };
                serde_wasm_bindgen::to_value(&js).unwrap_or(JsValue::NULL)
            }
            None => JsValue::NULL,
        }
    }

    /// Play-along verification (known-song mode). Pass the chart's expected
    /// MIDI notes for the current beat; returns each note's model confidence
    /// in [0,1] (rolling mean over the last ~93 ms of pitch posteriors).
    /// Treat confidence >= ~0.30 as a hit; tune ~0.25–0.40 strict↔lenient.
    ///
    /// JS: `const conf = det.verify_expected(new Int32Array([36, 40, 47, 52]));`
    pub fn verify_expected(&self, expected_midi: &[i32]) -> Vec<f32> {
        self.inner.verify_expected(expected_midi)
    }

    /// Which model is active. Setting this rebuilds the detector for the new
    /// instrument (thresholds/refractory/max_notes reset to that
    /// instrument's defaults; sample rate carries over).
    #[wasm_bindgen(getter)]
    pub fn instrument(&self) -> Instrument {
        self.instrument
    }
    #[wasm_bindgen(setter)]
    pub fn set_instrument(&mut self, v: Instrument) {
        if v != self.instrument {
            self.instrument = v;
            self.inner = build_backend(v, self.sample_rate);
        }
    }

    /// Pitch posterior threshold to report a note. Default 0.50.
    #[wasm_bindgen(getter)]
    pub fn pitch_threshold(&self) -> f32 {
        self.inner.pitch_threshold()
    }
    #[wasm_bindgen(setter)]
    pub fn set_pitch_threshold(&mut self, v: f32) {
        self.inner.set_pitch_threshold(v.clamp(0.05, 0.95));
    }

    /// Onset probability threshold to fire an onset. Default 0.60.
    #[wasm_bindgen(getter)]
    pub fn onset_threshold(&self) -> f32 {
        self.inner.onset_threshold()
    }
    #[wasm_bindgen(setter)]
    pub fn set_onset_threshold(&mut self, v: f32) {
        self.inner.set_onset_threshold(v.clamp(0.05, 0.95));
    }

    /// Minimum frames between onsets (debounce). Default 6.
    #[wasm_bindgen(getter)]
    pub fn refractory(&self) -> usize {
        self.inner.refractory()
    }
    #[wasm_bindgen(setter)]
    pub fn set_refractory(&mut self, v: usize) {
        self.inner.set_refractory(v.max(1));
    }

    /// Max simultaneous notes reported per frame. Default 6.
    #[wasm_bindgen(getter)]
    pub fn max_notes(&self) -> usize {
        self.inner.max_notes()
    }
    #[wasm_bindgen(setter)]
    pub fn set_max_notes(&mut self, v: usize) {
        self.inner.set_max_notes(v.clamp(1, 12));
    }

    /// Profiling: time of the last hop's CQT extraction (ms; front-end only,
    /// excludes the model forward pass).
    #[wasm_bindgen(getter)]
    pub fn cqt_ms(&self) -> f32 {
        self.inner.cqt_ms()
    }
}
