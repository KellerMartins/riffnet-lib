//! Self-contained RiffNet inference. The forward pass runs through the
//! hand-rolled streaming engine (`streaming.rs`, ~9x faster than replaying
//! the full 17-frame window through burn-flex every hop); burn is used only
//! at construction time to load the `.bpk` weights that `build.rs` compiled
//! out of `model/weights/{riffnet,bass_riffnet}.onnx` (the generated `Model`
//! + burn-flex forward is kept for the parity tests below). Reuses
//! `ModelDetector`'s CQT front-end and `decode_logits` unchanged — only the
//! source of onset/pitch logits changes.
//!
//! Generic over `BINS`/`P1`/`P2`/`NPITCH` (const generics) so guitar and
//! bass share this exact wrapper with zero runtime cost -- see
//! `streaming.rs`'s module doc for the reasoning. The one piece that can't
//! be generic is burn-onnx's codegen output itself (`generated_guitar`/
//! `generated_bass` below): it's a distinct Rust type per ONNX graph,
//! produced by `build.rs`'s `ModelGen`, so loading it is a small
//! per-instrument shim rather than hand-written engine logic.
use crate::model::ModelDetector;
use crate::streaming::{StreamingRiffNet, WeightMap};
use burn::tensor::Bytes;
use burn_flex::Flex;
use burn_store::ModuleSnapshot;
use riffnet_dsp::dsp::{BASS_FREQ_BASE, BASS_NUM_SEMITONES, GUITAR_FREQ_BASE, GUITAR_NUM_SEMITONES};
use riffnet_dsp::DetectionResult;

// `Model::forward` is only exercised by the parity tests; production inference
// goes through `StreamingRiffNet` (LTO strips the unused burn kernels from the
// wasm build).
#[allow(dead_code)]
mod generated_guitar {
    include!(concat!(env!("OUT_DIR"), "/model/riffnet.rs"));
}
#[allow(dead_code)]
mod generated_bass {
    include!(concat!(env!("OUT_DIR"), "/model/bass/bass_riffnet.rs"));
}

type B = Flex;

static GUITAR_WEIGHTS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/model/riffnet.bpk"));
static BASS_WEIGHTS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/model/bass/bass_riffnet.bpk"));

/// Load a burn-onnx `.bpk` through its generated module (the only public
/// burn-store path that can enumerate its tensors) and flatten it to plain
/// name -> (shape, f32) pairs for the streaming engine. The `Model` is
/// dropped afterwards — burn tensors never run at inference time.
fn guitar_weight_map() -> WeightMap {
    let device: <B as burn::tensor::backend::BackendTypes>::Device = Default::default();
    let model: generated_guitar::Model<B> =
        generated_guitar::Model::from_bytes(Bytes::from_bytes_vec(GUITAR_WEIGHTS.to_vec()), &device);
    let mut map = WeightMap::new();
    for snapshot in model.collect(None, None, false) {
        let data = snapshot.to_data().expect("materialize guitar weight tensor");
        let shape = data.shape.to_vec();
        let values: Vec<f32> = data.to_vec().expect("guitar weights must be f32");
        map.insert(snapshot.full_path(), (shape, values));
    }
    map
}

fn bass_weight_map() -> WeightMap {
    let device: <B as burn::tensor::backend::BackendTypes>::Device = Default::default();
    let model: generated_bass::Model<B> =
        generated_bass::Model::from_bytes(Bytes::from_bytes_vec(BASS_WEIGHTS.to_vec()), &device);
    let mut map = WeightMap::new();
    for snapshot in model.collect(None, None, false) {
        let data = snapshot.to_data().expect("materialize bass weight tensor");
        let shape = data.shape.to_vec();
        let values: Vec<f32> = data.to_vec().expect("bass weights must be f32");
        map.insert(snapshot.full_path(), (shape, values));
    }
    map
}

pub struct BurnRiffNet<const BINS: usize, const P1: usize, const P2: usize, const NPITCH: usize> {
    inner: ModelDetector<BINS, NPITCH>,
    engine: StreamingRiffNet<BINS, P1, P2, NPITCH>,
    last_result: Option<DetectionResult>,
}

impl<const BINS: usize, const P1: usize, const P2: usize, const NPITCH: usize>
    BurnRiffNet<BINS, P1, P2, NPITCH>
{
    pub fn new(
        sample_rate: f32,
        weights: &WeightMap,
        freq_base: f32,
        num_semitones: usize,
        midi_low: i32,
    ) -> Self {
        let engine = StreamingRiffNet::from_weights(weights).expect("RiffNet weights");
        Self {
            inner: ModelDetector::new(sample_rate, freq_base, num_semitones, midi_low),
            engine,
            last_result: None,
        }
    }

    pub fn process_block(&mut self, samples: &[f32]) {
        // Per-sample so every hop gets its own frame pushed into the streaming
        // engine (its time alignment relies on exactly one push per hop; the
        // old full-window path silently skipped frames on multi-hop blocks).
        for s in samples {
            self.inner.hop_occurred = false;
            self.inner.process_block(std::slice::from_ref(s));
            if self.inner.hop_occurred {
                let (onset_logit, pitch_logits) = {
                    let row = self.inner.extract_cqt_row();
                    self.engine.push_frame(row)
                };
                self.last_result = Some(self.inner.decode_logits(onset_logit, &pitch_logits));
            }
        }
    }

    /// Drain the latest decode result (populated every hop — this is a
    /// continuously-running per-frame estimator).
    pub fn poll(&mut self) -> Option<DetectionResult> {
        self.last_result.take()
    }

    pub fn verify_expected(&self, expected_midi: &[i32]) -> Vec<f32> {
        self.inner.verify_expected(expected_midi)
    }

    /// Profiling: time of the last hop's CQT extraction (ms, front-end only,
    /// excludes the model forward pass).
    pub fn cqt_ms(&self) -> f32 {
        self.inner.last_cqt_ms
    }
}

/// Guitar: MIDI 36-90, 55 pitch classes, 109 CQT bins.
pub type GuitarBurnRiffNet = BurnRiffNet<109, 36, 12, 55>;
/// Bass: MIDI 23-65, 43 pitch classes, 85 CQT bins (5-string low-B floor).
pub type BassBurnRiffNet = BurnRiffNet<85, 28, 9, 43>;

pub fn new_guitar(sample_rate: f32) -> GuitarBurnRiffNet {
    GuitarBurnRiffNet::new(sample_rate, &guitar_weight_map(), GUITAR_FREQ_BASE, GUITAR_NUM_SEMITONES, 36)
}

pub fn new_bass(sample_rate: f32) -> BassBurnRiffNet {
    BassBurnRiffNet::new(sample_rate, &bass_weight_map(), BASS_FREQ_BASE, BASS_NUM_SEMITONES, 23)
}

/// Uniform interface over both instruments' `BurnRiffNet<...>` instantiations
/// (distinct concrete types, since their const-generic shapes differ), so
/// `RiffNetDetector` can hold either behind one `Box<dyn Backend>` and swap
/// on an `instrument` change. Blanket-implemented below for every
/// `BurnRiffNet<BINS, P1, P2, NPITCH>`.
pub trait Backend {
    fn process_block(&mut self, samples: &[f32]);
    fn poll(&mut self) -> Option<DetectionResult>;
    fn verify_expected(&self, expected_midi: &[i32]) -> Vec<f32>;
    fn cqt_ms(&self) -> f32;
    fn pitch_threshold(&self) -> f32;
    fn set_pitch_threshold(&mut self, v: f32);
    fn onset_threshold(&self) -> f32;
    fn set_onset_threshold(&mut self, v: f32);
    fn refractory(&self) -> usize;
    fn set_refractory(&mut self, v: usize);
    fn max_notes(&self) -> usize;
    fn set_max_notes(&mut self, v: usize);
}

impl<const BINS: usize, const P1: usize, const P2: usize, const NPITCH: usize> Backend
    for BurnRiffNet<BINS, P1, P2, NPITCH>
{
    fn process_block(&mut self, samples: &[f32]) {
        BurnRiffNet::process_block(self, samples)
    }
    fn poll(&mut self) -> Option<DetectionResult> {
        BurnRiffNet::poll(self)
    }
    fn verify_expected(&self, expected_midi: &[i32]) -> Vec<f32> {
        BurnRiffNet::verify_expected(self, expected_midi)
    }
    fn cqt_ms(&self) -> f32 {
        BurnRiffNet::cqt_ms(self)
    }
    fn pitch_threshold(&self) -> f32 {
        self.inner.th_pi
    }
    fn set_pitch_threshold(&mut self, v: f32) {
        self.inner.th_pi = v;
    }
    fn onset_threshold(&self) -> f32 {
        self.inner.th_on
    }
    fn set_onset_threshold(&mut self, v: f32) {
        self.inner.th_on = v;
    }
    fn refractory(&self) -> usize {
        self.inner.refractory
    }
    fn set_refractory(&mut self, v: usize) {
        self.inner.refractory = v;
    }
    fn max_notes(&self) -> usize {
        self.inner.max_notes
    }
    fn set_max_notes(&mut self, v: usize) {
        self.inner.max_notes = v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::prelude::*;

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn lcg_input(seed_start: u64, len: usize) -> Vec<f32> {
        let mut seed = seed_start;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / u32::MAX as f32) * 0.03
        };
        (0..len).map(|_| next()).collect()
    }

    /// Push a window's frames ([c][t][b] layout) into a fresh engine, one
    /// `push_frame` at a time.
    fn stream_window<const BINS: usize, const P1: usize, const P2: usize, const NPITCH: usize>(
        engine: &mut StreamingRiffNet<BINS, P1, P2, NPITCH>,
        window: &[f32],
        n_chan: usize,
        win: usize,
    ) -> (f32, [f32; NPITCH]) {
        let mut out = (0.0, [0f32; NPITCH]);
        let mut frame = vec![0f32; n_chan * BINS];
        for t in 0..win {
            for c in 0..n_chan {
                frame[c * BINS..(c + 1) * BINS]
                    .copy_from_slice(&window[(c * win + t) * BINS..(c * win + t) * BINS + BINS]);
            }
            out = engine.push_frame(&frame);
        }
        out
    }

    mod guitar {
        use super::*;
        use crate::streaming::GuitarStreamingRiffNet;

        const N_CHAN: usize = 6;
        const WIN: usize = 17;
        const BINS: usize = 109;
        const NPITCH: usize = 55;

        /// Reference values from the burn-generated graph on the deployed
        /// guitar checkpoint's weights (109 bins x 55 pitch classes), using
        /// `lcg_input`'s LCG-seeded input. If the checkpoint or the model
        /// shape changes, regenerate these from the new weights — they must
        /// always match the currently deployed model.
        const REF_ONSET: f32 = 0.14580609;
        const REF_PITCH5: [f32; 5] = [0.030158741, 0.015911676, 0.003806027, 0.012532847, 0.012561091];

        fn burn_forward(window: &[f32]) -> (f32, Vec<f32>) {
            let device = Default::default();
            let model: generated_guitar::Model<B> =
                generated_guitar::Model::from_bytes(Bytes::from_bytes_vec(GUITAR_WEIGHTS.to_vec()), &device);
            let input = Tensor::<B, 1>::from_floats(window, &device).reshape([1, N_CHAN, WIN, BINS]);
            let (onset, pitch) = model.forward(input);
            let onset_data: Vec<f32> = onset.into_data().to_vec().unwrap();
            let pitch_data: Vec<f32> = pitch.into_data().to_vec().unwrap();
            (*onset_data.last().unwrap(), pitch_data[pitch_data.len() - NPITCH..].to_vec())
        }

        #[test]
        fn burn_forward_matches_onnxruntime_reference() {
            let (onset_logit, pitch_logits) = burn_forward(&lcg_input(42, N_CHAN * WIN * BINS));
            let onset_sig = sigmoid(onset_logit);
            assert!(
                (onset_sig - REF_ONSET).abs() < 1e-5,
                "onset: got {onset_sig:.8} expected {REF_ONSET:.8}"
            );
            for (i, (&got, &exp)) in pitch_logits[..5].iter().zip(REF_PITCH5.iter()).enumerate() {
                let got = sigmoid(got);
                let diff = (got - exp).abs();
                assert!(diff < 1e-5, "pitch[{i}]: got {got:.8} expected {exp:.8} diff {diff:.2e}");
            }
        }

        #[test]
        fn streaming_matches_onnxruntime_reference() {
            let mut engine = GuitarStreamingRiffNet::from_weights(&guitar_weight_map()).unwrap();
            let (onset_logit, pitch_logits) =
                stream_window(&mut engine, &lcg_input(42, N_CHAN * WIN * BINS), N_CHAN, WIN);
            let onset_sig = sigmoid(onset_logit);
            assert!(
                (onset_sig - REF_ONSET).abs() < 1e-5,
                "onset: got {onset_sig:.8} expected {REF_ONSET:.8}"
            );
            for (i, (&got, &exp)) in pitch_logits[..5].iter().zip(REF_PITCH5.iter()).enumerate() {
                let got = sigmoid(got);
                let diff = (got - exp).abs();
                assert!(diff < 1e-5, "pitch[{i}]: got {got:.8} expected {exp:.8} diff {diff:.2e}");
            }
        }

        /// The incremental path: after the initial fill, each additional frame
        /// must reproduce the full-window burn forward on the shifted window —
        /// this exercises the ring-buffer reuse, not just a cold fill.
        #[test]
        fn streaming_matches_burn_forward_incrementally() {
            let frames: Vec<Vec<f32>> = (0..20)
                .map(|i| lcg_input(7 + i * 1000, N_CHAN * BINS))
                .collect();
            let window_at = |end: usize| -> Vec<f32> {
                let mut w = vec![0f32; N_CHAN * WIN * BINS];
                for c in 0..N_CHAN {
                    for t in 0..WIN {
                        w[(c * WIN + t) * BINS..(c * WIN + t) * BINS + BINS]
                            .copy_from_slice(&frames[end - (WIN - 1) + t][c * BINS..(c + 1) * BINS]);
                    }
                }
                w
            };

            let mut engine = GuitarStreamingRiffNet::from_weights(&guitar_weight_map()).unwrap();
            for (i, f) in frames.iter().enumerate() {
                let got = engine.push_frame(f);
                if i >= WIN - 1 {
                    let (ref_onset, ref_pitch) = burn_forward(&window_at(i));
                    let d_on = (got.0 - ref_onset).abs();
                    let d_pi = got.1.iter().zip(ref_pitch.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                    assert!(
                        d_on < 1e-4 && d_pi < 1e-4,
                        "window ending at frame {i}: onset logit diff {d_on:.2e}, max pitch logit diff {d_pi:.2e}"
                    );
                }
            }
        }
    }

    mod bass {
        use super::*;
        use crate::streaming::BassStreamingRiffNet;

        const N_CHAN: usize = 6;
        const WIN: usize = 17;
        const BINS: usize = 85;
        const NPITCH: usize = 43;

        /// Reference values from the burn-generated graph on the deployed
        /// bass checkpoint's weights (85 bins x 43 pitch classes), using
        /// `lcg_input`'s LCG-seeded input. If the checkpoint or the model
        /// shape changes, regenerate these from the new weights — they must
        /// always match the currently deployed model.
        const REF_ONSET: f32 = 0.18909799;
        const REF_PITCH5: [f32; 5] =
            [0.06856193, 0.06780282, 0.04325456, 0.0536419, 0.020317571];

        fn burn_forward(window: &[f32]) -> (f32, Vec<f32>) {
            let device = Default::default();
            let model: generated_bass::Model<B> =
                generated_bass::Model::from_bytes(Bytes::from_bytes_vec(BASS_WEIGHTS.to_vec()), &device);
            let input = Tensor::<B, 1>::from_floats(window, &device).reshape([1, N_CHAN, WIN, BINS]);
            let (onset, pitch) = model.forward(input);
            let onset_data: Vec<f32> = onset.into_data().to_vec().unwrap();
            let pitch_data: Vec<f32> = pitch.into_data().to_vec().unwrap();
            (*onset_data.last().unwrap(), pitch_data[pitch_data.len() - NPITCH..].to_vec())
        }

        #[test]
        fn burn_forward_matches_onnxruntime_reference() {
            let (onset_logit, pitch_logits) = burn_forward(&lcg_input(42, N_CHAN * WIN * BINS));
            let onset_sig = sigmoid(onset_logit);
            assert!(
                (onset_sig - REF_ONSET).abs() < 1e-5,
                "onset: got {onset_sig:.8} expected {REF_ONSET:.8}"
            );
            for (i, (&got, &exp)) in pitch_logits[..5].iter().zip(REF_PITCH5.iter()).enumerate() {
                let got = sigmoid(got);
                let diff = (got - exp).abs();
                assert!(diff < 1e-5, "pitch[{i}]: got {got:.8} expected {exp:.8} diff {diff:.2e}");
            }
        }

        #[test]
        fn streaming_matches_onnxruntime_reference() {
            let mut engine = BassStreamingRiffNet::from_weights(&bass_weight_map()).unwrap();
            let (onset_logit, pitch_logits) =
                stream_window(&mut engine, &lcg_input(42, N_CHAN * WIN * BINS), N_CHAN, WIN);
            let onset_sig = sigmoid(onset_logit);
            assert!(
                (onset_sig - REF_ONSET).abs() < 1e-5,
                "onset: got {onset_sig:.8} expected {REF_ONSET:.8}"
            );
            for (i, (&got, &exp)) in pitch_logits[..5].iter().zip(REF_PITCH5.iter()).enumerate() {
                let got = sigmoid(got);
                let diff = (got - exp).abs();
                assert!(diff < 1e-5, "pitch[{i}]: got {got:.8} expected {exp:.8} diff {diff:.2e}");
            }
        }

        #[test]
        fn streaming_matches_burn_forward_incrementally() {
            let frames: Vec<Vec<f32>> = (0..20)
                .map(|i| lcg_input(7 + i * 1000, N_CHAN * BINS))
                .collect();
            let window_at = |end: usize| -> Vec<f32> {
                let mut w = vec![0f32; N_CHAN * WIN * BINS];
                for c in 0..N_CHAN {
                    for t in 0..WIN {
                        w[(c * WIN + t) * BINS..(c * WIN + t) * BINS + BINS]
                            .copy_from_slice(&frames[end - (WIN - 1) + t][c * BINS..(c + 1) * BINS]);
                    }
                }
                w
            };

            let mut engine = BassStreamingRiffNet::from_weights(&bass_weight_map()).unwrap();
            for (i, f) in frames.iter().enumerate() {
                let got = engine.push_frame(f);
                if i >= WIN - 1 {
                    let (ref_onset, ref_pitch) = burn_forward(&window_at(i));
                    let d_on = (got.0 - ref_onset).abs();
                    let d_pi = got.1.iter().zip(ref_pitch.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                    assert!(
                        d_on < 1e-4 && d_pi < 1e-4,
                        "window ending at frame {i}: onset logit diff {d_on:.2e}, max pitch logit diff {d_pi:.2e}"
                    );
                }
            }
        }
    }
}
