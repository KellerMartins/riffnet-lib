use crate::resample::Resampler;
use riffnet_dsp::{DetectedNote, DetectionResult, CqtFrontEnd};
use std::collections::VecDeque;

const WIN: usize = 17;
const CANON_SR: f32 = 44100.0;
const HOP: usize = 256; // must match the training hop (build_dataset.py hop=256)
const PITCH_WIN: usize = 16; // rolling pitch-mean window (tuned on val/test)
// HCQT: harmonics 0.5x/1x/2x/3x/4x -> log-freq bin offsets round(24*log2(h)),
// plus a 6th flux channel: relu(cqt[t] - cqt[t-1]) per bin — makes the pick
// attack explicit where distortion compresses magnitude contrast.
const N_CHAN: usize = 6;
const HCQT_OFFSETS: [i32; 5] = [-24, 0, 24, 38, 48];
// Harmonic-ghost intervals to suppress (3rd/4th/5th/6th harmonics); octave
// (+12) excluded — it's often a real chord tone (validated on GOAT polyphony).
const HARMONIC_INTERVALS: [usize; 4] = [19, 24, 28, 31];
const SIEVE_RATIO: f32 = 1.3;

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(target_arch = "wasm32")]
fn now_ms() -> f64 {
    web_sys::window().and_then(|w| w.performance()).map_or(0.0, |p| p.now())
}

#[cfg(not(target_arch = "wasm32"))]
fn now_ms() -> f64 {
    0.0 // profiling getter is wasm-only; native tests don't need wall time
}

#[inline]
fn midi_to_freq(midi: f32) -> f32 {
    440.0 * 2.0f32.powf((midi - 69.0) / 12.0)
}

/// Generic over `BINS`/`NPITCH` (const generics) so guitar and bass -- two
/// separately-trained models with different CQT ranges/pitch vocabularies --
/// share this exact featurizer/decoder with zero runtime cost (see
/// `streaming.rs`'s module doc for the same reasoning). `midi_low` is a
/// regular constructor arg, not a const generic: it's only ever used in
/// scalar arithmetic (not array sizing), so genericizing it would buy
/// nothing.
pub struct ModelDetector<const BINS: usize, const NPITCH: usize> {
    nd: CqtFrontEnd,
    resampler: Resampler,
    // Window rings feed only `extract_cqt_window`, the reference full-window
    // featurizer; production uses `extract_cqt_row` (streaming) instead.
    #[allow(dead_code)]
    ring: VecDeque<Vec<f32>>,
    #[allow(dead_code)]
    flux_ring: VecDeque<Vec<f32>>,
    prev_cqt: Vec<f32>,
    sample_count: usize,
    pub hop_occurred: bool,

    midi_low: i32,
    prev_onset: f32,
    prev_prev_onset: f32,
    frames_since_onset: usize,
    pitch_hist: VecDeque<Vec<f32>>,

    pub th_on: f32,
    pub th_pi: f32,
    pub refractory: usize,
    pub max_notes: usize,
    pub last_cqt_ms: f32,
    row_buf: Vec<f32>,
}

impl<const BINS: usize, const NPITCH: usize> ModelDetector<BINS, NPITCH> {
    pub fn new(sample_rate: f32, freq_base: f32, num_semitones: usize, midi_low: i32) -> Self {
        Self {
            nd: CqtFrontEnd::new(CANON_SR, freq_base, num_semitones),
            resampler: Resampler::new(sample_rate, CANON_SR),
            ring: VecDeque::with_capacity(WIN),
            flux_ring: VecDeque::with_capacity(WIN),
            prev_cqt: vec![0.0; BINS],
            sample_count: 0,
            hop_occurred: false,
            midi_low,
            prev_onset: 0.0,
            prev_prev_onset: 0.0,
            frames_since_onset: usize::MAX / 2,
            pitch_hist: VecDeque::with_capacity(PITCH_WIN),
            th_on: 0.6,
            // Same default for both instruments; not independently
            // grid-searched per instrument. Runtime-tunable via
            // RiffNetDetector.pitch_threshold.
            th_pi: 0.50,
            refractory: 6,
            max_notes: 6,
            last_cqt_ms: 0.0,
            row_buf: vec![0.0; N_CHAN * BINS],
        }
    }

    pub fn process_block(&mut self, samples: &[f32]) {
        for &s in samples {
            let (out, n) = self.resampler.push(s);
            for &y in &out[..n] {
                self.nd.process_sample(y);
                self.sample_count += 1;
                if self.sample_count % HOP == 0 {
                    self.hop_occurred = true;
                }
            }
        }
    }

    /// Build the HCQT window: 5 harmonic channels, each the CQT window
    /// log-freq-shifted so harmonic h aligns to the fundamental's bin
    /// (channel_c[b] = window[b + HCQT_OFFSETS[c]]), plus a 6th flux channel
    /// relu(cqt[t] - cqt[t-1]) computed on the raw magnitudes as frames arrive
    /// (prev = zeros at stream start). Mirrors cqt.build_hcqt's flux channel.
    /// Returns a flat [N_CHAN, WIN, BINS] tensor (channel-major) -> JS [1,6,17,BINS].
    ///
    /// Reference implementation, kept for the row-vs-window parity test —
    /// production inference uses `extract_cqt_row` + the streaming engine.
    #[allow(dead_code)]
    pub fn extract_cqt_window(&mut self) -> Vec<f32> {
        let t0 = now_ms();
        let cqt = self.nd.cqt_magnitude();
        let flux: Vec<f32> = cqt
            .iter()
            .zip(self.prev_cqt.iter())
            .map(|(&c, &p)| (c - p).max(0.0))
            .collect();
        self.prev_cqt.copy_from_slice(&cqt);
        if self.ring.len() == WIN {
            self.ring.pop_front();
            self.flux_ring.pop_front();
        }
        self.ring.push_back(cqt);
        self.flux_ring.push_back(flux);
        // Single-channel windows first (left zero-padded while warming up).
        let mut win = vec![0f32; WIN * BINS];
        let n = self.ring.len();
        for (i, frame) in self.ring.iter().enumerate() {
            let row = WIN - n + i;
            win[row * BINS..(row + 1) * BINS].copy_from_slice(frame);
        }
        // Harmonic stack + flux channel.
        let mut data = vec![0f32; N_CHAN * WIN * BINS];
        for (c, &off) in HCQT_OFFSETS.iter().enumerate() {
            let cbase = c * WIN * BINS;
            for t in 0..WIN {
                let tb = t * BINS;
                for b in 0..BINS {
                    let src = b as i32 + off;
                    if src >= 0 && src < BINS as i32 {
                        data[cbase + tb + b] = win[tb + src as usize];
                    }
                }
            }
        }
        let fbase = 5 * WIN * BINS;
        for (i, frame) in self.flux_ring.iter().enumerate() {
            let row = WIN - n + i;
            data[fbase + row * BINS..fbase + (row + 1) * BINS].copy_from_slice(frame);
        }
        self.last_cqt_ms = (now_ms() - t0) as f32;
        data
    }

    /// Build only the NEWEST frame's 6-channel HCQT+flux row (layout [c][b],
    /// c*BINS+b) — the streaming-path counterpart of `extract_cqt_window`,
    /// which rebuilds all 17 frames. Same channel semantics: 5 harmonic
    /// log-freq shifts of the current CQT frame plus the flux channel
    /// relu(cqt[t] - cqt[t-1]). Updates the flux state (`prev_cqt`), so call
    /// exactly one of `extract_cqt_row` / `extract_cqt_window` per hop.
    pub fn extract_cqt_row(&mut self) -> &[f32] {
        let t0 = now_ms();
        let cqt = self.nd.cqt_magnitude();
        for (c, &off) in HCQT_OFFSETS.iter().enumerate() {
            let cbase = c * BINS;
            for b in 0..BINS {
                let src = b as i32 + off;
                self.row_buf[cbase + b] = if src >= 0 && src < BINS as i32 {
                    cqt[src as usize]
                } else {
                    0.0
                };
            }
        }
        let fbase = 5 * BINS;
        for b in 0..BINS {
            self.row_buf[fbase + b] = (cqt[b] - self.prev_cqt[b]).max(0.0);
        }
        self.prev_cqt.copy_from_slice(&cqt);
        self.last_cqt_ms = (now_ms() - t0) as f32;
        &self.row_buf
    }

    /// Play-along verification (known-song mode). Given the chart's expected MIDI
    /// pitches for the current moment, return the model's rolling-mean confidence
    /// (0..1) for each, read from the same `PITCH_WIN` buffer the live decoder
    /// already maintains (so this costs nothing extra — just query it a few frames
    /// after the expected onset). Only the expected pitches are queried, so wrong
    /// notes can't be produced.
    ///
    /// Threshold: treat a returned confidence >= ~0.30 as a hit; tune ~0.25–0.40
    /// for strict↔lenient. Out-of-range pitches (outside this instance's
    /// MIDI range) → 0.
    pub fn verify_expected(&self, expected_midi: &[i32]) -> Vec<f32> {
        if self.pitch_hist.is_empty() {
            return vec![0.0; expected_midi.len()];
        }
        let inv = 1.0 / self.pitch_hist.len() as f32;
        expected_midi
            .iter()
            .map(|&m| {
                let idx = m - self.midi_low;
                if idx < 0 || idx as usize >= NPITCH {
                    return 0.0;
                }
                let idx = idx as usize;
                self.pitch_hist.iter().map(|f| f[idx]).sum::<f32>() * inv
            })
            .collect()
    }

    pub fn decode_logits(&mut self, onset_logit: f32, pitch_logits: &[f32]) -> DetectionResult {
        let onset_p = sigmoid(onset_logit);
        
        let refractory_ok = self.frames_since_onset >= self.refractory;
        let onset_detected = self.prev_onset > self.th_on
            && self.prev_onset >= self.prev_prev_onset
            && self.prev_onset > onset_p
            && refractory_ok;
        self.frames_since_onset = if onset_detected { 0 } else { self.frames_since_onset.saturating_add(1) };
        self.prev_prev_onset = self.prev_onset;
        self.prev_onset = onset_p;

        let mut cur_pitch = vec![0f32; NPITCH];
        for c in 0..NPITCH {
            cur_pitch[c] = sigmoid(pitch_logits[c]);
        }
        if self.pitch_hist.len() == PITCH_WIN {
            self.pitch_hist.pop_front();
        }
        self.pitch_hist.push_back(cur_pitch);
        let inv = 1.0 / self.pitch_hist.len() as f32;
        
        let v: Vec<f32> = (0..NPITCH)
            .map(|c| self.pitch_hist.iter().map(|f| f[c]).sum::<f32>() * inv)
            .collect();

        let mut cand: Vec<(usize, f32)> = (0..NPITCH)
            .map(|c| (c, v[c]))
            .filter(|&(_, p)| p > self.th_pi)
            .collect();
        cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Harmonic sieve: drop a candidate that sits at a (non-octave) harmonic
        // interval above a clearly-stronger kept candidate (kept >= ratio*cand),
        // so genuine chord tones survive but loud-note harmonic ghosts are removed.
        let mut kept: Vec<(usize, f32)> = Vec::with_capacity(cand.len());
        for &(c, p) in &cand {
            let is_ghost = kept.iter().any(|&(c2, p2)| {
                c > c2 && HARMONIC_INTERVALS.contains(&(c - c2)) && p2 >= SIEVE_RATIO * p
            });
            if !is_ghost {
                kept.push((c, p));
            }
        }
        let mut cand = kept;
        cand.truncate(self.max_notes);

        let peaks: Vec<DetectedNote> = cand
            .iter()
            .map(|&(c, p)| {
                let midi = (self.midi_low + c as i32) as f32;
                DetectedNote { midi_note: midi, frequency: midi_to_freq(midi), amplitude: p }
            })
            .collect();

        let chord_name = if peaks.len() >= 2 {
            riffnet_dsp::chord::recognize_chord(&peaks)
        } else {
            None
        };

        DetectionResult {
            onset_detected,
            peaks,
            chord_name,
        }
    }
}

/// Guitar: MIDI 36-90, 55 pitch classes, 109 CQT bins.
#[allow(dead_code)] // used by burn_model.rs's tests, not the non-test build
pub type GuitarModelDetector = ModelDetector<109, 55>;
/// Bass: MIDI 23-65, 43 pitch classes, 85 CQT bins (5-string low-B floor).
#[allow(dead_code)]
pub type BassModelDetector = ModelDetector<85, 43>;

#[cfg(test)]
mod tests {
    use super::*;
    use riffnet_dsp::dsp::{GUITAR_FREQ_BASE, GUITAR_NUM_SEMITONES};

    const TEST_BINS: usize = 109; // guitar's shape; this test is instrument-agnostic logic

    /// `extract_cqt_row` must produce exactly the newest frame of what
    /// `extract_cqt_window` builds (all 6 channels, bit-identical) — the
    /// streaming path feeds the model per-row instead of per-window.
    #[test]
    fn cqt_row_matches_window_last_frame() {
        let mut win_det = GuitarModelDetector::new(44100.0, GUITAR_FREQ_BASE, GUITAR_NUM_SEMITONES, 36);
        let mut row_det = GuitarModelDetector::new(44100.0, GUITAR_FREQ_BASE, GUITAR_NUM_SEMITONES, 36);
        let samples: Vec<f32> = (0..4096)
            .map(|i| {
                let t = i as f32 / 44100.0;
                (2.0 * std::f32::consts::PI * 110.0 * t).sin() * 0.4
                    + (2.0 * std::f32::consts::PI * 220.0 * t).sin() * 0.2
            })
            .collect();
        let mut hops = 0;
        for chunk in samples.chunks(128) {
            win_det.hop_occurred = false;
            win_det.process_block(chunk);
            row_det.hop_occurred = false;
            row_det.process_block(chunk);
            if win_det.hop_occurred {
                let window = win_det.extract_cqt_window();
                let row = row_det.extract_cqt_row();
                for c in 0..N_CHAN {
                    for b in 0..TEST_BINS {
                        let w = window[(c * WIN + (WIN - 1)) * TEST_BINS + b];
                        let r = row[c * TEST_BINS + b];
                        assert_eq!(w, r, "hop {hops}, channel {c}, bin {b}");
                    }
                }
                hops += 1;
            }
        }
        assert!(hops >= 8, "expected several hops, got {hops}");
    }
}
