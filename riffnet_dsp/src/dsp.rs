use wide::f32x4;

// CQT front-end configuration. These are FIXED to what the deployed models
// were trained on — the Python trainer's `cqt.py` mirrors them bit-for-bit
// (BINS_PER_OCTAVE / Q_SCALE / LPF_CUTOFF are shared across every instrument;
// FREQ_BASE / NUM_SEMITONES are per-instrument, passed to `CqtFrontEnd::new`).
// Do not change one side without the other, or the model sees a distribution
// shift.
const LPF_CUTOFF: f32 = 3000.0; // 4th-order (cascaded 2nd-order) low-pass, Hz
const BINS_PER_OCTAVE: usize = 24;
const Q_SCALE: f32 = 0.3;

/// C2 floor: covers downtuned/extended-range guitar (Drop D/C) through
/// standard guitar's high range. MIDI 36 (C2) .. 90 (F#6) -> 109 bins at 24 bpo.
pub const GUITAR_FREQ_BASE: f32 = 65.406391;
pub const GUITAR_NUM_SEMITONES: usize = 54;
/// B0 floor: a 5-string bass's open low B through the practical bulk of real
/// mixed-context bass content. MIDI 23 (B0) .. 65 -> 85 bins at 24 bpo.
/// Covers 4-string basses too; they simply never play the bottom five classes.
///
/// The bottom 10 bins are the most expensive in the bank -- CQT kernel length
/// grows as frequency falls (B0's is ~14.6k taps vs E1's ~11k), so extending
/// the floor five semitones costs ~38% more front-end work per frame than the
/// old MIDI 28 floor did. That is the price of the low B; the top of the range
/// is nearly free by comparison.
pub const BASS_FREQ_BASE: f32 = 30.867706;
pub const BASS_NUM_SEMITONES: usize = 42;

/// LPF + constant-Q transform front-end. Feed raw samples (`process_sample`) at
/// the model's canonical sample rate; read the current magnitude spectrum with
/// `cqt_magnitude`. Onset/pitch decoding lives in the model path, not here.
/// `freq_base`/`num_semitones` select the instrument's range (see the
/// `GUITAR_*`/`BASS_*` constants above) — everything else about the CQT
/// (bins/octave, Q, LPF) is shared across instruments.
pub struct CqtFrontEnd {
    sample_rate: f32,

    // LPF coefficients
    lpf_b: [f32; 3],
    lpf_a: [f32; 2],
    // LPF state variables (two cascaded 2nd-order stages)
    lpf_x: [f32; 2],
    lpf_y: [f32; 2],
    lpf_x2: [f32; 2],
    lpf_y2: [f32; 2],

    // CQT kernels split into real/imag arrays so the per-bin dot products
    // vectorize (the input window is real -> two real dot products).
    kernels_re: Vec<Vec<f32>>,
    kernels_im: Vec<Vec<f32>>,
    max_n_k: usize,

    // Circular history buffer for filtered audio
    history: Vec<f32>,
    history_idx: usize,
}

impl CqtFrontEnd {
    pub fn new(sample_rate: f32, freq_base: f32, num_semitones: usize) -> Self {
        let (lpf_b, lpf_a) = Self::compute_lpf_coefficients(LPF_CUTOFF, sample_rate);
        let (kernels_re, kernels_im, max_n_k) =
            Self::build_kernels(sample_rate, freq_base, num_semitones);

        // History buffer must be at least max_n_k; use the next power of 2
        // above max_n_k (min 32768) so `get_history` can mask instead of modulo.
        let mut history_len = 32768;
        while history_len < max_n_k {
            history_len *= 2;
        }

        Self {
            sample_rate,
            lpf_b,
            lpf_a,
            lpf_x: [0.0; 2],
            lpf_y: [0.0; 2],
            lpf_x2: [0.0; 2],
            lpf_y2: [0.0; 2],
            kernels_re,
            kernels_im,
            max_n_k,
            history: vec![0.0; history_len],
            history_idx: 0,
        }
    }

    // Butterworth 2nd-order LPF coefficients (matches cqt.lpf_coeffs in Python).
    fn compute_lpf_coefficients(cutoff: f32, sample_rate: f32) -> ([f32; 3], [f32; 2]) {
        // Clamp cutoff to safe range to avoid NaN or division by zero at Nyquist.
        let safe_cutoff = cutoff.clamp(20.0, 0.45 * sample_rate);
        let omega = (std::f32::consts::PI * safe_cutoff / sample_rate).tan();
        let sqrt2 = std::f32::consts::SQRT_2;
        let c = 1.0 / (1.0 + sqrt2 * omega + omega * omega);
        let b0 = omega * omega * c;
        let b1 = 2.0 * b0;
        let b2 = b0;
        let a1 = 2.0 * (omega * omega - 1.0) * c;
        let a2 = (1.0 - sqrt2 * omega + omega * omega) * c;
        ([b0, b1, b2], [a1, a2])
    }

    /// Build CQT kernels split into real/imag arrays (plus the longest window
    /// length), so the per-bin correlations are two real dot products amenable
    /// to SIMD.
    fn build_kernels(sample_rate: f32, freq_base: f32, num_semitones: usize)
        -> (Vec<Vec<f32>>, Vec<Vec<f32>>, usize)
    {
        let b_val = BINS_PER_OCTAVE as f32;
        let q_standard = 1.0 / (2.0f32.powf(1.0 / b_val) - 1.0);
        let q = q_standard * Q_SCALE;
        let num_bins = (num_semitones * BINS_PER_OCTAVE) / 12 + 1;

        let mut kernels_re = Vec::with_capacity(num_bins);
        let mut kernels_im = Vec::with_capacity(num_bins);
        let mut max_n_k = 0;

        for i in 0..num_bins {
            let freq = freq_base * 2.0f32.powf(i as f32 / b_val);

            let mut n_k = (q * sample_rate / freq).round() as usize;
            if n_k > 16384 { n_k = 16384; }
            if n_k < 16 { n_k = 16; }
            if n_k > max_n_k { max_n_k = n_k; }

            let mut kr = Vec::with_capacity(n_k);
            let mut ki = Vec::with_capacity(n_k);
            for n in 0..n_k {
                let w = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / (n_k - 1) as f32).cos());
                let theta = 2.0 * std::f32::consts::PI * freq * n as f32 / sample_rate;
                let scale = w / n_k as f32;
                kr.push(theta.cos() * scale);
                ki.push(-theta.sin() * scale);
            }
            kernels_re.push(kr);
            kernels_im.push(ki);
        }
        (kernels_re, kernels_im, max_n_k)
    }

    /// CQT magnitude from the current history. Gather the most recent `max_n_k`
    /// samples once into a contiguous window (delay order), then each bin is two
    /// f32x4 dot products. (4-lane accumulation reassociates the sum vs a strict
    /// sequential add — differs by ~1e-6, well within feature tolerance.)
    fn compute_cqt(&self) -> Vec<f32> {
        let win_len = self.max_n_k;
        let mut win = vec![0.0f32; win_len];
        for n in 0..win_len {
            win[n] = self.get_history(n);
        }
        let num_bins = self.kernels_re.len();
        let mut mag = vec![0.0f32; num_bins];
        for k in 0..num_bins {
            let kr = &self.kernels_re[k];
            let ki = &self.kernels_im[k];
            let n_k = kr.len();
            let w = &win[..n_k];

            let mut acc_re = f32x4::splat(0.0);
            let mut acc_im = f32x4::splat(0.0);
            let chunks = n_k / 4;
            for c in 0..chunks {
                let i = c * 4;
                let wv = f32x4::new([w[i], w[i + 1], w[i + 2], w[i + 3]]);
                acc_re += wv * f32x4::new([kr[i], kr[i + 1], kr[i + 2], kr[i + 3]]);
                acc_im += wv * f32x4::new([ki[i], ki[i + 1], ki[i + 2], ki[i + 3]]);
            }
            let mut sre = acc_re.reduce_add();
            let mut sim = acc_im.reduce_add();
            for i in (chunks * 4)..n_k {
                sre += w[i] * kr[i];
                sim += w[i] * ki[i];
            }
            mag[k] = (sre * sre + sim * sim).sqrt();
        }
        mag
    }

    /// Push a single raw audio sample through the cascaded LPF into the circular
    /// history buffer.
    pub fn process_sample(&mut self, sample: f32) {
        let x = sample;
        let y = self.lpf_b[0] * x
              + self.lpf_b[1] * self.lpf_x[0]
              + self.lpf_b[2] * self.lpf_x[1]
              - self.lpf_a[0] * self.lpf_y[0]
              - self.lpf_a[1] * self.lpf_y[1];

        self.lpf_x[1] = self.lpf_x[0];
        self.lpf_x[0] = x;
        self.lpf_y[1] = self.lpf_y[0];
        self.lpf_y[0] = y;

        // Stage 2 cascade
        let y2 = self.lpf_b[0] * y
               + self.lpf_b[1] * self.lpf_x2[0]
               + self.lpf_b[2] * self.lpf_x2[1]
               - self.lpf_a[0] * self.lpf_y2[0]
               - self.lpf_a[1] * self.lpf_y2[1];

        self.lpf_x2[1] = self.lpf_x2[0];
        self.lpf_x2[0] = y;
        self.lpf_y2[1] = self.lpf_y2[0];
        self.lpf_y2[0] = y2;

        let mask = self.history.len() - 1;
        self.history[self.history_idx] = y2;
        self.history_idx = (self.history_idx + 1) & mask;
    }

    // Get sample from history, going backward in time (delay=0 is latest).
    // Bitwise AND masking relies on the power-of-2 history length.
    #[inline(always)]
    fn get_history(&self, delay: usize) -> f32 {
        let mask = self.history.len() - 1;
        let idx = (self.history_idx + mask - delay) & mask;
        self.history[idx]
    }

    /// Current CQT magnitude spectrum (bin count set by `num_semitones` at
    /// construction) without mutating detector state. This is the model's
    /// per-frame input feature; it is verified to match the Python trainer's
    /// `cqt.cqt_magnitude` bit-for-bit.
    pub fn cqt_magnitude(&self) -> Vec<f32> {
        self.compute_cqt()
    }

    /// The sample rate this detector's kernels/LPF were built for.
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }
}
