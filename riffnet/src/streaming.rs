//! Hand-rolled streaming RiffNet inference — the deployed forward pass.
//!
//! The CNN's receptive field is a 17-frame sliding window that advances one
//! frame per hop, so 16 of the 17 frames are identical to the previous hop.
//! Every layer up to conv2d6 is convolutional along time with no time
//! padding, so per hop only ONE new output row per layer needs computing;
//! older rows are cached in per-layer ring buffers. That cuts the work from
//! ~10.1M to ~2.0M MACs/hop (~5x less), and this implementation is also
//! allocation-free with channel-last rows so the inner loops autovectorize
//! (wasm simd128) — considerably faster in practice than running the same
//! graph through burn-flex.
//!
//! Weights come in as a plain name -> (shape, f32 data) map extracted from
//! the burn-onnx `.bpk` at detector init (see `burn_model.rs`); this module
//! has no tensor-runtime dependency. The layer topology is hand-mirrored
//! from the generated `riffnet.rs` graph — shape asserts in `from_weights`
//! catch weight-file drift, and the parity tests in `burn_model.rs` compare
//! logits against the burn forward + the onnxruntime reference values, so a
//! retrain with an unchanged architecture needs no changes here while an
//! architecture change fails loudly.
//!
//! Time alignment (all convs "valid" in time):
//!   input rows  t-16..t (17)
//!   a1 = relu(conv2d1)              row t needs input rows {t-2, t-1, t}
//!   conv2d2 (depthwise, dil 2)      row t needs a1  rows {t-4, t-2, t}
//!   p1 = pool(relu(conv2d3 . ))     row t, freq BINS -> P1 (floor(BINS/3))
//!   conv2d4 (depthwise, dil 4)      row t needs p1  rows {t-8, t-4, t}
//!   p2 = pool(relu(conv2d5 . ))     row t, freq P1 -> P2 (floor(P1/3))
//!   conv2d6                         row t needs p2  rows {t-2, t-1, t}
//!   conv2d7 + heads collapse freq -> (onset logit, NPITCH pitch logits)
//!
//! Warmup: `reset()` primes the rings as if an infinite run of zero frames
//! preceded the stream, which reproduces the left-zero-padded warmup window
//! of the original full-window path exactly.
//!
//! Generic over `BINS`/`P1`/`P2`/`NPITCH` (const generics) so guitar and
//! bass -- separately trained models with different CQT ranges/pitch
//! vocabularies, but byte-identical channel widths/kernels/dilations
//! otherwise -- share this exact engine with zero runtime cost: Rust
//! monomorphizes each concrete instantiation (see the `Guitar*`/`Bass*`
//! type aliases at the bottom) into its own fully-specialized code, same as
//! hand-duplicating the file. C1-C7/N_CHAN are genuinely shared (identical
//! for both trained models), so they stay plain consts.

use std::collections::BTreeMap;

pub const N_CHAN: usize = 6;
const C1: usize = 32;
const C3: usize = 64;
const C5: usize = 96;
const C6: usize = 96;
const C7: usize = 128;
const BN_EPS: f32 = 1e-5; // batchnormalization1 epsilon (fixed in the ONNX graph)

pub type WeightMap = BTreeMap<String, (Vec<usize>, Vec<f32>)>;

/// Fixed-capacity ring of rows (flat storage, rotating head).
struct RowRing {
    rows: Vec<f32>,
    rowlen: usize,
    nrows: usize,
    head: usize,
}

impl RowRing {
    fn new(nrows: usize, rowlen: usize) -> Self {
        Self { rows: vec![0.0; nrows * rowlen], rowlen, nrows, head: 0 }
    }
    /// Advance and return the new head row for writing.
    fn push_row(&mut self) -> &mut [f32] {
        self.head = (self.head + 1) % self.nrows;
        let o = self.head * self.rowlen;
        &mut self.rows[o..o + self.rowlen]
    }
    /// Row from `dt` hops ago (0 = newest).
    fn get(&self, dt: usize) -> &[f32] {
        debug_assert!(dt < self.nrows);
        let idx = (self.head + self.nrows - dt) % self.nrows;
        let o = idx * self.rowlen;
        &self.rows[o..o + self.rowlen]
    }
}

pub struct StreamingRiffNet<const BINS: usize, const P1: usize, const P2: usize, const NPITCH: usize> {
    // pre: y = bn_a[c] * ln(1 + 256*x) + bn_b[c]   (BatchNorm folded into affine)
    bn_a: [f32; N_CHAN],
    bn_b: [f32; N_CHAN],
    // conv2d1 6->32 k[3,5]: w1[((kt*5+kw)*6+ci)*32+co]
    w1: Vec<f32>,
    b1: [f32; C1],
    // conv2d2 depthwise-32 k[3,5] dil_t 2: w2[(kt*5+kw)*32+c]
    w2: Vec<f32>,
    b2: [f32; C1],
    // conv2d3 1x1 32->64: w3[ci*64+co]
    w3: Vec<f32>,
    b3: [f32; C3],
    // conv2d4 depthwise-64 k[3,5] dil_t 4: w4[(kt*5+kw)*64+c]
    w4: Vec<f32>,
    b4: [f32; C3],
    // conv2d5 1x1 64->96: w5[ci*96+co]
    w5: Vec<f32>,
    b5: [f32; C5],
    // conv2d6 96->96 k[3,3] freq-pad 1: w6[((kt*3+kw)*96+ci)*96+co]
    w6: Vec<f32>,
    b6: [f32; C6],
    // conv2d7 96->128 k[1,P2]: w7[(kw*96+ci)*128+co]
    w7: Vec<f32>,
    b7: [f32; C7],
    // heads (1x1 conv1d): onset 128->1, pitch 128->NPITCH (w[ci*NPITCH+co])
    h_on_w: [f32; C7],
    h_on_b: f32,
    h_pi_w: Vec<f32>,
    h_pi_b: [f32; NPITCH],

    in_ring: RowRing, // 3 rows [BINS*6]  (log+BN-applied input, layout [b][ci])
    a1_ring: RowRing, // 5 rows [BINS*32] (relu(conv2d1), layout [b][c])
    p1_ring: RowRing, // 9 rows [P1*64]
    p2_ring: RowRing, // 3 rows [P2*96]

    tmp2: Vec<f32>, // [BINS*32] conv2d2 out
    tmp3: Vec<f32>, // [BINS*64] conv2d3+relu out
    tmp4: Vec<f32>, // [P1*64] conv2d4 out
    tmp5: Vec<f32>, // [P1*96] conv2d5+relu out
    r6: Vec<f32>,   // [P2*96] conv2d6+relu out
    v7: [f32; C7],  // conv2d7+relu out
}

/// Fetch a tensor from the map, checking its shape.
fn take(map: &WeightMap, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
    let (s, v) = map.get(name).ok_or_else(|| {
        format!(
            "missing weight tensor '{name}'; available: {:?}",
            map.keys().collect::<Vec<_>>()
        )
    })?;
    if s != shape {
        return Err(format!("weight '{name}' has shape {s:?}, expected {shape:?}"));
    }
    Ok(v.clone())
}

fn arr<const N: usize>(v: Vec<f32>) -> [f32; N] {
    v.try_into().unwrap()
}

impl<const BINS: usize, const P1: usize, const P2: usize, const NPITCH: usize>
    StreamingRiffNet<BINS, P1, P2, NPITCH>
{
    pub fn from_weights(map: &WeightMap) -> Result<Self, String> {
        // ---- BatchNorm fold: y = a*x + b ----
        let gamma = take(map, "batchnormalization1.gamma", &[N_CHAN])?;
        let beta = take(map, "batchnormalization1.beta", &[N_CHAN])?;
        let mean = take(map, "batchnormalization1.running_mean", &[N_CHAN])?;
        let var = take(map, "batchnormalization1.running_var", &[N_CHAN])?;
        let mut bn_a = [0f32; N_CHAN];
        let mut bn_b = [0f32; N_CHAN];
        for c in 0..N_CHAN {
            let a = gamma[c] / (var[c] + BN_EPS).sqrt();
            bn_a[c] = a;
            bn_b[c] = beta[c] - a * mean[c];
        }

        // Repack conv weights channel-last (output channel innermost) so the
        // per-bin inner loops are contiguous SIMD accumulations.
        // burn Conv2d weight layout: [cout, cin/groups, kh, kw]; Conv1d: [cout, cin, k].
        let raw = take(map, "conv2d1.weight", &[C1, N_CHAN, 3, 5])?;
        let mut w1 = vec![0f32; 3 * 5 * N_CHAN * C1];
        for co in 0..C1 {
            for ci in 0..N_CHAN {
                for kt in 0..3 {
                    for kw in 0..5 {
                        w1[((kt * 5 + kw) * N_CHAN + ci) * C1 + co] =
                            raw[((co * N_CHAN + ci) * 3 + kt) * 5 + kw];
                    }
                }
            }
        }
        let b1 = arr::<C1>(take(map, "conv2d1.bias", &[C1])?);

        let raw = take(map, "conv2d2.weight", &[C1, 1, 3, 5])?;
        let mut w2 = vec![0f32; 3 * 5 * C1];
        for c in 0..C1 {
            for kt in 0..3 {
                for kw in 0..5 {
                    w2[(kt * 5 + kw) * C1 + c] = raw[(c * 3 + kt) * 5 + kw];
                }
            }
        }
        let b2 = arr::<C1>(take(map, "conv2d2.bias", &[C1])?);

        let raw = take(map, "conv2d3.weight", &[C3, C1, 1, 1])?;
        let mut w3 = vec![0f32; C1 * C3];
        for co in 0..C3 {
            for ci in 0..C1 {
                w3[ci * C3 + co] = raw[co * C1 + ci];
            }
        }
        let b3 = arr::<C3>(take(map, "conv2d3.bias", &[C3])?);

        let raw = take(map, "conv2d4.weight", &[C3, 1, 3, 5])?;
        let mut w4 = vec![0f32; 3 * 5 * C3];
        for c in 0..C3 {
            for kt in 0..3 {
                for kw in 0..5 {
                    w4[(kt * 5 + kw) * C3 + c] = raw[(c * 3 + kt) * 5 + kw];
                }
            }
        }
        let b4 = arr::<C3>(take(map, "conv2d4.bias", &[C3])?);

        let raw = take(map, "conv2d5.weight", &[C5, C3, 1, 1])?;
        let mut w5 = vec![0f32; C3 * C5];
        for co in 0..C5 {
            for ci in 0..C3 {
                w5[ci * C5 + co] = raw[co * C3 + ci];
            }
        }
        let b5 = arr::<C5>(take(map, "conv2d5.bias", &[C5])?);

        let raw = take(map, "conv2d6.weight", &[C6, C5, 3, 3])?;
        let mut w6 = vec![0f32; 3 * 3 * C5 * C6];
        for co in 0..C6 {
            for ci in 0..C5 {
                for kt in 0..3 {
                    for kw in 0..3 {
                        w6[((kt * 3 + kw) * C5 + ci) * C6 + co] =
                            raw[((co * C5 + ci) * 3 + kt) * 3 + kw];
                    }
                }
            }
        }
        let b6 = arr::<C6>(take(map, "conv2d6.bias", &[C6])?);

        let raw = take(map, "conv2d7.weight", &[C7, C6, 1, P2])?;
        let mut w7 = vec![0f32; P2 * C6 * C7];
        for co in 0..C7 {
            for ci in 0..C6 {
                for kw in 0..P2 {
                    w7[(kw * C6 + ci) * C7 + co] = raw[(co * C6 + ci) * P2 + kw];
                }
            }
        }
        let b7 = arr::<C7>(take(map, "conv2d7.bias", &[C7])?);

        let h_on_w = arr::<C7>(take(map, "conv1d1.weight", &[1, C7, 1])?);
        let h_on_b = take(map, "conv1d1.bias", &[1])?[0];
        let raw = take(map, "conv1d2.weight", &[NPITCH, C7, 1])?;
        let mut h_pi_w = vec![0f32; C7 * NPITCH];
        for co in 0..NPITCH {
            for ci in 0..C7 {
                h_pi_w[ci * NPITCH + co] = raw[co * C7 + ci];
            }
        }
        let h_pi_b = arr::<NPITCH>(take(map, "conv1d2.bias", &[NPITCH])?);

        let mut s = Self {
            bn_a,
            bn_b,
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            w4,
            b4,
            w5,
            b5,
            w6,
            b6,
            w7,
            b7,
            h_on_w,
            h_on_b,
            h_pi_w,
            h_pi_b,
            in_ring: RowRing::new(3, BINS * N_CHAN),
            a1_ring: RowRing::new(5, BINS * C1),
            p1_ring: RowRing::new(9, P1 * C3),
            p2_ring: RowRing::new(3, P2 * C5),
            tmp2: vec![0.0; BINS * C1],
            tmp3: vec![0.0; BINS * C3],
            tmp4: vec![0.0; P1 * C3],
            tmp5: vec![0.0; P1 * C5],
            r6: vec![0.0; P2 * C6],
            v7: [0.0; C7],
        };
        s.reset();
        Ok(s)
    }

    /// Prime rings as if an infinite run of zero frames preceded the stream.
    pub fn reset(&mut self) {
        // Vec, not a fixed array: `N_CHAN * BINS` mixes a plain const with a
        // const-generic param, which stable Rust doesn't allow as an array
        // length (`generic_const_exprs` is nightly-only) -- fine here since
        // this only runs once at construction, not the per-hop hot path.
        let zero = vec![0f32; N_CHAN * BINS];
        for _ in 0..17 {
            self.push_frame(&zero);
        }
    }

    /// Push one hop's 6-channel input row (layout [c][b], c*BINS+b — the
    /// newest frame's slice of the HCQT+flux stack) and get the logits for
    /// the 17-frame window ending at this frame.
    pub fn push_frame(&mut self, frame: &[f32]) -> (f32, [f32; NPITCH]) {
        debug_assert_eq!(frame.len(), N_CHAN * BINS);

        // ---- pre (log) + BN -> in_ring (layout [b][ci]) ----
        {
            let row = self.in_ring.push_row();
            for b in 0..BINS {
                for c in 0..N_CHAN {
                    let x = frame[c * BINS + b];
                    row[b * N_CHAN + c] = self.bn_a[c] * (256.0 * x + 1.0).ln() + self.bn_b[c];
                }
            }
        }

        // ---- conv2d1 + relu -> a1_ring (row [b][co=32]) ----
        {
            let rows: [&[f32]; 3] =
                [self.in_ring.get(2), self.in_ring.get(1), self.in_ring.get(0)];
            let a1 = self.a1_ring.push_row();
            for b in 0..BINS {
                let mut acc = self.b1;
                for (kt, row) in rows.iter().enumerate() {
                    let kw_lo = 2usize.saturating_sub(b);
                    let kw_hi = if b + 2 < BINS { 5 } else { BINS + 2 - b };
                    for kw in kw_lo..kw_hi {
                        let bb = b + kw - 2;
                        let x6 = &row[bb * N_CHAN..bb * N_CHAN + N_CHAN];
                        let wbase = ((kt * 5 + kw) * N_CHAN) * C1;
                        for ci in 0..N_CHAN {
                            let x = x6[ci];
                            let w = &self.w1[wbase + ci * C1..wbase + ci * C1 + C1];
                            for (a, &wv) in acc.iter_mut().zip(w.iter()) {
                                *a += wv * x;
                            }
                        }
                    }
                }
                let out = &mut a1[b * C1..b * C1 + C1];
                for (o, a) in out.iter_mut().zip(acc.iter()) {
                    *o = a.max(0.0);
                }
            }
        }

        // ---- conv2d2 (depthwise-32, time-dilation 2) -> tmp2 ----
        {
            let rows: [&[f32]; 3] =
                [self.a1_ring.get(4), self.a1_ring.get(2), self.a1_ring.get(0)];
            for b in 0..BINS {
                let mut acc = self.b2;
                for (kt, row) in rows.iter().enumerate() {
                    let kw_lo = 2usize.saturating_sub(b);
                    let kw_hi = if b + 2 < BINS { 5 } else { BINS + 2 - b };
                    for kw in kw_lo..kw_hi {
                        let bb = b + kw - 2;
                        let x = &row[bb * C1..bb * C1 + C1];
                        let w = &self.w2[(kt * 5 + kw) * C1..(kt * 5 + kw) * C1 + C1];
                        for c in 0..C1 {
                            acc[c] += w[c] * x[c];
                        }
                    }
                }
                self.tmp2[b * C1..b * C1 + C1].copy_from_slice(&acc);
            }
        }

        // ---- conv2d3 (1x1, 32->64) + relu -> tmp3 ----
        for b in 0..BINS {
            let mut acc = self.b3;
            let x = &self.tmp2[b * C1..b * C1 + C1];
            for ci in 0..C1 {
                let xv = x[ci];
                let w = &self.w3[ci * C3..ci * C3 + C3];
                for (a, &wv) in acc.iter_mut().zip(w.iter()) {
                    *a += wv * xv;
                }
            }
            let out = &mut self.tmp3[b * C3..b * C3 + C3];
            for (o, a) in out.iter_mut().zip(acc.iter()) {
                *o = a.max(0.0);
            }
        }

        // ---- maxpool freq /3 (BINS -> P1; trailing bin(s) dropped if not
        // evenly divisible by 3) -> p1_ring ----
        {
            let p1 = self.p1_ring.push_row();
            for p in 0..P1 {
                let s0 = &self.tmp3[(3 * p) * C3..(3 * p) * C3 + C3];
                let s1 = &self.tmp3[(3 * p + 1) * C3..(3 * p + 1) * C3 + C3];
                let s2 = &self.tmp3[(3 * p + 2) * C3..(3 * p + 2) * C3 + C3];
                let out = &mut p1[p * C3..p * C3 + C3];
                for c in 0..C3 {
                    out[c] = s0[c].max(s1[c]).max(s2[c]);
                }
            }
        }

        // ---- conv2d4 (depthwise-64, time-dilation 4) -> tmp4 ----
        {
            let rows: [&[f32]; 3] =
                [self.p1_ring.get(8), self.p1_ring.get(4), self.p1_ring.get(0)];
            for b in 0..P1 {
                let mut acc = self.b4;
                for (kt, row) in rows.iter().enumerate() {
                    let kw_lo = 2usize.saturating_sub(b);
                    let kw_hi = if b + 2 < P1 { 5 } else { P1 + 2 - b };
                    for kw in kw_lo..kw_hi {
                        let bb = b + kw - 2;
                        let x = &row[bb * C3..bb * C3 + C3];
                        let w = &self.w4[(kt * 5 + kw) * C3..(kt * 5 + kw) * C3 + C3];
                        for c in 0..C3 {
                            acc[c] += w[c] * x[c];
                        }
                    }
                }
                self.tmp4[b * C3..b * C3 + C3].copy_from_slice(&acc);
            }
        }

        // ---- conv2d5 (1x1, 64->96) + relu -> tmp5 ----
        for b in 0..P1 {
            let mut acc = self.b5;
            let x = &self.tmp4[b * C3..b * C3 + C3];
            for ci in 0..C3 {
                let xv = x[ci];
                let w = &self.w5[ci * C5..ci * C5 + C5];
                for (a, &wv) in acc.iter_mut().zip(w.iter()) {
                    *a += wv * xv;
                }
            }
            let out = &mut self.tmp5[b * C5..b * C5 + C5];
            for (o, a) in out.iter_mut().zip(acc.iter()) {
                *o = a.max(0.0);
            }
        }

        // ---- maxpool freq /3 (P1 -> P2; trailing bin(s) dropped if not
        // evenly divisible by 3) -> p2_ring ----
        {
            let p2 = self.p2_ring.push_row();
            for p in 0..P2 {
                let s0 = &self.tmp5[(3 * p) * C5..(3 * p) * C5 + C5];
                let s1 = &self.tmp5[(3 * p + 1) * C5..(3 * p + 1) * C5 + C5];
                let s2 = &self.tmp5[(3 * p + 2) * C5..(3 * p + 2) * C5 + C5];
                let out = &mut p2[p * C5..p * C5 + C5];
                for c in 0..C5 {
                    out[c] = s0[c].max(s1[c]).max(s2[c]);
                }
            }
        }

        // ---- conv2d6 (96->96 k[3,3], freq-pad 1) + relu -> r6 ----
        {
            let rows: [&[f32]; 3] =
                [self.p2_ring.get(2), self.p2_ring.get(1), self.p2_ring.get(0)];
            for p in 0..P2 {
                let mut acc = self.b6;
                for (kt, row) in rows.iter().enumerate() {
                    let kw_lo = 1usize.saturating_sub(p);
                    let kw_hi = if p + 1 < P2 { 3 } else { P2 + 1 - p };
                    for kw in kw_lo..kw_hi {
                        let pp = p + kw - 1;
                        let x = &row[pp * C5..pp * C5 + C5];
                        let wbase = ((kt * 3 + kw) * C5) * C6;
                        for ci in 0..C5 {
                            let xv = x[ci];
                            let w = &self.w6[wbase + ci * C6..wbase + ci * C6 + C6];
                            for (a, &wv) in acc.iter_mut().zip(w.iter()) {
                                *a += wv * xv;
                            }
                        }
                    }
                }
                let out = &mut self.r6[p * C6..p * C6 + C6];
                for (o, a) in out.iter_mut().zip(acc.iter()) {
                    *o = a.max(0.0);
                }
            }
        }

        // ---- conv2d7 (96->128 k[1,P2]) + relu -> v7 ----
        {
            let mut acc = self.b7;
            for kw in 0..P2 {
                let x = &self.r6[kw * C6..kw * C6 + C6];
                for ci in 0..C6 {
                    let xv = x[ci];
                    let wbase = (kw * C6 + ci) * C7;
                    let w = &self.w7[wbase..wbase + C7];
                    for (a, &wv) in acc.iter_mut().zip(w.iter()) {
                        *a += wv * xv;
                    }
                }
            }
            for (o, a) in self.v7.iter_mut().zip(acc.iter()) {
                *o = a.max(0.0);
            }
        }

        // ---- heads ----
        let mut onset = self.h_on_b;
        for (x, w) in self.v7.iter().zip(self.h_on_w.iter()) {
            onset += x * w;
        }
        let mut pitch = self.h_pi_b;
        for ci in 0..C7 {
            let xv = self.v7[ci];
            let w = &self.h_pi_w[ci * NPITCH..ci * NPITCH + NPITCH];
            for (a, &wv) in pitch.iter_mut().zip(w.iter()) {
                *a += wv * xv;
            }
        }
        (onset, pitch)
    }
}

/// Guitar: MIDI 36-90, 55 pitch classes, 109 CQT bins (pool1=36, pool2=12).
#[allow(dead_code)] // used by burn_model.rs's tests, not the non-test build
pub type GuitarStreamingRiffNet = StreamingRiffNet<109, 36, 12, 55>;
/// Bass: MIDI 23-65, 43 pitch classes, 85 CQT bins (pool1=28, pool2=9).
#[allow(dead_code)]
pub type BassStreamingRiffNet = StreamingRiffNet<85, 28, 9, 43>;
