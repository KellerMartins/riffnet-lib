/// Linear-interpolation resampler from an arbitrary input sample rate (e.g. the
/// mic's native rate) to the model's canonical rate (44.1 kHz).
pub(crate) struct Resampler {
    step: f64,
    t: f64,
    k: i64,
    prev: f32,
    cur: f32,
}

impl Resampler {
    pub(crate) fn new(in_sr: f32, out_sr: f32) -> Self {
        Self { step: (in_sr as f64) / (out_sr as f64), t: 0.0, k: -1, prev: 0.0, cur: 0.0 }
    }

    pub(crate) fn push(&mut self, x: f32) -> ([f32; 8], usize) {
        self.k += 1;
        self.prev = self.cur;
        self.cur = x;
        if self.k == 0 { self.prev = x; }
        let mut out = [0.0f32; 8];
        let mut n = 0;
        while self.t <= self.k as f64 && n < out.len() {
            let i0 = self.t.floor() as i64;
            if i0 == self.k - 1 {
                let frac = (self.t - i0 as f64) as f32;
                out[n] = self.prev * (1.0 - frac) + self.cur * frac;
                n += 1;
            } else if i0 >= self.k {
                break;
            } else {
                out[n] = self.cur;
                n += 1;
            }
            self.t += self.step;
        }
        (out, n)
    }
}
