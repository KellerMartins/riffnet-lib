"""Python replica of the Rust LPF + CQT front-end (riffnet_dsp/src/dsp.rs).

This MUST match the Rust inference front-end bit-for-bit (within f32 rounding),
otherwise a model trained on these features will see a distribution shift when
it runs in WASM. The Rust side mirrors these constants in `dsp.rs`.
"""
import numpy as np
from scipy.signal import lfilter

# C2 floor: covers downtuned/extended-range guitar (Drop D/C) through
# standard guitar's high range.
FREQ_BASE = 65.406391         # C2
MIDI_BASE = 36                # C2
BINS_PER_OCTAVE = 24
NUM_SEMITONES = 54            # C2..F#6 -> 109 bins at 24 bpo
Q_SCALE = 0.3
LPF_CUTOFF = 3000.0

# HCQT input: stack the CQT log-freq-shifted so each channel aligns a given
# harmonic (or sub-harmonic) to the fundamental's bin. Harmonic h sits
# round(BINS_PER_OCTAVE*log2(h)) bins above the fundamental, so channel_h[b] =
# CQT[b + off_h]. Makes harmonic structure explicit -> pitch recall + octave
# disambiguation. Cheap: just shifts.
HCQT_HARMONICS = (0.5, 1.0, 2.0, 3.0, 4.0)

# Flux input channel: a 6th channel holding the half-wave-rectified
# temporal difference of the CQT. Heavy distortion compresses sustain up to
# attack level, erasing the magnitude contrast the onset head needs; the flux of
# the SAME frames makes the pick attack explicit. Causal: frame t needs only t-1.
# Total model input channels = len(HCQT_HARMONICS) + 1 flux = 6.


def _freq_shift(spec: np.ndarray, off: int) -> np.ndarray:
    """spec (T, F) -> out where out[:, b] = spec[:, b+off] (zero-filled edges)."""
    if off == 0:
        return spec
    out = np.zeros_like(spec)
    f = spec.shape[1]
    if off > 0:
        out[:, :f - off] = spec[:, off:]
    else:
        out[:, -off:] = spec[:, :f + off]
    return out


def build_hcqt(spec: np.ndarray, bpo: int = BINS_PER_OCTAVE) -> np.ndarray:
    """(T, F) CQT magnitude -> (6, T, F): 5 harmonic channels + 1 flux channel."""
    chans = [_freq_shift(spec, int(round(bpo * np.log2(h)))) for h in HCQT_HARMONICS]
    fx = np.zeros_like(spec)
    fx[1:] = np.maximum(spec[1:] - spec[:-1], 0)
    chans.append(fx)
    return np.stack(chans, axis=0).astype(np.float32)


def lpf_coeffs(cutoff: float, sr: float):
    """Matches CqtFrontEnd::compute_lpf_coefficients."""
    safe = min(max(cutoff, 20.0), 0.45 * sr)
    omega = np.tan(np.pi * safe / sr)
    s2 = np.sqrt(2.0)
    c = 1.0 / (1.0 + s2 * omega + omega * omega)
    b0 = omega * omega * c
    b = np.array([b0, 2.0 * b0, b0], dtype=np.float64)
    a = np.array([1.0, 2.0 * (omega * omega - 1.0) * c,
                  (1.0 - s2 * omega + omega * omega) * c], dtype=np.float64)
    return b, a


def midi_to_freq(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def build_kernels(sr: float, bins_per_octave=BINS_PER_OCTAVE,
                  num_semitones=NUM_SEMITONES, q_scale=Q_SCALE, freq_base=FREQ_BASE):
    """`freq_base` defaults to guitar's own floor (C2); pass a different
    floor (e.g. midi_to_freq(28) for bass's open E1) to build a different
    range's kernel bank -- see build_dataset.py's bass builders."""
    b = bins_per_octave
    q_std = 1.0 / (2.0 ** (1.0 / b) - 1.0)
    q = q_std * q_scale
    num_bins = num_semitones * bins_per_octave // 12 + 1
    kernels, freqs = [], []
    for i in range(num_bins):
        freq = freq_base * 2.0 ** (i / b)
        n_k = int(round(q * sr / freq))
        n_k = min(max(n_k, 16), 16384)
        n = np.arange(n_k)
        w = 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (n_k - 1)))
        theta = 2.0 * np.pi * freq * n / sr
        scale = w / n_k
        ker = (np.cos(theta) - 1j * np.sin(theta)) * scale
        kernels.append(ker.astype(np.complex64))
        freqs.append(freq)
    return kernels, np.array(freqs, dtype=np.float32)


def apply_lpf(audio: np.ndarray, sr: float, cutoff=LPF_CUTOFF) -> np.ndarray:
    b, a = lpf_coeffs(cutoff, sr)
    x = lfilter(b, a, audio.astype(np.float64))
    x = lfilter(b, a, x)
    return x.astype(np.float32)


def cqt_magnitude(audio: np.ndarray, sr: float, hop: int = 256,
                  cutoff=LPF_CUTOFF, bins_per_octave=BINS_PER_OCTAVE,
                  num_semitones=NUM_SEMITONES, q_scale=Q_SCALE,
                  freq_base=FREQ_BASE, kernels=None):
    """Returns (num_frames, num_bins) float32 magnitude spectrogram.

    Frames are taken at sample indices e = hop, 2*hop, ... (e <= N-1), matching
    the Rust loop `if i > 0 && i % hop == 0` over i in 0..N. For e-n < 0 the Rust
    circular buffer reads zeros, replicated here by left zero-padding.
    """
    x = apply_lpf(audio, sr, cutoff)
    if kernels is None:
        kernels, _ = build_kernels(sr, bins_per_octave, num_semitones, q_scale, freq_base)
    N = len(x)
    ends = np.arange(hop, N, hop)
    ends = ends[ends <= N - 1]
    num_bins = len(kernels)
    max_len = max(len(k) for k in kernels)

    # Pad all kernels into one (num_bins, max_len) matrix; padded taps are zero
    # so they contribute nothing (identical math to per-bin correlation).
    ker_mat = np.zeros((num_bins, max_len), dtype=np.complex64)
    for k, ker in enumerate(kernels):
        ker_mat[k, :len(ker)] = ker

    # Gather + batched matmul, chunked over frames: a long clip times the
    # wider/longer low-frequency kernels can build a (frames x max_len) index
    # array too large to allocate in one shot.
    xp = np.concatenate([np.zeros(max_len, dtype=np.float32), x])
    chunk = max(1, (1 << 26) // max_len)   # ~64M elements/chunk cap
    outs = []
    for start in range(0, len(ends), chunk):
        e = ends[start:start + chunk]
        idx = (e[:, None] - np.arange(max_len)[None, :]) + max_len
        M = xp[idx]
        vals = M @ ker_mat.T
        outs.append(np.abs(vals).astype(np.float32))
    out = np.concatenate(outs, axis=0) if outs else np.zeros((0, num_bins), dtype=np.float32)
    return out, ends
