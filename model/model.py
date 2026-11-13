"""Small causal streaming CNN: CQT context -> per-frame onset + 55 pitch logits.

Causality without any ONNX `Pad` node: the time dimension uses *valid*
convolutions (no time padding) while frequency uses the conv's own symmetric
padding attribute. Dilated time-convs (dilation 1/2/4) widen the receptive field
to TIME_CONTEXT=16 frames without extra params. The caller left-pads the input by
16 frames (real past context, or zeros at a clip start), so output[t] depends
only on input[t-16..t] and aligns 1:1 with the un-padded frames.

Input:  (B, 6, T+16, 109)  raw HCQT magnitude (5 harmonics + flux; 16 frames of
                           left context), log-compressed inside forward()
Output: onset_logits (B, T), pitch_logits (B, T, 55)
Runs entirely in Rust/WASM via a hand-rolled streaming engine
(riffnet/src/streaming.rs) — no onnxruntime, no separate ML
runtime; burn-flex is used only to load the exported weights, not at
inference time. This module is the PyTorch training side; export.py
converts a checkpoint to ONNX, which build.rs embeds at compile time.
NOTE: trained at hop 256 (see build_dataset.py / train.py) — the Rust front-end
must feed frames at the same hop.

Pitch vocabulary: MIDI 36-90 (55 classes), covering downtuned/extended-range
guitar (Drop D/C) through standard guitar's high range.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from cqt import HCQT_HARMONICS

NUM_PITCH = 55
NUM_BINS = 109
N_IN = len(HCQT_HARMONICS) + 1   # 5 harmonic channels + 1 flux channel = 6
TIME_CONTEXT = 16                # frames of left context the caller must prepend
FEAT_SCALE = 256.0               # log1p(mag * FEAT_SCALE); mirror in Rust


def log_compress(mag: torch.Tensor) -> torch.Tensor:
    return torch.log1p(mag * FEAT_SCALE)


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding, dilation=(1, 1)):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size, 
            padding=padding, dilation=dilation, groups=in_ch
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, (1, 1))
    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class RiffNet(nn.Module):
    # Depthwise-separable + dilated-time convs; receptive field = 16 time frames.
    def __init__(self, ch=(32, 64, 96, 96), emb=128):
        super().__init__()
        self.in_bn = nn.BatchNorm2d(N_IN)
        # padding=(time=0 -> valid, freq=kf//2 -> 'same'); input = HCQT (N_IN chans)
        self.c1 = nn.Conv2d(N_IN,  ch[0], (3, 5), padding=(0, 2), dilation=(1, 1))
        self.b1 = nn.BatchNorm2d(ch[0])
        # c2 and c3 use Depthwise Separable Conv2d with wider (3,5) kernel and same frequency padding (0,2)
        self.c2 = DepthwiseSeparableConv2d(ch[0], ch[1], (3, 5), padding=(0, 2), dilation=(2, 1))
        self.b2 = nn.BatchNorm2d(ch[1])
        self.c3 = DepthwiseSeparableConv2d(ch[1], ch[2], (3, 5), padding=(0, 2), dilation=(4, 1))
        self.b3 = nn.BatchNorm2d(ch[2])
        self.c4 = nn.Conv2d(ch[2], ch[3], (3, 3), padding=(0, 1), dilation=(1, 1))
        self.b4 = nn.BatchNorm2d(ch[3])
        # after two freq-pools of 3: 109 -> 36 -> 12
        self.freq_collapse = nn.Conv2d(ch[3], emb, (1, 12))   # 12 -> 1
        self.onset_head = nn.Conv1d(emb, 1, 1)
        self.pitch_head = nn.Conv1d(emb, NUM_PITCH, 1)

    def forward(self, mag):
        # mag: (B, N_IN, T+16, 109) raw HCQT magnitude
        x = log_compress(mag)
        x = self.in_bn(x)
        x = F.relu(self.b1(self.c1(x)))            # time -2
        x = F.relu(self.b2(self.c2(x)))            # time -2
        x = F.max_pool2d(x, (1, 3))                # freq 109 -> 36
        x = F.relu(self.b3(self.c3(x)))            # time -2
        x = F.max_pool2d(x, (1, 3))                # freq 36 -> 12
        x = F.relu(self.b4(self.c4(x)))            # time -2
        x = F.relu(self.freq_collapse(x))          # (B, emb, T, 1)
        x = x.squeeze(-1)                          # (B, emb, T)
        onset = self.onset_head(x).squeeze(1)      # (B, T)
        pitch = self.pitch_head(x).transpose(1, 2)  # (B, T, 55)
        return onset, pitch

    def receptive_time(self):
        return TIME_CONTEXT + 1



def load_model(ckpt_data):
    """Instantiate RiffNet and load a checkpoint saved by train.py."""
    m = RiffNet()
    m.load_state_dict(ckpt_data["model"])
    return m


# ---- Bass: a second, separately-trained model (not a shared backbone) --
# same channel widths/kernels/dilations as RiffNet, only the frequency range
# differs (MIDI 23-65, 43 classes, 85 CQT bins -- a 5-string's open low B
# through the practical bulk of real mixed-context bass content).
#
# The MIDI 23 floor covers 5-string (BEADG) as well as 4-string basses; a
# 4-string simply never plays the bottom five classes. MIDI 23-27 has no real
# training audio anywhere -- neither IDMT-SMT-Bass (floor 28) nor SlakhBass
# (floor 47) reaches it -- so that register is trained on pitch-shifted real
# recordings, shifted only from weak-fundamental source notes so the harmonic
# balance stays physical (scripts/build_dataset.py, build_idmt_shifted).

BASS_NUM_PITCH = 43
BASS_NUM_BINS = 85


class RiffNetBass(nn.Module):
    # Same architecture as RiffNet; only num_bins/num_pitch (and the
    # freq_collapse/pitch_head widths derived from them) differ.
    def __init__(self, ch=(32, 64, 96, 96), emb=128):
        super().__init__()
        self.in_bn = nn.BatchNorm2d(N_IN)
        self.c1 = nn.Conv2d(N_IN,  ch[0], (3, 5), padding=(0, 2), dilation=(1, 1))
        self.b1 = nn.BatchNorm2d(ch[0])
        self.c2 = DepthwiseSeparableConv2d(ch[0], ch[1], (3, 5), padding=(0, 2), dilation=(2, 1))
        self.b2 = nn.BatchNorm2d(ch[1])
        self.c3 = DepthwiseSeparableConv2d(ch[1], ch[2], (3, 5), padding=(0, 2), dilation=(4, 1))
        self.b3 = nn.BatchNorm2d(ch[2])
        self.c4 = nn.Conv2d(ch[2], ch[3], (3, 3), padding=(0, 1), dilation=(1, 1))
        self.b4 = nn.BatchNorm2d(ch[3])
        # after two freq-pools of 3: 85 -> 28 -> 9
        self.freq_collapse = nn.Conv2d(ch[3], emb, (1, 9))   # 9 -> 1
        self.onset_head = nn.Conv1d(emb, 1, 1)
        self.pitch_head = nn.Conv1d(emb, BASS_NUM_PITCH, 1)

    def forward(self, mag):
        # mag: (B, N_IN, T+16, 85) raw HCQT magnitude
        x = log_compress(mag)
        x = self.in_bn(x)
        x = F.relu(self.b1(self.c1(x)))            # time -2
        x = F.relu(self.b2(self.c2(x)))            # time -2
        x = F.max_pool2d(x, (1, 3))                # freq 85 -> 28
        x = F.relu(self.b3(self.c3(x)))            # time -2
        x = F.max_pool2d(x, (1, 3))                # freq 28 -> 9
        x = F.relu(self.b4(self.c4(x)))            # time -2
        x = F.relu(self.freq_collapse(x))          # (B, emb, T, 1)
        x = x.squeeze(-1)                          # (B, emb, T)
        onset = self.onset_head(x).squeeze(1)      # (B, T)
        pitch = self.pitch_head(x).transpose(1, 2)  # (B, T, 43)
        return onset, pitch

    def receptive_time(self):
        return TIME_CONTEXT + 1


def load_model_bass(ckpt_data):
    """Instantiate RiffNetBass and load a checkpoint saved by train.py
    --instrument bass."""
    m = RiffNetBass()
    m.load_state_dict(ckpt_data["model"])
    return m
