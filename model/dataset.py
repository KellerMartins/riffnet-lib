"""Dataset over the cached CQT clips: train/val/test splits, a curated
exclusion list, and a loader with random fixed-length crops and light
feature-space augmentation (pitch-shift + gain + noise + soft-onset
targets). Guitar (EGDB + GOAT, `split_clips`) and bass (IDMTBass +
SlakhBass, `split_clips_bass`) are separate models with separate splits --
see model.py's RiffNet/RiffNetBass.
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from model import TIME_CONTEXT as PAD
from cqt import build_hcqt

TRAIN_TIMBRES = ["DI", "Marshall", "Ftwin", "Mesa", "GOAT_DI", "GOAT_amp_1", "GOAT_amp_2", "GOAT_amp_3"]
TEST_TIMBRES = ["JCjazz", "Plexi", "GOAT_amp_4", "GOAT_amp_5"]
TRAIN_CLIP_MAX = 215


# ---- Quarantined clips: excluded from cache builds, splits, and evals ---------
# The 10 GOAT re-amps in NEAR_SILENT are near-silent (CQT peak ~25x below a
# healthy clip): the re-amped audio is effectively missing while the labels are
# intact, so they are pure label noise. The LABEL_MISALIGNED clips have onset
# labels offset beyond the 50 ms scorer tolerance vs the acoustic attack (some
# misaligned at the source, some single broken-stream renders). Either way the
# labels don't match the audio, so the clips are unusable for training or scoring.
NEAR_SILENT = {
    "GOAT_amp_1/item_102", "GOAT_amp_1/item_132", "GOAT_amp_2/item_115",
    "GOAT_amp_2/item_119", "GOAT_amp_3/item_79", "GOAT_amp_3/item_141",
    "GOAT_amp_4/item_111", "GOAT_amp_4/item_112", "GOAT_amp_4/item_128",
    "GOAT_amp_5/item_159",
}
LABEL_MISALIGNED = {
    "GOAT_DI/item_102", "GOAT_DI/item_150", "GOAT_DI/item_167", "GOAT_DI/item_35",
    "GOAT_DI/item_4", "GOAT_DI/item_41", "GOAT_DI/item_50", "GOAT_DI/item_52",
    "GOAT_DI/item_93",
    "GOAT_amp_1/item_110", "GOAT_amp_1/item_114", "GOAT_amp_1/item_135",
    "GOAT_amp_1/item_150", "GOAT_amp_1/item_35", "GOAT_amp_1/item_41",
    "GOAT_amp_1/item_49", "GOAT_amp_1/item_52", "GOAT_amp_1/item_76",
    "GOAT_amp_1/item_9", "GOAT_amp_1/item_93", "GOAT_amp_1/item_95",
    "GOAT_amp_2/item_102", "GOAT_amp_2/item_111", "GOAT_amp_2/item_120",
    "GOAT_amp_2/item_167", "GOAT_amp_2/item_35", "GOAT_amp_2/item_41",
    "GOAT_amp_2/item_50", "GOAT_amp_2/item_52", "GOAT_amp_2/item_55",
    "GOAT_amp_2/item_59", "GOAT_amp_2/item_73", "GOAT_amp_2/item_9",
    "GOAT_amp_2/item_93",
    "GOAT_amp_3/item_110", "GOAT_amp_3/item_111", "GOAT_amp_3/item_113",
    "GOAT_amp_3/item_118", "GOAT_amp_3/item_120", "GOAT_amp_3/item_171",
    "GOAT_amp_3/item_35", "GOAT_amp_3/item_41", "GOAT_amp_3/item_49",
    "GOAT_amp_3/item_50", "GOAT_amp_3/item_52", "GOAT_amp_3/item_55",
    "GOAT_amp_3/item_93", "GOAT_amp_3/item_96",
    "GOAT_amp_4/item_110", "GOAT_amp_4/item_160", "GOAT_amp_4/item_167",
    "GOAT_amp_4/item_2", "GOAT_amp_4/item_35", "GOAT_amp_4/item_4",
    "GOAT_amp_4/item_41", "GOAT_amp_4/item_50", "GOAT_amp_4/item_52",
    "GOAT_amp_4/item_73", "GOAT_amp_4/item_77", "GOAT_amp_4/item_93",
    "GOAT_amp_5/item_102", "GOAT_amp_5/item_118", "GOAT_amp_5/item_120",
    "GOAT_amp_5/item_167", "GOAT_amp_5/item_171", "GOAT_amp_5/item_35",
    "GOAT_amp_5/item_4", "GOAT_amp_5/item_41", "GOAT_amp_5/item_50",
    "GOAT_amp_5/item_90", "GOAT_amp_5/item_93", "GOAT_amp_5/item_96",
}
QUARANTINED = NEAR_SILENT | LABEL_MISALIGNED


def is_quarantined(path):
    """True if a cache path (…/<timbre_dir>/<item>.npz) is quarantined."""
    d, base = os.path.split(path)
    return f"{os.path.basename(d)}/{os.path.splitext(base)[0]}" in QUARANTINED


def shift_freq(x, bins):
    """Shift a (T, num_bins) CQT up (+) or down (-) by `bins`, zero-filling the edge."""
    if bins == 0:
        return x
    out = np.zeros_like(x)
    if bins > 0:
        out[:, bins:] = x[:, :x.shape[1] - bins]
    else:
        out[:, :bins] = x[:, -bins:]
    return out


def shift_classes(p, s):
    """Shift a (T, num_pitch) pitch label by `s` classes; notes out of range are dropped."""
    if s == 0:
        return p
    out = np.zeros_like(p)
    if s > 0:
        out[:, s:] = p[:, :p.shape[1] - s]
    else:
        out[:, :s] = p[:, -s:]
    return out


def split_clips(data_dir):
    train = []
    val = []
    test_timbre = []
    for t in TRAIN_TIMBRES:
        for f in glob.glob(os.path.join(data_dir, t, "*.npz")):
            if is_quarantined(f):
                continue
            name = os.path.splitext(os.path.basename(f))[0]
            clip = int(name.replace("item_", "")) if name.startswith("item_") else int(name)
            is_val = (clip > 150) if "GOAT" in t else (clip > TRAIN_CLIP_MAX)
            if is_val: val.append(f)
            else: train.append(f)

    for t in TEST_TIMBRES:
        for f in glob.glob(os.path.join(data_dir, t, "*.npz")):
            if is_quarantined(f):
                continue
            name = os.path.splitext(os.path.basename(f))[0]
            clip = int(name.replace("item_", "")) if name.startswith("item_") else int(name)
            is_val = (clip > 150) if "GOAT" in t else (clip > TRAIN_CLIP_MAX)
            if is_val: test_timbre.append(f)

    # Real EGDB/GOAT audio pitch-shifted down into the downtuned register
    # (build_dataset.py's build_egdb_shifted/build_goat_shifted), extending
    # coverage down to MIDI 36. Built exclusively from each corpus's own
    # TRAIN-split clips, so every file here goes to train -- no leakage into
    # either corpus's existing validation split.
    for f in glob.glob(os.path.join(data_dir, "EGDB_Shifted", "*.npz")):
        if not is_quarantined(f):
            train.append(f)
    for f in glob.glob(os.path.join(data_dir, "GOAT_Shifted", "*.npz")):
        if not is_quarantined(f):
            train.append(f)

    return sorted(train), sorted(val), sorted(test_timbre)


# IDMTBass (Fraunhofer IDMT-SMT-Bass, Zenodo 7188892) has no natural
# clip-number split (single isolated notes), so one whole bass instrument is
# held out entirely -- the same "hold out a whole timbre, not random clips"
# principle EGDB/GOAT use above.
IDMT_HELD_OUT_BASS = "BS_3"


def split_clips_bass(data_dir):
    """Bass-only dataset -- a separate model from guitar (RiffNetBass in
    model.py), with its own cache built at its own practical range directly
    (MIDI 23-65, see build_dataset.py's bass builders). IDMTBass held out by
    instrument (BS_3); SlakhBass comes pre-split into
    train/validation/test folders by the source corpus.

    IDMTBassShifted (the MIDI 23..27 low-B augmentation) goes into TRAIN ONLY.
    It is synthesized, so scoring on it would measure the augmentation rather
    than the model; and its sources are real IDMT train clips, so val entries
    would be near-duplicates of training data. build_dataset.py excludes BS_3
    as a shift source for the same reason, so no held-out recording has a
    shifted copy in train either.

    Consequence worth knowing: MIDI 23..27 has NO held-out real audio anywhere
    -- no corpus here has any audio below MIDI 28 at all, which is why the
    register needs augmenting in the first place. Nothing in this split can
    validate that register; only recordings from a real 5-string can, and
    none have been made yet."""
    train, val, test_timbre = [], [], []

    for f in glob.glob(os.path.join(data_dir, "IDMTBass", "*.npz")):
        if is_quarantined(f):
            continue
        if f"_{IDMT_HELD_OUT_BASS}_EQ_" in os.path.basename(f):
            val.append(f)
        else:
            train.append(f)

    for f in glob.glob(os.path.join(data_dir, "IDMTBassShifted", "*.npz")):
        if not is_quarantined(f):
            train.append(f)

    for f in glob.glob(os.path.join(data_dir, "SlakhBass", "train", "*.npz")):
        if not is_quarantined(f):
            train.append(f)
    for f in glob.glob(os.path.join(data_dir, "SlakhBass", "validation", "*.npz")):
        if not is_quarantined(f):
            val.append(f)
    for f in glob.glob(os.path.join(data_dir, "SlakhBass", "test", "*.npz")):
        if not is_quarantined(f):
            test_timbre.append(f)

    return sorted(train), sorted(val), sorted(test_timbre)


class ClipChunks(Dataset):
    def __init__(self, files, chunk=256, augment=False, crops_per_clip=4):
        self.files = files
        self.chunk = chunk
        self.augment = augment
        self.crops = crops_per_clip
        # preload into RAM (dataset is ~1GB)
        self.data = []
        for f in files:
            d = np.load(f)
            self.data.append((d["cqt"], d["onset"], d["pitch"]))

    def __len__(self):
        return len(self.files) * self.crops

    def __getitem__(self, i):
        cqt_full, onset_full, pitch_full = self.data[i % len(self.files)]
        T = cqt_full.shape[0]
        c = self.chunk
        if T <= c:
            s = 0
            onset = np.pad(onset_full, (0, c - T))
            pitch = np.pad(pitch_full, ((0, c - T), (0, 0)))
            cqt = np.pad(cqt_full, ((PAD, c - T), (0, 0)))         # PAD left + tail pad
        else:
            s = np.random.randint(0, T - c)
            lo = s - PAD
            left = max(0, -lo)
            cqt = cqt_full[max(0, lo):s + c]
            if left:
                cqt = np.pad(cqt, ((left, 0), (0, 0)))             # zero context at clip start
            onset = onset_full[s:s + c]
            pitch = pitch_full[s:s + c]
        cqt = cqt.astype(np.float32).copy()                       # length c + PAD
        onset = onset.astype(np.float32).copy()                   # length c
        pitch = pitch.copy()
        if self.augment:
            # pitch-shift (CQT log-freq: +1 semitone = +2 bins, labels +1 class).
            # Both directions: down fills the low-E region, up fills the high range.
            if np.random.rand() < 0.6:
                s = np.random.randint(-8, 11)
                cqt = shift_freq(cqt, 2 * s)
                pitch = shift_classes(pitch, s)
            cqt *= np.float32(np.random.uniform(0.5, 1.8))         # random gain
            cqt += np.random.uniform(0, 0.0015) * np.random.randn(*cqt.shape).astype(np.float32)
            np.maximum(cqt, 0, out=cqt)
            # soft onset target: widen by one frame on each side (triangular)
            onset = np.maximum.reduce([
                onset,
                0.5 * np.roll(onset, 1), 0.5 * np.roll(onset, -1),
            ]).astype(np.float32)
        mag = build_hcqt(cqt)                                      # (6, T, num_bins)
        mag = torch.from_numpy(mag)
        return mag, torch.from_numpy(onset), torch.from_numpy(pitch.astype(np.float32))
