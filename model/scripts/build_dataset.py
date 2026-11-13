"""Build the training cache: per-clip CQT spectrograms + frame-aligned
onset/pitch labels, using the parity-verified Python front-end (cqt.py).
Guitar (EGDB/GOAT, real + pitch-shifted) and bass (IDMTBass/SlakhBass) are
separate models with separate caches/ranges -- see model.py's
RiffNet/RiffNetBass:

  --dataset egdb   EGDB wav+midi (audio_<timbre>/*.wav)     -> data/hop256/<timbre>/*.npz
  --dataset goat   GOAT re-amps (metadata.csv streams)       -> data/hop256/GOAT_*/*.npz
  --dataset idmt   IDMT-SMT-Bass isolated notes               -> data/hop256_bass/IDMTBass/*.npz
                   + its low-B shifted augmentation           -> data/hop256_bass/IDMTBassShifted/*.npz
  --dataset slakh  SlakhBass per-track bass stems             -> data/hop256_bass/SlakhBass/<split>/*.npz
  --dataset all    egdb + goat (+ their shifted derivatives, see build_egdb_shifted/build_goat_shifted)
  --dataset bass   idmt + slakh

Frame f covers sample index e_f = (f+1)*hop, timestamp t_f = e_f / sr (matching
the Rust frame convention). Guitar pitch class c = midi - 36, range MIDI 36..90
(55 classes, covering downtuned/extended-range guitar through standard guitar's
high range); bass pitch class c = midi - 23, range MIDI 23..65 (43 classes,
covering a 5-string bass's open low B through the practical bulk of real
mixed-context bass content). Quarantined GOAT clips (see dataset.QUARANTINED)
are skipped.

EGDB_Shifted/GOAT_Shifted extend real-recording coverage down into the
downtuned register: EGDB/GOAT's own audio rarely goes below standard guitar's
open low E (MIDI 40), so a stratified/full subset of each corpus's own
TRAIN-split DI recordings is pitch-shifted -1..-4 semitones (Pedalboard
PitchShift, duration-preserving) down to MIDI 36 -- real recorded audio with
correct attack/pickup/amp character, rather than synthesizing the register
from scratch. Only ever built from TRAIN-split source clips, so nothing here
leaks into either corpus's held-out val/test split.
"""
import argparse
import csv
import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import soundfile as sf
from pedalboard import Pedalboard, PitchShift

import cqt
from dataset import QUARANTINED

MIDI_LOW = 36            # C2
NUM_PITCH = 55           # MIDI 36..90
SHIFTS = (-1, -2, -3, -4)  # semitones; reaches EGDB/GOAT's real floor down to MIDI 36

BASS_MIDI_LOW = 23       # B0 (5-string bass's open low string)
BASS_NUM_PITCH = 43      # MIDI 23..65

# Bass low-register augmentation. Neither IDMT-SMT-Bass (floor MIDI 28) nor
# SlakhBass (floor 47) has any audio below a 4-string's open low E, so MIDI
# 23..27 -- the 5-string range this model was widened for -- is synthesized by
# shifting real IDMT notes down, the same way EGDB_Shifted/GOAT_Shifted cover
# downtuned guitar.
#
# BASS_SHIFT_MAX_SOURCE is the part that matters and is easy to get wrong. A
# pitch shifter moves whatever spectrum it is handed; it does not know that
# real bass notes lose their fundamental as they descend. Measured on IDMT
# (median fundamental / 2nd-harmonic energy at the labeled pitch): MIDI 28 =
# 0.22, 30 = 0.67, 31 = 0.85, 35 = 1.02, 37 = 2.06. Shifting a MIDI 37 note
# down to 27 therefore manufactures a low note whose fundamental is ~10x
# stronger than anything real down there, teaching the model to look for
# something the instrument never produces. Capping sources at 30 keeps the
# shifted corpus at 0.22-0.38, against real open-E1's 0.199.
#
# Targets stop at 27 rather than blending into 28+ on purpose: real audio
# already covers 28+ with the correct spectral balance, and shifted clips
# there would put contradictory examples (real MIDI 28 at 0.22 vs
# shifted-to-28 at ~0.85) on the same pitch class.
BASS_SHIFTS = (-1, -2, -3, -4, -5, -6, -7)
BASS_SHIFT_MAX_SOURCE = 30   # only shift FROM weak-fundamental notes
BASS_SHIFT_MAX_TARGET = 27   # top of the register with no real audio
# dataset.split_clips_bass holds this instrument out for validation, so it
# must never be a shift source -- a shifted copy of a held-out recording is a
# near-duplicate of validation data. NOTE this is a *basename prefix*, not
# dataset.py's `_BS_3_EQ_` pattern: that one matches cache filenames, where
# the source's folder path has been prefixed (`ES__NO__BS_3_EQ_...`). A raw
# IDMT wav's basename *starts* with `BS_3_EQ_`, so the underscore-led pattern
# matches none of the 936 held-out source files.
BASS_HELD_OUT_PREFIX = "BS_3_EQ_"

# IDMT-SMT-Bass: string 1-4 = E/A/D/G (standard 4-string tuning), fret 0 = open.
IDMT_OPEN_STRING_MIDI = {1: 28, 2: 33, 3: 38, 4: 43}
# Subfolders (under ES/) with a genuinely time-varying or off-formula pitch
# during the note -- excluded, this script only labels one static pitch per
# clip. HA=harmonics (rings at 2x/3x the fretted node, not this formula),
# DN=dead-notes (muted, no fundamental), BE/BEQ/BES=bending, VI/VIF/VIS=
# vibrato, SLD/SLU=slide (all move pitch during the note).
IDMT_EXCLUDED_ES_SUBDIRS = {"HA", "DN", "BE", "BEQ", "BES", "VI", "VIF", "VIS", "SLD", "SLU"}
IDMT_ONSET_RMS_FRAC = 0.20
IDMT_OFFSET_RMS_FRAC = 0.03
IDMT_RMS_HOP_S = 0.010
IDMT_FILENAME_RE = re.compile(r"_(\d)_(\d+)\.wav$")

EGDB_TRAIN_TIMBRES = ["DI", "Marshall", "Ftwin", "Mesa"]
EGDB_TEST_TIMBRES = ["JCjazz", "Plexi"]
EGDB_TRAIN_CLIP_MAX = 215      # matches dataset.TRAIN_CLIP_MAX
EGDB_SHIFT_N_CLIPS = 100       # stratified subset of DI train clips to shift

# GOAT metadata.csv column per output stream name
GOAT_STREAMS = {
    "DI": "di_audio_path",
    "amp_1": "amp_audio_path_1", "amp_2": "amp_audio_path_2",
    "amp_3": "amp_audio_path_3", "amp_4": "amp_audio_path_4",
    "amp_5": "amp_audio_path_5", "gp": "gp_audio_path",
}
GOAT_VAL_THRESHOLD = 150   # matches dataset.split_clips's GOAT train/val boundary


def parse_notes(midi_path):
    """EGDB/GOAT midi -> [(midi, start_s, end_s)] via mido's tempo-aware timing."""
    import mido
    mid = mido.MidiFile(midi_path)
    notes = []
    active = {}          # key -> start_time
    t = 0.0
    for msg in mid:      # msg.time is delta seconds (tempo-applied)
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            active.setdefault(msg.note, t)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            if msg.note in active:
                notes.append((msg.note, active.pop(msg.note), t))
    for k, s in active.items():
        notes.append((k, s, t))
    return notes


def parse_notes_pretty(midi_path):
    """GOAT fine-aligned midi -> [(midi, start_s, end_s)] via pretty_midi."""
    import pretty_midi
    notes = []
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        for instr in pm.instruments:
            for note in instr.notes:
                notes.append((note.pitch, note.start, note.end))
    except Exception as e:
        print(f"Error parsing {midi_path}: {e}")
    return notes


def make_labels(notes, num_frames, sr, hop=256, onset_win_ms=32.0,
                midi_low=MIDI_LOW, num_pitch=NUM_PITCH):
    onset = np.zeros(num_frames, dtype=np.float32)
    pitch = np.zeros((num_frames, num_pitch), dtype=np.float32)
    t_f = (np.arange(num_frames) + 1) * hop / sr          # frame timestamps
    onset_win = onset_win_ms / 1000.0
    for midi, s, e in notes:
        c = int(round(midi)) - midi_low
        if c < 0 or c >= num_pitch:
            continue
        pitch[(t_f >= s) & (t_f <= e), c] = 1.0            # sustained activation
        on = (t_f >= s) & (t_f < s + onset_win)            # onset window
        if not on.any():                                   # guarantee >=1 onset frame
            idx = np.searchsorted(t_f, s)
            if idx < num_frames:
                on[idx] = True
        onset[on] = 1.0
    return onset, pitch


def _is_valid_output(path, num_pitch=NUM_PITCH):
    """True if path is a readable, non-empty, correctly-shaped cache file --
    not just present, so an interrupted run can resume without rewriting a
    truncated/corrupt npz as if it were done."""
    if not os.path.exists(path):
        return False
    try:
        d = np.load(path)
        return (d["cqt"].shape[0] > 0 and d["pitch"].shape[0] > 0
                and d["cqt"].shape[0] == d["onset"].shape[0] == d["pitch"].shape[0]
                and d["pitch"].shape[1] == num_pitch)
    except Exception:
        return False


def _cache_clip(audio_path, notes, odir, name, hop, kernels_box,
                midi_low=MIDI_LOW, num_pitch=NUM_PITCH):
    """Returns (num_frames, num_notes, skipped). Skips recomputation if a
    valid cache file already exists at this path; a truncated/corrupt one
    (e.g. from an interrupted run) is treated as missing and rebuilt.
    kernels_box holds a kernel bank built for the CALLER's own range (guitar
    vs bass callers each pass their own box, since the ranges differ)."""
    out_path = os.path.join(odir, f"{name}.npz")
    if _is_valid_output(out_path, num_pitch=num_pitch):
        d = np.load(out_path)
        return d["cqt"].shape[0], len(notes), True

    audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    if kernels_box[0] is None:
        if midi_low == MIDI_LOW and num_pitch == NUM_PITCH:
            # Guitar's default range: build_kernels' own hardcoded FREQ_BASE
            # literal, not a recomputed midi_to_freq(MIDI_LOW) -- they agree
            # to ~1e-7 but aren't bit-identical, and this cache must stay
            # reproducible against the deployed guitar weights' training data.
            kernels_box[0], _ = cqt.build_kernels(sr)
        else:
            kernels_box[0], _ = cqt.build_kernels(
                sr, num_semitones=num_pitch - 1, freq_base=cqt.midi_to_freq(midi_low))
    spec, _ = cqt.cqt_magnitude(audio, sr, hop=hop, kernels=kernels_box[0])
    onset, pitch = make_labels(notes, spec.shape[0], sr, hop=hop,
                                midi_low=midi_low, num_pitch=num_pitch)
    np.savez_compressed(out_path, cqt=spec, onset=onset, pitch=pitch)
    return spec.shape[0], len(notes), False


def build_egdb(args, kernels_box):
    lo, hi = (int(x) for x in args.clips.split("-"))
    timbres = args.timbres.split(",") if args.timbres else (EGDB_TRAIN_TIMBRES + EGDB_TEST_TIMBRES)
    n_done = n_skipped = 0
    for timbre in timbres:
        adir = os.path.join(args.egdb, f"audio_{timbre}")
        odir = os.path.join(args.out, timbre)
        os.makedirs(odir, exist_ok=True)
        count = 0
        for clip in range(lo, hi + 1):
            wav = os.path.join(adir, f"{clip}.wav")
            midi = os.path.join(args.egdb, "audio_label", f"{clip}.midi")
            if not (os.path.exists(wav) and os.path.exists(midi)):
                continue
            if args.limit and count >= args.limit:
                break
            frames, n_notes, skipped = _cache_clip(wav, parse_notes(midi), odir, str(clip), args.hop, kernels_box)
            count += 1; n_done += 1; n_skipped += skipped
            if n_done % 50 == 0:
                print(f"  {n_done} EGDB clips (last: {timbre}/{clip}, {frames} frames, {n_notes} notes)")
        print(f"EGDB timbre {timbre}: {count} clips -> {odir}")
    print(f"EGDB done: {n_done} clips ({n_skipped} already cached, skipped)")


def build_goat(args, kernels_box):
    for s_name in GOAT_STREAMS:
        os.makedirs(os.path.join(args.out, f"GOAT_{s_name}"), exist_ok=True)
    n_done = n_skipped = 0
    with open(os.path.join(args.goat, "metadata.csv"), newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            midi_rel = row.get("finealigned_midi_path", "")
            if not midi_rel:
                continue
            midi_path = os.path.join("..", midi_rel.replace("GOAT/item", "GOAT/data/item"))
            if not os.path.exists(midi_path):
                continue
            notes = None
            for s_name, col in GOAT_STREAMS.items():
                if f"GOAT_{s_name}/{row['item']}" in QUARANTINED:
                    continue
                audio_rel = row.get(col, "")
                if not audio_rel:
                    continue
                audio_path = os.path.join("..", audio_rel.replace("GOAT/item", "GOAT/data/item"))
                if not os.path.exists(audio_path):
                    continue
                out_path = os.path.join(args.out, f"GOAT_{s_name}", f"{row['item']}.npz")
                if _is_valid_output(out_path):
                    n_done += 1; n_skipped += 1
                    continue
                if notes is None:
                    notes = parse_notes_pretty(midi_path) or []
                if not notes:
                    break
                _cache_clip(audio_path, notes, os.path.join(args.out, f"GOAT_{s_name}"),
                            row["item"], args.hop, kernels_box)
                n_done += 1
            print(f"Processed GOAT {row['item']}")
    print(f"GOAT done: {n_done} clips ({n_skipped} already cached, skipped)")


def build_egdb_shifted(args, kernels_box):
    """Pitch-shift a stratified subset of EGDB's own DI-timbre TRAIN clips
    (1..EGDB_TRAIN_CLIP_MAX) down by SHIFTS semitones. Pedalboard's PitchShift
    is duration-preserving, so only the note pitches need shifting, not the
    timing."""
    odir = os.path.join(args.out, "EGDB_Shifted")
    os.makedirs(odir, exist_ok=True)

    all_clips = [c for c in range(1, EGDB_TRAIN_CLIP_MAX + 1)
                 if os.path.exists(os.path.join(args.egdb, "audio_DI", f"{c}.wav"))
                 and os.path.exists(os.path.join(args.egdb, "audio_label", f"{c}.midi"))]
    n_clips = min(EGDB_SHIFT_N_CLIPS, len(all_clips))
    idx = np.linspace(0, len(all_clips) - 1, n_clips).round().astype(int)
    clips = [all_clips[i] for i in sorted(set(idx.tolist()))]
    print(f"EGDB_Shifted: {len(clips)}/{len(all_clips)} available train DI clips x {len(SHIFTS)} shifts")

    n_done = n_skipped = n_dropped = 0
    for clip in clips:
        wav = os.path.join(args.egdb, "audio_DI", f"{clip}.wav")
        midi = os.path.join(args.egdb, "audio_label", f"{clip}.midi")
        notes = None
        audio = None
        for shift in SHIFTS:
            out_path = os.path.join(odir, f"{clip}_shift{shift:+03d}.npz")
            if _is_valid_output(out_path):
                n_skipped += 1
                continue
            if audio is None:
                audio, sr = sf.read(wav, dtype="float32", always_2d=False)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1).astype(np.float32)
                notes = parse_notes(midi)
                if kernels_box[0] is None:
                    kernels_box[0], _ = cqt.build_kernels(sr)

            shifted_notes = [(m + shift, s, e) for (m, s, e) in notes]
            shifted_notes = [(m, s, e) for (m, s, e) in shifted_notes
                              if MIDI_LOW <= m < MIDI_LOW + NUM_PITCH]
            if not shifted_notes:
                n_dropped += 1
                continue

            board = Pedalboard([PitchShift(semitones=float(shift))])
            y = np.asarray(board(audio, sr), dtype=np.float32).reshape(-1)
            if len(y) != len(audio):   # guard against any off-by-one
                y = y[:len(audio)] if len(y) > len(audio) else np.pad(y, (0, len(audio) - len(y)))

            spec, _ = cqt.cqt_magnitude(y, sr, hop=args.hop, kernels=kernels_box[0])
            onset, pitch = make_labels(shifted_notes, spec.shape[0], sr, hop=args.hop)
            np.savez_compressed(out_path, cqt=spec, onset=onset, pitch=pitch)
            n_done += 1
        if n_done % 100 == 0 and n_done:
            print(f"  {n_done} EGDB_Shifted clips cached so far")
    print(f"EGDB_Shifted done: {n_done} newly cached, {n_skipped} already valid, "
          f"{n_dropped} shift/clip combos dropped (no notes left in range) -> {odir}")


_goat_shifted_kernels = None


def _init_goat_shifted_worker():
    global _goat_shifted_kernels
    _goat_shifted_kernels, _ = cqt.build_kernels(44100.0)


def _goat_shifted_job(job):
    """job = (item, wav_path, mid_path, shift, hop, out_dir). Runs in a worker
    process; returns a short status string."""
    item, wav_path, mid_path, shift, hop, out_dir = job
    out_path = os.path.join(out_dir, f"{item}_shift{shift:+03d}.npz")
    if _is_valid_output(out_path):
        return "skipped"

    audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    notes = parse_notes_pretty(mid_path)
    if not notes:
        return "dropped"

    shifted_notes = [(m + shift, s, e) for (m, s, e) in notes]
    shifted_notes = [(m, s, e) for (m, s, e) in shifted_notes
                      if MIDI_LOW <= m < MIDI_LOW + NUM_PITCH]
    if not shifted_notes:
        return "dropped"

    board = Pedalboard([PitchShift(semitones=float(shift))])
    y = np.asarray(board(audio, sr), dtype=np.float32).reshape(-1)
    if len(y) != len(audio):
        y = y[:len(audio)] if len(y) > len(audio) else np.pad(y, (0, len(audio) - len(y)))

    global _goat_shifted_kernels
    spec, _ = cqt.cqt_magnitude(y, sr, hop=hop, kernels=_goat_shifted_kernels)
    onset, pitch = make_labels(shifted_notes, spec.shape[0], sr, hop=hop)
    np.savez_compressed(out_path, cqt=spec, onset=onset, pitch=pitch)
    return "done"


def build_goat_shifted(args, kernels_box):
    """Pitch-shift every standard-tuning GOAT TRAIN item's DI stream down by
    SHIFTS semitones. Parallelized across (item, shift) jobs: GOAT's long
    clips combined with this range's low-frequency kernels can need 500MB+
    per in-flight job for the real/complex matmul upcast in
    cqt.cqt_magnitude, so keep --goat-shift-workers modest (default 6) --
    running many more concurrently risked an OOM in practice."""
    odir = os.path.join(args.out, "GOAT_Shifted")
    os.makedirs(odir, exist_ok=True)

    rows = {}
    with open(os.path.join(args.goat, "metadata.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["item"]] = row

    # Standard-tuning, TRAIN-split items only: non-standard tunings are
    # already-downtuned real content (no need to synthesize what's already
    # there), and items past GOAT_VAL_THRESHOLD are held out for val/test.
    items = []
    n_missing = 0
    for item, row in rows.items():
        if row.get("tuning_class") != "standard":
            continue
        if int(item.replace("item_", "")) > GOAT_VAL_THRESHOLD:
            continue
        # An empty path joined via os.path.join collapses to the (existing)
        # parent dir, so os.path.exists() alone won't catch a missing path --
        # check non-empty explicitly first.
        if not row.get("di_audio_path") or not row.get("finealigned_midi_path"):
            n_missing += 1
            continue
        wav = os.path.join("..", row["di_audio_path"].replace("GOAT/item", "GOAT/data/item"))
        mid = os.path.join("..", row["finealigned_midi_path"].replace("GOAT/item", "GOAT/data/item"))
        if os.path.exists(wav) and os.path.exists(mid):
            items.append((item, wav, mid))
    if n_missing:
        print(f"GOAT_Shifted: skipped {n_missing} standard items with no fine-aligned MIDI")
    items.sort(key=lambda t: t[0])
    print(f"GOAT_Shifted: {len(items)} standard-tuning train items x {len(SHIFTS)} shifts, "
          f"{args.goat_shift_workers} workers")

    jobs = [(item, wav, mid, shift, args.hop, odir)
            for (item, wav, mid) in items for shift in SHIFTS]
    n_done = n_skipped = n_dropped = 0
    with ProcessPoolExecutor(max_workers=args.goat_shift_workers,
                              initializer=_init_goat_shifted_worker) as ex:
        futures = [ex.submit(_goat_shifted_job, job) for job in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            if result == "skipped": n_skipped += 1
            elif result == "dropped": n_dropped += 1
            else: n_done += 1
            if i % 60 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} jobs complete ({n_done} newly cached, "
                      f"{n_skipped} already valid, {n_dropped} dropped)")
    print(f"GOAT_Shifted done: {n_done} newly cached, {n_skipped} already valid, "
          f"{n_dropped} dropped -> {odir}")


# ---- Bass: IDMT-SMT-Bass + SlakhBass (separate model/range from guitar) --

def is_static_pitch_file(path):
    """True for IDMT PS/* (always expression style NO) and ES/NO -- the only
    subfolders where the fretted pitch is constant for the note's sustain."""
    parts = os.path.normpath(path).split(os.sep)
    if "PS" in parts:
        return True
    if "ES" in parts:
        i = parts.index("ES")
        subdir = parts[i + 1] if i + 1 < len(parts) else None
        return subdir not in IDMT_EXCLUDED_ES_SUBDIRS
    return False


def _idmt_parse_label(path):
    """Returns midi, or None if this file should be excluded (a
    time-varying-pitch subfolder, or an unparseable/out-of-range string
    number)."""
    if not is_static_pitch_file(path):
        return None
    m = IDMT_FILENAME_RE.search(os.path.basename(path))
    if not m:
        return None
    string, fret = int(m.group(1)), int(m.group(2))
    if string not in IDMT_OPEN_STRING_MIDI:
        return None
    return IDMT_OPEN_STRING_MIDI[string] + fret


def _idmt_estimate_note_span(audio, sr):
    """(start_s, end_s) from a short-time RMS envelope -- each IDMT file is a
    single isolated note with silence lead-in/trail and no MIDI note-on/off,
    so onset/offset are estimated from the audio itself. Returns None if the
    clip is silent throughout."""
    hop = max(1, int(IDMT_RMS_HOP_S * sr))
    n_hops = len(audio) // hop
    if n_hops < 2:
        return None
    frames = audio[:n_hops * hop].reshape(n_hops, hop)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    peak = rms.max()
    if peak <= 1e-9:
        return None
    onset_idx = np.argmax(rms >= IDMT_ONSET_RMS_FRAC * peak)
    above_floor = np.nonzero(rms >= IDMT_OFFSET_RMS_FRAC * peak)[0]
    offset_idx = above_floor[-1] if len(above_floor) else n_hops - 1
    if offset_idx <= onset_idx:
        offset_idx = n_hops - 1
    return onset_idx * hop / sr, (offset_idx + 1) * hop / sr


def build_idmt(args, bass_kernels_box):
    odir = os.path.join(args.out, "IDMTBass")
    os.makedirs(odir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.idmt, "**", "*.wav"), recursive=True))
    n_done, n_skipped_cached, n_skipped_label, n_skipped_silent = 0, 0, 0, 0
    for path in files:
        if args.limit and n_done >= args.limit:
            break
        midi = _idmt_parse_label(path)
        if midi is None:
            n_skipped_label += 1
            continue

        rel = os.path.relpath(path, args.idmt)
        name = rel.replace(os.sep, "__").replace(".wav", "")
        out_path = os.path.join(odir, f"{name}.npz")
        if _is_valid_output(out_path, num_pitch=BASS_NUM_PITCH):
            n_done += 1; n_skipped_cached += 1
            continue

        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1).astype(np.float32)

        span = _idmt_estimate_note_span(audio, sr)
        if span is None:
            n_skipped_silent += 1
            continue
        start_s, end_s = span

        if bass_kernels_box[0] is None:
            bass_kernels_box[0], _ = cqt.build_kernels(
                sr, num_semitones=BASS_NUM_PITCH - 1, freq_base=cqt.midi_to_freq(BASS_MIDI_LOW))
        spec, _ = cqt.cqt_magnitude(audio, sr, hop=args.hop, kernels=bass_kernels_box[0])
        onset, pitch = make_labels([(midi, start_s, end_s)], spec.shape[0], sr, hop=args.hop,
                                    midi_low=BASS_MIDI_LOW, num_pitch=BASS_NUM_PITCH)

        np.savez_compressed(out_path, cqt=spec, onset=onset, pitch=pitch)
        n_done += 1
        if n_done % 200 == 0:
            print(f"  {n_done} IDMT clips done (last: {rel}, midi={midi}, "
                  f"span={start_s:.2f}-{end_s:.2f}s)")
    print(f"IDMT done: {n_done} clips ({n_skipped_cached} already cached, skipped; "
          f"skipped {n_skipped_label} HA/DN/unparseable, {n_skipped_silent} silent) -> {odir}")


def build_idmt_shifted(args, bass_kernels_box):
    """IDMTBassShifted: real IDMT notes pitch-shifted down into MIDI 23..27,
    the 5-string low-B register no corpus here covers natively. See
    BASS_SHIFT_MAX_SOURCE's comment for why the source pitch is capped -- it
    is the difference between a plausible low note and one with a fundamental
    no real bass produces.

    Train-only by construction (dataset.split_clips_bass never puts it in
    val/test): it is synthesized, so scoring on it would measure the
    augmentation rather than the model."""
    # args.out is the bass cache root here -- see main()'s repurposing note.
    odir = os.path.join(args.out, "IDMTBassShifted")
    os.makedirs(odir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.idmt, "**", "*.wav"), recursive=True))

    n_done, n_skipped, n_holdout, n_silent = 0, 0, 0, 0
    for path in files:
        if args.limit and n_done >= args.limit:
            break
        base = os.path.basename(path)
        midi = _idmt_parse_label(path)
        if midi is None or midi > BASS_SHIFT_MAX_SOURCE:
            continue
        if base.startswith(BASS_HELD_OUT_PREFIX):
            n_holdout += 1
            continue

        rel = os.path.relpath(path, args.idmt)
        stem = rel.replace(os.sep, "__").replace(".wav", "")
        audio, sr = None, None
        for shift in BASS_SHIFTS:
            target = midi + shift
            if not (BASS_MIDI_LOW <= target <= BASS_SHIFT_MAX_TARGET):
                continue
            out_path = os.path.join(odir, f"{stem}__shift{shift:+03d}.npz")
            if _is_valid_output(out_path, num_pitch=BASS_NUM_PITCH):
                n_skipped += 1
                continue
            if audio is None:
                audio, sr = sf.read(path, dtype="float32", always_2d=False)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1).astype(np.float32)
                if bass_kernels_box[0] is None:
                    bass_kernels_box[0], _ = cqt.build_kernels(
                        sr, num_semitones=BASS_NUM_PITCH - 1,
                        freq_base=cqt.midi_to_freq(BASS_MIDI_LOW))

            board = Pedalboard([PitchShift(semitones=float(shift))])
            y = np.asarray(board(audio, sr), dtype=np.float32).reshape(-1)
            if len(y) != len(audio):   # guard against any off-by-one
                y = y[:len(audio)] if len(y) > len(audio) else np.pad(y, (0, len(audio) - len(y)))

            # Estimate the span on the SHIFTED audio -- that is what the CQT
            # is computed from, so labels and features come off one waveform.
            span = _idmt_estimate_note_span(y, sr)
            if span is None:
                n_silent += 1
                continue
            start_s, end_s = span

            spec, _ = cqt.cqt_magnitude(y, sr, hop=args.hop, kernels=bass_kernels_box[0])
            onset, pitch = make_labels([(target, start_s, end_s)], spec.shape[0], sr,
                                        hop=args.hop, midi_low=BASS_MIDI_LOW,
                                        num_pitch=BASS_NUM_PITCH)
            np.savez_compressed(out_path, cqt=spec, onset=onset, pitch=pitch)
            n_done += 1
            if n_done % 100 == 0:
                print(f"  {n_done} IDMTBassShifted clips cached so far")
    print(f"IDMTBassShifted done: {n_done} newly cached, {n_skipped} already valid, "
          f"{n_holdout} held-out {BASS_HELD_OUT_PREFIX}* sources excluded, "
          f"{n_silent} silent -> {odir}")


def build_slakh(args, bass_kernels_box):
    tracks = sorted(glob.glob(os.path.join(args.slakh, "*", "Track*")))
    n_done, n_skipped_cached, n_skipped_missing = 0, 0, 0
    for track_dir in tracks:
        if args.limit and n_done >= args.limit:
            break
        split = os.path.basename(os.path.dirname(track_dir))
        track = os.path.basename(track_dir)
        wav_path = os.path.join(track_dir, "bass.flac")
        mid_path = os.path.join(track_dir, "bass.mid")
        if not (os.path.exists(wav_path) and os.path.exists(mid_path)):
            n_skipped_missing += 1
            continue

        odir = os.path.join(args.out, "SlakhBass", split)
        os.makedirs(odir, exist_ok=True)
        frames, n_notes, skipped = _cache_clip(
            wav_path, parse_notes(mid_path), odir, track, args.hop, bass_kernels_box,
            midi_low=BASS_MIDI_LOW, num_pitch=BASS_NUM_PITCH)
        n_done += 1; n_skipped_cached += skipped
        if n_done % 50 == 0:
            print(f"  {n_done} Slakh tracks done (last: {split}/{track}, {frames} frames, {n_notes} notes)")
    print(f"Slakh done: {n_done} tracks ({n_skipped_cached} already cached, skipped; "
          f"skipped {n_skipped_missing} missing files) -> {os.path.join(args.out, 'SlakhBass')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all",
                    choices=["egdb", "goat", "idmt", "slakh", "all", "bass"],
                    help="all = egdb+goat (guitar); bass = idmt+slakh")
    ap.add_argument("--out", default="data/hop256", help="guitar cache dir")
    ap.add_argument("--hop", type=int, default=256)
    # EGDB options
    ap.add_argument("--egdb", default="../EGDB")
    ap.add_argument("--timbres", default="", help="EGDB timbres (default: all)")
    ap.add_argument("--clips", default="1-240", help="EGDB clip range lo-hi")
    ap.add_argument("--limit", type=int, default=0, help="cap clips/tracks per source (0=all)")
    # GOAT options
    ap.add_argument("--goat", default="../GOAT")
    ap.add_argument("--goat-shift-workers", type=int, default=6,
                    help="parallel workers for GOAT_Shifted (see build_goat_shifted docstring)")
    # Bass options. --idmt expects the IDMT-SMT-Bass extracted tree (Zenodo
    # 7188892). --slakh expects a `train/`, `validation/`, `test/` layout of
    # Track*/{bass.flac,bass.mid} dirs.
    ap.add_argument("--bass-out", default="data/hop256_bass", help="bass cache dir")
    ap.add_argument("--idmt", default="../IDMTSMTBass/extracted")
    ap.add_argument("--slakh", default="../SlakhBass")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    kernels_box = [None]      # kernels are sr-dependent; built once, shared across builders
    if args.dataset in ("egdb", "all"):
        build_egdb(args, kernels_box)
        build_egdb_shifted(args, kernels_box)
    if args.dataset in ("goat", "all"):
        build_goat(args, kernels_box)
        build_goat_shifted(args, kernels_box)

    if args.dataset in ("idmt", "slakh", "bass"):
        os.makedirs(args.bass_out, exist_ok=True)
        bass_kernels_box = [None]
        # args.out is repurposed as the bass builders' output root for this
        # invocation, so the shared _cache_clip/_is_valid_output helpers
        # don't need a separate "which root" parameter.
        args.out = args.bass_out
        if args.dataset in ("idmt", "bass"):
            build_idmt(args, bass_kernels_box)
            build_idmt_shifted(args, bass_kernels_box)
        if args.dataset in ("slakh", "bass"):
            build_slakh(args, bass_kernels_box)

    print(f"DONE: cache in {args.out}")


if __name__ == "__main__":
    main()
