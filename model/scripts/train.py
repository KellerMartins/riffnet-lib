"""Train RiffNet: guitar on the cached EGDB+GOAT dataset (default), or bass on
IDMTBass+SlakhBass via --instrument bass. Reports frame-level onset/pitch F1
on the held-out val split each epoch; with --event-select it also scores the
event level via decode.py's shared decode+scorer and selects on event Combined.

Defaults are the deployed recipe: focal gamma 1.0, octave penalty 0.15,
6-channel HCQT+flux input.
"""
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import RiffNet, RiffNetBass, TIME_CONTEXT as PAD
from cqt import build_hcqt
import dataset as ds
from decode import decode, score as event_score, group_events
from .build_dataset import parse_notes, BASS_MIDI_LOW, BASS_NUM_PITCH

# Loss hyperparameters
OCT_PEN = 0.15        # harmonic octave-loss penalty weight
FOCAL_GAMMA = 1.0     # focal-loss gamma (onset recall/precision knob)
VAL_CHUNK = 16384     # validate() GPU forward-pass chunk size, in frames


def midi_path_for(f):
    """Ground-truth midi path for a cached clip (EGDB integer clips, GOAT items)."""
    s = os.path.splitext(os.path.basename(f))[0]
    if s.startswith("item_"):
        for c in (f"{s}_fine_aligned.mid", f"{s}.mid"):
            p = os.path.join("..", "GOAT", "data", s, c)
            if os.path.exists(p):
                return p
        return None
    return os.path.join("..", "EGDB", "audio_label", f"{int(s)}.midi")


def idmt_gt_notes(npz_path):
    """Single note per IDMTBass file, by construction (build_dataset.py's
    build_idmt): pitch from the cached label's active class, timing from the
    same cached onset/pitch arrays (both exact -- this is how the cache was
    written)."""
    d = np.load(npz_path)
    active = np.nonzero(d["pitch"].any(axis=0))[0]
    if len(active) == 0:
        return []
    c = int(active[0])
    midi = c + BASS_MIDI_LOW
    onset_frames = np.nonzero(d["onset"])[0]
    active_frames = np.nonzero(d["pitch"][:, c])[0]
    if len(onset_frames) == 0 or len(active_frames) == 0:
        return []
    return [(midi, onset_frames[0] * 256 / 44100, active_frames[-1] * 256 / 44100)]


def slakh_gt_path(npz_path):
    """GT midi path for a cached SlakhBass clip (data_dir/SlakhBass/{split}/
    {track}.npz -> ../SlakhBass/{split}/{track}/bass.mid)."""
    parts = os.path.normpath(npz_path).split(os.sep)
    split, track = parts[-2], os.path.splitext(parts[-1])[0]
    return os.path.join("..", "SlakhBass", split, track, "bass.mid")


def bass_gt_notes_for(f):
    """GT notes for a bass-cached clip -- IDMTBass (single note, read off the
    cache itself) or SlakhBass (bass.mid)."""
    if "IDMTBass" in f:
        return idmt_gt_notes(f)
    p = slakh_gt_path(f)
    return parse_notes(p) if os.path.exists(p) else []


def focal_bce(logits, targets, alpha, gamma):
    """Focal BCE: down-weights easy (mostly negative) frames, focuses on hard
    positives — better than a fixed pos_weight for the sparse onset/pitch labels."""
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * (1 - p_t).pow(gamma)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss
    return loss.mean()


@torch.no_grad()
def validate(model, files, device, gt_events=None, dec_cfg=None):
    """Frame-level F1s; if gt_events/dec_cfg given, also the event-level
    (onsetF1, pitchF1, Combined) per-file mean using the shared decode+scorer.

    Accumulates TP/FP/FN incrementally and runs the GPU forward pass in
    temporal chunks (VAL_CHUNK frames) instead of concatenating whole-file
    prediction arrays / running one giant forward — a long enough clip can
    OOM both host RAM and VRAM in the unchunked version."""
    model.eval()
    ev = np.zeros(3)
    on_tp = on_fp = on_fn = 0
    pi_tp = pi_fp = pi_fn = 0
    chunk_size = VAL_CHUNK
    for i, f in enumerate(files):
        d = np.load(f)
        cqt = np.pad(d["cqt"], ((PAD, 0), (0, 0))).astype(np.float32)   # left context
        mag_np = build_hcqt(cqt)                                       # (6, T+PAD, 109)

        T = mag_np.shape[1] - PAD
        onset_parts, pitch_parts = [], []
        for start in range(0, T, chunk_size):
            end = min(T, start + chunk_size)
            mag_chunk = torch.from_numpy(mag_np[None, :, start:end + PAD]).to(device)
            on_l_chunk, pi_l_chunk = model(mag_chunk)
            onset_parts.append(torch.sigmoid(on_l_chunk)[0].cpu().numpy())
            pitch_parts.append(torch.sigmoid(pi_l_chunk)[0].cpu().numpy())
        onset_p = np.concatenate(onset_parts, axis=0)
        pitch_p = np.concatenate(pitch_parts, axis=0)

        on_p, on_t = (onset_p >= 0.5), (d["onset"] >= 0.5)
        on_tp += (on_p & on_t).sum(); on_fp += (on_p & ~on_t).sum(); on_fn += (~on_p & on_t).sum()
        pi_p, pi_t = (pitch_p >= 0.5), (d["pitch"] >= 0.5)
        pi_tp += (pi_p & pi_t).sum(); pi_fp += (pi_p & ~pi_t).sum(); pi_fn += (~pi_p & pi_t).sum()

        if gt_events is not None:
            dets = decode(onset_p, pitch_p, **dec_cfg)
            s = event_score(gt_events[i], dets)
            ev += (s[2], s[5], s[6])                                    # onF1, piF1, comb

    on_prec = on_tp / (on_tp + on_fp + 1e-9); on_rec = on_tp / (on_tp + on_fn + 1e-9)
    of = (float(2 * on_prec * on_rec / (on_prec + on_rec + 1e-9)), float(on_prec), float(on_rec))
    pi_prec = pi_tp / (pi_tp + pi_fp + 1e-9); pi_rec = pi_tp / (pi_tp + pi_fn + 1e-9)
    pf = (float(2 * pi_prec * pi_rec / (pi_prec + pi_rec + 1e-9)), float(pi_prec), float(pi_rec))
    if device == "cuda":
        torch.cuda.empty_cache()
    return of, pf, (ev / len(files) if gt_events is not None else None)


def train_loss(on_l, pi_l, onset_s, pitch_s, onset_alpha, pitch_alpha, gamma):
    loss_on = focal_bce(on_l, onset_s, onset_alpha, gamma)
    loss_pi = focal_bce(pi_l, pitch_s, pitch_alpha, gamma)
    total = loss_on + loss_pi

    if OCT_PEN == 0.0:
        return total

    # Penalize putting probability on the octave (+/-12 classes) of a real note.
    probs_pi = torch.sigmoid(pi_l)
    targets_plus_12 = torch.zeros_like(pitch_s)
    targets_plus_12[:, :, 12:] = pitch_s[:, :, :-12]
    targets_minus_12 = torch.zeros_like(pitch_s)
    targets_minus_12[:, :, :-12] = pitch_s[:, :, 12:]
    octave_targets = (targets_plus_12 + targets_minus_12).clamp(0.0, 1.0)
    octave_penalty = (probs_pi * octave_targets).mean()
    return total + OCT_PEN * octave_penalty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", default="guitar", choices=["guitar", "bass"],
                    help="guitar = RiffNet, MIDI 36-90, ds.split_clips (EGDB+GOAT); "
                         "bass = RiffNetBass, MIDI 23-65, ds.split_clips_bass "
                         "(IDMTBass+SlakhBass) -- a separately-trained model, not a shared "
                         "backbone (see model.py)")
    ap.add_argument("--data", default="data/hop256",
                    help="cache dir (default matches --instrument guitar; pass "
                         "data/hop256_bass for --instrument bass)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--out", default="riffnet.pt")
    ap.add_argument("--seed", type=int, default=None, help="fix RNG seed for reproducible runs")
    ap.add_argument("--event-select", action="store_true",
                    help="also score val at EVENT level each epoch (shared decode+scorer) "
                         "and select --out on best event Combined; the frame-selected "
                         "checkpoint is saved alongside as *_frame.pt")
    args = ap.parse_args()
    print(f"oct_pen {OCT_PEN} | focal gamma {FOCAL_GAMMA}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.seed is not None:
        import random as _random
        _random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print(f"seed: {args.seed}")

    is_bass = args.instrument == "bass"
    split_fn = ds.split_clips_bass if is_bass else ds.split_clips
    train_files, val_files, test_files = split_fn(args.data)
    print(f"train {len(train_files)} | val {len(val_files)} | test(timbre) {len(test_files)} clips")

    gt_events, dec_cfg = None, None
    if args.event_select:
        dec_cfg = dict(th_on=0.6, th_pi=0.50, refr=6, win=16, topk=6,
                       peak=True, agg="mean", offset=6, nms=False,
                       midi_low=BASS_MIDI_LOW if is_bass else 36)
        gt_events = []
        for f in val_files:
            if is_bass:
                gt_events.append(group_events(bass_gt_notes_for(f)))
            else:
                mp = midi_path_for(f)
                assert mp and os.path.exists(mp), f"no GT midi for val clip {f}"
                gt_events.append(group_events(parse_notes(mp)))
        print(f"event-select: ON (decode {dec_cfg})")

    train_set = ds.ClipChunks(train_files, chunk=args.chunk, augment=True, crops_per_clip=6)
    loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=0, drop_last=True)

    model = (RiffNetBass() if is_bass else RiffNet()).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e3:.0f}K | receptive time: {model.receptive_time()} frames")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    # focal loss: alpha weights the rare positive class up
    onset_alpha, pitch_alpha, gamma = 0.85, 0.75, FOCAL_GAMMA

    best = -1.0
    best_ev = -1.0
    frame_out = os.path.splitext(args.out)[0] + "_frame.pt"
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for mag, onset, pitch in loader:
            mag, onset, pitch = mag.to(device), onset.to(device), pitch.to(device)
            on_l, pi_l = model(mag)
            onset_s = onset * 0.9 + 0.05
            pitch_s = pitch * 0.9 + 0.05
            loss = train_loss(on_l, pi_l, onset_s, pitch_s, onset_alpha, pitch_alpha, gamma)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()
        (of, op, orc), (pf, pp, prc), ev = validate(model, val_files, device,
                                                    gt_events=gt_events, dec_cfg=dec_cfg)
        score = of * pf
        line = (f"ep{ep:02d} loss {tot/len(loader):.3f} | "
                f"onset F1 {of:.3f}(P{op:.2f}/R{orc:.2f}) | "
                f"pitch F1 {pf:.3f}(P{pp:.2f}/R{prc:.2f}) | frameComb {score:.3f}")
        if ev is not None:
            line += f" | evOn {ev[0]:.3f} evPi {ev[1]:.3f} evComb {ev[2]:.4f}"
        print(line)
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "args": vars(args)},
                       frame_out if args.event_select else args.out)
        if ev is not None and ev[2] > best_ev:
            best_ev = ev[2]
            torch.save({"model": model.state_dict(), "args": vars(args)}, args.out)
    if args.event_select:
        print(f"best event-combined {best_ev:.4f} -> {args.out} | "
              f"best frame-combined {best:.3f} -> {frame_out}")
    else:
        print(f"best frame-combined {best:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
