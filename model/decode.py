"""Shared decode + scoring for offline evaluation and in-training validation:
group ground-truth notes into events, peak-pick onsets, aggregate pitch over a
window, optional octave-NMS / harmonic-sieve, and score onset/pitch F1.

Imported by train.py (in-training event scoring) and eval.py (offline eval).
"""
import numpy as np

SR, HOP, MIDI_LOW = 44100, 256, 36  # C2 floor: downtuned guitar (Drop D/C)


def group_events(notes, tol=0.03):
    """(midi, start, end) notes -> [(time, [midi...])] grouped by onset proximity."""
    notes = sorted(notes, key=lambda n: n[1])
    if not notes:
        return []
    events = []
    cur_t, cur_p = notes[0][1], [notes[0][0]]
    for midi, s, _ in notes[1:]:
        if abs(s - cur_t) < tol:
            if midi not in cur_p:
                cur_p.append(midi)
        else:
            events.append((cur_t, cur_p))
            cur_t, cur_p = s, [midi]
    events.append((cur_t, cur_p))
    return events


def octave_nms(cands, ratio=1.6):
    """cands: list of (class, prob) sorted desc. Drop a note if a stronger note
    sits an octave (12/24) below/above and is >= ratio x stronger."""
    keep = []
    for c, p in cands:
        drop = False
        for c2, p2 in keep:
            if abs(c - c2) in (12, 24) and p2 >= ratio * p:
                drop = True
                break
        if not drop:
            keep.append((c, p))
    return keep


# Harmonic intervals (semitones) above a fundamental to treat as ghosts:
# 12th(3f)=19, 2-oct(4f)=24, 17th(5f)=28, 19th(6f)=31. The OCTAVE (+12, 2f) is
# deliberately EXCLUDED — on dense polyphony it's frequently a real chord tone
# (verified on GOAT: including +12 dropped 3-note recall 0.842->0.804), whereas
# +19/+24/+28/+31 are almost never real chord tones (they're the loud-note
# harmonic ghosts HCQT introduces). Mirrors HARMONIC_INTERVALS in model.rs.
HARMONIC_INTERVALS = (19, 24, 28, 31)


def harmonic_sieve(cands, ratio=1.3):
    """cands sorted desc by prob. Drop a note that is a harmonic interval ABOVE a
    kept note that is >= ratio x stronger (so genuine equal-strength chord tones
    survive, but weaker harmonic ghosts are removed)."""
    keep = []
    for c, p in cands:
        drop = False
        for c2, p2 in keep:
            if c > c2 and (c - c2) in HARMONIC_INTERVALS and p2 >= ratio * p:
                drop = True
                break
        if not drop:
            keep.append((c, p))
    return keep


def decode(onset_p, pitch_p, th_on, th_pi, refr, win, topk,
           agg="mean", peak=True, nms=False, nms_ratio=1.6, offset=0, hop=256,
           sieve=False, sieve_ratio=1.3, midi_low=MIDI_LOW):
    T = len(onset_p)
    dets = []
    last = -10 ** 9
    for f in range(1, T - 1):
        if peak:
            fire = (onset_p[f] > th_on and onset_p[f] >= onset_p[f - 1]
                    and onset_p[f] > onset_p[f + 1] and f - last >= refr)
        else:
            fire = (onset_p[f] > th_on and onset_p[f - 1] <= th_on and f - last >= refr)
        if not fire:
            continue
        last = f
        s0 = min(f + offset, T)
        seg = pitch_p[s0:min(s0 + win, T)]
        if len(seg) == 0:
            seg = pitch_p[f:f + 1]
        v = seg.max(axis=0) if agg == "max" else seg.mean(axis=0)
        v = v.copy()
        cand = []
        for c in range(len(v)):
            if v[c] > th_pi:
                cand.append((c, v[c]))
        cand.sort(key=lambda x: -x[1])
        if nms:
            cand = octave_nms(cand, nms_ratio)
        if sieve:
            cand = harmonic_sieve(cand, sieve_ratio)
        cand = cand[:topk]
        dets.append(((f + 1) * hop / SR, [c + midi_low for c, _ in cand]))
    return dets


def score(gt, dets, tol=0.05):
    gt_m = [False] * len(gt); det_m = [False] * len(dets)
    tp = fn = 0; tot_gt = tot_det = cor = 0
    for gi, (gtt, gp) in enumerate(gt):
        tot_gt += len(gp)
        best, bd = None, tol
        for di, (dt, _) in enumerate(dets):
            if det_m[di]:
                continue
            dd = abs(dt - gtt)
            if dd <= tol and dd < bd:
                bd, best = dd, di
        if best is not None:
            gt_m[gi] = det_m[best] = True; tp += 1
            dp = dets[best][1]; tot_det += len(dp)
            cor += sum(1 for p in gp if p in dp)
        else:
            fn += 1
    fp = det_m.count(False)
    op = tp / (tp + fp + 1e-9); orr = tp / (tp + fn + 1e-9)
    of = 2 * op * orr / (op + orr + 1e-9)
    pp = cor / (tot_det + 1e-9); pr = cor / (tot_gt + 1e-9)
    pf = 2 * pp * pr / (pp + pr + 1e-9)
    return np.array([op, orr, of, pp, pr, pf, of * pf])


def evaluate(data, **kw):
    accs = np.zeros(7)
    for onset_p, pitch_p, gt in data:
        dets = decode(onset_p, pitch_p, **kw)
        accs += score(gt, dets)
    return accs / len(data)
