"""Deep-dive model quality evaluation.

Runs a RiffNet checkpoint against the cached dataset (EGDB + GOAT, any split)
and prints a detailed breakdown of failure modes:

 1. Overall + per-timbre onset/pitch F1 and Combined
 2. Train vs val vs test split aggregates
 3. Per-pitch-range accuracy (low / mid / high)
 4. Polyphony analysis (mono vs chords vs rapid passages)
 5. Onset timing error distribution (signed, shows early/late bias)
 6. Octave confusion matrix (GT note vs detected +/-12/+/-24)
 7. Per-note pitch recall
 8. False positive analysis (what pitches get hallucinated)
 9. Confidence calibration (model confidence vs actual accuracy)
10. Worst-performing clips (for manual inspection)
11. Inter-onset-interval failure rate (rapid passages)

Output: printed tables + an optional JSON summary.
"""
import argparse
import os
import json
import glob
import numpy as np
import torch
from collections import defaultdict

from model import RiffNet, load_model, TIME_CONTEXT as PAD
from cqt import build_hcqt
import dataset as ds
from decode import decode, group_events
from .build_dataset import parse_notes

SR, MIDI_LOW, NUM_PITCH = 44100, 36, 55
NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Default decode config (recall-balanced; matches train.py --event-select).
DECODE_CFG = dict(th_on=0.6, th_pi=0.50, refr=6, win=16, topk=6,
                  peak=True, agg="mean", offset=6, nms=False)


def midi_name(m):
    return NAMES[m % 12] + str(m // 12 - 1)


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


def score_file(gt_events, dets, tol=0.05):
    """Extended scoring that returns per-event match details."""
    gt_m = [False] * len(gt_events)
    det_m = [False] * len(dets)
    matches = []  # per-GT-event match detail

    for gi, (gt, gp) in enumerate(gt_events):
        best, bd = None, tol
        for di, (dt, _) in enumerate(dets):
            if det_m[di]:
                continue
            d = abs(dt - gt)
            if d <= tol and d < bd:
                bd, best = d, di
        if best is not None:
            gt_m[gi] = det_m[best] = True
            dt, dp = dets[best]
            matches.append({
                'gt_time': gt, 'det_time': dt, 'time_error': dt - gt,
                'gt_pitches': gp, 'det_pitches': dp,
                'correct_pitches': [p for p in gp if p in dp],
                'missed_pitches': [p for p in gp if p not in dp],
                'extra_pitches': [p for p in dp if p not in gp],
                'octave_errors': [p for p in gp if p not in dp and (p+12 in dp or p-12 in dp)],
            })
        else:
            matches.append({
                'gt_time': gt, 'det_time': None, 'time_error': None,
                'gt_pitches': gp, 'det_pitches': [],
                'correct_pitches': [], 'missed_pitches': gp, 'extra_pitches': [],
                'octave_errors': [],
            })

    false_positives = [(dets[di][0], dets[di][1]) for di in range(len(dets)) if not det_m[di]]

    tp = sum(1 for m in matches if m['det_time'] is not None)
    fn = sum(1 for m in matches if m['det_time'] is None)
    fp = len(false_positives)
    o_p = tp / (tp + fp + 1e-9)
    o_r = tp / (tp + fn + 1e-9)
    o_f = 2 * o_p * o_r / (o_p + o_r + 1e-9)

    tot_gt = sum(len(m['gt_pitches']) for m in matches)
    tot_det = sum(len(m['det_pitches']) for m in matches if m['det_time'] is not None)
    cor = sum(len(m['correct_pitches']) for m in matches)
    p_p = cor / (tot_det + 1e-9)
    p_r = cor / (tot_gt + 1e-9)
    p_f = 2 * p_p * p_r / (p_p + p_r + 1e-9)

    return {
        'onset_f1': o_f, 'onset_prec': o_p, 'onset_rec': o_r,
        'pitch_f1': p_f, 'pitch_prec': p_p, 'pitch_rec': p_r,
        'combined': o_f * p_f,
        'matches': matches, 'false_positives': false_positives,
        'tp': tp, 'fp': fp, 'fn': fn,
        'pitch_correct': cor, 'pitch_total_gt': tot_gt, 'pitch_total_det': tot_det,
    }


@torch.no_grad()
def run_full_eval(model, files, device, decode_cfg, hop):
    """Run model on all files and collect detailed per-file results."""
    results = []
    for f in files:
        d = np.load(f)
        cqt = np.pad(d["cqt"].astype(np.float32), ((PAD, 0), (0, 0)))
        mag = torch.from_numpy(build_hcqt(cqt)[None]).to(device)        # (1, 6, T, 109)
        on_l, pi_l = model(mag)
        onset_p = torch.sigmoid(on_l)[0].cpu().numpy()
        pitch_p = torch.sigmoid(pi_l)[0].cpu().numpy()

        clip_str = os.path.splitext(os.path.basename(f))[0]
        clip = int(clip_str.split("_")[-1]) if clip_str.startswith("item_") else int(clip_str)
        midi_path = midi_path_for(f)
        gt_notes = parse_notes(midi_path) if (midi_path and os.path.exists(midi_path)) else []
        gt_events = group_events(gt_notes)

        dets = decode(onset_p, pitch_p, **decode_cfg, hop=hop)
        detail = score_file(gt_events, dets)
        detail['file'] = f
        detail['timbre'] = os.path.basename(os.path.dirname(f))
        detail['clip'] = clip
        detail['onset_prob'] = onset_p
        detail['pitch_prob'] = pitch_p
        detail['gt_notes'] = gt_notes
        detail['n_gt_events'] = len(gt_events)
        detail['n_det_events'] = len(dets)
        results.append(detail)
    return results


def analyze_per_timbre(results):
    print("\n" + "="*80)
    print("SECTION 1: PER-TIMBRE BREAKDOWN")
    print("="*80)
    timbres = sorted(set(r['timbre'] for r in results))
    header = f"{'Timbre':10s} {'Files':>5s} {'OnF1':>6s} {'OnP':>5s} {'OnR':>5s}  {'PiF1':>6s} {'PiP':>5s} {'PiR':>5s}  {'Comb':>7s}"
    print(header)
    print("-" * len(header))
    for t in timbres:
        sub = [r for r in results if r['timbre'] == t]
        n = len(sub)
        of = np.mean([r['onset_f1'] for r in sub])
        op = np.mean([r['onset_prec'] for r in sub])
        orc = np.mean([r['onset_rec'] for r in sub])
        pf = np.mean([r['pitch_f1'] for r in sub])
        pp = np.mean([r['pitch_prec'] for r in sub])
        pr = np.mean([r['pitch_rec'] for r in sub])
        c = np.mean([r['combined'] for r in sub])
        print(f"{t:10s} {n:5d} {of:6.1%} {op:5.1%} {orc:5.1%}  {pf:6.1%} {pp:5.1%} {pr:5.1%}  {c:7.4f}")


def analyze_per_pitch_range(results):
    print("\n" + "="*80)
    print("SECTION 3: PITCH ACCURACY BY MIDI RANGE")
    print("="*80)
    ranges = [
        ("Low  (C2-B2,  36-47)", 36, 47),
        ("Mid-L(C3-B3,  48-59)", 48, 59),
        ("Mid-H(C4-B4,  60-71)", 60, 71),
        ("High (C5-F#6, 72-90)", 72, 90),
    ]
    for label, lo, hi in ranges:
        correct = total_gt = total_det = octave_err = 0
        for r in results:
            for m in r['matches']:
                for p in m['gt_pitches']:
                    if lo <= p <= hi:
                        total_gt += 1
                        if p in m['correct_pitches']:
                            correct += 1
                        elif p in m['octave_errors']:
                            octave_err += 1
                for p in m['extra_pitches']:
                    if lo <= p <= hi:
                        total_det += 1
        recall = correct / (total_gt + 1e-9)
        oct_rate = octave_err / (total_gt + 1e-9)
        fp_rate = total_det / (total_gt + 1e-9)
        print(f"  {label}: recall={recall:5.1%} | octave-err={oct_rate:5.1%} | FP-notes={total_det:4d} (rate={fp_rate:5.1%}) | n={total_gt}")


def analyze_polyphony(results):
    print("\n" + "="*80)
    print("SECTION 4: POLYPHONY ANALYSIS")
    print("="*80)
    poly_stats = defaultdict(lambda: {'correct': 0, 'total_gt': 0, 'events_matched': 0, 'events_total': 0})
    for r in results:
        for m in r['matches']:
            n_notes = len(m['gt_pitches'])
            key = min(n_notes, 4)
            poly_stats[key]['total_gt'] += n_notes
            poly_stats[key]['correct'] += len(m['correct_pitches'])
            poly_stats[key]['events_total'] += 1
            if m['det_time'] is not None:
                poly_stats[key]['events_matched'] += 1
    print(f"  {'Polyphony':12s} {'Events':>7s} {'OnsetDet':>9s} {'PiRecall':>9s} {'Notes':>6s}")
    print("  " + "-" * 50)
    for k in sorted(poly_stats.keys()):
        s = poly_stats[k]
        label = f"{k} notes" if k < 4 else "4+ notes"
        det_rate = s['events_matched'] / (s['events_total'] + 1e-9)
        recall = s['correct'] / (s['total_gt'] + 1e-9)
        print(f"  {label:12s} {s['events_total']:7d} {det_rate:9.1%} {recall:9.1%} {s['total_gt']:6d}")


def analyze_onset_timing(results):
    print("\n" + "="*80)
    print("SECTION 5: ONSET TIMING ERROR DISTRIBUTION")
    print("="*80)
    errors = [m['time_error'] for r in results for m in r['matches'] if m['time_error'] is not None]
    errors = np.array(errors)
    if len(errors) == 0:
        print("  No matched events found.")
        return
    print(f"  Total matched onsets:  {len(errors)}")
    print(f"  Mean time error:       {errors.mean()*1000:+.1f} ms (+ = late)")
    print(f"  Median time error:     {np.median(errors)*1000:+.1f} ms")
    print(f"  Std dev:               {errors.std()*1000:.1f} ms")
    for tol in (0.01, 0.02, 0.03, 0.05):
        print(f"  Within +/-{int(tol*1000):2d}ms:          {np.mean(np.abs(errors) <= tol):.1%}")
    bins = np.linspace(-0.05, 0.05, 21)
    counts, _ = np.histogram(errors, bins)
    print("\n  Timing error histogram (ms):")
    max_c = max(counts) if max(counts) > 0 else 1
    for i in range(len(counts)):
        bar = "#" * int(40 * counts[i] / max_c)
        print(f"  [{bins[i]*1000:+5.0f},{bins[i+1]*1000:+5.0f}) {counts[i]:5d} {bar}")


def analyze_octave_confusion(results):
    print("\n" + "="*80)
    print("SECTION 6: OCTAVE CONFUSION ANALYSIS")
    print("="*80)
    confusion = defaultdict(int)
    for r in results:
        for m in r['matches']:
            if m['det_time'] is None:
                continue
            dp = set(m['det_pitches'])
            for p in m['gt_pitches']:
                if p in dp:
                    confusion[(p // 12, 'correct')] += 1
                elif p + 12 in dp:
                    confusion[(p // 12, '+1oct')] += 1
                elif p - 12 in dp:
                    confusion[(p // 12, '-1oct')] += 1
                elif p + 24 in dp:
                    confusion[(p // 12, '+2oct')] += 1
                elif p - 24 in dp:
                    confusion[(p // 12, '-2oct')] += 1
                else:
                    confusion[(p // 12, 'other_miss')] += 1
    octaves = sorted(set(k[0] for k in confusion.keys()))
    types = ['correct', '+1oct', '-1oct', '+2oct', '-2oct', 'other_miss']
    print(f"  {'Octave':>8s}", end="")
    for t in types:
        print(f" {t:>10s}", end="")
    print(f" {'Total':>8s}  {'ErrRate':>8s}")
    print("  " + "-" * 80)
    for o in octaves:
        total = sum(confusion.get((o, t), 0) for t in types)
        corr = confusion.get((o, 'correct'), 0)
        err_rate = 1 - corr / (total + 1e-9)
        print(f"  Oct{o:2d}   ", end="")
        for t in types:
            print(f" {confusion.get((o, t), 0):10d}", end="")
        print(f" {total:8d}  {err_rate:8.1%}")


def analyze_per_note_recall(results):
    print("\n" + "="*80)
    print("SECTION 7: PER-NOTE PITCH RECALL (MIDI 36-90)")
    print("="*80)
    note_correct = np.zeros(NUM_PITCH)
    note_total = np.zeros(NUM_PITCH)
    note_octave_err = np.zeros(NUM_PITCH)
    for r in results:
        for m in r['matches']:
            if m['det_time'] is None:
                continue
            dp = set(m['det_pitches'])
            for p in m['gt_pitches']:
                if MIDI_LOW <= p <= MIDI_LOW + NUM_PITCH - 1:
                    idx = p - MIDI_LOW
                    note_total[idx] += 1
                    if p in dp:
                        note_correct[idx] += 1
                    elif (p + 12 in dp) or (p - 12 in dp):
                        note_octave_err[idx] += 1
    print(f"  {'Note':>5s}  {'MIDI':>4s}  {'Total':>6s}  {'Recall':>7s}  {'OctErr':>7s}  Bar")
    print("  " + "-" * 60)
    for i in range(NUM_PITCH):
        total = int(note_total[i])
        if total == 0:
            continue
        recall = note_correct[i] / total
        oct_err = note_octave_err[i] / total
        bar = "#" * int(30 * recall) + "." * int(30 * oct_err)
        print(f"  {midi_name(i + MIDI_LOW):>5s}  {i + MIDI_LOW:4d}  {total:6d}  {recall:7.1%}  {oct_err:7.1%}  {bar}")


def analyze_false_positives(results):
    print("\n" + "="*80)
    print("SECTION 8: FALSE POSITIVE ANALYSIS")
    print("="*80)
    fp_pitch_counts = defaultdict(int)
    total_fps = 0
    for r in results:
        for t, pitches in r['false_positives']:
            for p in pitches:
                fp_pitch_counts[p] += 1
                total_fps += 1
    extra_pitch_counts = defaultdict(int)
    total_extras = 0
    for r in results:
        for m in r['matches']:
            for p in m['extra_pitches']:
                extra_pitch_counts[p] += 1
                total_extras += 1
    print(f"\n  A) Phantom onsets (no GT match): {sum(r['fp'] for r in results)} events, {total_fps} total notes")
    for p, c in sorted(fp_pitch_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"     {midi_name(p):>5s} (MIDI {p:2d}): {c:5d} occurrences")
    print(f"\n  B) Wrong pitches on matched onsets: {total_extras} extra notes")
    for p, c in sorted(extra_pitch_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"     {midi_name(p):>5s} (MIDI {p:2d}): {c:5d} occurrences")


def analyze_confidence(results, hop):
    print("\n" + "="*80)
    print("SECTION 9: CONFIDENCE CALIBRATION")
    print("="*80)
    gt_onset_probs, fp_onset_probs = [], []
    for r in results:
        onset_p = r['onset_prob']
        for m in r['matches']:
            if m['det_time'] is not None:
                f = max(0, int(round(m['det_time'] * SR / hop)) - 1)
                if f < len(onset_p):
                    gt_onset_probs.append(onset_p[f])
        for t, _ in r['false_positives']:
            f = max(0, int(round(t * SR / hop)) - 1)
            if f < len(onset_p):
                fp_onset_probs.append(onset_p[f])
    gt_onset_probs = np.array(gt_onset_probs) if gt_onset_probs else np.array([0.0])
    print(f"  Onset confidence at TRUE onsets:  mean={np.mean(gt_onset_probs):.3f} median={np.median(gt_onset_probs):.3f} (n={len(gt_onset_probs)})")
    if fp_onset_probs:
        fp_onset_probs = np.array(fp_onset_probs)
        print(f"  Onset confidence at FALSE pos:   mean={np.mean(fp_onset_probs):.3f} median={np.median(fp_onset_probs):.3f} (n={len(fp_onset_probs)})")
    else:
        print("  Onset confidence at FALSE pos:   (no false positives)")

    correct_pitch_probs, wrong_pitch_probs = [], []
    for r in results:
        pitch_p = r['pitch_prob']
        for m in r['matches']:
            if m['det_time'] is None:
                continue
            f = max(0, int(round(m['det_time'] * SR / hop)) - 1)
            s0 = min(f + 4, len(pitch_p))
            seg = pitch_p[s0:min(s0 + 12, len(pitch_p))]
            if len(seg) == 0:
                seg = pitch_p[f:f+1]
            v = seg.mean(axis=0) if len(seg) > 0 else np.zeros(NUM_PITCH)
            for p in m['correct_pitches']:
                idx = p - MIDI_LOW
                if 0 <= idx < NUM_PITCH:
                    correct_pitch_probs.append(v[idx])
            for p in m['missed_pitches']:
                idx = p - MIDI_LOW
                if 0 <= idx < NUM_PITCH:
                    wrong_pitch_probs.append(v[idx])
    if correct_pitch_probs:
        cpp = np.array(correct_pitch_probs)
        print(f"  Pitch confidence for CORRECT notes: mean={cpp.mean():.3f} median={np.median(cpp):.3f}")
    if wrong_pitch_probs:
        wpp = np.array(wrong_pitch_probs)
        print(f"  Pitch confidence for MISSED  notes: mean={wpp.mean():.3f} median={np.median(wpp):.3f}")
        if correct_pitch_probs:
            print(f"  Confidence gap (correct - missed): {cpp.mean() - wpp.mean():+.3f}")


def analyze_worst_clips(results, n=15):
    print("\n" + "="*80)
    print(f"SECTION 10: WORST {n} CLIPS (by combined score)")
    print("="*80)
    sorted_r = sorted(results, key=lambda r: r['combined'])
    print(f"  {'Timbre':10s} {'Clip':>5s} {'Comb':>7s} {'OnF1':>6s} {'PiF1':>6s} {'GT':>4s} {'Det':>4s} {'FP':>4s} {'FN':>4s}")
    print("  " + "-" * 65)
    for r in sorted_r[:n]:
        print(f"  {r['timbre']:10s} {r['clip']:5d} {r['combined']:7.4f} "
              f"{r['onset_f1']:6.1%} {r['pitch_f1']:6.1%} "
              f"{r['n_gt_events']:4d} {r['n_det_events']:4d} {r['fp']:4d} {r['fn']:4d}")


def analyze_splits(results):
    print("\n" + "="*80)
    print("SECTION 2: TRAIN vs VAL vs TEST SPLITS")
    print("="*80)
    train_timbres = set(ds.TRAIN_TIMBRES)
    test_timbres = set(ds.TEST_TIMBRES)
    splits = {
        'Train (seen clips+timbres)': [],
        'Val   (unseen clips, train timbres)': [],
        'Test  (unseen clips+timbres)': [],
    }
    for r in results:
        clip, timbre = r['clip'], r['timbre']
        if timbre in train_timbres and clip <= ds.TRAIN_CLIP_MAX:
            splits['Train (seen clips+timbres)'].append(r)
        elif timbre in train_timbres:
            splits['Val   (unseen clips, train timbres)'].append(r)
        elif timbre in test_timbres:
            splits['Test  (unseen clips+timbres)'].append(r)
    for name, sub in splits.items():
        if not sub:
            continue
        of = np.mean([r['onset_f1'] for r in sub])
        pf = np.mean([r['pitch_f1'] for r in sub])
        c = np.mean([r['combined'] for r in sub])
        print(f"  {name:42s}: OnF1={of:.1%}  PiF1={pf:.1%}  Comb={c:.4f}  (n={len(sub)})")


def analyze_ioi(results):
    print("\n" + "="*80)
    print("SECTION 11: INTER-ONSET INTERVAL (IOI) FAILURE RATE")
    print("="*80)
    ioi_buckets = {
        '<50ms': (0, 0.05), '50-100ms': (0.05, 0.10), '100-200ms': (0.10, 0.20),
        '200-500ms': (0.20, 0.50), '>500ms': (0.50, 10.0),
    }
    bucket_stats = {k: {'total': 0, 'detected': 0} for k in ioi_buckets}
    for r in results:
        for i, m in enumerate(r['matches']):
            ioi = (m['gt_time'] - r['matches'][i-1]['gt_time']) if i > 0 else 10.0
            for label, (lo, hi) in ioi_buckets.items():
                if lo <= ioi < hi:
                    bucket_stats[label]['total'] += 1
                    if m['det_time'] is not None:
                        bucket_stats[label]['detected'] += 1
                    break
    print(f"  {'IOI bucket':>14s}  {'Total':>7s}  {'Detected':>9s}  {'DetRate':>8s}")
    print("  " + "-" * 45)
    for label in ioi_buckets:
        s = bucket_stats[label]
        rate = s['detected'] / (s['total'] + 1e-9)
        print(f"  {label:>14s}  {s['total']:7d}  {s['detected']:9d}  {rate:8.1%}")


def overall_summary(results):
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    of = np.mean([r['onset_f1'] for r in results])
    op = np.mean([r['onset_prec'] for r in results])
    orc = np.mean([r['onset_rec'] for r in results])
    pf = np.mean([r['pitch_f1'] for r in results])
    pp = np.mean([r['pitch_prec'] for r in results])
    pr = np.mean([r['pitch_rec'] for r in results])
    c = np.mean([r['combined'] for r in results])
    print(f"  Files evaluated: {len(results)}")
    print(f"  Onset F1:  {of:.1%}  (P={op:.1%}, R={orc:.1%})")
    print(f"  Pitch F1:  {pf:.1%}  (P={pp:.1%}, R={pr:.1%})")
    print(f"  Combined:  {c:.4f}")
    print(f"  GT events: {sum(r['n_gt_events'] for r in results)}  "
          f"Detections: {sum(r['n_det_events'] for r in results)}")
    print(f"  TP: {sum(r['tp'] for r in results)}  "
          f"FP: {sum(r['fp'] for r in results)}  FN: {sum(r['fn'] for r in results)}")


def main():
    ap = argparse.ArgumentParser(description="Deep-dive model evaluation")
    ap.add_argument("--ckpt", default="riffnet.pt", help="model checkpoint")
    ap.add_argument("--data", default="data/hop256", help="cached dataset dir")
    ap.add_argument("--hop", type=int, default=256, help="hop size in samples")
    ap.add_argument("--splits", default="all", choices=["all", "val", "test", "train"],
                    help="which data split(s) to evaluate")
    ap.add_argument("--timbres", default="", help="comma-separated timbres (default: all)")
    ap.add_argument("--json", default="", help="output JSON summary path")
    ap.add_argument("--th_pi", type=float, default=None, help="override decode pitch threshold")
    ap.add_argument("--th_on", type=float, default=None, help="override decode onset threshold")
    args = ap.parse_args()
    if args.th_pi is not None:
        DECODE_CFG["th_pi"] = args.th_pi
        print(f"decode th_pi override: {args.th_pi}")
    if args.th_on is not None:
        DECODE_CFG["th_on"] = args.th_on
        print(f"decode th_on override: {args.th_on}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\nCheckpoint: {args.ckpt}")

    ckpt_data = torch.load(args.ckpt, map_location=device)
    model = load_model(ckpt_data).to(device)
    model.eval()
    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e3:.0f}K")

    if args.splits == "all":
        timbres = args.timbres.split(",") if args.timbres else (ds.TRAIN_TIMBRES + ds.TEST_TIMBRES)
        files = []
        for t in timbres:
            files.extend(sorted(glob.glob(os.path.join(args.data, t, "*.npz"))))
        files = [f for f in files if not ds.is_quarantined(f)]
    else:
        train_f, val_f, test_f = ds.split_clips(args.data)
        files = {"train": train_f, "val": val_f, "test": test_f}[args.splits]

    print(f"Evaluating {len(files)} clips...\n")
    results = run_full_eval(model, files, device, DECODE_CFG, args.hop)

    overall_summary(results)
    analyze_per_timbre(results)
    analyze_splits(results)
    analyze_per_pitch_range(results)
    analyze_polyphony(results)
    analyze_onset_timing(results)
    analyze_octave_confusion(results)
    analyze_per_note_recall(results)
    analyze_false_positives(results)
    analyze_confidence(results, args.hop)
    analyze_worst_clips(results)
    analyze_ioi(results)

    if args.json:
        summary = {
            'checkpoint': args.ckpt,
            'n_files': len(files),
            'overall': {
                'onset_f1': float(np.mean([r['onset_f1'] for r in results])),
                'pitch_f1': float(np.mean([r['pitch_f1'] for r in results])),
                'combined': float(np.mean([r['combined'] for r in results])),
            },
            'per_timbre': {},
        }
        for t in sorted(set(r['timbre'] for r in results)):
            sub = [r for r in results if r['timbre'] == t]
            summary['per_timbre'][t] = {
                'onset_f1': float(np.mean([r['onset_f1'] for r in sub])),
                'pitch_f1': float(np.mean([r['pitch_f1'] for r in sub])),
                'combined': float(np.mean([r['combined'] for r in sub])),
            }
        with open(args.json, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nJSON summary saved to {args.json}")


if __name__ == "__main__":
    main()
