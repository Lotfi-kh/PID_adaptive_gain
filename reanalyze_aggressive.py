#!/usr/bin/env python3
"""
Reanalyze aggressive eval NPZ files using the correct rate axis per condition.
  dist_axis=roll  -> measure roll_rates
  dist_axis=pitch -> measure pitch_rates
  dist_axis=both  -> measure sqrt(roll² + pitch²) combined angular rate
"""
import os
import numpy as np

NPZ_ROOT = "test_results/aggressive_eval"
OUT_TXT  = os.path.join(NPZ_ROOT, "aggressive_eval_corrected.txt")

INIT_NOISES = [0.05, 0.10, 0.15, 0.20]
DIST_MAGNITUDES = [
    ("low",     4.3e-2),
    ("medium",  1.7e-1),
    ("high",    3.0e-1),
    ("extreme", 4.0e-1),
]
DIST_AXES = ["roll", "pitch", "both"]


def load_rate(data, key_base):
    """Pick the right angular rate array based on dist_axis."""
    dist_axis = str(data["dist_axis"])
    if dist_axis == "roll":
        return data[f"{key_base}_roll_rates"]
    elif dist_axis == "pitch":
        return data[f"{key_base}_pitch_rates"]
    else:  # both, combined angular rate magnitude
        rr = data[f"{key_base}_roll_rates"]
        pr = data[f"{key_base}_pitch_rates"]
        return np.sqrt(rr**2 + pr**2)


def rms_val(arr_pad, ep_lens):
    vals = []
    for i in range(arr_pad.shape[0]):
        ep = arr_pad[i, :int(ep_lens[i])]
        ep = ep[~np.isnan(ep)]
        vals.extend(ep.tolist())
    sq = np.array(vals) ** 2
    return float(np.sqrt(sq.mean())) if sq.size else float("nan")


def peak_post(arr_pad, ep_lens, dist_start):
    peaks = []
    for i in range(arr_pad.shape[0]):
        end = int(ep_lens[i])
        if end > dist_start:
            w = arr_pad[i, dist_start:end]
            w = w[~np.isnan(w)]
            if w.size:
                peaks.append(float(np.max(np.abs(w))))
    return float(np.mean(peaks)) if peaks else float("nan")


def recovery(arr_pad, ep_lens, dist_end, thr=0.10, window=10):
    out = []
    for i in range(arr_pad.shape[0]):
        ep_len = int(ep_lens[i])
        seg = arr_pad[i, min(dist_end, ep_len):ep_len]
        consec, rec = 0, float("nan")
        for j, v in enumerate(seg):
            if not np.isnan(v) and abs(v) < thr:
                consec += 1
                if consec >= window:
                    rec = float(j - window + 1)
                    break
            else:
                consec = 0
        out.append(rec)
    arr = np.array(out, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(finite.mean()) if finite.size else float("nan")


def summarise(data):
    dist_end   = int(data["dist_step"]) + int(data["dist_duration"])
    dist_start = int(data["dist_step"])

    bl_rate = load_rate(data, "bl")
    rl_rate = load_rate(data, "rl")
    bl_lens = data["bl_ep_lens"]
    rl_lens = data["rl_ep_lens"]
    n = bl_rate.shape[0]
    bl_crashes = int(np.sum(data["bl_crashes"].astype(float)))
    rl_crashes = int(np.sum(data["rl_crashes"].astype(float)))

    bl_rms  = rms_val(bl_rate, bl_lens)
    rl_rms  = rms_val(rl_rate, rl_lens)
    bl_peak = peak_post(bl_rate, bl_lens, dist_start)
    rl_peak = peak_post(rl_rate, rl_lens, dist_start)
    bl_rec  = recovery(bl_rate, bl_lens, dist_end)
    rl_rec  = recovery(rl_rate, rl_lens, dist_end)

    def pct(bl, rl):
        return (bl - rl) / (abs(bl) + 1e-12) * 100.0

    peak_pct = pct(bl_peak, rl_peak)
    rms_pct  = pct(bl_rms,  rl_rms)
    rec_pct  = pct(bl_rec,  rl_rec) if not (np.isnan(bl_rec) or np.isnan(rl_rec)) else float("nan")

    winner = "RL"   if (rl_crashes <= bl_crashes and peak_pct >= 10.0
                        and (rl_rms - bl_rms) / (abs(bl_rms) + 1e-12) * 100 <= 5.0) else \
             "BASE" if (bl_crashes < rl_crashes or
                        ((rl_rms - bl_rms)/(abs(bl_rms)+1e-12)*100 > 10.0 and bl_peak <= rl_peak)) else \
             "MIXED"

    return {
        "n": n, "bl_crash": bl_crashes, "rl_crash": rl_crashes,
        "bl_rms": bl_rms, "rl_rms": rl_rms, "rms_pct": rms_pct,
        "bl_peak": bl_peak, "rl_peak": rl_peak, "peak_pct": peak_pct,
        "bl_rec": bl_rec, "rl_rec": rl_rec, "rec_pct": rec_pct,
        "winner": winner,
    }


def fmt_pct(v):
    if np.isnan(v): return "   nan  "
    return f"{v:+6.1f}%"


def main():
    rows = {}
    for noise in INIT_NOISES:
        for mag_label, mag_val in DIST_MAGNITUDES:
            for dist_axis in DIST_AXES:
                tag = f"noise{int(noise*100):02d}_{mag_label}_{dist_axis}"
                npz = os.path.join(NPZ_ROOT, tag, "disturbance_eval_results.npz")
                if not os.path.exists(npz):
                    rows[tag] = None
                    continue
                data = np.load(npz, allow_pickle=True)
                r = summarise(data)
                r.update({"noise": noise, "mag_label": mag_label,
                           "mag_val": mag_val, "dist_axis": dist_axis})
                rows[tag] = r

    sep = "-" * 112
    header = (f"{'Condition':<38}  {'Crash':^10}  {'RMS Δ%':>8}  "
              f"{'Peak Δ%':>8}  {'Rec Δ%':>8}  {'BL rec':>7}  {'RL rec':>7}  {'Winner':^7}")

    lines = []
    lines.append("=" * 112)
    lines.append("CORRECTED AGGRESSIVE EVAL, c860k  [rate axis matched to disturbance axis]")
    lines.append(f"roll/both -> roll_rate or combined;  pitch -> pitch_rate")
    lines.append("=" * 112)
    lines.append(header)
    lines.append(sep)

    rl_wins = base_wins = mixed = 0
    all_peak = []; all_rms = []; all_rec = []

    for tag, r in rows.items():
        if r is None:
            lines.append(f"  {tag:<36}  MISSING")
            continue
        crash_str = f"{r['bl_crash']}/{r['n']} vs {r['rl_crash']}/{r['n']}"
        rec_bl = f"{r['bl_rec']:.1f}" if not np.isnan(r['bl_rec']) else "  nan"
        rec_rl = f"{r['rl_rec']:.1f}" if not np.isnan(r['rl_rec']) else "  nan"
        lines.append(f"  {tag:<36}  {crash_str:^10}  {fmt_pct(r['rms_pct']):>8}  "
                     f"{fmt_pct(r['peak_pct']):>8}  {fmt_pct(r['rec_pct']):>8}  "
                     f"{rec_bl:>7}  {rec_rl:>7}  {r['winner']:^7}")
        if r['winner'] == 'RL':   rl_wins += 1
        elif r['winner'] == 'BASE': base_wins += 1
        else: mixed += 1
        all_peak.append(r['peak_pct'])
        all_rms.append(r['rms_pct'])
        if not np.isnan(r['rec_pct']): all_rec.append(r['rec_pct'])

    lines.append(sep)
    n_valid = rl_wins + base_wins + mixed
    lines.append(f"\nSUMMARY  ({n_valid} valid conditions)")
    lines.append(f"  RL wins : {rl_wins}   Baseline wins: {base_wins}   Mixed: {mixed}")
    if all_peak: lines.append(f"  Mean peak improvement : {np.nanmean(all_peak):+.1f}%")
    if all_rms:  lines.append(f"  Mean RMS  improvement : {np.nanmean(all_rms):+.1f}%")
    if all_rec:  lines.append(f"  Mean recovery improve : {np.nanmean(all_rec):+.1f}%")

    lines.append("\n--- Per disturbance axis ---")
    for axis in DIST_AXES:
        sub = [r for r in rows.values() if r and r.get("dist_axis") == axis]
        rls  = sum(1 for r in sub if r['winner'] == 'RL')
        peak_m = np.nanmean([r['peak_pct'] for r in sub])
        rec_m  = np.nanmean([r['rec_pct']  for r in sub if not np.isnan(r['rec_pct'])])
        lines.append(f"  {axis:6s}: RL wins {rls}/{len(sub)}  "
                     f"peak Δ {peak_m:+.1f}%  rec Δ {rec_m:+.1f}%")

    lines.append("\n--- Per disturbance magnitude ---")
    for mag_label, mag_val in DIST_MAGNITUDES:
        sub = [r for r in rows.values() if r and r.get("mag_label") == mag_label]
        rls = sum(1 for r in sub if r['winner'] == 'RL')
        peak_m = np.nanmean([r['peak_pct'] for r in sub])
        crash_rl = sum(r['rl_crash'] for r in sub)
        crash_bl = sum(r['bl_crash'] for r in sub)
        lines.append(f"  {mag_label:8s} ({mag_val:.2f} N.m): RL wins {rls}/{len(sub)}  "
                     f"peak Δ {peak_m:+.1f}%  crashes BL={crash_bl} RL={crash_rl}")

    lines.append("\n--- Per noise level ---")
    for noise in INIT_NOISES:
        sub = [r for r in rows.values() if r and r.get("noise") == noise]
        rls = sum(1 for r in sub if r['winner'] == 'RL')
        peak_m = np.nanmean([r['peak_pct'] for r in sub])
        rms_m  = np.nanmean([r['rms_pct']  for r in sub])
        lines.append(f"  noise={noise:.2f}: RL wins {rls}/{len(sub)}  "
                     f"peak Δ {peak_m:+.1f}%  RMS Δ {rms_m:+.1f}%")

    lines.append("=" * 112)

    report = "\n".join(lines)
    print(report)

    with open(OUT_TXT, "w") as f:
        f.write(report + "\n")
    print(f"\n[OK] Corrected report -> {OUT_TXT}")


if __name__ == "__main__":
    main()
