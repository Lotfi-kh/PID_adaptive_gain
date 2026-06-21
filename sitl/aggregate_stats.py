#!/usr/bin/env python3
"""
aggregate_stats.py — turn the per-run metrics into a statistical verdict.

Reads the master CSV written by sitl_disturb_batch.py (one row per flight,
tagged tag/seed/scale/controller). For the seeded paired batch (tag=pair) it
pairs baseline vs adaptive BY SEED and reports, per metric:

  baseline mean ± 95% CI,  adaptive mean ± 95% CI,  improvement %,
  paired Wilcoxon signed-rank p,  paired Cohen's d,  win-rate.

This is what moves the claim from "looks better in one flight" to a tested
result, echoing the eval-suite's win-count framing.

For the bundled sweep (tag=sweep) it plots a chosen metric vs disturbance
magnitude for both controllers, mapping where adaptive helps and where the
recoverable envelope ends.

Outputs (next to the input CSV):
  stats_summary.csv, stats_summary.png, sweep_curve.png

Usage:
    python sitl/aggregate_stats.py test_results/gazebo/metrics_master.csv
"""
import argparse
import csv
import os

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

META = ["tag", "seed", "scale", "controller", "ulog"]

# lower-is-better performance metrics that get the full paired test
PERF = ["peak|roll| deg", "peak|pitch| deg", "peak|rollrate|", "peak|pitchrate|",
        "RMS roll deg", "RMS pitch deg", "RMS rollrate", "RMS pitchrate",
        "mean recovery s", "max recovery s",
        "IAE rate_roll", "IAE rate_pitch", "RMS rate_err",
        "alt_dev m", "peak|vz|"]
# cost metrics — reported the same way but flagged (improvement must be "free")
COST = ["effort_roll Nms", "effort_pitch Nms", "peak|tau_roll|", "peak|tau_pitch|",
        "motor_sat %"]
# headline metrics for the bar figure
HEADLINE = ["peak|roll| deg", "peak|pitch| deg", "RMS rate_err", "mean recovery s"]


def load(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit(f"[stats] {path} is empty")
    return rows


def fval(row, k):
    try:
        return float(row[k])
    except (KeyError, ValueError):
        return np.nan


def ci95(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan
    m = x.mean()
    if n == 1:
        return m, np.nan
    sem = x.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * sem
    return m, h


def paired(base, adap):
    """base, adap: arrays aligned by seed. Lower is better."""
    b, a = np.asarray(base, float), np.asarray(adap, float)
    ok = ~(np.isnan(b) | np.isnan(a))
    b, a = b[ok], a[ok]
    n = len(b)
    res = {"n": n, "p": np.nan, "d": np.nan, "win": np.nan}
    if n == 0:
        return res
    diff = b - a                      # >0 ⇒ adaptive lower ⇒ better
    res["win"] = float(np.mean(diff > 0))
    if n >= 2 and np.any(diff != 0):
        sd = diff.std(ddof=1)
        res["d"] = float(diff.mean() / sd) if sd > 0 else np.nan
        try:
            res["p"] = float(stats.wilcoxon(b, a).pvalue)
        except ValueError:
            res["p"] = np.nan
    return res


def pair_by_seed(rows, tag):
    """Return {seed: {controller: row}} for the given tag."""
    out = {}
    for r in rows:
        if r["tag"] != tag:
            continue
        out.setdefault(r["seed"], {})[r["controller"]] = r
    return {s: d for s, d in out.items() if "baseline" in d and "adaptive" in d}


def summarize(rows, metrics_present, out_csv):
    pairs = pair_by_seed(rows, "pair")
    seeds = sorted(pairs, key=lambda s: int(s))
    print(f"\n[stats] {len(seeds)} matched pairs in the seeded batch")
    if not seeds:
        print("[stats] no complete pairs — nothing to test")
        return [], []

    table = [["metric", "kind", "baseline_mean", "baseline_ci95",
              "adaptive_mean", "adaptive_ci95", "improvement_%",
              "paired_p", "cohens_d", "win_rate", "n"]]
    hdr = f"  {'metric':<18}{'baseline':>12}{'adaptive':>12}{'impr%':>8}{'p':>9}{'d':>7}{'win':>6}"
    print(hdr); print("  " + "-" * len(hdr))

    for kind, group in (("perf", PERF), ("cost", COST)):
        for k in group:
            if k not in metrics_present:
                continue
            bvals = [fval(pairs[s]["baseline"], k) for s in seeds]
            avals = [fval(pairs[s]["adaptive"], k) for s in seeds]
            bm, bh = ci95(bvals); am, ah = ci95(avals)
            pr = paired(bvals, avals)
            imp = (bm - am) / bm * 100 if bm not in (0.0,) and not np.isnan(bm) else np.nan
            tagk = "" if kind == "perf" else "(cost)"
            print(f"  {k:<18}{bm:>12.4g}{am:>12.4g}{imp:>+8.1f}"
                  f"{pr['p']:>9.3g}{pr['d']:>7.2f}{pr['win']:>6.2f} {tagk}")
            table.append([k, kind, f"{bm:.6g}", f"{bh:.6g}", f"{am:.6g}", f"{ah:.6g}",
                          f"{imp:+.2f}", f"{pr['p']:.4g}", f"{pr['d']:.3f}",
                          f"{pr['win']:.3f}", pr["n"]])

    # crash counts (safety)
    if "crash" in metrics_present:
        bc = sum(int(fval(pairs[s]["baseline"], "crash") > 0.5) for s in seeds)
        ac = sum(int(fval(pairs[s]["adaptive"], "crash") > 0.5) for s in seeds)
        print(f"\n  crashes:  baseline {bc}/{len(seeds)}   adaptive {ac}/{len(seeds)}")
        table.append(["crash_count", "safety", bc, "", ac, "", "", "", "", "",
                      len(seeds)])

    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(table)
    print(f"\n[stats] summary → {out_csv}")
    return seeds, pairs


def bar_figure(seeds, pairs, metrics_present, out_png):
    keys = [k for k in HEADLINE if k in metrics_present]
    if not keys or not seeds:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.8))
    if len(keys) == 1:
        axes = [axes]
    for ax, k in zip(axes, keys):
        bvals = [fval(pairs[s]["baseline"], k) for s in seeds]
        avals = [fval(pairs[s]["adaptive"], k) for s in seeds]
        bm, bh = ci95(bvals); am, ah = ci95(avals)
        ax.bar([0, 1], [bm, am],
               yerr=[[0, 0], [bh if bh == bh else 0, ah if ah == ah else 0]],
               color=["gray", "C0"], capsize=5, alpha=0.85)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["baseline", "adaptive"])
        ax.set_title(k, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Baseline vs adaptive — {len(seeds)} matched pairs "
                 f"(mean ± 95% CI), SITL/Gazebo F450", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print(f"[stats] bar figure → {out_png}")


def _driven_peak(r):
    """Peak attitude deviation on the axis the kick actually drove (deg)."""
    tx, ty = abs(fval(r, "torque_x")), abs(fval(r, "torque_y"))
    pr, pp = fval(r, "peak_roll_deg"), fval(r, "peak_pitch_deg")
    if tx > 0 and ty > 0:
        return max(pr, pp)
    if ty > 0:
        return pp
    return pr


def sweep_figure(events_path, out_png):
    """In-flight magnitude sweep: each --safe flight contains graduated kicks
    (0.15 and 0.40 single-axis, 0.15+0.15 and 0.20+0.20 combined), so the
    per-event rows give a peak/recovery-vs-magnitude curve for both controllers
    pooled across every seed — no separate sweep flights needed."""
    if not (events_path and os.path.exists(events_path)):
        return
    ev = list(csv.DictReader(open(events_path)))
    if not ev:
        return

    # pool events by (controller, magnitude)
    agg = {}
    for r in ev:
        key = (r["controller"], round(fval(r, "mag"), 3))
        d = agg.setdefault(key, {"peak": [], "rec": []})
        d["peak"].append(_driven_peak(r))
        d["rec"].append(fval(r, "recovery_s"))
    mags = sorted({m for (_, m) in agg})
    if not mags:
        return

    panels = [("peak attitude on driven axis (deg)", "peak"),
              ("recovery time (s)", "rec")]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.0))
    for ax, (ylabel, field) in zip(axes, panels):
        for ctrl, color, ls in (("baseline", "gray", "--"), ("adaptive", "C0", "-")):
            ys, es = [], []
            for m in mags:
                m_, h = ci95(agg.get((ctrl, m), {"peak": [], "rec": []})[field])
                ys.append(m_); es.append(h if h == h else 0.0)
            ax.errorbar(mags, ys, yerr=es, fmt="o" + ls, color=color,
                        capsize=4, label=ctrl)
        ax.axvspan(0.40, max(mags) + 0.05, color="red", alpha=0.07)
        ax.set_xlabel("kick magnitude (N·m)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("In-flight magnitude sweep (pooled per-event, mean ± 95% CI) — "
                 "shaded = past documented 0.40 single-axis", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print(f"[stats] sweep figure → {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("csv", nargs="?",
                    default=os.path.join("test_results", "gazebo", "metrics_master.csv"))
    ap.add_argument("--events", default=None,
                    help="Per-event CSV for the sweep (default: <csv>_events.csv).")
    args = ap.parse_args()

    rows = load(args.csv)
    metrics_present = [k for k in rows[0] if k not in META]
    outdir = os.path.dirname(os.path.abspath(args.csv))

    seeds, pairs = summarize(rows, metrics_present,
                             os.path.join(outdir, "stats_summary.csv"))
    if seeds:
        bar_figure(seeds, pairs, metrics_present,
                   os.path.join(outdir, "stats_summary.png"))
    events_path = args.events or (os.path.splitext(os.path.abspath(args.csv))[0]
                                  + "_events.csv")
    sweep_figure(events_path, os.path.join(outdir, "sweep_curve.png"))


if __name__ == "__main__":
    main()
