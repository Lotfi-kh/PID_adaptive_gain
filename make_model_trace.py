#!/usr/bin/env python3
"""
Per-model roll-rate TRACE figure, consistent across all models/conditions.

Top panel : one representative roll kick, static baseline (red dashed) vs
            adaptive RL (blue solid).
Bottom    : mean |roll rate| over the 20 episodes of the condition.
Box       : the three GRID-MEAN improvements (peak / RMS / recovery) read
            verbatim from the axis-aware corrected summary header, so the
            figure's headline numbers are the model's real grid means.

Usage:
    python make_model_trace.py <corrected.txt> <npz_dir> <condition> <out.png> "<title>"
e.g.
    python make_model_trace.py test_results/aggressive_eval/aggressive_eval_corrected.txt \
        test_results/aggressive_eval noise10_high_roll \
        ~/thesis/images/trace_clean_nominal.png "Clean-trained model — nominal"
"""
import re, sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BL = "#d62728"   # red  dashed -> static-gain baseline
RL = "#1f77b4"   # blue solid  -> adaptive (RL)


def read_means(path):
    txt = open(path).read()
    def g(label):
        m = re.search(label + r"\s*([+-]?[\d.]+)%", txt)
        return float(m.group(1)) if m else float("nan")
    return (g("Mean peak rate improvement:"),
            g("Mean RMS rate improvement:"),
            g("Mean recovery improvement:"))


def main():
    corrected, npz_dir, cond, out, title = sys.argv[1:6]
    pk, rms, rec = read_means(corrected)

    d = np.load(os.path.join(npz_dir, cond, "disturbance_eval_results.npz"),
                allow_pickle=True)
    fs = int(d["ctrl_freq"]); ds = int(d["dist_step"]); dd = int(d["dist_duration"])
    R2D = 180.0 / np.pi
    bl = d["bl_roll_rates"] * R2D
    rl = d["rl_roll_rates"] * R2D
    T = bl.shape[1]; t = np.arange(T) / fs

    # representative episode: baseline peak nearest the median
    blpk = np.abs(bl).max(1)
    ep = int(np.argmin(np.abs(blpk - np.median(blpk))))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True,
                                   gridspec_kw=dict(height_ratios=[2.1, 1]))

    ax1.axvspan(ds / fs, (ds + dd) / fs, color="#999", alpha=.18, lw=0)
    ax1.text((ds + dd / 2) / fs, ax1.get_ylim()[1], "torque\nkick",
             ha="center", va="top", fontsize=8.5, color="#555")
    ax1.plot(t, bl[ep], color=BL, lw=1.7, ls="--", label="Static-gain baseline")
    ax1.plot(t, rl[ep], color=RL, lw=2.0, ls="-",  label="Adaptive (RL)")
    ax1.axhline(0, color="#bbb", lw=.8, zorder=0)
    ib = np.abs(bl[ep]).argmax(); ir = np.abs(rl[ep]).argmax()
    ax1.annotate(f"peak {bl[ep, ib]:+.0f}/s", (t[ib], bl[ep, ib]),
                 textcoords="offset points", xytext=(6, -2), color=BL, fontsize=8.5)
    ax1.annotate(f"peak {rl[ep, ir]:+.0f}/s", (t[ir], rl[ep, ir]),
                 textcoords="offset points", xytext=(6, 4), color=RL, fontsize=8.5)
    ax1.set_ylabel(r"roll rate  ($^\circ$/s)")
    ax1.set_title(title, fontsize=11, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=9, framealpha=.95)

    ax2.fill_between(t, 0, np.abs(bl).mean(0), color=BL, alpha=.15)
    ax2.fill_between(t, 0, np.abs(rl).mean(0), color=RL, alpha=.25)
    ax2.plot(t, np.abs(bl).mean(0), color=BL, lw=1.5, ls="--")
    ax2.plot(t, np.abs(rl).mean(0), color=RL, lw=1.7, ls="-")
    ax2.axvspan(ds / fs, (ds + dd) / fs, color="#999", alpha=.18, lw=0)
    ax2.set_ylabel("mean |roll rate|\n($^\\circ$/s, 20 eps)", fontsize=9)
    ax2.set_xlabel("time (s)")
    ax2.set_xlim(t[0], t[-1])

    box = (f"Grid mean (48 conditions)\n"
           f"Peak-rate      {pk:+.1f}%\n"
           f"RMS-rate       {rms:+.1f}%\n"
           f"Recovery-time  {rec:+.1f}%")
    ax2.text(0.985, 0.92, box, transform=ax2.transAxes, ha="right", va="top",
             fontsize=9, family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#888", alpha=.95))

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[OK] {out}  peak {pk:+.1f}%  rms {rms:+.1f}%  rec {rec:+.1f}%  (ep {ep}, cond {cond})")


if __name__ == "__main__":
    main()
