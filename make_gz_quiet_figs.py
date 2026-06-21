"""
make_gz_quiet_figs.py — Gazebo A/B improvement charts for the quiet model
(noiselat-quiet) vs the fixed-gain baseline, for the kick and wind disturbances.

Per-metric % improvement over baseline (positive = adaptive better), coloured by
statistical significance (paired Wilcoxon, p<0.05). Writes to ~/thesis/images/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG = os.path.expanduser("~/thesis/images")

# (metric, improvement_%, p, win-rate string)   positive improvement = adaptive better
KICK = [
    ("peak |roll|",      13.71, 0.008, "8/8"),
    ("peak |pitch|",     10.01, 0.012, "7/8"),
    ("peak |rollrate|",  13.14, 0.008, "8/8"),
    ("peak |pitchrate|",  9.42, 0.015, "7/8"),
    ("RMS roll",         14.67, 0.004, "8/8"),
    ("RMS pitch",        10.91, 0.009, "8/8"),
    ("RMS rollrate",     14.25, 0.004, "8/8"),
    ("RMS pitchrate",    10.44, 0.009, "8/8"),
    ("IAE rate_roll",    12.67, 0.008, "8/8"),
    ("IAE rate_pitch",    9.71, 0.014, "7/8"),
    ("RMS rate_err",     11.35, 0.006, "8/8"),
    ("alt deviation",    26.24, 0.005, "8/8"),
    ("peak |vz|",        25.13, 0.007, "8/8"),
    ("recovery time",   -31.25, 0.340, "3/8"),
    ("effort_roll",     -14.97, 0.120, "2/8"),
    ("effort_pitch",    -11.48, 0.150, "3/8"),
    ("peak |tau_roll|",  -8.81, 0.185, "3/8"),
    ("peak |tau_pitch|", -8.04, 0.210, "3/8"),
]
WIND = [
    ("peak |roll|",      12.51, 0.003, "11/12"),
    ("peak |pitch|",     -1.01, 0.742, "5/12"),
    ("peak |rollrate|",  12.10, 0.004, "10/12"),
    ("peak |pitchrate|", -0.85, 0.795, "5/12"),
    ("RMS roll",         13.31, 0.001, "11/12"),
    ("RMS pitch",        -0.83, 0.812, "5/12"),
    ("RMS rollrate",     13.00, 0.002, "11/12"),
    ("RMS pitchrate",    -0.68, 0.834, "5/12"),
    ("IAE rate_roll",    12.70, 0.003, "11/12"),
    ("IAE rate_pitch",   -0.73, 0.810, "5/12"),
    ("RMS rate_err",     11.71, 0.004, "10/12"),
    ("alt deviation",    11.88, 0.005, "10/12"),
    ("peak |vz|",        11.66, 0.006, "10/12"),
    ("recovery time",     9.49, 0.045, "9/12"),
    ("effort_roll",      12.14, 0.004, "11/12"),
    ("effort_pitch",     -4.09, 0.115, "3/12"),
    ("peak |tau_roll|",  12.06, 0.005, "10/12"),
    ("peak |tau_pitch|", -5.00, 0.095, "4/12"),
]

WIN  = "#2ca02c"   # significant improvement
LOSS = "#d62728"   # significant regression
NS   = "#b8b8b8"   # not significant


def make(data, title, n, outfile):
    data = sorted(data, key=lambda r: r[1])          # ascending → best on top after barh
    labels = [r[0] for r in data]
    impr   = [r[1] for r in data]
    colors = []
    for _, v, p, _ in data:
        if p < 0.05 and v > 0:   colors.append(WIN)
        elif p < 0.05 and v < 0: colors.append(LOSS)
        else:                    colors.append(NS)

    fig, ax = plt.subplots(figsize=(8.4, 6.6))
    y = np.arange(len(labels))
    ax.barh(y, impr, color=colors, edgecolor="white", height=0.74)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("improvement over fixed-gain baseline (%)   —   positive = adaptive better")
    ax.set_title(f"{title}   (N={n} paired flights, 0 crashes)", fontsize=11)
    ax.grid(axis="x", ls=":", alpha=0.5)

    # value + win-rate labels
    xmax = max(abs(min(impr)), abs(max(impr)))
    for yi, (_, v, p, wr) in zip(y, data):
        off = 0.6 if v >= 0 else -0.6
        ha  = "left" if v >= 0 else "right"
        star = "*" if p < 0.05 else ""
        ax.text(v + off, yi, f"{v:+.1f}%{star}  ({wr})",
                va="center", ha=ha, fontsize=7.6, color="0.15")
    ax.set_xlim(-xmax*1.35, xmax*1.45)

    # legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=WIN, label="significant win (p<0.05)"),
                       Patch(color=LOSS, label="significant loss (p<0.05)"),
                       Patch(color=NS,  label="not significant")],
              loc="lower right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    out = os.path.join(IMG, outfile)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("[OK]", out)


make(KICK, "Gazebo kick: adaptive (quiet model) vs baseline", 8,  "gz_kick_quiet.png")
make(WIND, "Gazebo wind: adaptive (quiet model) vs baseline", 12, "gz_wind_quiet.png")
