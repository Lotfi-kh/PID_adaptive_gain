#!/usr/bin/env python3
"""
Generate the 48-condition evaluation-grid heatmap from the AXIS-AWARE CORRECTED
summary (never the buggy roll-only aggressive_eval_summary.txt).

Layout per grid: 1x3 panels (roll / pitch / both). Each panel is a
noise (rows) x magnitude (cols) grid, coloured by the per-condition winner
(RL win / mixed / baseline) and annotated with the peak-rate improvement %.

Usage:
    python make_grid_heatmap.py <corrected.txt> <out.png> "<title>"
"""
import re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

NOISES = ["05", "10", "15", "20"]              # rows  (sensor noise sigma)
MAGS   = ["low", "medium", "high", "extreme"]  # cols  (disturbance magnitude)
MAG_NM = {"low": "0.043", "medium": "0.17", "high": "0.30", "extreme": "0.40"}
AXES   = ["roll", "pitch", "both"]
WIN_CODE = {"BASE": 0, "MIXED": 1, "RL": 2}
# baseline-red, mixed-amber, RL-green
CMAP = ListedColormap(["#d96459", "#f2c14e", "#5fa860"])

LINE = re.compile(
    r"noise(\d+)_(low|medium|high|extreme)_(roll|pitch|both)\s+"
    r"\d+/\d+ vs \d+/\d+\s+(-?[\d.]+)%\s+(-?[\d.]+)%\s+(-?[\d.]+)%\s+"
    r"[\d.]+\s+[\d.]+\s+(\w+)")


def parse(path):
    """Return dict[(noise,mag,axis)] = (peak_pct, winner)."""
    d = {}
    for ln in open(path):
        m = LINE.search(ln)
        if not m:
            continue
        noise, mag, axis, rms, peak, rec, winner = m.groups()
        d[(noise, mag, axis)] = (float(peak), winner.upper())
    return d


def main():
    src, out, title = sys.argv[1], sys.argv[2], sys.argv[3]
    data = parse(src)
    if len(data) != 48:
        print(f"[WARN] parsed {len(data)} cells (expected 48) from {src}")

    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for ax, axis in zip(axs, AXES):
        codes = np.zeros((len(NOISES), len(MAGS)))
        peaks = np.zeros_like(codes)
        for i, n in enumerate(NOISES):
            for j, mg in enumerate(MAGS):
                peak, win = data.get((n, mg, axis), (0.0, "BASE"))
                codes[i, j] = WIN_CODE.get(win, 0)
                peaks[i, j] = peak
        ax.imshow(codes, cmap=CMAP, vmin=0, vmax=2, aspect="auto")
        for i in range(len(NOISES)):
            for j in range(len(MAGS)):
                ax.text(j, i, f"{peaks[i,j]:+.0f}%", ha="center", va="center",
                        fontsize=10, fontweight="bold", color="#111111")
        ax.set_xticks(range(len(MAGS)))
        ax.set_xticklabels([f"{m}\n{MAG_NM[m]}" for m in MAGS], fontsize=9)
        ax.set_yticks(range(len(NOISES)))
        ax.set_yticklabels([f"0.{n}" for n in NOISES], fontsize=9)
        ax.set_title(axis.capitalize(), fontsize=12, fontweight="bold")
        ax.set_xlabel("disturbance magnitude (N·m)", fontsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-.5, len(MAGS), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(NOISES), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)
    axs[0].set_ylabel("initial tilt (rad)", fontsize=9)

    legend = [Patch(facecolor="#5fa860", label="RL wins"),
              Patch(facecolor="#f2c14e", label="Mixed"),
              Patch(facecolor="#d96459", label="Baseline wins")]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.text(0.5, 0.93, "cell value = peak angular-rate improvement vs static-gain PID",
             ha="center", fontsize=9, style="italic", color="#555")
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
