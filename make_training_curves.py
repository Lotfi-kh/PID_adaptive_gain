#!/usr/bin/env python3
"""Three-model training-return figure with warm-start branch points.

c860k has no eval npz; its curve is the same second-phase eval reward already shown
in Chapter 3 (fig:learning_curve), reused here for consistency. noiselat and quiet
use their evaluations.npz (absolute timesteps, so the warm-starts line up directly).
Each model is solid up to its selected (best) checkpoint and faded afterwards, to show
the divergence / over-optimization tails that justify not using the final checkpoint.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# c860k second-phase eval points (same as Chapter 3), steps in thousands
c860k = np.array([
    (20,-6.7),(60,13.7),(100,12.3),(140,4.0),(180,10.9),(220,8.6),(260,12.7),
    (300,-9.3),(340,-25.5),(380,12.6),(420,17.9),(460,3.1),(500,12.3),(540,16.2),
    (580,17.1),(620,7.7),(660,-19.2),(700,16.6),(740,16.6),(780,2.3),(820,10.4),
    (860,19.6),(900,17.6),(940,-11.4),(980,19.6)])

def smooth(y, w=5):
    """Clip eval-crash outliers and apply a short centred moving average."""
    y = np.clip(np.asarray(y, float), -20, 25)
    out = y.copy()
    for i in range(len(y)):
        lo, hi = max(0, i - w//2), min(len(y), i + w//2 + 1)
        out[i] = y[lo:hi].mean()
    return out

def load(run):
    d = np.load(f"runs/{run}/eval/evaluations.npz")
    return d["timesteps"]/1e3, smooth(d["results"].mean(axis=1))

nl_x, nl_y = load("2026-06-06_14-20-35")   # hardened (noiselat)
q_x,  q_y  = load("2026-06-09_05-54-54")   # quiet

BEST = {"clean": 860, "hardened": 1420, "quiet": 1760}
COL  = {"clean": "#1f77b4", "hardened": "#ff7f0e", "quiet": "#2ca060"}

fig, ax = plt.subplots(figsize=(10, 5.2))

def seg(x, y, best, color, label):
    x = np.asarray(x); y = np.asarray(y)
    s = x <= best
    ax.plot(x[s], y[s], color=color, lw=2.0, label=label, zorder=3)
    # faded continuation past the selected checkpoint
    tail = x >= best
    ax.plot(x[tail], y[tail], color=color, lw=1.6, alpha=0.30, ls="--", zorder=2)

seg(c860k[:,0], smooth(c860k[:,1]), BEST["clean"], COL["clean"],    "clean (c860k)")
seg(nl_x, nl_y,                     BEST["hardened"], COL["hardened"], "hardened (noiselat)")
seg(q_x,  q_y,                      BEST["quiet"],    COL["quiet"],    "quiet (noiselat-quiet)")

# selected-checkpoint lines
for name, xb in BEST.items():
    ax.axvline(xb, color=COL[name], ls=":", lw=1.0, alpha=0.7, zorder=1)
    ax.text(xb, 26.2, f"best\n{xb}k", ha="center", va="bottom", fontsize=7.5, color=COL[name])
ax.annotate("warm-start from clean", xy=(862, 21), xytext=(940, 24.2),
            fontsize=8, color=COL["hardened"],
            arrowprops=dict(arrowstyle="->", color=COL["hardened"], lw=0.8))
ax.annotate("warm-start from hardened", xy=(1422, 16), xytext=(1480, 23),
            fontsize=8, color=COL["quiet"],
            arrowprops=dict(arrowstyle="->", color=COL["quiet"], lw=0.8))
ax.text(1505, -18, "diverges", color=COL["hardened"], fontsize=8, alpha=0.9)
ax.text(2025, 9, "over-\noptimizes", color=COL["quiet"], fontsize=8, alpha=0.9, va="center")

ax.set_xlabel("training steps (thousands)")
ax.set_ylabel("mean evaluation return (smoothed)")
ax.set_xlim(0, 2130); ax.set_ylim(-24, 30)
ax.grid(True, ls="--", alpha=0.35)
ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
ax.set_title("Training return of the three models (solid = up to selected checkpoint, "
             "faded = discarded tail)", fontsize=10.5)
fig.tight_layout()
fig.savefig("test_results/figures/training_curves.png", dpi=200, bbox_inches="tight")
print("[OK] wrote test_results/figures/training_curves.png")
