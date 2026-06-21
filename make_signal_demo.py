"""
make_signal_demo.py — time-series 'signal' figure: baseline vs adaptive roll
response to a kick, aligned on the disturbance window (dist_active flag).
Demonstrates the format; regenerate from a clean re-run for the final thesis figure.
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.expanduser("~/rl_pid_tuner/test_results/gazebo/_archive_kick_quiet")
SEED = 1
b = pd.read_csv(f"{D}/pair_seed{SEED}_baseline_wide.csv")
a = pd.read_csv(f"{D}/pair_seed{SEED}_adaptive_wide.csv")

def biggest_event(df):
    """Find the dist_active window with the largest |roll| excursion."""
    da = (df["dist_active"].values > 0.5).astype(int)
    edges = np.where(np.diff(da) == 1)[0]
    if len(edges) == 0:
        i = int(np.argmax(np.abs(df["roll"].values))); return df["t"].values[i]
    best_t, best_amp = df["t"].values[edges[0]], -1
    t = df["t"].values
    for e in edges:
        m = (t > t[e] - 0.2) & (t < t[e] + 3.0)
        amp = np.abs(df["roll"].values[m]).max()
        if amp > best_amp: best_amp, best_t = amp, t[e]
    return best_t

t0 = biggest_event(a)
W0, W1 = t0 - 0.6, t0 + 4.0

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6.0), sharex=True)
for df, color, lbl, ls in [(b, "#888888", "baseline (fixed gains)", "--"),
                           (a, "#1f77b4", "adaptive (quiet model)", "-")]:
    t = df["t"].values
    m = (t >= W0) & (t <= W1)
    tt = t[m] - t0
    ax1.plot(tt, np.rad2deg(df["roll"].values[m]),      color=color, ls=ls, lw=1.8, label=lbl)
    ax2.plot(tt, df["roll_rate"].values[m],             color=color, ls=ls, lw=1.8, label=lbl)

# shade the disturbance window
da = (a["dist_active"].values > 0.5)
ta = a["t"].values
seg = ta[(da) & (ta >= W0) & (ta <= W1)]
if len(seg):
    for ax in (ax1, ax2):
        ax.axvspan(seg.min() - t0, seg.max() - t0, color="0.9", zorder=0, label="_")

for ax in (ax1, ax2):
    ax.axhline(0, color="k", lw=0.6, alpha=0.4); ax.grid(True, ls=":", alpha=0.5)
ax1.set_ylabel("roll angle (deg)"); ax2.set_ylabel("roll rate (rad/s)")
ax2.set_xlabel("time since disturbance onset (s)")
ax1.legend(loc="upper right", fontsize=9)
ax1.set_title("Gazebo kick — roll response to a disturbance (quiet model vs baseline)")
fig.tight_layout()
out = os.path.expanduser("~/thesis/images/gz_kick_signal_demo.png")
fig.savefig(out, dpi=160); print("[OK]", out)
print(f"baseline peak|roll|={np.rad2deg(np.abs(b['roll'])).max():.2f} deg  "
      f"adaptive peak|roll|={np.rad2deg(np.abs(a['roll'])).max():.2f} deg")
