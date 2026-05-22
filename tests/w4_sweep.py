"""
tests/w4_sweep.py — Oscillation-penalty weight (w4) sweep
===========================================================
Sweeps w4 ∈ {0.5, 0.05, 0.005, 0.001} with 20k training steps each.
All other hyperparameters are identical across all four runs.

Outputs
-------
- Per-run training summary (from SB3 Monitor CSV)
- Per-run deterministic evaluation (5 episodes, init_noise=0.05)
- Comparison table printed to stdout
- Two plots saved to tests/results/w4_sweep/

Usage:
    cd ~/rl_pid_tuner && python tests/w4_sweep.py
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from envs import PyBulletPIDTunerEnv

# ── Config ─────────────────────────────────────────────────────────────────────

W4_VALUES   = [0.5, 0.05, 0.005, 0.001]
TRAIN_STEPS = 20_000
EVAL_EPS    = 5
SEED        = 42

RESULTS = os.path.join(os.path.dirname(__file__), "results", "w4_sweep")
os.makedirs(RESULTS, exist_ok=True)

# Fixed across all runs
ENV_KWARGS = dict(
    max_steps       = 500,
    target_alt      = 1.0,
    init_noise      = 0.05,
    reward_w1       = 1.0,
    reward_w2       = 2.0,
    reward_w3       = 0.1,
    crash_penalty   = 50.0,
    stability_bonus = 200.0,
)

TD3_KWARGS = dict(
    learning_rate   = 1e-3,
    buffer_size     = 20_000,
    batch_size      = 128,
    gamma           = 0.97,
    tau             = 0.005,
    policy_delay    = 2,
    learning_starts = 500,
    train_freq      = (1, "step"),
    gradient_steps  = 1,
    policy_kwargs   = {"net_arch": [256, 256]},
    device          = "cpu",
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def read_monitor(run_dir, max_steps=500):
    """Parse SB3 Monitor CSV → training curve metrics."""
    csv_path = os.path.join(run_dir, "monitor.csv")
    rows = []
    with open(csv_path) as f:
        for i, line in enumerate(f):
            if i < 2:
                continue
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    rows.append((float(parts[0]), int(parts[1])))
                except ValueError:
                    pass
    if not rows:
        return None
    rewards = np.array([r for r, _ in rows])
    lengths = np.array([l for _, l in rows])
    q       = max(1, len(rewards) // 4)
    # ep length == max_steps means survived (truncated); shorter means crash
    n_crashes_train = int((lengths < max_steps).sum())
    return dict(
        n_ep           = len(rewards),
        first_q_rew    = rewards[:q].mean(),
        last_q_rew     = rewards[-q:].mean(),
        first_q_len    = lengths[:q].mean(),
        last_q_len     = lengths[-q:].mean(),
        n_crashes_train = n_crashes_train,
        crash_rate_train = n_crashes_train / max(1, len(rewards)),
        rewards        = rewards,
        lengths        = lengths,
    )


def run_eval(w4, model_path):
    """5 deterministic episodes, init_noise=0.05 — same conditions for all runs."""
    model = TD3.load(model_path, device="cpu")
    env   = PyBulletPIDTunerEnv(**ENV_KWARGS, reward_w4=w4)

    rr2_buf, rew_buf, act_buf = [], [], []
    ep_lens, ep_crashes = [], []
    kp_list, ki_list, kd_list = [], [], []

    for ep in range(EVAL_EPS):
        obs, _ = env.reset(seed=SEED + ep)
        done = False
        ep_rr2, ep_rew, ep_act = [], [], []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_rr2.append(info["roll_rate"] ** 2)
            ep_rew.append(reward)
            ep_act.append(float(np.mean(np.abs(action))))
            done = terminated or truncated

        ep_crashes.append(bool(info["crashed"]))
        ep_lens.append(int(info["step"]))
        rr2_buf.extend(ep_rr2)
        rew_buf.extend(ep_rew)
        act_buf.extend(ep_act)
        kp_list.append(info["Kp"])
        ki_list.append(info["Ki"])
        kd_list.append(info["Kd"])

    env.close()

    return dict(
        crashes      = int(sum(ep_crashes)),
        mean_ep_len  = float(np.mean(ep_lens)),
        mean_rr2     = float(np.mean(rr2_buf)),
        mean_rew     = float(np.mean(rew_buf)),
        mean_act     = float(np.mean(act_buf)),
        kp           = float(np.mean(kp_list)),
        ki           = float(np.mean(ki_list)),
        kd           = float(np.mean(kd_list)),
    )


# ── Main sweep ─────────────────────────────────────────────────────────────────

all_train = {}
all_eval  = {}

for w4 in W4_VALUES:
    run_dir    = os.path.join(RESULTS, f"w4_{w4}")
    model_path = os.path.join(run_dir, "model")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  w4 = {w4}   ({TRAIN_STEPS} steps)")
    print(f"{'='*62}")

    # ── Train ──────────────────────────────────────────────────────────────────
    train_env = Monitor(
        PyBulletPIDTunerEnv(**ENV_KWARGS, reward_w4=w4),
        run_dir,
    )
    n_actions    = train_env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean  = np.zeros(n_actions),
        sigma = 0.1 * np.ones(n_actions),
    )
    model = TD3(
        policy       = "MlpPolicy",
        env          = train_env,
        action_noise = action_noise,
        verbose      = 0,
        seed         = SEED,
        **TD3_KWARGS,
    )

    t0 = time.time()
    model.learn(total_timesteps=TRAIN_STEPS, progress_bar=True)
    elapsed = time.time() - t0
    model.save(model_path)
    train_env.close()
    print(f"  Training done in {elapsed:.0f}s")

    # ── Monitor metrics ────────────────────────────────────────────────────────
    tm = read_monitor(run_dir)
    all_train[w4] = tm
    if tm:
        trend = "IMPROVING" if tm["last_q_rew"] > tm["first_q_rew"] else "not improving"
        print(f"  Train: {tm['n_ep']} eps  "
              f"first-q={tm['first_q_rew']:+.1f}  last-q={tm['last_q_rew']:+.1f}  "
              f"[{trend}]  crashes={tm['n_crashes_train']}/{tm['n_ep']}")

    # ── Eval ───────────────────────────────────────────────────────────────────
    print(f"  Eval: {EVAL_EPS} deterministic episodes …")
    em = run_eval(w4, model_path)
    all_eval[w4] = em
    print(f"  Eval: crashes={em['crashes']}/{EVAL_EPS}  "
          f"ep_len={em['mean_ep_len']:.0f}  "
          f"rr²={em['mean_rr2']:.5f}  "
          f"rew/step={em['mean_rew']:+.3f}  "
          f"|act|={em['mean_act']:.3f}")


# ── Comparison table ───────────────────────────────────────────────────────────

W = 8
print("\n\n" + "="*90)
print("W4 SWEEP — RESULTS  (20k training steps · 5 deterministic eval episodes each)")
print("="*90)

header = (
    f"{'w4':>6}  "
    f"{'n_ep':>5}  "
    f"{'1st-q rew':>11}  "
    f"{'last-q rew':>11}  "
    f"{'last-q len':>10}  "
    f"{'tr crash%':>9}  "
    f"{'ev crash':>8}  "
    f"{'rr²':>10}  "
    f"{'rew/step':>9}  "
    f"{'|act|':>7}"
)
print(header)
print("-"*90)

for w4 in W4_VALUES:
    tm = all_train[w4]
    em = all_eval[w4]
    print(
        f"{w4:>6.3f}  "
        f"{tm['n_ep']:>5}  "
        f"{tm['first_q_rew']:>+11.1f}  "
        f"{tm['last_q_rew']:>+11.1f}  "
        f"{tm['last_q_len']:>10.1f}  "
        f"{tm['crash_rate_train']*100:>8.1f}%  "
        f"{em['crashes']:>2}/{EVAL_EPS}     "
        f"{em['mean_rr2']:>10.6f}  "
        f"{em['mean_rew']:>+9.3f}  "
        f"{em['mean_act']:>7.4f}"
    )

print("="*90)
print("Columns: rr² = mean roll_rate² in eval (↓ better)  "
      "rew/step = mean per-step eval reward (↑ better)  "
      "|act| = mean |action| (↑ more aggressive gain changes)")

# ── Plots ──────────────────────────────────────────────────────────────────────

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COLORS = {"0.5": "#e74c3c", "0.05": "#f39c12",
              "0.005": "#2ecc71", "0.001": "#3498db"}

    # ── Plot 1: training curves, 2×2 grid ─────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"w4 Sweep — Episode Reward During Training  "
                 f"({TRAIN_STEPS} steps · init_noise={ENV_KWARGS['init_noise']})",
                 fontsize=13)

    for ax, w4 in zip(axes.flat, W4_VALUES):
        tm    = all_train[w4]
        color = COLORS[str(w4)]
        ax.plot(tm["rewards"], alpha=0.3, color=color, linewidth=0.6, label="raw")
        win = max(3, len(tm["rewards"]) // 8)
        if len(tm["rewards"]) >= win:
            kernel   = np.ones(win) / win
            smoothed = np.convolve(tm["rewards"], kernel, mode="valid")
            ax.plot(range(win - 1, len(tm["rewards"])), smoothed,
                    color=color, linewidth=2.0, label=f"smooth (w={win})")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(
            f"w4={w4}   last-q: {tm['last_q_rew']:+.1f}   "
            f"crashes: {tm['n_crashes_train']}/{tm['n_ep']}",
            fontsize=10,
        )
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p1 = os.path.join(RESULTS, "w4_sweep_training_curves.png")
    plt.savefig(p1, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nPlot 1 saved → {p1}")

    # ── Plot 2: eval metrics bar chart ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"w4 Sweep — Eval Metrics  "
                 f"({EVAL_EPS} deterministic episodes · init_noise={ENV_KWARGS['init_noise']})",
                 fontsize=12)

    labels = [str(w) for w in W4_VALUES]
    colors = [COLORS[l] for l in labels]

    metrics = [
        ("mean_rr2",    "Mean roll_rate²  (↓ better)",    False),
        ("mean_rew",    "Mean rew/step    (↑ better)",     True),
        ("mean_ep_len", "Mean ep length   (↑ better)",     True),
        ("mean_act",    "Mean |action|  (gain aggressiv.)", None),
    ]
    for ax, (key, title, higher_better) in zip(axes, metrics):
        vals = [all_eval[w][key] for w in W4_VALUES]
        bars = ax.bar(labels, vals, color=colors)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("w4")
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + abs(bar.get_height()) * 0.02,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    p2 = os.path.join(RESULTS, "w4_sweep_eval_metrics.png")
    plt.savefig(p2, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Plot 2 saved → {p2}")

except Exception as e:
    print(f"\n[WARN] Plotting failed: {e}")

print("\nSweep complete.")
