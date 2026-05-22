"""
Generate deterministic test vectors for the frozen actor ONNX model.

Outputs:
  export/actor_joint_1p05M_test_vectors.json  — full structured vectors
  export/actor_joint_1p05M_test_vectors.csv   — flat table for C/STM32Cube.AI tests

Each vector records:
  - obs[13]        : exact float32 input fed to the ONNX
  - action[3]      : float32 output from the ONNX
  - delta_gains[3] : action * DELTA_SCALE
  - gains_prev[3]  : running [Kp, Ki, Kd] before this step
  - gains_new[3]   : after clip(gains_prev + delta, lo, hi)

Usage:
    cd ~/rl_pid_tuner && python export/gen_test_vectors.py
"""
import csv
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

ONNX_PATH = os.path.join(_HERE, "actor_joint_1p05M_shared3D.onnx")
OUT_JSON  = os.path.join(_HERE, "actor_joint_1p05M_test_vectors.json")
OUT_CSV   = os.path.join(_HERE, "actor_joint_1p05M_test_vectors.csv")

# ── Runtime contract constants ─────────────────────────────────────────────────
KP_DEFAULT  = 0.171
KI_DEFAULT  = 8.6e-3
KD_DEFAULT  = 1.71e-3

KP_LO, KP_HI = 0.0, 1.72
KI_LO, KI_HI = 0.0, 0.172
KD_LO, KD_HI = 0.0, 8.6e-3

DELTA_SCALE = np.array([3.4e-2, 3.4e-3, 1.7e-4], dtype=np.float64)

KP_ATT    = 3.0
STEP_PROG = 1.0    # deployment default: hold at 1.0


def build_obs(roll, pitch, roll_rate, pitch_rate, kp, ki, kd, step_prog=STEP_PROG):
    """Construct 13-D obs exactly as _computeObs() does in training."""
    roll_rate_err  = KP_ATT * (-roll)  - roll_rate
    pitch_rate_err = KP_ATT * (-pitch) - pitch_rate
    kp_n = kp / KP_HI
    ki_n = ki / KI_HI
    kd_n = kd / KD_HI
    return np.array([
        roll, pitch,
        roll_rate, pitch_rate,
        roll_rate_err, pitch_rate_err,
        kp_n, ki_n, kd_n,
        kp_n, ki_n, kd_n,   # pitch == roll (shared action)
        step_prog,
    ], dtype=np.float32)


def apply_action(action, kp_prev, ki_prev, kd_prev):
    delta  = action.astype(np.float64) * DELTA_SCALE
    kp_new = float(np.clip(kp_prev + delta[0], KP_LO, KP_HI))
    ki_new = float(np.clip(ki_prev + delta[1], KI_LO, KI_HI))
    kd_new = float(np.clip(kd_prev + delta[2], KD_LO, KD_HI))
    return kp_new, ki_new, kd_new, delta


DG = (KP_DEFAULT, KI_DEFAULT, KD_DEFAULT)

CASES = [
    ("zero_obs",
     "All states zero. Gains at default. step_prog=1.0 (deployment nominal).",
     0.0, 0.0, 0.0, 0.0, *DG, 1.0),
    ("nominal_hover",
     "Tiny residual roll/pitch and rates. Realistic calm hover.",
     0.02, 0.01, 0.05, 0.03, *DG, 1.0),
    ("roll_disturbance",
     "Roll impulse: roll=0.15 rad (8.6 deg), roll_rate=0.8 rad/s. Pitch calm.",
     0.15, 0.01, 0.80, 0.05, *DG, 1.0),
    ("pitch_disturbance",
     "Pitch impulse: pitch=0.15 rad (8.6 deg), pitch_rate=0.8 rad/s. Roll calm.",
     0.01, 0.15, 0.05, 0.80, *DG, 1.0),
    ("combined_disturbance",
     "Both axes: roll=0.12/0.6 rad/s, pitch=0.10/0.5 rad/s.",
     0.12, 0.10, 0.60, 0.50, *DG, 1.0),
    ("high_rate_edge",
     "Near obs limits: large attitude and rates. Tests numerical robustness.",
     0.50, -0.40, 5.0, -4.0, *DG, 1.0),
    ("adapted_gains",
     "Gains shifted after adaptation: Kp=0.30, Ki=0.015, Kd=2.5e-3. step_prog=0.5.",
     0.08, 0.06, 0.30, 0.20, 0.30, 0.015, 2.5e-3, 0.5),
    ("start_of_episode",
     "step_prog=0.0. Tests network at episode start.",
     0.0, 0.0, 0.0, 0.0, *DG, 0.0),
    ("gains_near_lower_bound",
     "Gains near zero. Tests clip at lower bound.",
     0.10, 0.08, 0.40, 0.35, 0.005, 0.001, 5.0e-5, 1.0),
    ("gains_near_upper_bound",
     "Gains near ceiling. Tests clip at upper bound.",
     0.05, 0.04, 0.20, 0.18, 1.70, 0.168, 8.4e-3, 1.0),
]


def main():
    try:
        import onnxruntime as ort
    except ImportError:
        sys.exit("[VECGEN] pip install onnxruntime")

    if not os.path.isfile(ONNX_PATH):
        sys.exit(f"[VECGEN] ONNX not found: {ONNX_PATH}")

    sess     = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    in_name  = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    print(f"[VECGEN] ONNX  : {ONNX_PATH}")
    print(f"[VECGEN] Cases : {len(CASES)}\n")

    records = []
    for name, desc, roll, pitch, rr, pr, kp, ki, kd, sp in CASES:
        obs    = build_obs(roll, pitch, rr, pr, kp, ki, kd, sp)
        action = sess.run([out_name], {in_name: obs.reshape(1, 13)})[0].flatten().astype(np.float32)
        kp_new, ki_new, kd_new, delta = apply_action(action, kp, ki, kd)

        rec = {
            "name"       : name,
            "description": desc,
            "obs"        : [round(float(v), 8) for v in obs],
            "action"     : [round(float(v), 8) for v in action],
            "delta_gains": {"dKp": round(float(delta[0]), 10),
                            "dKi": round(float(delta[1]), 10),
                            "dKd": round(float(delta[2]), 10)},
            "gains_prev" : {"Kp": kp, "Ki": ki, "Kd": kd},
            "gains_new"  : {"Kp": round(kp_new, 8),
                            "Ki": round(ki_new, 8),
                            "Kd": round(kd_new, 10)},
        }
        records.append(rec)
        print(f"  [{name}]")
        print(f"    obs    : {[round(float(v), 4) for v in obs]}")
        print(f"    action : {[round(float(v), 6) for v in action]}")
        print(f"    delta  : dKp={delta[0]:.6f}  dKi={delta[1]:.7f}  dKd={delta[2]:.9f}")
        print(f"    gains  : Kp {kp:.4f}->{kp_new:.6f}  "
              f"Ki {ki:.5f}->{ki_new:.7f}  Kd {kd:.6f}->{kd_new:.8f}\n")

    # JSON
    payload = {
        "metadata": {
            "onnx_model"           : os.path.basename(ONNX_PATH),
            "input_shape"          : [1, 13],
            "output_shape"         : [1, 3],
            "input_dtype"          : "float32",
            "output_range"         : [-1.0, 1.0],
            "DELTA_SCALE"          : list(DELTA_SCALE),
            "KP_DEFAULT"           : KP_DEFAULT,
            "KI_DEFAULT"           : KI_DEFAULT,
            "KD_DEFAULT"           : KD_DEFAULT,
            "KP_BOUNDS"            : [KP_LO, KP_HI],
            "KI_BOUNDS"            : [KI_LO, KI_HI],
            "KD_BOUNDS"            : [KD_LO, KD_HI],
            "KP_ATT"               : KP_ATT,
            "step_prog_deployment" : STEP_PROG,
            "obs_layout": [
                "obs[0]  roll_rad",
                "obs[1]  pitch_rad",
                "obs[2]  roll_rate_rad_s  (body frame: R.T @ omega_world)",
                "obs[3]  pitch_rate_rad_s (body frame: R.T @ omega_world)",
                "obs[4]  roll_rate_err  = KP_ATT*(-roll)  - roll_rate",
                "obs[5]  pitch_rate_err = KP_ATT*(-pitch) - pitch_rate",
                "obs[6]  kp_roll_norm  = Kp_roll  / 1.72",
                "obs[7]  ki_roll_norm  = Ki_roll  / 0.172",
                "obs[8]  kd_roll_norm  = Kd_roll  / 8.6e-3",
                "obs[9]  kp_pitch_norm = Kp_pitch / 1.72  (== obs[6])",
                "obs[10] ki_pitch_norm = Ki_pitch / 0.172 (== obs[7])",
                "obs[11] kd_pitch_norm = Kd_pitch / 8.6e-3 (== obs[8])",
                "obs[12] step_prog = step/max_steps  (hold at 1.0 in deployment)",
            ],
            "action_layout": [
                "action[0] in [-1,1] -> dKp = action[0] * 3.4e-2",
                "action[1] in [-1,1] -> dKi = action[1] * 3.4e-3",
                "action[2] in [-1,1] -> dKd = action[2] * 1.7e-4",
            ],
        },
        "vectors": records,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[VECGEN] Wrote JSON : {OUT_JSON}")

    header = (["name"]
              + [f"obs_{i:02d}" for i in range(13)]
              + ["action_0", "action_1", "action_2"]
              + ["dKp", "dKi", "dKd"]
              + ["Kp_prev", "Ki_prev", "Kd_prev"]
              + ["Kp_new",  "Ki_new",  "Kd_new"])
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in records:
            w.writerow(
                [r["name"]]
                + [f"{v:.8f}" for v in r["obs"]]
                + [f"{v:.8f}" for v in r["action"]]
                + [f"{r['delta_gains']['dKp']:.10f}",
                   f"{r['delta_gains']['dKi']:.10f}",
                   f"{r['delta_gains']['dKd']:.10f}"]
                + [f"{r['gains_prev']['Kp']:.8f}",
                   f"{r['gains_prev']['Ki']:.8f}",
                   f"{r['gains_prev']['Kd']:.8f}"]
                + [f"{r['gains_new']['Kp']:.8f}",
                   f"{r['gains_new']['Ki']:.8f}",
                   f"{r['gains_new']['Kd']:.10f}"]
            )
    print(f"[VECGEN] Wrote CSV  : {OUT_CSV}")
    print(f"[VECGEN] Done — {len(records)} vectors.")


if __name__ == "__main__":
    main()
