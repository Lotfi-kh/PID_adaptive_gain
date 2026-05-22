"""
Generate deterministic test vectors for the FROZEN 12-D c860k actor ONNX.

NEW FILE, does not modify or replace gen_test_vectors.py (13-D 1p05M),
which stays frozen for traceability.

Model under test:
    export/actor_joint_12d_c860k.onnx   (12-D obs, step_prog removed)

Outputs (all c860k-specific, new names, nothing old is overwritten):
    export/actor_joint_12d_c860k_test_vectors.json   full structured vectors
    export/actor_joint_12d_c860k_test_vectors.csv    flat table (human/debug)
    export/actor_joint_12d_c860k_valinput.csv        raw [N,12], no header  -> stedgeai --valinput
    export/actor_joint_12d_c860k_valoutput.csv       raw [N,3],  no header  -> stedgeai --valoutput

12-D observation layout (step_prog GONE, invariance by construction):
    [ roll, pitch,
      roll_rate, pitch_rate,
      roll_rate_err, pitch_rate_err,
      Kp_roll_n, Ki_roll_n, Kd_roll_n,
      Kp_pitch_n, Ki_pitch_n, Kd_pitch_n ]

DEPLOYMENT CONTRACT, every constant below is copied verbatim from
sitl/sitl_gain_injector.py so the C-side oracle matches the live injector.
Two distinct concepts, kept strictly separate (the old 13-D generator
conflated them because both happened to equal 1.72):
  • *_NORM   : obs-normalization divisor   (Kp/1.72, Ki/0.172, Kd/8.6e-3)
  • clamp    : applied-gain hard floor/ceiling (the new safety clamp band)

Usage:
    python export/gen_test_vectors_12d_c860k.py
"""
import csv
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

ONNX_PATH       = os.path.join(_HERE, "actor_joint_12d_c860k.onnx")
OUT_JSON        = os.path.join(_HERE, "actor_joint_12d_c860k_test_vectors.json")
OUT_CSV         = os.path.join(_HERE, "actor_joint_12d_c860k_test_vectors.csv")
OUT_VALINPUT    = os.path.join(_HERE, "actor_joint_12d_c860k_valinput.csv")
OUT_VALOUTPUT   = os.path.join(_HERE, "actor_joint_12d_c860k_valoutput.csv")

OBS_DIM = 12
ACT_DIM = 3


KP_DEFAULT = 0.171
KI_DEFAULT = 0.0086
KD_DEFAULT = 0.00171

# Obs-normalization divisors (sitl_gain_injector.py KP_NORM/KI_NORM/KD_NORM).
KP_NORM = 1.72
KI_NORM = 0.172
KD_NORM = 8.6e-3

KP_ATT = 3.0

# Deployment safety clamp, applied-gain hard floor/ceiling.
# sitl_gain_injector.py: CLAMP_LO_MULT=0.5, CLAMP_HI_MULT=2.5 on the defaults.
CLAMP_LO_MULT = 0.5
CLAMP_HI_MULT = 2.5
KP_MIN, KP_MAX = CLAMP_LO_MULT * KP_DEFAULT, CLAMP_HI_MULT * KP_DEFAULT  # [0.0855, 0.4275]
KI_MIN, KI_MAX = CLAMP_LO_MULT * KI_DEFAULT, CLAMP_HI_MULT * KI_DEFAULT  # [0.00430, 0.02150]
KD_MIN, KD_MAX = CLAMP_LO_MULT * KD_DEFAULT, CLAMP_HI_MULT * KD_DEFAULT  # [0.000855, 0.004275]

DELTA_SCALE = np.array([3.4e-2, 3.4e-3, 1.7e-4], dtype=np.float64)


def build_obs(roll, pitch, roll_rate, pitch_rate, kp, ki, kd):
    """Construct 12-D obs exactly as sitl_gain_injector.build_obs() does."""
    roll_rate_err  = KP_ATT * (-roll)  - roll_rate
    pitch_rate_err = KP_ATT * (-pitch) - pitch_rate
    kp_n = kp / KP_NORM
    ki_n = ki / KI_NORM
    kd_n = kd / KD_NORM
    return np.array([
        roll, pitch,
        roll_rate, pitch_rate,
        roll_rate_err, pitch_rate_err,
        kp_n, ki_n, kd_n,
        kp_n, ki_n, kd_n,    # pitch == roll (shared joint action)
    ], dtype=np.float32)


def apply_action(action, kp_prev, ki_prev, kd_prev):
    """Apply DELTA_SCALE then clip to the deployment safety clamp band."""
    delta  = action.astype(np.float64) * DELTA_SCALE
    kp_new = float(np.clip(kp_prev + delta[0], KP_MIN, KP_MAX))
    ki_new = float(np.clip(ki_prev + delta[1], KI_MIN, KI_MAX))
    kd_new = float(np.clip(kd_prev + delta[2], KD_MIN, KD_MAX))
    return kp_new, ki_new, kd_new, delta


DG = (KP_DEFAULT, KI_DEFAULT, KD_DEFAULT)

# (name, description, roll, pitch, roll_rate, pitch_rate, kp, ki, kd)
# NO step_prog anywhere, removed from the model entirely.
CASES = [
    ("zero_obs",
     "All states zero. Gains at default.",
     0.0, 0.0, 0.0, 0.0, *DG),
    ("nominal_hover",
     "Tiny residual roll/pitch and rates. Realistic calm hover.",
     0.02, 0.01, 0.05, 0.03, *DG),
    ("roll_disturbance",
     "Roll impulse: roll=0.15 rad (8.6 deg), roll_rate=0.8 rad/s. Pitch calm.",
     0.15, 0.01, 0.80, 0.05, *DG),
    ("pitch_disturbance",
     "Pitch impulse: pitch=0.15 rad (8.6 deg), pitch_rate=0.8 rad/s. Roll calm.",
     0.01, 0.15, 0.05, 0.80, *DG),
    ("combined_disturbance",
     "Both axes: roll=0.12/0.6 rad/s, pitch=0.10/0.5 rad/s.",
     0.12, 0.10, 0.60, 0.50, *DG),
    ("high_rate_edge",
     "Near obs limits: large attitude and rates. Numerical robustness.",
     0.50, -0.40, 5.0, -4.0, *DG),
    ("adapted_gains",
     "Gains shifted mid-adaptation, well inside the clamp band: "
     "Kp=0.30, Ki=0.015, Kd=2.5e-3.",
     0.08, 0.06, 0.30, 0.20, 0.30, 0.015, 2.5e-3),
    ("gains_at_lower_clamp",
     "Gains AT the new safety-clamp floor (Kp/Ki/Kd MIN). "
     "Tests clip at the lower clamp bound.",
     0.10, 0.08, 0.40, 0.35, KP_MIN, KI_MIN, KD_MIN),
    ("gains_at_upper_clamp",
     "Gains AT the new safety-clamp ceiling (Kp/Ki/Kd MAX). "
     "Tests clip at the upper clamp bound.",
     0.05, 0.04, 0.20, 0.18, KP_MAX, KI_MAX, KD_MAX),
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

    sin = sess.get_inputs()[0].shape
    if list(sin)[-1] != OBS_DIM:
        sys.exit(f"[VECGEN] ONNX input shape {sin} != expected (...,{OBS_DIM}). "
                 f"Wrong model? Expected the 12-D c860k actor.")

    print(f"[VECGEN] ONNX  : {ONNX_PATH}")
    print(f"[VECGEN] Shape : in={sin} -> out={sess.get_outputs()[0].shape}")
    print(f"[VECGEN] Clamp : Kp[{KP_MIN:.6g},{KP_MAX:.6g}] "
          f"Ki[{KI_MIN:.6g},{KI_MAX:.6g}] Kd[{KD_MIN:.6g},{KD_MAX:.6g}]")
    print(f"[VECGEN] Cases : {len(CASES)}\n")

    records = []
    for name, desc, roll, pitch, rr, pr, kp, ki, kd in CASES:
        obs    = build_obs(roll, pitch, rr, pr, kp, ki, kd)
        action = sess.run([out_name],
                          {in_name: obs.reshape(1, OBS_DIM)})[0].flatten().astype(np.float32)
        kp_new, ki_new, kd_new, delta = apply_action(action, kp, ki, kd)

        records.append({
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
        })
        print(f"  [{name}]")
        print(f"    obs    : {[round(float(v), 4) for v in obs]}")
        print(f"    action : {[round(float(v), 6) for v in action]}")
        print(f"    delta  : dKp={delta[0]:.6f}  dKi={delta[1]:.7f}  dKd={delta[2]:.9f}")
        print(f"    gains  : Kp {kp:.4f}->{kp_new:.6f}  "
              f"Ki {ki:.5f}->{ki_new:.7f}  Kd {kd:.6f}->{kd_new:.8f}\n")


    payload = {
        "metadata": {
            "onnx_model"   : os.path.basename(ONNX_PATH),
            "input_shape"  : [1, OBS_DIM],
            "output_shape" : [1, ACT_DIM],
            "input_dtype"  : "float32",
            "output_range" : [-1.0, 1.0],
            "DELTA_SCALE"  : list(DELTA_SCALE),
            "KP_DEFAULT"   : KP_DEFAULT,
            "KI_DEFAULT"   : KI_DEFAULT,
            "KD_DEFAULT"   : KD_DEFAULT,
            "obs_norm"     : {"KP_NORM": KP_NORM, "KI_NORM": KI_NORM, "KD_NORM": KD_NORM},
            "clamp_band"   : {"KP": [KP_MIN, KP_MAX],
                              "KI": [KI_MIN, KI_MAX],
                              "KD": [KD_MIN, KD_MAX],
                              "CLAMP_LO_MULT": CLAMP_LO_MULT,
                              "CLAMP_HI_MULT": CLAMP_HI_MULT},
            "KP_ATT"       : KP_ATT,
            "obs_layout": [
                "obs[0]  roll_rad",
                "obs[1]  pitch_rad",
                "obs[2]  roll_rate_rad_s  (body frame)",
                "obs[3]  pitch_rate_rad_s (body frame)",
                "obs[4]  roll_rate_err  = KP_ATT*(-roll)  - roll_rate",
                "obs[5]  pitch_rate_err = KP_ATT*(-pitch) - pitch_rate",
                "obs[6]  kp_roll_norm  = Kp_roll  / 1.72",
                "obs[7]  ki_roll_norm  = Ki_roll  / 0.172",
                "obs[8]  kd_roll_norm  = Kd_roll  / 8.6e-3",
                "obs[9]  kp_pitch_norm = Kp_pitch / 1.72  (== obs[6])",
                "obs[10] ki_pitch_norm = Ki_pitch / 0.172 (== obs[7])",
                "obs[11] kd_pitch_norm = Kd_pitch / 8.6e-3 (== obs[8])",
            ],
            "action_layout": [
                "action[0] in [-1,1] -> dKp = action[0] * 3.4e-2",
                "action[1] in [-1,1] -> dKi = action[1] * 3.4e-3",
                "action[2] in [-1,1] -> dKd = action[2] * 1.7e-4",
            ],
            "note": ("step_prog removed entirely (12-D model). obs is "
                     "normalized by *_NORM; applied gains clip to clamp_band "
                     "-- these are distinct, do not conflate."),
        },
        "vectors": records,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[VECGEN] Wrote JSON     : {OUT_JSON}")


    header = (["name"]
              + [f"obs_{i:02d}" for i in range(OBS_DIM)]
              + [f"action_{i}" for i in range(ACT_DIM)]
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
    print(f"[VECGEN] Wrote CSV      : {OUT_CSV}")


    with open(OUT_VALINPUT, "w", newline="") as f:
        w = csv.writer(f)
        for r in records:
            w.writerow([f"{v:.8f}" for v in r["obs"]])         # 12 cols
    print(f"[VECGEN] Wrote valinput : {OUT_VALINPUT}  ([{len(records)},{OBS_DIM}], no header)")

    with open(OUT_VALOUTPUT, "w", newline="") as f:
        w = csv.writer(f)
        for r in records:
            w.writerow([f"{v:.8f}" for v in r["action"]])      # 3 cols
    print(f"[VECGEN] Wrote valoutput: {OUT_VALOUTPUT}  ([{len(records)},{ACT_DIM}], no header)")

    print(f"[VECGEN] Done, {len(records)} vectors. step_prog: ABSENT (12-D).")


if __name__ == "__main__":
    main()
