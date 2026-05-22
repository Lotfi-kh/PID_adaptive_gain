"""
Verify the exported ONNX actor:
  1. Load with onnxruntime, print I/O tensor names and shapes.
  2. Run inference on 4 dummy inputs (zeros, ones, random, edge).
  3. Compare against the PyTorch actor (policy.actor.mu) on the same inputs.
  4. Report max absolute error and pass/fail (tolerance 1e-5).

Read-only. Does not modify the source model or the ONNX file.

Usage:
    python export/verify_actor_onnx.py
"""
import argparse
import os
import sys

import numpy as np
import torch

from stable_baselines3 import TD3

# Resolve all paths relative to the project root (parent of this script's dir)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

DEFAULT_MODEL = os.path.join(_ROOT, "results/frozen_joint_1p05M_shared3D/td3_pid_interrupted.zip")
DEFAULT_ONNX  = os.path.join(_HERE, "actor_joint_1p05M_shared3D.onnx")
TOLERANCE     = 1e-5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--onnx",  default=DEFAULT_ONNX)
    ap.add_argument("--tol",   type=float, default=TOLERANCE)
    args = ap.parse_args()

    try:
        import onnxruntime as ort
    except ImportError:
        sys.exit("[VERIFY] onnxruntime not installed. Install with:\n"
                 "         pip install onnxruntime")

    if not os.path.isfile(args.onnx):
        sys.exit(f"[VERIFY] ONNX file not found: {args.onnx}\n"
                 f"         Run export/export_actor_onnx.py first.")
    if not os.path.isfile(args.model):
        sys.exit(f"[VERIFY] Source model not found: {args.model}")

    print(f"[VERIFY] ONNX file  : {args.onnx}")
    print(f"[VERIFY] PyTorch ref: {args.model}")

    # --- ONNX runtime session
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_meta  = sess.get_inputs()[0]
    out_meta = sess.get_outputs()[0]
    print(f"[VERIFY] ONNX input : name={in_meta.name!r}, shape={in_meta.shape}, dtype={in_meta.type}")
    print(f"[VERIFY] ONNX output: name={out_meta.name!r}, shape={out_meta.shape}, dtype={out_meta.type}")

    # --- PyTorch reference
    model = TD3.load(args.model, device="cpu")
    actor_mu = model.policy.actor.mu
    actor_mu.eval()
    obs_dim = int(model.observation_space.shape[0])
    act_dim = int(model.action_space.shape[0])

    expected_in  = [1, obs_dim]
    expected_out = [1, act_dim]
    in_shape  = [int(d) if isinstance(d, int) else 1 for d in in_meta.shape]
    out_shape = [int(d) if isinstance(d, int) else 1 for d in out_meta.shape]
    if in_shape != expected_in or out_shape != expected_out:
        print(f"[VERIFY] WARN: shape mismatch, expected in={expected_in}, out={expected_out}")
    else:
        print(f"[VERIFY] Shapes match expected ({expected_in} -> {expected_out}).")

    rng = np.random.default_rng(0)
    test_cases = {
        "zeros"  : np.zeros((1, obs_dim), dtype=np.float32),
        "ones"   : np.ones((1, obs_dim),  dtype=np.float32),
        "random" : rng.standard_normal((1, obs_dim)).astype(np.float32),
        "edge"   : (rng.standard_normal((1, obs_dim)) * 10.0).astype(np.float32),
    }

    print(f"[VERIFY] Comparing 4 inputs (tol = {args.tol}):")
    all_pass = True
    with torch.no_grad():
        for name, x in test_cases.items():
            y_onnx  = sess.run([out_meta.name], {in_meta.name: x})[0]
            y_torch = actor_mu(torch.from_numpy(x)).numpy()
            max_err = float(np.max(np.abs(y_onnx - y_torch)))
            ok = max_err < args.tol
            all_pass = all_pass and ok
            tag = "PASS" if ok else "FAIL"
            print(f"  [{tag}] {name:<8s} max|Δ| = {max_err:.2e}   "
                  f"onnx={y_onnx.flatten().round(4).tolist()}  "
                  f"torch={y_torch.flatten().round(4).tolist()}")

    print(f"[VERIFY] Overall: {'PASS' if all_pass else 'FAIL'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
