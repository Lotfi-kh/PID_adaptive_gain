"""
Export the deployable actor (policy.actor.mu) from the frozen TD3 model to ONNX.

Read-only on the source checkpoint. Refuses to overwrite an existing ONNX file
unless --force is passed.

Target: policy.actor.mu — a clean nn.Sequential of
    Linear(12,64) → ReLU → Linear(64,64) → ReLU → Linear(64,3) → Tanh

Why .mu and not .actor:
    SB3's Actor wraps .mu with a FlattenExtractor (no-op for a (12,) Box obs)
    plus SB3 forward-pass plumbing. Exporting .mu gives the minimal pure
    Linear/ReLU/Tanh graph — maximum STM32Cube.AI compatibility, no SB3
    training-mode artifacts in the trace.

Usage:
    cd ~/rl_pid_tuner && python export/export_actor_onnx.py
    cd ~/rl_pid_tuner && python export/export_actor_onnx.py --force   # overwrite
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import torch

from stable_baselines3 import TD3

# Resolve all paths relative to the project root (parent of this script's dir)
_HERE        = os.path.dirname(os.path.abspath(__file__))
_ROOT        = os.path.dirname(_HERE)

FROZEN_MODEL = os.path.join(_ROOT, "results/frozen_joint_1p05M_shared3D/td3_pid_interrupted.zip")
OUT_DIR      = _HERE
OUT_ONNX     = os.path.join(OUT_DIR, "actor_joint_1p05M_shared3D.onnx")
OPSET        = 13   # STM32Cube.AI supports opset 13+; widely compatible


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  default=FROZEN_MODEL)
    ap.add_argument("--out",    default=OUT_ONNX)
    ap.add_argument("--opset",  type=int, default=OPSET)
    ap.add_argument("--force",  action="store_true",
                    help="Overwrite the ONNX file if it already exists.")
    args = ap.parse_args()

    if not os.path.isfile(args.model):
        sys.exit(f"[EXPORT] Source model not found: {args.model}")
    if os.path.exists(args.out) and not args.force:
        sys.exit(f"[EXPORT] Refusing to overwrite existing file: {args.out}\n"
                 f"         Pass --force to overwrite.")

    print(f"[EXPORT] Source model : {args.model}")
    print(f"[EXPORT] Output ONNX  : {args.out}")
    print(f"[EXPORT] Opset        : {args.opset}")

    # Load on CPU; read-only — we never call .save() on this model.
    model = TD3.load(args.model, device="cpu")

    actor_mu = model.policy.actor.mu
    actor_mu.eval()

    obs_dim = int(model.observation_space.shape[0])
    act_dim = int(model.action_space.shape[0])
    n_params = sum(p.numel() for p in actor_mu.parameters())

    print(f"[EXPORT] Input dim    : {obs_dim}")
    print(f"[EXPORT] Output dim   : {act_dim}")
    print(f"[EXPORT] Param count  : {n_params:,}")
    print(f"[EXPORT] Architecture :")
    for line in str(actor_mu).splitlines():
        print(f"           {line}")

    dummy_input = torch.zeros(1, obs_dim, dtype=torch.float32)

    torch.onnx.export(
        actor_mu,
        dummy_input,
        args.out,
        export_params  = True,
        opset_version  = args.opset,
        do_constant_folding = True,
        input_names    = ["obs"],
        output_names   = ["action"],
        # Fixed batch=1 for STM32Cube.AI memory-analysis friendliness.
        # If you later need batched inference, re-export with:
        #   dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}}
    )

    size_kb = os.path.getsize(args.out) / 1024.0
    print(f"[EXPORT] Wrote        : {args.out}  ({size_kb:.1f} KB)")

    meta = {
        "source_model"     : os.path.abspath(args.model),
        "exported_at_utc"  : datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds") + "Z",
        "exported_module"  : "policy.actor.mu",
        "framework"        : "stable-baselines3 TD3",
        "opset"            : args.opset,
        "input_name"       : "obs",
        "input_shape"      : [1, obs_dim],
        "input_dtype"      : "float32",
        "output_name"      : "action",
        "output_shape"     : [1, act_dim],
        "output_range"     : [-1.0, 1.0],
        "output_activation": "tanh",
        "param_count"      : n_params,
        "architecture"     : [str(line) for line in str(actor_mu).splitlines()],
        "deployment_note"  : (
            "Output is normalized action in [-1, 1]. Multiply by DELTA_SCALE = "
            "[3.4e-2, 3.4e-3, 1.7e-4] on the device to recover per-step gain "
            "deltas, then add to running gains and clip to KP/KI/KD bounds. "
            "Obs ordering must match envs/pybullet_pid_tuner_env.py 12-D layout."
        ),
    }
    with open(args.out.replace(".onnx", ".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[EXPORT] Wrote        : {args.out.replace('.onnx', '.meta.json')}")
    print(f"[EXPORT] Done.")


if __name__ == "__main__":
    main()
