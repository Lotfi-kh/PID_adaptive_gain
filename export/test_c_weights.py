"""
Offline C-weight test for the STM32Cube.AI generated model.

Two independent checks:
  1. Weight parity: binary blob in network_data_params.c == ONNX weights
  2. Forward-pass parity: inference using C-extracted weights == test vectors JSON

Architecture: Linear(13,64)->ReLU->Linear(64,64)->ReLU->Linear(64,3)->Tanh

Weight layout in the u64 blob (from network_configure_weights in network.c):
  offset     0: W0  [64,13] float32  (832  floats, 3328 bytes)
  offset  3328: b0  [64]    float32  (64   floats,  256 bytes)
  offset  3584: W1  [64,64] float32  (4096 floats, 16384 bytes)
  offset 19968: b1  [64]    float32  (64   floats,  256 bytes)
  offset 20224: W2  [3,64]  float32  (192  floats,  768 bytes)
  offset 20992: b2  [3]     float32  (3    floats,   12 bytes)
  total: 21004 bytes (u64 array: 2626 x 8 = 21008, 4 bytes padding)

Usage:
    python export/test_c_weights.py
"""
import os
import re
import sys
import json
import struct
import math
import numpy as np

_HERE   = os.path.dirname(os.path.abspath(__file__))
C_PARAMS = os.path.join(_HERE, "stm32_c_code", "network_data_params.c")
ONNX_PATH = os.path.join(_HERE, "actor_joint_1p05M_shared3D.onnx")
JSON_PATH = os.path.join(_HERE, "actor_joint_1p05M_test_vectors.json")

TOL_WEIGHT = 0.0        # weights must match exactly (same float32 bits)
TOL_ACTION = 1e-5       # forward-pass output tolerance (FP32 rounding)


LAYOUT = {
    "W0": (   0,  832),  # [64, 13]
    "b0": (3328,   64),  # [64]
    "W1": (3584, 4096),  # [64, 64]
    "b1": (19968,  64),  # [64]
    "W2": (20224, 192),  # [3, 64]
    "b2": (20992,   3),  # [3]
}


def parse_c_weight_blob(path: str) -> bytes:
    """Extract s_network_weights_array_u64 from network_data_params.c -> raw bytes."""
    with open(path) as f:
        src = f.read()
    m = re.search(
        r's_network_weights_array_u64\[2626\]\s*=\s*\{(.*?)\};',
        src, re.DOTALL)
    if not m:
        sys.exit("[FAIL] Could not find s_network_weights_array_u64 in " + path)
    tokens = re.findall(r'0x([0-9a-fA-F]+)U', m.group(1))
    if len(tokens) != 2626:
        sys.exit(f"[FAIL] Expected 2626 u64 tokens, got {len(tokens)}")
    raw = b"".join(struct.pack("<Q", int(t, 16)) for t in tokens)
    return raw  # 21008 bytes


def extract_weights(blob: bytes) -> dict:
    """Slice the blob at known offsets -> dict of float32 numpy arrays."""
    w = {}
    for name, (off, n) in LAYOUT.items():
        vals = struct.unpack_from(f"<{n}f", blob, off)
        w[name] = np.array(vals, dtype=np.float32)
    w["W0"] = w["W0"].reshape(64, 13)
    w["W1"] = w["W1"].reshape(64, 64)
    w["W2"] = w["W2"].reshape(3, 64)
    return w


def forward(x: np.ndarray, w: dict) -> np.ndarray:
    """Pure-numpy forward pass matching the generated C graph."""
    h = x.astype(np.float32)
    h = np.maximum(0, w["W0"] @ h + w["b0"])   # Gemm + ReLU
    h = np.maximum(0, w["W1"] @ h + w["b1"])   # Gemm + ReLU
    h = np.tanh(w["W2"] @ h + w["b2"])          # Gemm + Tanh
    return h.astype(np.float32)


def onnx_weights(path: str) -> dict:
    """Extract initialiser weights from ONNX."""
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:
        sys.exit("[SKIP] pip install onnx  (needed for weight parity check)")
    model = onnx.load(path)
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    # Map ONNX names -> our layout names; ONNX stores W as [out, in] (transB=1)
    mapping = {
        "W0": None, "b0": None,
        "W1": None, "b1": None,
        "W2": None, "b2": None,
    }
    for name, arr in inits.items():
        lname = name.lower()
        if   "0.weight" in lname or ("0" in lname and arr.shape == (64, 13)):
            mapping["W0"] = arr.astype(np.float32)
        elif "0.bias"   in lname or ("0" in lname and arr.shape == (64,)):
            mapping["b0"] = arr.astype(np.float32)
        elif "2.weight" in lname or ("2" in lname and arr.shape == (64, 64)):
            mapping["W1"] = arr.astype(np.float32)
        elif "2.bias"   in lname or ("2" in lname and arr.shape == (64,)):
            mapping["b1"] = arr.astype(np.float32)
        elif "4.weight" in lname or ("4" in lname and arr.shape == (3, 64)):
            mapping["W2"] = arr.astype(np.float32)
        elif "4.bias"   in lname or ("4" in lname and arr.shape == (3,)):
            mapping["b2"] = arr.astype(np.float32)
    return mapping


def main():
    print("=" * 60)
    print("Offline C-weight test")
    print("=" * 60)


    print(f"\n[1] Parsing weight blob from: {os.path.basename(C_PARAMS)}")
    blob = parse_c_weight_blob(C_PARAMS)
    print(f"    Blob size: {len(blob)} bytes  (expected 21008)")
    c_weights = extract_weights(blob)
    for name, arr in c_weights.items():
        print(f"    {name:3s}: shape={arr.shape}  "
              f"min={arr.min():.4f}  max={arr.max():.4f}  "
              f"norm={np.linalg.norm(arr):.4f}")


    print(f"\n[2] Weight parity vs ONNX ({os.path.basename(ONNX_PATH)})")
    onnx_w = onnx_weights(ONNX_PATH)
    parity_ok = True
    for name in ("W0", "b0", "W1", "b1", "W2", "b2"):
        ow = onnx_w[name]
        cw = c_weights[name]
        if ow is None:
            print(f"    {name}: [SKIP] not found in ONNX initialisers")
            continue
        if ow.shape != cw.shape:
            print(f"    {name}: [SHAPE MISMATCH] ONNX={ow.shape} C={cw.shape}")
            parity_ok = False
            continue
        max_diff = float(np.max(np.abs(ow - cw)))
        match = "PASS" if max_diff <= TOL_WEIGHT else "FAIL"
        print(f"    {name}: {match}  max|delta|={max_diff:.2e}  "
              f"(shapes: ONNX={ow.shape}, C={cw.shape})")
        if match == "FAIL":
            parity_ok = False


    print(f"\n[3] Forward-pass vs JSON test vectors ({os.path.basename(JSON_PATH)})")
    with open(JSON_PATH) as f:
        data = json.load(f)
    vectors = data["vectors"]
    all_ok = True
    print(f"    {'Name':<28} {'MaxErr':>10}  {'Status'}")
    print(f"    {'-'*28} {'-'*10}  {'-'*6}")
    max_errs = []
    for vec in vectors:
        obs    = np.array(vec["obs"], dtype=np.float32)
        expect = np.array(vec["action"], dtype=np.float32)
        got    = forward(obs, c_weights)
        err    = float(np.max(np.abs(got - expect)))
        max_errs.append(err)
        status = "PASS" if err <= TOL_ACTION else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"    {vec['name']:<28} {err:>10.2e}  {status}")


    print(f"\n{'='*60}")
    print(f"Weight parity:   {'PASS' if parity_ok else 'FAIL'}")
    print(f"Forward pass:    {'PASS' if all_ok else 'FAIL'}  "
          f"(worst err: {max(max_errs):.2e}, tol: {TOL_ACTION:.0e})")
    if parity_ok and all_ok:
        print("OVERALL: PASS, C weight blob is bit-exact and inference matches")
    else:
        print("OVERALL: FAIL, see details above")
    print("=" * 60)

    sys.exit(0 if (parity_ok and all_ok) else 1)


if __name__ == "__main__":
    main()
