"""
Emit a portable, dependency-free C implementation of the frozen 12-D c860k
actor, with weights extracted from the SAME frozen ONNX
(export/actor_joint_12d_c860k.onnx).

Why this exists: the stedgeai-generated network.c links ST's ARM-only
NetworkRuntime, it cannot compile into PX4 SITL (x86 posix). This tiny MLP
(Linear(12,64)->ReLU->Linear(64,64)->ReLU->Linear(64,3)->Tanh, 5187 params)
is reimplemented as plain C: identical math, runs on x86 SITL AND STM32, zero
non-portable dependency. The stedgeai C stays the embedded cross-check artifact.

Outputs (NEW, namespaced, nothing old touched):
    export/portable_c_12d_c860k/rl_actor_weights.h   weights as const float[]
    export/portable_c_12d_c860k/rl_actor.h           API
    export/portable_c_12d_c860k/rl_actor.c           portable forward pass
    export/portable_c_12d_c860k/test_rl_actor.c      harness vs ONNX vectors

Usage:
    python export/gen_portable_c_12d_c860k.py
"""
import os
import numpy as np
import onnx
from onnx import numpy_helper

_HERE = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = os.path.join(_HERE, "actor_joint_12d_c860k.onnx")
OUT_DIR   = os.path.join(_HERE, "portable_c_12d_c860k")

IN_DIM, H1, H2, OUT_DIM = 12, 64, 64, 3


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    m = onnx.load(ONNX_PATH)
    inits = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}

    # Walk Gemm nodes in topological order, deterministic, no shape guessing.
    # Each Gemm: input[0]=x, input[1]=weight, input[2]=bias; respect transB.
    W, B = [], []
    for nd in m.graph.node:
        if nd.op_type != "Gemm":
            continue
        wn, bn = nd.input[1], nd.input[2]
        w = inits[wn].astype(np.float64)
        transB = next((a.i for a in nd.attribute if a.name == "transB"), 0)
        # Gemm computes x @ (B^T if transB else B). nn.Linear weight is
        # [out,in] and PyTorch exports it with transB=1, so y[o]=sum_i w[o,i]x[i]
        # i.e. w is already row-major [out][in]; only transpose if transB==0.
        if not transB:
            w = w.T
        W.append(w)                                  # [out, in]
        B.append((bn, inits[bn].astype(np.float64)))

    assert len(W) == 3, f"expected 3 Gemm layers, got {len(W)}"
    assert [w.shape for w in W] == [(H1, IN_DIM), (H2, H1), (OUT_DIM, H2)], \
        f"unexpected layer shapes: {[w.shape for w in W]}"


    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    iname, oname = sess.get_inputs()[0].name, sess.get_outputs()[0].name

    def np_forward(x):
        for k in range(3):
            x = W[k] @ x + B[k][1]
            if k < 2:
                x = np.maximum(x, 0.0)
        return np.tanh(x)

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(64):
        xv = rng.standard_normal(IN_DIM).astype(np.float32)
        ref = sess.run([oname], {iname: xv.reshape(1, IN_DIM)})[0].flatten()
        mine = np_forward(xv.astype(np.float64))
        worst = max(worst, float(np.max(np.abs(mine - ref))))
    if worst > 1e-5:
        raise SystemExit(f"[PORTABLE] ABORT, extraction wrong, "
                         f"max|numpy - onnx| = {worst:.3e} (> 1e-5)")
    print(f"[PORTABLE] Self-check OK: max|numpy - onnx| = {worst:.3e} over 64 random inputs")

    def carr(name, arr):
        flat = arr.reshape(-1)
        body = ",\n  ".join(
            ", ".join(f"{v:.9e}f" for v in flat[k:k + 8])
            for k in range(0, len(flat), 8))
        return f"static const float {name}[{flat.size}] = {{\n  {body}\n}};\n"


    h = ['/* AUTO-GENERATED from export/actor_joint_12d_c860k.onnx, do not edit. */',
         '#ifndef RL_ACTOR_WEIGHTS_H', '#define RL_ACTOR_WEIGHTS_H', '',
         f'#define RL_IN_DIM  {IN_DIM}', f'#define RL_H1_DIM  {H1}',
         f'#define RL_H2_DIM  {H2}', f'#define RL_OUT_DIM {OUT_DIM}', '',
         '/* Row-major [out][in]. y = W.x + b, per nn.Linear. */',
         carr('RL_W0', W[0]), carr('RL_B0', B[0][1]),
         carr('RL_W1', W[1]), carr('RL_B1', B[1][1]),
         carr('RL_W2', W[2]), carr('RL_B2', B[2][1]),
         '#endif']
    with open(os.path.join(OUT_DIR, "rl_actor_weights.h"), "w") as f:
        f.write("\n".join(h))


    with open(os.path.join(OUT_DIR, "rl_actor.h"), "w") as f:
        f.write(
            "/* Portable, dependency-free forward pass of the frozen 12-D\n"
            " * c860k actor. Bit-validated against the ONNX (see test_rl_actor.c).\n"
            " * Layout: Linear(12,64)->ReLU->Linear(64,64)->ReLU->Linear(64,3)->tanh\n"
            " */\n"
            "#ifndef RL_ACTOR_H\n#define RL_ACTOR_H\n"
            "#ifdef __cplusplus\nextern \"C\" {\n#endif\n"
            "/* obs[12] -> action[3], action in [-1,1]. No malloc, no deps. */\n"
            "void rl_actor_forward(const float obs[12], float action[3]);\n"
            "#ifdef __cplusplus\n}\n#endif\n#endif\n")


    with open(os.path.join(OUT_DIR, "rl_actor.c"), "w") as f:
        f.write(
            '#include "rl_actor.h"\n#include "rl_actor_weights.h"\n'
            "#include <math.h>\n\n"
            "static void dense(const float *x, int in, int out,\n"
            "                  const float *W, const float *b,\n"
            "                  float *y, int relu) {\n"
            "  for (int o = 0; o < out; ++o) {\n"
            "    float acc = b[o];\n"
            "    const float *w = W + (long)o * in;\n"
            "    for (int i = 0; i < in; ++i) acc += w[i] * x[i];\n"
            "    y[o] = (relu && acc < 0.0f) ? 0.0f : acc;\n"
            "  }\n}\n\n"
            "void rl_actor_forward(const float obs[12], float action[3]) {\n"
            "  float h1[RL_H1_DIM], h2[RL_H2_DIM];\n"
            "  dense(obs, RL_IN_DIM, RL_H1_DIM, RL_W0, RL_B0, h1, 1);\n"
            "  dense(h1,  RL_H1_DIM, RL_H2_DIM, RL_W1, RL_B1, h2, 1);\n"
            "  dense(h2,  RL_H2_DIM, RL_OUT_DIM, RL_W2, RL_B2, action, 0);\n"
            "  for (int o = 0; o < RL_OUT_DIM; ++o) action[o] = tanhf(action[o]);\n"
            "}\n")


    with open(os.path.join(OUT_DIR, "test_rl_actor.c"), "w") as f:
        f.write(
            '#include "rl_actor.h"\n#include <stdio.h>\n#include <stdlib.h>\n'
            "#include <math.h>\n\n"
            "/* Usage: ./test_rl_actor valinput.csv valoutput.csv */\n"
            "int main(int argc, char **argv) {\n"
            "  if (argc != 3) { fprintf(stderr, \"need valinput valoutput\\n\"); return 2; }\n"
            "  FILE *fi = fopen(argv[1], \"r\"), *fo = fopen(argv[2], \"r\");\n"
            "  if (!fi || !fo) { perror(\"fopen\"); return 2; }\n"
            "  float obs[12], ref[3], act[3]; int n = 0; double mx = 0.0;\n"
            "  while (fscanf(fi, \"%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\",\n"
            "                &obs[0],&obs[1],&obs[2],&obs[3],&obs[4],&obs[5],\n"
            "                &obs[6],&obs[7],&obs[8],&obs[9],&obs[10],&obs[11]) == 12) {\n"
            "    if (fscanf(fo, \"%f,%f,%f\", &ref[0],&ref[1],&ref[2]) != 3) break;\n"
            "    rl_actor_forward(obs, act);\n"
            "    for (int k = 0; k < 3; ++k) {\n"
            "      double d = fabs((double)act[k] - (double)ref[k]);\n"
            "      if (d > mx) mx = d;\n"
            "    }\n"
            "    printf(\"sample %d: act=[% .7f % .7f % .7f] ref=[% .7f % .7f % .7f]\\n\",\n"
            "           n, act[0],act[1],act[2], ref[0],ref[1],ref[2]);\n"
            "    n++;\n"
            "  }\n"
            "  printf(\"\\n%d samples, max|portableC - ONNX| = %.3e\\n\", n, mx);\n"
            "  printf(\"%s\\n\", mx < 1e-5 ? \"PASS (bit-equivalent, tol 1e-5)\"\n"
            "                              : \"FAIL\");\n"
            "  return mx < 1e-5 ? 0 : 1;\n"
            "}\n")

    print(f"[PORTABLE] Wrote 4 files to {OUT_DIR}")
    print(f"[PORTABLE] Arch {IN_DIM}->{H1}->{H2}->{OUT_DIM}, "
          f"params={sum(w.size for w in W) + sum(b[1].size for b in B)}")


if __name__ == "__main__":
    main()
