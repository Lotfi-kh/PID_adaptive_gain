#include "rl_actor.h"
#include "rl_actor_weights.h"
#include <math.h>

static void dense(const float *x, int in, int out,
                  const float *W, const float *b,
                  float *y, int relu) {
  for (int o = 0; o < out; ++o) {
    float acc = b[o];
    const float *w = W + (long)o * in;
    for (int i = 0; i < in; ++i) acc += w[i] * x[i];
    y[o] = (relu && acc < 0.0f) ? 0.0f : acc;
  }
}

void rl_actor_forward(const float obs[12], float action[3]) {
  float h1[RL_H1_DIM], h2[RL_H2_DIM];
  dense(obs, RL_IN_DIM, RL_H1_DIM, RL_W0, RL_B0, h1, 1);
  dense(h1,  RL_H1_DIM, RL_H2_DIM, RL_W1, RL_B1, h2, 1);
  dense(h2,  RL_H2_DIM, RL_OUT_DIM, RL_W2, RL_B2, action, 0);
  for (int o = 0; o < RL_OUT_DIM; ++o) action[o] = tanhf(action[o]);
}
