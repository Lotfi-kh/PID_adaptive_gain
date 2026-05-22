/* Portable, dependency-free forward pass of the frozen 12-D
 * c860k actor. Bit-validated against the ONNX (see test_rl_actor.c).
 * Layout: Linear(12,64)->ReLU->Linear(64,64)->ReLU->Linear(64,3)->tanh
 */
#ifndef RL_ACTOR_H
#define RL_ACTOR_H
#ifdef __cplusplus
extern "C" {
#endif
/* obs[12] -> action[3], action in [-1,1]. No malloc, no deps. */
void rl_actor_forward(const float obs[12], float action[3]);
#ifdef __cplusplus
}
#endif
#endif
