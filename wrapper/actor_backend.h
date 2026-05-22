/*
 * actor_backend.h — internal backend interface.
 *
 * Link exactly one of:
 *   actor_stai_backend.c  : ST AI runtime, for on-target builds
 *   (test_harness.c provides a mock inline for host CI)
 */
#ifndef ACTOR_BACKEND_H
#define ACTOR_BACKEND_H

int  actor_backend_init(void);
int  actor_backend_run(const float obs[13], float action[3]);
void actor_backend_deinit(void);

#endif /* ACTOR_BACKEND_H */
