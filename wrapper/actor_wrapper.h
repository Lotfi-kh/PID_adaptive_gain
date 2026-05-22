/*
 * actor_wrapper.h — clean public API for the RL actor.
 *
 * Do NOT call the ST AI network_* functions directly.
 * All project logic goes through this interface.
 */
#ifndef ACTOR_WRAPPER_H
#define ACTOR_WRAPPER_H

#include <stdbool.h>

typedef enum {
    ACTOR_OK            = 0,
    ACTOR_ERR_INIT      = 1,  /* backend initialisation failed */
    ACTOR_ERR_DISABLED  = 2,  /* actor_set_enabled(false) is active */
    ACTOR_ERR_BAD_INPUT = 3,  /* NaN or Inf in obs[] */
    ACTOR_ERR_INFERENCE = 4,  /* backend returned an error */
} actor_status_t;

/* Call once at startup. Returns ACTOR_OK or ACTOR_ERR_INIT. */
actor_status_t actor_init(void);

/* Run one inference step.
 *   obs[13]   — observation vector; must be finite (see deployment_contract.md)
 *   action[3] — output in [-1,1], written only on ACTOR_OK
 * Returns ACTOR_OK on success; action[] is untouched on any error. */
actor_status_t actor_run(const float obs[13], float action[3]);

/* Release resources. */
void actor_deinit(void);

/* Enable / disable the actor globally.
 * Disabled at startup — caller must explicitly enable.
 * When disabled, actor_run() returns ACTOR_ERR_DISABLED immediately. */
void actor_set_enabled(bool enabled);
bool actor_is_enabled(void);

#endif /* ACTOR_WRAPPER_H */
