/*
 * actor_wrapper.c — safety layer over the ST AI backend.
 *
 * Responsibilities:
 *   - enable/disable gate
 *   - NaN/Inf rejection on all 13 inputs
 *   - defensive output clamp to [-1, 1]
 *   - NaN replacement in output (should never happen after Tanh, but defensive)
 */
#include "actor_wrapper.h"
#include "actor_backend.h"
#include <math.h>
#include <string.h>

#define OBS_DIM 13
#define ACT_DIM  3

static bool s_enabled = false;
static bool s_ready   = false;

actor_status_t actor_init(void)
{
    if (actor_backend_init() != 0)
        return ACTOR_ERR_INIT;
    s_ready = true;
    return ACTOR_OK;
}

actor_status_t actor_run(const float obs[13], float action[3])
{
    if (!s_enabled) return ACTOR_ERR_DISABLED;
    if (!s_ready)   return ACTOR_ERR_INIT;

    for (int i = 0; i < OBS_DIM; i++) {
        if (!isfinite(obs[i]))
            return ACTOR_ERR_BAD_INPUT;
    }

    float raw[ACT_DIM];
    if (actor_backend_run(obs, raw) != 0)
        return ACTOR_ERR_INFERENCE;

    for (int i = 0; i < ACT_DIM; i++) {
        float v = raw[i];
        /* Tanh guarantees [-1,1]; clamp defensively in case of bit corruption */
        if (!isfinite(v)) v = 0.0f;
        if (v >  1.0f)    v =  1.0f;
        if (v < -1.0f)    v = -1.0f;
        action[i] = v;
    }
    return ACTOR_OK;
}

void actor_deinit(void)
{
    actor_backend_deinit();
    s_ready = false;
}

void actor_set_enabled(bool enabled) { s_enabled = enabled; }
bool actor_is_enabled(void)          { return s_enabled; }
