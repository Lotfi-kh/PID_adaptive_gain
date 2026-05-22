/*
 * gain_updater.c — DELTA_SCALE, gain clip, per-second rate limit.
 *
 * All numeric constants from deployment_contract.md.
 * Shared action: roll and pitch always receive the same Kp/Ki/Kd.
 */
#include "gain_updater.h"
#include <math.h>
#include <string.h>
#include <stdbool.h>

/* ── Contract constants ────────────────────────────────────────────────────── */
#define KP_DEFAULT  0.171f
#define KI_DEFAULT  8.6e-3f
#define KD_DEFAULT  1.71e-3f

static const float GAIN_MIN[3] = { 0.0f,   0.0f,   0.0f    };
static const float GAIN_MAX[3] = { 1.72f,  0.172f, 8.6e-3f };
static const float DELTA_SCALE[3] = { 3.4e-2f, 3.4e-3f, 1.7e-4f };

/* Per-second rate limit: no more than 10 full-scale steps per second.
 * At 48 Hz this equals ~21% duty cycle of max action — reasonable ceiling. */
#define CTRL_HZ 48
static const float MAX_CHANGE_PER_SEC[3] = {
    10.0f * 3.4e-2f,   /* Kp: 0.34 / s  (≈ 20 % of full range) */
    10.0f * 3.4e-3f,   /* Ki: 0.034 / s */
    10.0f * 1.7e-4f,   /* Kd: 1.7e-3 / s */
};

/* ── State ─────────────────────────────────────────────────────────────────── */
static float s_gains[3];        /* current [Kp, Ki, Kd] */
static float s_accum[3];        /* accumulated |delta| in current 1-s window */
static int   s_tick;            /* ticks elapsed in current window */
static bool  s_rate_frozen;     /* true when window limit was hit */

static float clampf(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

static void reset_rate_state(void)
{
    memset(s_accum, 0, sizeof(s_accum));
    s_tick       = 0;
    s_rate_frozen = false;
}

void gain_updater_init(void)
{
    gain_updater_reset();
}

void gain_updater_reset(void)
{
    s_gains[0] = KP_DEFAULT;
    s_gains[1] = KI_DEFAULT;
    s_gains[2] = KD_DEFAULT;
    reset_rate_state();
}

void gain_updater_set_gains(float kp, float ki, float kd)
{
    s_gains[0] = clampf(kp, GAIN_MIN[0], GAIN_MAX[0]);
    s_gains[1] = clampf(ki, GAIN_MIN[1], GAIN_MAX[1]);
    s_gains[2] = clampf(kd, GAIN_MIN[2], GAIN_MAX[2]);
    reset_rate_state();
}

bool gain_updater_apply(const float action[3])
{
    /* Advance 1-second window */
    s_tick++;
    if (s_tick >= CTRL_HZ) {
        reset_rate_state();
    }

    if (s_rate_frozen) return false;

    /* Validate and compute deltas */
    float delta[3];
    for (int i = 0; i < 3; i++) {
        if (!isfinite(action[i])) return false;
        delta[i] = action[i] * DELTA_SCALE[i];
        s_accum[i] += fabsf(delta[i]);
        if (s_accum[i] > MAX_CHANGE_PER_SEC[i]) {
            s_rate_frozen = true;
            return false;
        }
    }

    /* Apply — gain bounds clip handles the hard limits */
    for (int i = 0; i < 3; i++)
        s_gains[i] = clampf(s_gains[i] + delta[i], GAIN_MIN[i], GAIN_MAX[i]);

    return true;
}

pid_gains_t gain_updater_get_gains(void)
{
    pid_gains_t g = { s_gains[0], s_gains[1], s_gains[2] };
    return g;
}
