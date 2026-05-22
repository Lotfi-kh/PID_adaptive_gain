/*
 * gain_updater.h — deployment-side gain update function.
 *
 * Applies action[3] from the RL actor to the shared roll/pitch PID gains.
 * All constants match deployment_contract.md exactly.
 */
#ifndef GAIN_UPDATER_H
#define GAIN_UPDATER_H

#include <stdbool.h>

typedef struct {
    float kp;
    float ki;
    float kd;
} pid_gains_t;

/* Initialise to default gains. Call once at startup / arm event. */
void gain_updater_init(void);

/* Reset gains to defaults and clear rate-limit state.
 * Call on arm, disarm, or any safety-triggered reset. */
void gain_updater_reset(void);

/* Override gains directly (for testing; also clears rate-limit state). */
void gain_updater_set_gains(float kp, float ki, float kd);

/* Apply one actor output step.
 *   action[3] — from actor_run(), must be in [-1,1]
 * Returns true  : gains updated normally.
 * Returns false : update blocked (rate-limit window exceeded, or non-finite
 *                 action). Gains are unchanged; caller should log the fault
 *                 and continue with the last good gains. */
bool gain_updater_apply(const float action[3]);

/* Read current shared gains (applied identically to roll and pitch). */
pid_gains_t gain_updater_get_gains(void);

#endif /* GAIN_UPDATER_H */
