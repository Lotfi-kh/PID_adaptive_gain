/*
 * test_harness.c — wrapper + gain_updater integration test.
 *
 * Compile (host, no ST AI runtime needed):
 *   cd wrapper && make test_host
 *   ./test_host
 *
 * This file provides its own mock actor_backend_* so no ST AI library is needed.
 * Network correctness is separately validated by export/test_actor_standalone.c.
 *
 * Tests:
 *   Section A — actor_wrapper safety checks
 *   Section B — gain_updater math (DELTA_SCALE, clip, bounds)
 *   Section C — full stack: 10 deterministic test vectors end-to-end
 */

#include "actor_wrapper.h"
#include "actor_backend.h"
#include "gain_updater.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>

/* ── Tolerance ─────────────────────────────────────────────────────────────── */
#define ACT_TOL  2e-5f   /* float32 vs Python float64 reference */
#define GAIN_TOL 1e-6f   /* gain math is pure float32 arithmetic */

/* ── Reference data (from export/actor_joint_1p05M_test_vectors.csv) ────────── */
#define N_VECS 10

static const char *vec_name[N_VECS] = {
    "zero_obs",
    "nominal_hover",
    "roll_disturbance",
    "pitch_disturbance",
    "combined_disturbance",
    "high_rate_edge",
    "adapted_gains",
    "start_of_episode",
    "gains_near_lower_bound",
    "gains_near_upper_bound",
};

static const float obs_ref[N_VECS][13] = {
    {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,9.941860e-02f,5.0e-02f,1.988372e-01f,9.941860e-02f,5.0e-02f,1.988372e-01f,1.0f},
    {2e-02f,1e-02f,5e-02f,3e-02f,-1.1e-01f,-6e-02f,9.941860e-02f,5e-02f,1.988372e-01f,9.941860e-02f,5e-02f,1.988372e-01f,1.0f},
    {1.5e-01f,1e-02f,8e-01f,5e-02f,-1.25f,-8e-02f,9.941860e-02f,5e-02f,1.988372e-01f,9.941860e-02f,5e-02f,1.988372e-01f,1.0f},
    {1e-02f,1.5e-01f,5e-02f,8e-01f,-8e-02f,-1.25f,9.941860e-02f,5e-02f,1.988372e-01f,9.941860e-02f,5e-02f,1.988372e-01f,1.0f},
    {1.2e-01f,1e-01f,6e-01f,5e-01f,-9.6e-01f,-8e-01f,9.941860e-02f,5e-02f,1.988372e-01f,9.941860e-02f,5e-02f,1.988372e-01f,1.0f},
    {5e-01f,-4e-01f,5.0f,-4.0f,-6.5f,5.2f,9.941860e-02f,5e-02f,1.988372e-01f,9.941860e-02f,5e-02f,1.988372e-01f,1.0f},
    {8e-02f,6e-02f,3e-01f,2e-01f,-5.4e-01f,-3.8e-01f,1.744186e-01f,8.720930e-02f,2.906977e-01f,1.744186e-01f,8.720930e-02f,2.906977e-01f,5e-01f},
    {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,9.941860e-02f,5e-02f,1.988372e-01f,9.941860e-02f,5e-02f,1.988372e-01f,0.0f},
    {1e-01f,8e-02f,4e-01f,3.5e-01f,-7e-01f,-5.9e-01f,2.906980e-03f,5.813950e-03f,5.813950e-03f,2.906980e-03f,5.813950e-03f,5.813950e-03f,1.0f},
    {5e-02f,4e-02f,2e-01f,1.8e-01f,-3.5e-01f,-3e-01f,9.883721e-01f,9.767442e-01f,9.767442e-01f,9.883721e-01f,9.767442e-01f,9.767442e-01f,1.0f},
};

/* Expected action outputs from ONNX (Python float64 reference, rounded float32) */
static const float act_ref[N_VECS][3] = {
    { 1.00000000e+00f, -9.99476910e-01f, -1.00000000e+00f},
    { 1.00000000e+00f, -9.98581650e-01f, -1.00000000e+00f},
    {-1.00000000e+00f, -9.99998810e-01f, -1.00000000e+00f},
    {-9.98967290e-01f,  9.99985870e-01f,  1.00000000e+00f},
    {-9.99995590e-01f,  9.99999520e-01f,  9.98529610e-01f},
    {-1.00000000e+00f, -1.00000000e+00f,  1.00000000e+00f},
    { 1.00000000e+00f, -9.25831440e-01f,  9.57543250e-01f},
    { 9.99337910e-01f, -5.47394510e-01f, -1.00000000e+00f},
    { 9.98453500e-01f,  1.00000000e+00f,  1.00000000e+00f},
    {-1.00000000e+00f, -1.00000000e+00f,  1.00000000e+00f},
};

/* Starting gains before each vector (from CSV gains_prev columns) */
static const float kp_prev[N_VECS] = {0.171f,0.171f,0.171f,0.171f,0.171f,0.171f,0.300f,0.171f,0.005f,1.700f};
static const float ki_prev[N_VECS] = {8.6e-3f,8.6e-3f,8.6e-3f,8.6e-3f,8.6e-3f,8.6e-3f,1.5e-2f,8.6e-3f,1e-3f,0.168f};
static const float kd_prev[N_VECS] = {1.71e-3f,1.71e-3f,1.71e-3f,1.71e-3f,1.71e-3f,1.71e-3f,2.5e-3f,1.71e-3f,5e-5f,8.4e-3f};

/* Expected gains after applying act_ref[i] to prev gains (from CSV gains_new columns) */
static const float kp_new_ref[N_VECS] = {0.20500f,0.20500f,0.13700f,0.137035f,0.137000f,0.13700f,0.33400f,0.204977f,0.038947f,1.66600f};
static const float ki_new_ref[N_VECS] = {5.20178e-3f,5.20482e-3f,5.20000e-3f,1.199995e-2f,1.200000e-2f,5.20000e-3f,1.185217e-2f,6.73886e-3f,4.40000e-3f,1.646000e-1f};
static const float kd_new_ref[N_VECS] = {1.54e-3f,1.54e-3f,1.54e-3f,1.88e-3f,1.879750e-3f,1.88e-3f,2.66278e-3f,1.54e-3f,2.20e-4f,8.57e-3f};

/* ── Mock backend (host only — returns precomputed actions in call order) ──── */
static int s_mock_call = 0;
static int s_mock_mode = 0;   /* 0 = normal table, 1 = return zeros, 2 = fail */

int actor_backend_init(void)    { s_mock_call = 0; return (s_mock_mode == 2) ? -1 : 0; }
void actor_backend_deinit(void) {}
int actor_backend_run(const float obs[13], float action[3])
{
    (void)obs;
    if (s_mock_mode == 2) return -1;
    if (s_mock_call >= N_VECS) { memset(action, 0, 3*sizeof(float)); return 0; }
    memcpy(action, act_ref[s_mock_call++], 3*sizeof(float));
    return 0;
}

/* ── Helpers ────────────────────────────────────────────────────────────────── */
static int n_fail = 0;
static int n_pass = 0;

static void check(const char *label, bool ok)
{
    if (ok) { printf("  [PASS] %s\n", label); n_pass++; }
    else    { printf("  [FAIL] %s\n", label); n_fail++; }
}

static bool feq(float a, float b, float tol)
{
    return fabsf(a - b) <= tol;
}

/* ── Section A: actor_wrapper safety checks ─────────────────────────────────── */
static void test_actor_safety(void)
{
    float obs[13] = {0};
    float action[3];
    actor_status_t st;

    printf("\n[Section A] actor_wrapper safety\n");

    /* disabled at startup */
    actor_init();
    st = actor_run(obs, action);
    check("disabled-at-startup returns ACTOR_ERR_DISABLED", st == ACTOR_ERR_DISABLED);

    /* enable and run zeros — should pass */
    actor_set_enabled(true);
    s_mock_call = 0;
    st = actor_run(obs, action);
    check("enabled run succeeds", st == ACTOR_OK);

    /* NaN in obs[4] */
    float obs_nan[13] = {0};
    obs_nan[4] = 0.0f / 0.0f;
    s_mock_call = 0;
    st = actor_run(obs_nan, action);
    check("NaN in obs rejected", st == ACTOR_ERR_BAD_INPUT);

    /* Inf in obs[0] */
    float obs_inf[13] = {0};
    obs_inf[0] = 1.0f / 0.0f;
    s_mock_call = 0;
    st = actor_run(obs_inf, action);
    check("Inf in obs rejected", st == ACTOR_ERR_BAD_INPUT);

    /* disable then re-enable */
    actor_set_enabled(false);
    st = actor_run(obs, action);
    check("disable flag blocks run", st == ACTOR_ERR_DISABLED);
    actor_set_enabled(true);
    check("is_enabled reflects state", actor_is_enabled());

    actor_deinit();
}

/* ── Section B: gain_updater math ───────────────────────────────────────────── */
static void test_gain_updater(void)
{
    printf("\n[Section B] gain_updater math\n");

    gain_updater_init();
    pid_gains_t g = gain_updater_get_gains();
    check("default Kp = 0.171",   feq(g.kp, 0.171f,   1e-6f));
    check("default Ki = 8.6e-3",  feq(g.ki, 8.6e-3f,  1e-7f));
    check("default Kd = 1.71e-3", feq(g.kd, 1.71e-3f, 1e-7f));

    /* zero action: gains unchanged */
    float a0[3] = {0.0f, 0.0f, 0.0f};
    gain_updater_apply(a0);
    g = gain_updater_get_gains();
    check("zero action leaves Kp unchanged", feq(g.kp, 0.171f, 1e-6f));

    /* full positive action on Kp only */
    gain_updater_reset();
    float a_pos[3] = {1.0f, 0.0f, 0.0f};
    gain_updater_apply(a_pos);
    g = gain_updater_get_gains();
    float expected_kp = 0.171f + 3.4e-2f;
    check("Kp += DELTA_SCALE[0] on action[0]=+1", feq(g.kp, expected_kp, GAIN_TOL));

    /* clip at upper Kp bound */
    gain_updater_set_gains(1.71f, 8.6e-3f, 1.71e-3f);
    gain_updater_apply(a_pos);
    g = gain_updater_get_gains();
    check("Kp clipped at 1.72", feq(g.kp, 1.72f, GAIN_TOL));

    /* clip at lower Kp bound */
    gain_updater_set_gains(0.01f, 8.6e-3f, 1.71e-3f);
    float a_neg[3] = {-1.0f, 0.0f, 0.0f};
    gain_updater_apply(a_neg);
    g = gain_updater_get_gains();
    check("Kp clipped at 0.0", feq(g.kp, 0.0f, GAIN_TOL));

    /* NaN action blocked */
    gain_updater_reset();
    float a_nan[3] = {0.0f / 0.0f, 0.0f, 0.0f};
    bool ok = gain_updater_apply(a_nan);
    check("NaN action blocked", !ok);
    g = gain_updater_get_gains();
    check("gains unchanged after NaN block", feq(g.kp, 0.171f, 1e-6f));

    /* per-second rate limit fires after MAX_CHANGE_PER_SEC accumulated */
    gain_updater_reset();
    float a_full[3] = {1.0f, 1.0f, 1.0f};
    bool froze = false;
    for (int t = 0; t < 48; t++) {
        bool r = gain_updater_apply(a_full);
        if (!r) { froze = true; break; }
    }
    check("per-second rate limit triggers within 48 ticks", froze);
}

/* ── Section C: 10 deterministic test vectors end-to-end ────────────────────── */
static void test_full_stack(void)
{
    printf("\n[Section C] Full stack — 10 deterministic vectors\n");
    printf("  %-26s  %8s  %8s  %6s\n",
           "Vector", "ActErr", "GainErr", "Status");
    printf("  %-26s  %8s  %8s  %6s\n",
           "--------------------------", "--------", "--------", "------");

    s_mock_mode  = 0;
    s_mock_call  = 0;
    actor_init();
    actor_set_enabled(true);

    float worst_act  = 0.0f;
    float worst_gain = 0.0f;
    int   vec_fail   = 0;

    for (int v = 0; v < N_VECS; v++) {
        /* Set the starting gains for this vector */
        gain_updater_set_gains(kp_prev[v], ki_prev[v], kd_prev[v]);

        /* Run actor */
        float action[3];
        actor_status_t st = actor_run(obs_ref[v], action);
        if (st != ACTOR_OK) {
            printf("  %-26s  actor_run failed: %d\n", vec_name[v], (int)st);
            n_fail++;
            vec_fail++;
            continue;
        }

        /* Check action matches reference */
        float act_err = 0.0f;
        for (int k = 0; k < 3; k++) {
            float e = fabsf(action[k] - act_ref[v][k]);
            if (e > act_err) act_err = e;
        }
        if (act_err > worst_act) worst_act = act_err;

        /* Apply to gain updater */
        gain_updater_apply(action);
        pid_gains_t g = gain_updater_get_gains();

        /* Check resulting gains */
        float gain_err = 0.0f;
        float ekp = fabsf(g.kp - kp_new_ref[v]);
        float eki = fabsf(g.ki - ki_new_ref[v]);
        float ekd = fabsf(g.kd - kd_new_ref[v]);
        gain_err = ekp > eki ? ekp : eki;
        if (ekd > gain_err) gain_err = ekd;
        if (gain_err > worst_gain) worst_gain = gain_err;

        bool ok = (act_err <= ACT_TOL) && (gain_err <= GAIN_TOL);
        if (!ok) vec_fail++;
        printf("  %-26s  %8.2e  %8.2e  %s\n",
               vec_name[v], (double)act_err, (double)gain_err,
               ok ? "PASS" : "FAIL");
        if (ok) n_pass++; else n_fail++;
    }

    printf("\n  Worst action error : %.2e  (tol %.0e)\n",
           (double)worst_act,  (double)ACT_TOL);
    printf("  Worst gain error   : %.2e  (tol %.0e)\n",
           (double)worst_gain, (double)GAIN_TOL);

    actor_deinit();
}

/* ── main ────────────────────────────────────────────────────────────────────── */
int main(void)
{
    printf("========================================\n");
    printf("  Actor Wrapper + Gain Updater Test\n");
    printf("========================================\n");

    test_actor_safety();
    test_gain_updater();
    test_full_stack();

    printf("\n========================================\n");
    printf("  PASS: %d   FAIL: %d\n", n_pass, n_fail);
    printf("  OVERALL: %s\n", n_fail == 0 ? "PASS" : "FAIL");
    printf("========================================\n");

    return n_fail > 0 ? 1 : 0;
}
