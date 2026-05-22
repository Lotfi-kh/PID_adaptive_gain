/*
 * actor_stai_backend.c — ST AI runtime backend.
 *
 * Compile only for on-target builds; links against:
 *   NetworkRuntime*.a  (from x_cube/Utilities)
 *   network.c / network_data.c / network_data_params.c  (from export/stm32_c_code/)
 *
 * Include paths required:
 *   -I../export/stm32_c_code
 *   -I<path-to-ai_platform.h>   (e.g. x_cube/Middlewares/ST/AI/Inc/)
 *
 * Do NOT modify the files under export/stm32_c_code/. They are generated.
 */
#include "actor_backend.h"
#include "network.h"
#include "network_data.h"
#include <string.h>

static ai_handle net = AI_HANDLE_NULL;
static ai_buffer *ai_in;
static ai_buffer *ai_out;

/* Activations buffer — size from generated network_data_params.h */
AI_ALIGNED(32)
static ai_u8 activations[AI_NETWORK_DATA_ACTIVATION_1_SIZE];

int actor_backend_init(void)
{
    const ai_handle acts[] = { activations };
    ai_error err = ai_network_create_and_init(&net, acts, NULL);
    if (err.type != AI_ERROR_NONE)
        return -1;
    ai_in  = ai_network_inputs_get(net, NULL);
    ai_out = ai_network_outputs_get(net, NULL);
    return 0;
}

int actor_backend_run(const float obs[13], float action[3])
{
    /* Both I/O are in the activations buffer (allocate-inputs/allocate-outputs).
     * Write directly into the pre-allocated input slot, read from output slot. */
    memcpy(ai_in[0].data,  obs,    13 * sizeof(float));
    ai_i32 n = ai_network_run(net, &ai_in[0], &ai_out[0]);
    if (n != 1) return -1;
    memcpy(action, ai_out[0].data, 3 * sizeof(float));
    return 0;
}

void actor_backend_deinit(void)
{
    if (net != AI_HANDLE_NULL) {
        ai_network_destroy(net);
        net = AI_HANDLE_NULL;
    }
}
