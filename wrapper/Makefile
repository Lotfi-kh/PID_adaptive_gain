CC     = gcc
CFLAGS = -O2 -Wall -Wextra -I.
LDLIBS = -lm

# ── Host test (no ST AI runtime needed) ─────────────────────────────────────
# test_harness.c provides a mock backend inline; tests wrapper + gain math.
# Run:  make && ./test_host
.PHONY: all clean

all: test_host

test_host: actor_wrapper.o gain_updater.o test_harness.o
	$(CC) -o $@ $^ $(LDLIBS)

actor_wrapper.o : actor_wrapper.c  actor_wrapper.h actor_backend.h
	$(CC) $(CFLAGS) -c -o $@ $<

gain_updater.o  : gain_updater.c   gain_updater.h
	$(CC) $(CFLAGS) -c -o $@ $<

test_harness.o  : test_harness.c   actor_wrapper.h gain_updater.h actor_backend.h
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f *.o test_host

# ── On-target build notes ────────────────────────────────────────────────────
# In STM32CubeIDE / your embedded Makefile, add:
#   SOURCES += wrapper/actor_wrapper.c
#   SOURCES += wrapper/gain_updater.c
#   SOURCES += wrapper/actor_stai_backend.c
#   SOURCES += export/stm32_c_code/network.c
#   SOURCES += export/stm32_c_code/network_data.c
#   SOURCES += export/stm32_c_code/network_data_params.c
#
#   INCLUDES += -Iwrapper
#   INCLUDES += -Iexport/stm32_c_code
#   INCLUDES += -I<path-to-ai_platform.h>
#
#   LIBS += NetworkRuntime1020_CM7_GCC.a   (from x_cube/Middlewares/ST/AI/Lib/)
#
# Do NOT add actor_stai_backend.c to the host test build.
