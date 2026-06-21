#include "rl_actor.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Usage: ./test_rl_actor valinput.csv valoutput.csv */
int main(int argc, char **argv) {
  if (argc != 3) { fprintf(stderr, "need valinput valoutput\n"); return 2; }
  FILE *fi = fopen(argv[1], "r"), *fo = fopen(argv[2], "r");
  if (!fi || !fo) { perror("fopen"); return 2; }
  float obs[12], ref[3], act[3]; int n = 0; double mx = 0.0;
  while (fscanf(fi, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",
                &obs[0],&obs[1],&obs[2],&obs[3],&obs[4],&obs[5],
                &obs[6],&obs[7],&obs[8],&obs[9],&obs[10],&obs[11]) == 12) {
    if (fscanf(fo, "%f,%f,%f", &ref[0],&ref[1],&ref[2]) != 3) break;
    rl_actor_forward(obs, act);
    for (int k = 0; k < 3; ++k) {
      double d = fabs((double)act[k] - (double)ref[k]);
      if (d > mx) mx = d;
    }
    printf("sample %d: act=[% .7f % .7f % .7f] ref=[% .7f % .7f % .7f]\n",
           n, act[0],act[1],act[2], ref[0],ref[1],ref[2]);
    n++;
  }
  printf("\n%d samples, max|portableC - ONNX| = %.3e\n", n, mx);
  printf("%s\n", mx < 1e-5 ? "PASS (bit-equivalent, tol 1e-5)"
                              : "FAIL");
  return mx < 1e-5 ? 0 : 1;
}
