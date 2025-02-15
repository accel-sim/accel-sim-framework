#include "MaxFlops_double.h"

int main() {
  intilizeDeviceProp(0);

  dpu_max_flops();

  return 1;
}
