#include "tensor_lat_half.h"

int main() {

  intilizeDeviceProp(0);

  if (deviceProp.major < 6) // tesnore unit was added since Volta
    return 1;

  std::cout << "FP16 operand, FP32 accumalte:\n";
  tensor_lat<half, float>();

  std::cout << "\nFP16 operand, FP16 accumalte:\n";
  tensor_lat<half, half>();

  // tensor_lat<char,int>();

  return 1;
}
