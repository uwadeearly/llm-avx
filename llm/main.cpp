#include <iostream>
#include <vector>

#include "tensor.hpp"
using namespace std;

/* compiler
 * gcc -mavx2 fun.c
 * g++ source.cpp -mavx512bw -msse -o source
 */

int main() {
  vector<float> arr = {0.1, 0.4, 0.7, 1.3, 3.4, 4.5};
  auto tensor = one::Tensor<float>(arr);
  tensor.print();
  cout << "this is tensor...\n";
  tensor.reShape({2, 3});

  uint32_t dims = tensor.getDims();
  cout << "tensor dims: " << dims << "\n";
  size_t length = tensor.getLen();
  cout << "tensor length: " << length << "\n";
  auto shape = tensor.getShape();

  cout << "shape[";
  for (const auto& elem : shape) {
    cout << elem << ", ";
  }
  cout << "]\n";

  return 0;
}