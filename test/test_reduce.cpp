#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "../llm/matmul.hpp"
#include "../llm/reduce.hpp"
#include "../llm/tensor.hpp"
#include "../llm/utils.hpp"
using namespace std;

void testReduceSum() {
  const uint32_t len = 10;
  float input[len] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
  float ret = one::reduceSum(input, len);
  cout << "sum result: " << ret << "\n";
}

int main() {
  testReduceSum();
  return 0;
}