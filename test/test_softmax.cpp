#include <iostream>
#include <random>
#include <vector>

#include "../llm/softmax.hpp"
#include "../llm/tensor.hpp"
#include "../llm/utils.hpp"
using namespace std;

template<typename T>
void testSoftmax(){
  const uint32_t m = 2;
  const uint32_t n = 2;

  auto input = one::Tensor<T>(m * n);
  input.reShape({m, n});
  one::genRandom(input.data(), m * n);
  cout << "left: \n";
  input.print();

  auto out = one::softmaxFunc(input);
  cout << "out:\n";
  out.print();
}

int main(){
  testSoftmax<float>();

  vector<float> arr = {1.0, 3.0, 0.1};
  float ret = std::accumulate(arr.begin(), arr.end(), 0.0f, std::plus<float>());

  printf("ret: %.3f\n", ret);
  return 0;
}