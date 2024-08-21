#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "../llm/matmul.hpp"
#include "../llm/tensor.hpp"
#include "../llm/utils.hpp"
using namespace std;

template <typename T>
void testMatmul() {
  const uint32_t m = 2;
  const uint32_t n = 2;
  const uint32_t k = 2;
  auto left = one::Tensor<T>(m * k);
  left.reShape({m, k});
  auto right = one::Tensor<T>(k * n);
  right.reShape({k, n});
  one::genRandom(left.data(), m * k);
  cout << "left: \n";
  left.print();
  one::genRandom(right.data(), k * n);
  cout << "right:\n";
  right.print();

  auto out = one::matmulFunc(left, right);
  cout << "out:\n";
  out.print();
}

template <typename T>
void testMatmulPro() {
  const uint32_t m = 4;
  const uint32_t n = 4;
  const uint32_t k = 4;
  auto left = one::Tensor<T>(m * k);
  left.reShape({m, k});
  auto right = one::Tensor<T>(k * n);
  right.reShape({k, n});
  one::genRandom(left.data(), m * k, true);
  cout << "left: \n";
  left.print();
  one::genRandom(right.data(), k * n, true);
  cout << "right: \n";
  right.print();

  auto out = one::mamtulPro(left, right);
  cout << "out pro:\n";
  out.print();
}

void testRandom() {
  const int len = 8;
  float input[len] = {0};
  cout << "gen random value:\n";
  one::genRandom(input, len, true);
  one::print(input, len);
}

template <typename T>
void maxtrixProfiling() {
  const uint32_t m = 1024;
  const uint32_t n = 1024;
  const uint32_t k = 1024;
  auto left = one::Tensor<T>(m * k);
  left.reShape({m, k});
  auto right = one::Tensor<T>(k * n);
  right.reShape({k, n});
  one::genRandom(left.data(), m * k, true);

  auto start = std::chrono::high_resolution_clock::now();
  auto outPro = one::mamtulPro(left, right);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "matrix AVX took " << duration.count()
            << " milliseconds to execute." << std::endl;
  start = std::chrono::high_resolution_clock::now();
  auto out = one::matmulFunc(left, right);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "matrix took " << duration.count()
            << " milliseconds to execute." << std::endl;

  one::checkValue(out, outPro);
}

int main() {
  // testRandom();
  // testMatmulPro<float>();
  maxtrixProfiling<float>();
  // _tile_loadd()
  return 0;
}