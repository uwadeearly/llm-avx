#ifndef UTILS_HPP
#define UTILS_HPP
#include <random>
#include <iostream>
#include <immintrin.h>
#include "tensor.hpp"

namespace one {
template <typename T>
void genRandom(T* data, uint32_t len, bool isInt = false) {
  std::uniform_real_distribution<T> dist(-1.0, 1.0);
  std::uniform_int_distribution<int> distInt(-3, 4);

  std::random_device rd;
  std::mt19937 gen(rd());
  for (uint32_t i = 0; i < len; ++i) {
    data[i] = dist(gen);
    if (isInt) {
      data[i] = distInt(gen);
    }
  }
}

template <typename T>
void genOne(T* data, uint32_t len) {
  for (uint32_t i = 0; i < len; ++i) {
    data[i] = 1.0;
  }
}

template <typename T>
void print(T* data, uint32_t len) {
  for (uint32_t i = 0; i < len; ++i) {
    std::cout << data[i] << ", ";
    if ((i + 1) % 8 == 0) {
      std::cout << "\n";
    }
  }
  std::cout << "\n";
}


void inline debugData(__m256 data, uint32_t len=8){
  float temp[len];
  _mm256_store_ps(temp, data);
  for(uint i=0; i<len; ++i){
    std::cout << temp[i] << ", ";
  }
  std::cout << "\n";
}

template<typename T>
void checkValue(const Tensor<T>& golden, const Tensor<T>& out, float esp=1e-5){
  T* gorldD = golden.data();
  T* outD = out.data();
  uint32_t len = golden.getLen();
  assert(len == out.getLen());
  for(uint32_t i=0; i<len; ++i){
    float error = std::abs(gorldD[i] - outD[i]);
    if(error > esp){
      cout << "out data is bad, error value: " << error << "\n";
      assert(0);
    }
  }
}



}  // namespace one
#endif