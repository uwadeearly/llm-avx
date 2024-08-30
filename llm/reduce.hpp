#ifndef REDUCE_HPP
#define REDUCE_HPP
#include <immintrin.h>

#include <iostream>

#include "context.hpp"
#include "tensor.hpp"
namespace one {
float reduceSum(float* input, uint32_t size) {
  __m256 sumVal = _mm256_setzero_ps();
  uint32_t sizeAlign = ALIAGN256(size);
  uint32_t sizeRemain = size - sizeAlign;
  for (uint32_t i = 0; i < sizeAlign; i += 8) {
    __m256 tmp = _mm256_loadu_ps(input + i);
    sumVal = _mm256_add_ps(sumVal, tmp);
  }
  float ret = sumVal[0] + sumVal[1] + sumVal[2] + sumVal[3] + sumVal[4] +
              sumVal[5] + sumVal[6] + sumVal[7];
  for(uint32_t i=0; i<sizeRemain; ++i){
    ret += input[sizeAlign+i];
  }
  return ret;
}

float reduceMax(float* input, uint32_t size) {
  __m256 maxVal = _mm256_setzero_ps();
  uint32_t sizeAlign = ALIAGN256(size);
  uint32_t sizeRemain = size - sizeAlign;
  for (uint32_t i = 0; i < sizeAlign; i += 8) {
    __m256 tmp = _mm256_loadu_ps(input + i);
    maxVal = _mm256_max_ps(maxVal, tmp);
  }
  
  float one = std::max(maxVal[0], maxVal[1]);
  float two = std::max(maxVal[2],  maxVal[3]);
  float three = std::max(maxVal[4],  maxVal[5]);
  float four = std::max(maxVal[6],  maxVal[7]);

  float five = std::max(one, three);
  float six = std::max(two, four);
  
  float ret = std::max(five, six);
  
  for(uint32_t i=0; i<sizeRemain; ++i){
    ret = std::max(ret, input[sizeAlign+i]);
  }
  return ret;
}
}  // namespace one
#endif  // REDUCE_HPP