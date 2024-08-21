#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP
#include <immintrin.h>

#include <cmath>

#include "context.hpp"
#include "tensor.hpp"

namespace one {
template <typename T>
Tensor<T> softmaxFunc(const Tensor<T>& input) {
  uint32_t dims = input.getDims();
  if (dims != 2) {
    assert(0);
  }
  auto shape = input.getShape();
  uint32_t cols = shape[0];
  uint32_t rows = shape[1];
  T* inD = input.data();
  Tensor<T> out = Tensor<T>(cols * rows);
  out.reShape({rows, cols});
  T* outD = out.data();

  for (uint32_t i = 0; i < rows; ++i) {
    // get max val per cols
    T* maxVal = std::max_element(inD + i * cols, inD + (i + 1) * cols);

     T sumVal = 0;
    for (uint32_t j = 0; j < cols; ++j) {
      outD[i * cols + j] = inD[i * cols + j] - *maxVal;
    }

    // exp
    for (uint32_t j = 0; j < cols; ++j) {
      T tmp = std::exp(outD[i * cols + j]);
      outD[i * cols + j] = tmp;
      sumVal += tmp;
    }
  
  //  T sumVal = std::accumulate(outD + i * cols, outD + (i + 1) * cols, 0.0f,
  //                              st::plus<T>());
    cout << "sum: " << sumVal << "\n";
    for (uint32_t j = 0; j < cols; ++j) {
      outD[i * cols + j] = outD[i * cols + j] / sumVal;
    }
  }
  return out;
}

template <typename T>
Tensor<T> softmaxFuncPro(const Tensor<T>& input) {
  uint32_t dims = input.getDims();
  if (dims != 2) {
    assert(0);
  }

  auto shape = input.getShape();
  uint32_t cols = shape[0];
  uint32_t rows = shape[1];
  T* inD = input.data();
  Tensor<T> out = Tensor<T>(cols * rows);
  out.reShape({rows, cols});
  T* outD = out.data();

  // __m256 a = _mm256_exp_ps();

  
}

}  // namespace one
#endif