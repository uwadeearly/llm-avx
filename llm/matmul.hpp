#ifndef MATMUL_HPP
#define MATMUL_HPP
#include <assert.h>
#include <immintrin.h>

#include <type_traits>

#include "context.hpp"
#include "tensor.hpp"
#include "utils.hpp"

namespace one {
template <typename T>
Tensor<T> matmulFunc(const Tensor<T>& left, const Tensor<T>& right) {
  auto leftShape = left.getShape();
  auto rightShape = right.getShape();

  const uint32_t M = leftShape[0];
  const uint32_t K = leftShape[1];
  assert(K == rightShape[0]);
  const uint32_t N = rightShape[1];

  Tensor<T> out(M * N);
  out.reShape({M, N});
  T* outD = out.data();
  T* lD = left.data();
  T* rD = right.data();

  for (uint32_t i = 0; i < M; ++i) {
    for (uint32_t j = 0; j < N; ++j) {
      T temp = 0;
      for (uint32_t q = 0; q < K; ++q) {
        temp += lD[i * K + q] * rD[q * N + j];
      }
      outD[i * N + j] = temp;
    }
  }
  return out;
}

void matmulAvxF(float* A, float* B, float* C, int m, int n, int p) {
  for (int i = 0; i < m; i += 8) {
    for (int j = 0; j < p; ++j) {
      __m256 sum = _mm256_setzero_ps();
      for (int k = 0; k < n; ++k) {
        // load left and right matrix
        __m256 a = _mm256_loadu_ps(A + i * n + k);
        // cout << "load left matrix...\n";
        __m256 b = _mm256_broadcast_ss(B + k * n + j);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(a, b));

        // debugData(sum);
      }
      _mm256_storeu_ps(C + i * p + j, sum);
    }
  }
}

void matmulAvxI(int32_t* A, int32_t* B, int32_t* C, int m, int n, int p) {
  return;
}

template <typename T>
Tensor<T> mamtulPro(const Tensor<T>& left, const Tensor<T>& right) {
  auto leftShape = left.getShape();
  auto rightShape = right.getShape();

  const uint32_t M = leftShape[0];
  const uint32_t K = leftShape[1];
  assert(K == rightShape[0]);
  const uint32_t N = rightShape[1];

  Tensor<T> out(M * N);
  out.reShape({M, N});

  if constexpr (std::is_same<T, float>::value) {
    cout << "avx float...\n";
    matmulAvxF(left.data(), right.data(), out.data(), M, K, N);
    return out;
  } else if constexpr (std::is_same<T, float>::value) {
    (left.data(), right.data(), out.data(), M, K, N);
    return out;
  } else {
    assert(0);
  }
}

}  // namespace one

#endif  // MATMUL_HPP