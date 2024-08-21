#include <immintrin.h>

#include <iostream>

#include "../llm/utils.hpp"
using namespace std;

void addAVX(float* a, float* b, float* c, uint32_t len) {
  __m256 am, bm, cm;
  for (uint i = 0; i < len; i += 8) {
    am = _mm256_load_ps(a + i);
    bm = _mm256_load_ps(b + i);
    cm = _mm256_add_ps(am, bm);
    _mm256_store_ps(c + i, cm);
  }
}

void mulAVX(float* a, float* b, float* c, uint32_t len) {
  __m256 am, bm, cm;
  for (uint i = 0; i < len; i += 8) {
    am = _mm256_load_ps(a + i);
    bm = _mm256_load_ps(b + i);
    cm = _mm256_mul_ps(am, bm);
    _mm256_store_ps(c + i, cm);
  }
}

void loadAVX(float* input, float* out, uint32_t stride) {
  __m256 inm;
  inm = _mm256_load_ps(input + stride);
  // inm = _mm256_set1_ps(10.0);
  _mm256_store_ps(out + stride, inm);
}
#include <iostream>
#include <immintrin.h>


void matrixAVX(float* A, float* B, float* C, int n) {
    __m256 a, b, c;
    for (int i = 0; i < n; i += 8) {
        for (int j = 0; j < n; j++) {
            c = _mm256_setzero_ps();
            for (int k = 0; k < n; k++) {
                a = _mm256_loadu_ps(&A[i * n + k]);
                b = _mm256_broadcast_ss(&B[k * n + j]);
                c = _mm256_add_ps(c, _mm256_mul_ps(a, b));
            }
            _mm256_storeu_ps(&C[i * n + j], c);
        }
    }
}

void testMatrix() {
    int n = 2;
    float A[] = {1.0f, 2.0f, 
                3.0f, 4.0f,};

    float B[] = {1.0f, 2.0f, 
                 3.0f, 4.0f,};

    float C[16] = {0};

    matrixAVX(A, B, C, n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << C[i * n + j] << " ";
        }
        std::cout << std::endl;
    }

}

int main() {
  const uint32_t len = 8*8;
  // #pragma pack(8) 
  // float a[len] __attribute__((aligned(16)))
  // float a[len] __attribute__((aligned(512))) = {0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
  // float a[len];
  // for(uint32_t i=0; i<len; ++i){
  //   a[i] = 0.1 * i;
  // }
  // float out[len] = {0};
  // loadAVX(a, out, 0);
  // one::print(out, len);

  testMatrix();

  return 0;
}