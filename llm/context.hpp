#ifndef CONTEXT_HPP
#define CONTEXT_HPP
#include <iostream>

#define ALIAGN256(x) ((x / 8) * 8)


namespace one {
enum class Device { CPU, GPU, XPU };

enum class DTYPE { ISFLOAT, ISINT32_T };

template <typename T>
struct IsType {};

template <>
struct IsType<float> {
  const static DTYPE value = DTYPE::ISFLOAT;
};

template <>
struct IsType<int32_t> {
  const static DTYPE value = DTYPE::ISINT32_T;
};

}  // namespace one
#endif  // CONTEXT_HPP