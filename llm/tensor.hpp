#ifndef TENSOR_HPP
#define TENSOR_HPP
#include <assert.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include "context.hpp"

namespace one {
using namespace std;

template <typename T>
class Tensor {
 public:
  // using BaseTensor<T>::BaseTensor;
  ~Tensor() { delete[] data_; }
  Tensor(size_t len) : len_(len) { make(len); }

  Tensor(size_t len, Device device) : len_(len), device_(device) { make(len); }

  Tensor(const vector<T>& arr) {
    len_ = arr.size();
    make(len_);
    std::copy(arr.begin(), arr.end(), this->data_);
  }

  Tensor(const Tensor<T>& other) {
    if (this != &other) {
      len_ = other.len_;
      shape_ = other.shape_;
      data_ = other.data_;
      dims_ = other.dims_;
    }
  }

  Tensor<T>& operator=(const Tensor<T>& other) {
    if (this != &other) {
      len_ = other.len_;
      shape_ = other.shape_;
      data_ = other.data_;
      dims_ = other.dims_;
    }
    return *this;
  }

  void fill(const vector<T>& input) {
    size_t len = input.size();
    if (len > this->len_) {
      assert(0);
    }
    len_ = len;
    std::copy(input.begin(), input.end(), this->data_);
  }

  void reShape(const vector<uint32_t>& newShape) {
    size_t length = std::accumulate(newShape.begin(), newShape.end(), 1,
                                    std::multiplies<uint32_t>());
    cout<< "length: " << length << "\n";
    if (length != this->len_) {
      assert(0);
    }
    shape_.resize(newShape.size());
    std::copy(newShape.begin(), newShape.end(), shape_.begin());
    dims_ = newShape.size();
  }

  void print() {
    std::cout << ">>>>>>>>\n";
    uint32_t cols = shape_[dims_ - 1];
    uint32_t rows = std::accumulate(shape_.begin(), shape_.end() - 1, 1,
                                     std::multiplies<uint32_t>());
    for (uint32_t i = 0; i < rows; ++i) {
      for (uint32_t j = 0; j < cols; ++j) {
        std::cout << data_[i * cols + j] << ", ";
      }
      std::cout << "\n";
    }
    std::cout << "<<<<<<<<\n";
  }

  T* data() const { return data_; }
  size_t getLen() const { return len_; }
  uint32_t getDims() const { return dims_; }
  auto getShape() const { return shape_; }

 private:
  void make(size_t len) {
    data_ = new T[len];
    dims_ = 1;
    shape_.resize(dims_);
    shape_[0] = len;
  }

 private:
  T* data_;
  size_t len_;
  uint32_t dims_ = 1;
  std::vector<uint32_t> shape_ = {1, 1, 1, 1, 1};
  std::string name_;
  Device device_ = Device::CPU;
};

}  // namespace one

#endif  // TENSOR_HPP