#include <iostream>
#ifndef ONLY_CPU
#include "cuda_runtime.h"
#define EIGEN_USE_GPU
#endif


#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
template <typename T>
using Vec = Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>,
  Eigen::Aligned>;

template <typename T>
using Matrix = Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex>,
  Eigen::Aligned>;

template <typename T>
void print(T* input, int size) {
  for (int i = 0; i < size; i++) {
    std::cout << input[i] << " ";
  }
  std::cout << std::endl;
}

int main() {

  int size = 10;
  float* t_a = (float*)malloc(size * sizeof(float));
  float* t_b = (float*)malloc(size * sizeof(float));
  float* t_c = (float*)malloc(size * sizeof(float));
  for (int i = 0; i < size; i++) {
    t_a[i] = i;
    t_b[i] = i;
  }
  
#ifndef ONLY_CPU
  float* d_a;
  float* d_b;
  float* d_c;
  cudaMalloc((void**)&d_a, size * sizeof(float));
  cudaMalloc((void**)&d_b, size * sizeof(float));
  cudaMalloc((void**)&d_c, size * sizeof(float));

  cudaMemcpy(d_a, t_a, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, t_b, size * sizeof(float), cudaMemcpyHostToDevice);
  
  Vec<float> a(d_a, size);
  Vec<float> b(d_b, size);
  Vec<float> c(d_c, size);
 
  Eigen::CudaStreamDevice sd;
  Eigen::GpuDevice dd(&sd);
  c.device(dd) = a + b;

  cudaMemcpy(t_c, d_c, size * sizeof(float), cudaMemcpyDeviceToHost);
  print<float>(t_c, size);

#else

  Vec<float> a(t_a, size);
  Vec<float> b(t_b, size);
  Vec<float> c(t_c, size);

  Eigen::DefaultDevice dd;
  c.device(dd) = a + b;
  print<float>(t_c, size);

#endif

  return 0;
}
