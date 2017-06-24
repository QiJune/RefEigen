#ifndef ONLY_CPU
#include "cuda_runtime.h"
#define EIGEN_USE_GPU
#endif

#include <iostream>
#include <cmath>
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


void test_add(int size, float* t_a, float* t_b, float* t_c) {

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

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

#else

  Vec<float> a(t_a, size);
  Vec<float> b(t_b, size);
  Vec<float> c(t_c, size);

  Eigen::DefaultDevice dd;
  c.device(dd) = a + b;
  print<float>(t_c, size);

#endif
}

void test_mul(int size, float* t_a, float* t_b, float* t_c) {

  int width = sqrt(size);
  int height = width;

  Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
  dim_pair[0].first = 1;
  dim_pair[0].second = 0;

#ifndef ONLY_CPU
  float* d_a;
  float* d_b;
  float* d_c;
  cudaMalloc((void**)&d_a, size * sizeof(float));
  cudaMalloc((void**)&d_b, size * sizeof(float));
  cudaMalloc((void**)&d_c, size * sizeof(float));

  cudaMemcpy(d_a, t_a, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, t_b, size * sizeof(float), cudaMemcpyHostToDevice);

  Matrix<float> a(d_a, height, width);
  Matrix<float> b(d_b, height, width);
  Matrix<float> c(d_c, height, width);

  Eigen::CudaStreamDevice sd;
  Eigen::GpuDevice dd(&sd);
  c.device(dd) = a.contract(b, dim_pair);

  cudaMemcpy(t_c, d_c, size * sizeof(float), cudaMemcpyDeviceToHost);
  print<float>(t_c, size);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

#else

  Matrix<float> a(t_a, height, width);
  Matrix<float> b(t_b, height, width);
  Matrix<float> c(t_c, height, width);

  Eigen::DefaultDevice dd;
  c.device(dd) = a.contract(b, dim_pair);
  print<float>(t_c, size);

#endif
}

int main() {
  int size = 4;
  
  float* t_a = (float*)malloc(size * sizeof(float));
  float* t_b = (float*)malloc(size * sizeof(float));
  float* t_c = (float*)malloc(size * sizeof(float));
  for (int i = 0; i < size; i++) {
    t_a[i] = i;
    t_b[i] = i;
  }

  test_add(size, t_a, t_b, t_c);
  test_mul(size, t_a, t_b, t_c);

  free(t_a);
  free(t_b);
  free(t_c);

  return 0;
}
