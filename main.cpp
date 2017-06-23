#include <iostream>
// #define EIGEN_USE_THREADS
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>


template <typename T>
using EigenVectorArrayMap = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> >;

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

  Vec<float> a(t_a, size);
  Vec<float> b(t_b, size);
  Vec<float> c(t_c, size);

  {
    Eigen::DefaultDevice dd;
    c.device(dd) = a + b;
  }

//  {
//    Eigen::ThreadPoolDevice dd(4 /* number of threads to use */);
//    c.device(dd) = a + b;
//  }

  return 0;
}