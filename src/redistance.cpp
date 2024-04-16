#include "redistance.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdio>

namespace py = pybind11;
namespace cuda {
    void redistance(float* phi, int shapex, int shapey, float h, int init_boundary);
    void redistance(float *phi_np, int shapex, int shapey, int shapez, float h, int init_boundary);
    void redistance_rb(float *phi_np, int shapex, int shapey, int shapez, float h, int init_boundary);
}

Redistance::Redistance(int dim): dim(dim) {}

Eigen::MatrixXf& Redistance::redistance(Eigen::MatrixXf &phi, float h, int init_boundary){
    cuda::redistance(phi.data(), phi.rows(), phi.cols(), h, init_boundary);
    printf("args: rows, cols = %d, %d\n", phi.rows(), phi.cols());
    return phi;
}

pybind11::array_t<float> &Redistance::redistance_3d(pybind11::array_t<float> &phi, float h, int init_boundary){
    auto phi_ptr = phi.data();
    auto shape = phi.shape();
    cuda::redistance(static_cast<float*>(phi.mutable_data()), int(shape[0]), int(shape[1]), int(shape[2]), h, init_boundary);
    return phi;
}

pybind11::array_t<float> &Redistance::redistance_rb(pybind11::array_t<float> &phi, float h, int init_boundary){
    auto phi_ptr = phi.data();
    auto shape = phi.shape();
    cuda::redistance_rb(static_cast<float*>(phi.mutable_data()), int(shape[0]), int(shape[1]), int(shape[2]), h, init_boundary);
    return phi;
}
