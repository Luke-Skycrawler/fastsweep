#include "redistance.h"
#include <cstdio>
namespace cuda {
    void redistance(float* phi, int shapex, int shapey, float h, int init_boundary);
}

Redistance::Redistance(int dim): dim(dim) {}

Eigen::MatrixXf& Redistance::redistance(Eigen::MatrixXf &phi, float h, int init_boundary){
    cuda::redistance(phi.data(), phi.rows(), phi.cols(), h, init_boundary);
    printf("args: rows, cols = %d, %d\n", phi.rows(), phi.cols());
    return phi;
}
