#include "redistance.h"
#include <Eigen/Dense>
float data[128 * 128];
using namespace Eigen;
int main() {
    Redistance r(2);
    Eigen::MatrixXf phi = MatrixXf::Map(data, 128, 128);
    float* p = phi.data();
    phi.fill(-1.0f);
    phi.block<64, 128>(0, 0).fill(1.0f);
    r.redistance(phi, 1.0 / 128, 0);
}