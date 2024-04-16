#include "redistance.h"
#include <Eigen/Dense>
static const int res = 64;
float data[res * res * res];
// float data[res * res];
using namespace Eigen;
using namespace pybind11;
int main() {
    // Redistance r(2);
    // Eigen::MatrixXf phi = MatrixXf::Map(data, 128, 128);
    // float* p = phi.data();
    // phi.fill(-1.0f);
    // phi.block<64, 128>(0, 0).fill(1.0f);
    // r.redistance(phi, 1.0 / 128, 0);

    Redistance r(3);
    array_t<float> phi = array_t<float>({res, res, res});
    r.redistance_3d(phi, 1.0 / res, 0);
    return 0;


}