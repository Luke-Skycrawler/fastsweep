#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
class Redistance{
    public:

        Redistance(int dim = 2);
        Eigen::MatrixXf& redistance(Eigen::MatrixXf &phi, float h, int init_boundary = 0);
        pybind11::array_t<float> &redistance_3d(pybind11::array_t<float> &phi, float h, int init_boundary = 0);
        pybind11::array_t<float> &redistance_rb(pybind11::array_t<float> &phi, float h, int init_boundary = 0);
        const int dim;
    // private: 
    //     void init_distance(Eigen::MatrixXf &phi);
    //     void sweep(Eigen::MatrixXf &phi, int incx, int incy, float h, int init_boundary);
};