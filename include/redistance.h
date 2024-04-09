#include <Eigen/Dense>

class Redistance{
    public:

        Redistance(int dim = 2);
        Eigen::MatrixXf& redistance(Eigen::MatrixXf &phi, float h, int init_boundary = 0);
        const int dim;
    // private: 
    //     void init_distance(Eigen::MatrixXf &phi);
    //     void sweep(Eigen::MatrixXf &phi, int incx, int incy, float h, int init_boundary);
};