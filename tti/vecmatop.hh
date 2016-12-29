#ifndef _VECMATOP_HH
#define _VECMATOP_HH
#include <Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <iostream>
#include <vector>
extern "C" {
#include <rsf.h>
}

#ifdef EIGEN_DEFAULT_DENSE_INDEX_TYPE
#undef EIGEN_DEFAULT_DENSE_INDEX_TYPE
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE std::size_t
#endif

using namespace Eigen;
using std::cerr;
using std::endl;
using std::vector;

int lowrank(size_t m, size_t n,
            int (*sample)(vector<size_t>&, vector<size_t>&, MatrixXf&,
                          void*),
            float eps, size_t npk, vector<size_t>& cidx,
            vector<size_t>& ridx, MatrixXf& mid, void* ebuf);
#endif
