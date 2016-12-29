/*
 * module to sample phase-term of acousti TI media
 */
#ifndef _SAMPLE_HH
#define _SAMPLE_HH
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include "lrdecomp.hh"

#ifdef EIGEN_DEFAULT_DENSE_INDEX_TYPE
#undef EIGEN_DEFAULT_DENSE_INDEX_TYPE
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE std::size_t
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace Eigen;
using std::vector;

typedef struct lrpar_ti_t lrpar_ti;
struct lrpar_ti_t
{
  lrpar_base* base;
  float *vpz, *vpx, *eta;
  char* atype;
  float* theta;
  int nkx, nkz;
};

int sample(vector<size_t>&, vector<size_t>&, MatrixXf&, void*);

#endif
