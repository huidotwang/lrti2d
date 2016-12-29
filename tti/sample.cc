/* TI phase-term sampling module */
#include "sample.hh"
#include <cstdarg>
#include <cstring>
#include "vecmatop.hh"

/*
 * Implementation of lrdecomp_ti constructor
 */
void*
lrdecomp_new(size_t m, size_t n)
{
  lrpar_ti* tilrpar = new lrpar_ti;
  tilrpar->base = new lrpar_base;
  tilrpar->base->kk = new float*[2];
  tilrpar->base->m = m;
  tilrpar->base->n = n;
  return tilrpar;
}
// vp0, vhor, eta
void
lrdecomp_init(void* lrpar, int seed_, int npk_, float eps_, float dt_, ...)
{
  lrpar_ti* tilrpar = (lrpar_ti*)lrpar;
  tilrpar->base->seed = seed_;
  tilrpar->base->npk = npk_;
  tilrpar->base->eps = eps_;
  tilrpar->base->dt = dt_;
  va_list ap;
  va_start(ap, dt_);
  tilrpar->nkz = va_arg(ap, int);
  tilrpar->nkx = va_arg(ap, int);
  tilrpar->base->kk[0] = va_arg(ap, float*);
  tilrpar->base->kk[1] = va_arg(ap, float*);
  tilrpar->vpz = va_arg(ap, float*);
  tilrpar->vpx = va_arg(ap, float*);
  tilrpar->eta = va_arg(ap, float*);
  tilrpar->atype = va_arg(ap, char*);
  if (tilrpar->atype[0] == 't')
  tilrpar->theta = va_arg(ap, float*);
  va_end(ap);
  return;
}

lrmat*
lrdecomp_compute(void* param)
{
  lrpar_ti* tilrpar = (lrpar_ti*)param;
  size_t m = tilrpar->base->m;
  size_t n = tilrpar->base->n;
  int seed = tilrpar->base->seed;
  int npk = tilrpar->base->npk;
  float eps = tilrpar->base->eps;
  srand48(seed);
  vector<size_t> lidx, ridx;
  MatrixXf mid;
  lowrank(m, n, sample, eps, npk, lidx, ridx, mid, param);
  size_t n2 = mid.cols();
  size_t m2 = mid.rows();
  vector<size_t> midx(m), nidx(n);
  for (size_t i = 0; i < m; i++) midx[i] = i;
  for (size_t i = 0; i < n; i++) nidx[i] = i;

  MatrixXf lmat_(m, m2);
  sample(midx, lidx, lmat_, param);
  MatrixXf lmat(m, n2);
  lmat = lmat_ * mid;

  MatrixXf rmat(n2, n);
  sample(ridx, nidx, rmat, param);
  rmat.transposeInPlace();

  lrmat* p = new lrmat;
  p->nrank = n2;
  p->lft_data = new float[m * n2];
  p->rht_data = new float[n * n2];
  memcpy(p->lft_data, lmat.data(), sizeof(float) * m * n2);
  memcpy(p->rht_data, rmat.data(), sizeof(float) * n * n2);
  return p;
}

void
lrdecomp_delete(void* lr_, void* fltmat_)
{
  lrpar_ti* lr = (lrpar_ti*)lr_;
  lr->vpz = NULL;
  lr->vpx = NULL;
  lr->eta = NULL;
  delete lr->base->kk;
  delete lr->base;
  delete lr;
  lrmat* fltmat = (lrmat*)fltmat_;
  delete fltmat->lft_data;
  delete fltmat->rht_data;
  delete fltmat;
}

/*
 * map 1d array index to 3d array index
 */
static inline void
index1dto2d(size_t i, int n1, int n2, int& i1, int& i2)
{
  size_t ii = i;
  i1 = ii % n1;
  ii /= n1;
  i2 = ii % n2;
}

int
sample(vector<size_t>& rows, vector<size_t>& cols, MatrixXf& res, void* ebuf)
{
  lrpar_ti* m_param = (lrpar_ti*)ebuf;
  float dt = m_param->base->dt;
  float* kz = m_param->base->kk[0];
  float* kx = m_param->base->kk[1];
  float* vpz = m_param->vpz;
  float* vpx = m_param->vpx;
  float* eta = m_param->eta;
  char* atype = m_param->atype;
  float* theta = m_param->theta;
  int nkx = m_param->nkx;
  int nkz = m_param->nkz;

  size_t nrow = rows.size();
  size_t ncol = cols.size();
  res.resize(nrow, ncol);
  res.setZero(nrow, ncol);
  for (size_t ir = 0; ir < nrow; ir++) {
    float vpz_ = vpz[rows[ir]]; // vpz
    float vpx_ = vpx[rows[ir]]; // vpx
    float eta_ = eta[rows[ir]]; // eta
    float scalar = (8.f * eta_) / (1.f + 2.f * eta_);

    float cos_theta = 1.0f;
    float sin_theta = 0.0f;
    if (atype[0] == 't') {
      cos_theta = cosf(theta[rows[ir]]);
      sin_theta = sinf(theta[rows[ir]]);
    }

    for (size_t ic = 0; ic < ncol; ic++) {
      int ikx, ikz;
      index1dto2d(cols[ic], nkz, nkx, ikz, ikx);
      float kx_ = kx[ikx] * cos_theta + kz[ikz] * sin_theta;
      float kz_ = -kx[ikx] * sin_theta + kz[ikz] * cos_theta;
      float vhkh = vpx_ * vpx_ * kx_ * kx_;
      float vvkz = vpz_ * vpz_ * kz_ * kz_;
      float psi = (vhkh + vvkz) * (vhkh + vvkz) - scalar * vhkh * vvkz;
      psi = .5f * (vhkh + vvkz + sqrtf(psi));
      psi = sqrtf(psi);
      res(ir, ic) = 2.f * cosf(psi * dt) - 2.f;
    }
  }
  return 0;
}
