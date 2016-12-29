/* lowrank P-wave propagation in TTI media */
#include <iostream>
#include <Eigen/Dense>
extern "C" {
#include <complex.h>
#include <fftw3.h>
#include <omp.h>
#include "rsf.h"
#include "Grid.h"
#include "sinc.h"
#include "abcutil.h"
#include "ts_kernel.h"
}
#include "vecmatop.hh"
#include "sample.hh"

#ifdef EIGEN_DEFAULT_DENSE_INDEX_TYPE
#undef EIGEN_DEFAULT_DENSE_INDEX_TYPE
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE std::size_t
#endif

using namespace Eigen;
using std::vector;

static void
wfld2d_inject(float** uo, float** ui, int nzo, int nxo, int nb)
{
#pragma omp parallel for schedule(dynamic, 1)
  for (int ix = 0; ix < nxo; ix++) {
    for (int iz = 0; iz < nzo; iz++) {
      uo[ix + nb][iz + nb] += ui[ix][iz];
    }
  }
  return;
}

int
main(int argc, char* argv[])
{
  bool verb, snap, adj;
  int nz, nx, nt, ns, nr;
  float dz, dx, dt, oz, ox;
  int nz0, nx0, nb;
  float oz0, ox0;
  int nkz, nkx;
  int nzpad, nxpad;

  float **u1, **u0;
  float *ws, *wr;
  char* atype;

  sf_file file_src = NULL;
  sf_file file_rec = NULL;
  sf_file file_inp = NULL;
  sf_file file_mdl = NULL;
  sf_file file_out = NULL;
  sf_axis az = NULL, ax = NULL;
  sf_axis az0 = NULL, ax0 = NULL;
  sf_axis at = NULL, as = NULL, ar = NULL;
  pt2d* src2d = NULL;
  pt2d* rec2d = NULL;
  scoef2d cssinc = NULL;
  scoef2d crsinc = NULL;

  float *wi = NULL, *wo = NULL;
  sf_axis ai = NULL, ao = NULL;
  scoef2d cisinc = NULL, cosinc = NULL;
  bool spt = false, rpt = false;
  bool ipt = false, opt = false;

  int seed, npk;
  float eps;
  Eigen::setNbThreads(omp_get_max_threads());

  sf_init(argc, argv);

  if (NULL == (atype = sf_getstring("atype"))) atype = const_cast<char*>("v");
  switch (atype[0]) {
    case 't':
      sf_warning("TTI model");
      break;
    case 'v':
      sf_warning("VTI model");
      break;
  }
  if (!sf_getbool("verb", &verb)) verb = false;
  if (!sf_getbool("snap", &snap)) snap = false;
  if (!sf_getbool("adj", &adj)) adj = false;
  if (!sf_getint("nb", &nb)) nb = 4;
  if (!sf_getint("seed", &seed)) seed = time(NULL);
  if (!sf_getfloat("eps", &eps)) eps = 1e-7;
  if (!sf_getint("npk", &npk)) npk = 20;
  if (sf_getstring("sou") != NULL) {
    spt = true;
    if (adj)
      opt = true;
    else
      ipt = true;
  }
  if (sf_getstring("rec") != NULL) {
    rpt = true;
    if (adj)
      ipt = true;
    else
      opt = true;
  }

  file_inp = sf_input("in");
  file_mdl = sf_input("model");
  if (spt) file_src = sf_input("sou");
  if (rpt) file_rec = sf_input("rec");
  file_out = sf_output("out");

  if (ipt)
    at = sf_iaxa(file_inp, 2);
  else
    at = sf_iaxa(file_inp, 3);

  if (spt) as = sf_iaxa(file_src, 2);
  if (rpt) ar = sf_iaxa(file_rec, 2);
  az0 = sf_iaxa(file_mdl, 1);
  ax0 = sf_iaxa(file_mdl, 2);
  nt = sf_n(at);
  dt = sf_d(at); // ot = sf_o(at);
  nz0 = sf_n(az0);
  dz = sf_d(az0);
  oz0 = sf_o(az0);
  nx0 = sf_n(ax0);
  dx = sf_d(ax0);
  ox0 = sf_o(ax0);

  if (spt) ns = sf_n(as);
  if (rpt) nr = sf_n(ar);
  nz = nz0 + 2 * nb;
  nx = nx0 + 2 * nb;
  oz = oz0 - nb * dz;
  ox = ox0 - nb * dx;
  az = sf_maxa(nz, oz, dz);
  ax = sf_maxa(nx, ox, dx);
  // sf_error("ox=%f ox0=%f oz=%f oz0=%f",ox,ox0,oz,oz0);

  nzpad = kiss_fft_next_fast_size(((nz + 1) >> 1) << 1);
  nkx = nxpad = kiss_fft_next_fast_size(nx);
  nkz = nzpad / 2 + 1;
  /* float okx = - 0.5f / dx; */
  float okx = 0.f;
  float okz = 0.f;
  float dkx = 1.f / (nxpad * dx);
  float dkz = 1.f / (nzpad * dz);

  // (1,2,3) = (z,x,y)
  float** vpz = sf_floatalloc2(nz, nx);
  float** vpx = sf_floatalloc2(nz, nx);
  float** eta = sf_floatalloc2(nz, nx);
  float** theta = NULL;
  if (atype[0] == 't') {
    theta = sf_floatalloc2(nz, nx);
  }
  float** tmparray = sf_floatalloc2(nz0, nx0);
  sf_floatread(tmparray[0], nz0 * nx0, file_mdl);
  expand2d(tmparray, vpz, az0, ax0, az, ax);
  sf_floatread(tmparray[0], nz0 * nx0, file_mdl);
  expand2d(tmparray, vpx, az0, ax0, az, ax);
  sf_floatread(tmparray[0], nz0 * nx0, file_mdl);
  expand2d(tmparray, eta, az0, ax0, az, ax);
  if (atype[0] == 't') {
    sf_floatread(tmparray[0], nz0 * nx0, file_mdl);
    expand2d(tmparray, theta, az0, ax0, az, ax);
  }
  free(*tmparray);
  free(tmparray);

  if (atype[0] == 't') {
    for (int ix = 0; ix < nx; ix++) {
      for (int iz = 0; iz < nz; iz++) {
        theta[ix][iz] *= SF_PI / 180.f;
      }
    }
  }

  float* kk[2];
  kk[0] = sf_floatalloc(nkz);
  kk[1] = sf_floatalloc(nkx);

  for (int ikx = 0; ikx < nkx; ++ikx) {
    kk[1][ikx] = okx + ikx * dkx;
    if (ikx >= nkx / 2) kk[1][ikx] = (ikx - nkx) * dkx;
    kk[1][ikx] *= 2 * SF_PI;
    // kk[1][ikx] *= kk[1][ikx];
  }
  for (int ikz = 0; ikz < nkz; ++ikz) {
    kk[0][ikz] = okz + ikz * dkz;
    kk[0][ikz] *= 2 * SF_PI;
    // kk[0][ikz] *= kk[0][ikz];
  }

  if (adj) {
    ai = ar;
    ao = as;
  } else {
    ai = as;
    ao = ar;
  }

  if (opt) {
    sf_oaxa(file_out, ao, 1);
    sf_oaxa(file_out, at, 2);
  } else {
    sf_oaxa(file_out, az0, 1);
    sf_oaxa(file_out, ax0, 2);
    sf_oaxa(file_out, at, 3);
  }
  sf_fileflush(file_out, NULL);

  if (spt) {
    src2d = pt2dalloc1(ns);
    pt2dread1(file_src, src2d, ns, 2);
    cssinc = sinc2d_make(ns, src2d, az, ax);
    ws = sf_floatalloc(ns);
    if (adj) {
      cosinc = cssinc;
      wo = ws;
    } else {
      cisinc = cssinc;
      wi = ws;
    }
  }
  if (rpt) {
    rec2d = pt2dalloc1(nr);
    pt2dread1(file_rec, rec2d, nr, 2);
    crsinc = sinc2d_make(nr, rec2d, az, ax);
    wr = sf_floatalloc(nr);
    if (adj) {
      cisinc = crsinc;
      wi = wr;
    } else {
      cosinc = crsinc;
      wo = wr;
    }
  }

  // lowrank decomposition
  size_t m = nz * nx;
  size_t n = nkz * nkx;
  void* tti_lrparam = lrdecomp_new(m, n);
  if (atype[0] == 't') {
    lrdecomp_init(tti_lrparam, seed, npk, eps, dt, nkz, nkx, kk[0],
                  kk[1], vpz[0], vpx[0], eta[0], atype, theta[0]);
  } else {
    lrdecomp_init(tti_lrparam, seed, npk, eps, dt, nkz, nkx, kk[0],
                  kk[1], vpz[0], vpx[0], eta[0], atype);
  }
  lrmat* tti_lrmat = lrdecomp_compute(tti_lrparam);
  int nrank = tti_lrmat->nrank;
  float** lft = sf_floatalloc2(m, nrank);
  float** rht = sf_floatalloc2(n, nrank);
  memcpy(lft[0], tti_lrmat->lft_data, sizeof(float) * m * nrank);
  memcpy(rht[0], tti_lrmat->rht_data, sizeof(float) * n * nrank);
  lrdecomp_delete(tti_lrparam, tti_lrmat);

  u0 = sf_floatalloc2(nz, nx);
  u1 = sf_floatalloc2(nz, nx);

  float* rwavem = (float*)fftwf_malloc(nzpad * nxpad * sizeof(float));
  fftwf_complex* cwave =
    (fftwf_complex*)fftwf_malloc(nkz * nkx * sizeof(fftwf_complex));
  fftwf_complex* cwavem =
    (fftwf_complex*)fftwf_malloc(nkz * nkx * sizeof(fftwf_complex));

  /* boundary conditions */
  float** ucut = NULL;
  if (!(ipt && opt)) ucut = sf_floatalloc2(nz0, nx0);
  float* damp = damp_make(nb);

  float wt = 1.0 / (nxpad * nzpad);
  fftwf_plan forward_plan;
  fftwf_plan inverse_plan;
  fftwf_init_threads();
  fftwf_plan_with_nthreads(omp_get_max_threads());
  forward_plan =
    fftwf_plan_dft_r2c_2d(nxpad, nzpad, rwavem, cwave, FFTW_MEASURE);
  fftwf_plan_with_nthreads(omp_get_max_threads());
  inverse_plan =
    fftwf_plan_dft_c2r_2d(nxpad, nzpad, cwavem, rwavem, FFTW_MEASURE);

  int itb, ite, itc;
  if (adj) {
    itb = nt - 1;
    ite = -1;
    itc = -1;
  } else {
    itb = 0;
    ite = nt;
    itc = 1;
  }

  if (adj) {
    for (int it = 0; it < nt; it++) {
      if (opt)
        sf_floatwrite(wo, sf_n(ao), file_out);
      else
        sf_floatwrite(ucut[0], nz0 * nx0, file_out);
    }
    sf_seek(file_out, 0, SEEK_SET);
  }

  float** ptrtmp = NULL;
  memset(u0[0], 0, sizeof(float) * nz * nx);
  memset(u1[0], 0, sizeof(float) * nz * nx);

  for (int it = itb; it != ite; it += itc) {
    if (verb) sf_warning("it = %d;", it);
    double tic = omp_get_wtime();

    if (ipt) {
      if (adj)
        sf_seek(file_inp, (off_t)(it) * sizeof(float) * sf_n(ai), SEEK_SET);
      sf_floatread(wi, sf_n(ai), file_inp);
    } else {
      if (adj)
        sf_seek(file_inp, (off_t)(it) * sizeof(float) * nz0 * nx0, SEEK_SET);
      sf_floatread(ucut[0], nz0 * nx0, file_inp);
    }

    /* apply absorbing boundary condition: E \times u@n-1 */
    damp2d_apply(u0, damp, nz, nx, nb);

    lr_fft_stepforward(u0, u1, rwavem, cwave, cwavem, lft, rht, forward_plan,
                       inverse_plan, nz, nx, nzpad, nxpad, nkz, nkx, nrank, wt,
                       adj);
    if (ipt) /* sinc3d_inject_with_vv(u0, wi, cisinc, vcomposite); */
      sinc2d_inject(u0, wi, cisinc);
    else
      wfld2d_inject(u0, ucut, nz0, nx0, nb);

    /* apply absorbing boundary condition: E \times u@n+1 */
    damp2d_apply(u0, damp, nz, nx, nb);

    /* loop over pointers */
    ptrtmp = u0;
    u0 = u1;
    u1 = ptrtmp;

    if (opt) {
      if (adj)
        sf_seek(file_out, (off_t)(it) * sizeof(float) * sf_n(ao), SEEK_SET);
      sinc2d_extract(u0, wo, cosinc);
      sf_floatwrite(wo, sf_n(ao), file_out);
    } else {
      if (adj)
        sf_seek(file_out, (off_t)(it) * sizeof(float) * nz0 * nx0, SEEK_SET);
      wwin2d(ucut, u0, nz0, nx0, nb);
      sf_floatwrite(ucut[0], nz0 * nx0, file_out);
    }

    double toc = omp_get_wtime();
    if (verb) fprintf(stderr, " clock = %lf;", toc - tic);
  } /* END OF FORWARD TIME LOOP */

  return 0;
}
