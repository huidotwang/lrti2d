#include "ts_kernel.h"

void
lr_fft_stepforward(float** u0, float** u1, float* rwavem, fftwf_complex* cwave,
                   fftwf_complex* cwavem, float** lft, float** rht,
                   fftwf_plan forward_plan, fftwf_plan inverse_plan, int nz,
                   int nx, int nzpad, int nxpad, int nkz, int nkx, int nrank,
                   float wt, bool adj)
{
#pragma omp parallel for schedule(dynamic, 1)
  for (int ix = 0; ix < nxpad; ix++) {
    memset(&rwavem[ix * nzpad], 0, sizeof(float) * nzpad);
    memset(&cwave[ix * nkz], 0, sizeof(fftwf_complex) * nkz);
    memset(&cwavem[ix * nkz], 0, sizeof(fftwf_complex) * nkz);
  }

  if (adj) { /* adjoint modeling */
    for (int im = 0; im < nrank; im++) {
#pragma omp parallel for schedule(dynamic, 1)
      /* rwavem = L^T_i \schur_dot rwave */
      for (int j = 0; j < nx; j++) {
        for (int i = 0; i < nz; i++) {
          int ii = j * nz + i;
          int jj = j * nzpad + i;
          rwavem[jj] = lft[im][ii] * u1[j][i];
        }
      }
      /* --- 2D forward Fourier transform ---*/
      fftwf_execute(forward_plan);

#pragma omp parallel for schedule(dynamic, 1)
      /* cwavem += R^T_i \schur_dot cwave */
      for (int j = 0; j < nkx; j++) {
        for (int ii = 0; ii < nkz; ii++) {
          int idx = j * nkz + ii;
          cwavem[idx] += rht[im][idx] * cwave[idx];
        }
      }
    }
    /* --- 2D backward Fourier transform ---*/
    fftwf_execute(inverse_plan);

#pragma omp parallel for schedule(dynamic, 1)
    for (int j = 0; j < nx; j++) {
      for (int i = 0; i < nz; i++) {
        int jj = j * nzpad + i;
        u0[j][i] = 2.0f * u1[j][i] - u0[j][i];
        /* FFT normalization */
        u0[j][i] += rwavem[jj] * wt;
      }
    }
  } else { /* forward modeling */
#pragma omp parallel for schedule(dynamic, 1)
    for (int j = 0; j < nx; j++) {
      for (int i = 0; i < nz; i++) {
        int jj = j * nzpad + i;
        u0[j][i] = 2.0f * u1[j][i] - u0[j][i];
        rwavem[jj] = u1[j][i];
      }
    }

    /* --- 2D forward Fourier transform ---*/
    fftwf_execute(forward_plan);

    for (int im = 0; im < nrank; im++) {
/* element-wise vector multiplication: u@t(kz,kx) * M3(im,:) */
#pragma omp parallel for schedule(dynamic, 1)
      for (int j = 0; j < nkx; j++) {
        for (int ii = 0; ii < nkz; ii++) {
          int idx = j * nkz + ii;
          cwavem[idx] = rht[im][idx] * cwave[idx];
        }
      }

      /* --- 2D backward Fourier transform ---*/
      fftwf_execute(inverse_plan);

/* element-wise vector multiplication: M1(:,im) * u@t(z,x) */
#pragma omp parallel for schedule(dynamic, 1)
      for (int j = 0; j < nx; j++) {
        for (int i = 0; i < nz; i++) {
          int ii = j * nz + i;
          int jj = j * nzpad + i;
          /* FFT normalization \times wt */
          u0[j][i] += lft[im][ii] * rwavem[jj] * wt;
        }
      }
    }
  }
  return;
}
