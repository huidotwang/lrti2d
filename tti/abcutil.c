/* Absorbing boundary utility */
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include "abcutil.h"

float*
damp_make(int nb)
{
  float* damp = malloc(nb * sizeof(float));
  assert(damp);
  for (int ib = 0; ib < nb; ib++) {
    float fb = 1.f * ib / nb;
    damp[nb - 1 - ib] = exp(-fb * fb / 8.f);
  }
  return damp;
}

void
damp2d_apply(float** uu, float* damp, int nz, int nx, int nb)
{
  if (damp != NULL) {
/* left/right boundary */
#pragma omp parallel for schedule(dynamic, 1)
      for (int ib = 0; ib < nb; ib++) {
        for (int iz = 0; iz < nz; iz++) {
          uu[ib][iz] *= damp[ib];
          uu[nx - 1 - ib][iz] *= damp[ib];
        }
      }

/* top/bottom boundary */
#pragma omp parallel for schedule(dynamic, 1)
      for (int ix = 0; ix < nx; ix++) {
        for (int ib = 0; ib < nb; ib++) {
          uu[ix][ib] *= damp[ib];
          uu[ix][nz - 1 - ib] *= damp[ib];
        }
      }
  }
  return;
}


void
damp3d_apply(float*** uu, float* damp, int nz, int nx, int ny, int nb)
{
  if (damp != NULL) {
/* left/right boundary */
#pragma omp parallel for schedule(dynamic, 1)
    for (int iy = 0; iy < ny; iy++) {
      for (int ib = 0; ib < nb; ib++) {
        for (int iz = 0; iz < nz; iz++) {
          uu[iy][ib][iz] *= damp[ib];
          uu[iy][nx - 1 - ib][iz] *= damp[ib];
        }
      }
    }

/* top/bottom boundary */
#pragma omp parallel for schedule(dynamic, 1)
    for (int iy = 0; iy < ny; iy++) {
      for (int ix = 0; ix < nx; ix++) {
        for (int ib = 0; ib < nb; ib++) {
          uu[iy][ix][ib] *= damp[ib];
          uu[iy][ix][nz - 1 - ib] *= damp[ib];
        }
      }
    }

/* front/rear boundary */
#pragma omp parallel for schedule(dynamic, 1)
    for (int ib = 0; ib < nb; ib++) {
      for (int ix = 0; ix < nx; ix++) {
        for (int iz = 0; iz < nz; iz++) {
          uu[ib][ix][iz] *= damp[ib];
          uu[ny - 1 - ib][ix][iz] *= damp[ib];
        }
      }
    }
  }
  return;
}

