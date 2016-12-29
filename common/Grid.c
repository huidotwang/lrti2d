#include "Grid.h"

void
expand2d(float** a, float** b, sf_axis az, sf_axis ax, sf_axis azpad,
         sf_axis axpad)
/*< expand domain >*/
{
  int iz, ix;
  int nz = sf_n(az);
  int nx = sf_n(ax);
  int nzpad = sf_n(azpad);
  int nxpad = sf_n(axpad);
  int nb = (nzpad - nz) / 2;

  for (ix = 0; ix < nx; ix++) {
    for (iz = 0; iz < nz; iz++) {
      b[nb + ix][nb + iz] = a[ix][iz];
    }
  }

  for (ix = 0; ix < nxpad; ix++) {
    for (iz = 0; iz < nb; iz++) {
      b[ix][iz] = b[ix][nb];
      b[ix][nzpad - iz - 1] = b[ix][nzpad - nb - 1];
    }
  }

  for (ix = 0; ix < nb; ix++) {
    for (iz = 0; iz < nzpad; iz++) {
      b[ix][iz] = b[nb][iz];
      b[nxpad - ix - 1][iz] = b[nxpad - nb - 1][iz];
    }
  }
}

void
expand3d(float*** a, float*** b, sf_axis az, sf_axis ax, sf_axis ay,
         sf_axis azpad, sf_axis axpad, sf_axis aypad)
/*< expand domain >*/
{
  int iz, ix, iy;
  int nz = sf_n(az);
  int nx = sf_n(ax);
  int ny = sf_n(ay);
  int nzpad = sf_n(azpad);
  int nxpad = sf_n(axpad);
  int nypad = sf_n(aypad);
  int nb = (nzpad - nz) / 2;

  for (iy = 0; iy < ny; iy++) {
    for (ix = 0; ix < nx; ix++) {
      for (iz = 0; iz < nz; iz++) {
        b[nb + iy][nb + ix][nb + iz] = a[iy][ix][iz];
      }
    }
  }

  for (iy = 0; iy < nypad; iy++) {
    for (ix = 0; ix < nxpad; ix++) {
      for (iz = 0; iz < nb; iz++) {
        b[iy][ix][iz] = b[iy][ix][nb];
        b[iy][ix][nzpad - iz - 1] = b[iy][ix][nzpad - nb - 1];
      }
    }
  }

  for (iy = 0; iy < nypad; iy++) {
    for (ix = 0; ix < nb; ix++) {
      for (iz = 0; iz < nzpad; iz++) {
        b[iy][ix][iz] = b[iy][nb][iz];
        b[iy][nxpad - ix - 1][iz] = b[iy][nxpad - nb - 1][iz];
      }
    }
  }

  for (iy = 0; iy < nb; iy++) {
    for (ix = 0; ix < nxpad; ix++) {
      for (iz = 0; iz < nzpad; iz++) {
        b[iy][ix][iz] = b[nb][ix][iz];
        b[nypad - iy - 1][ix][iz] = b[nypad - nb - 1][ix][iz];
      }
    }
  }
}

void
window3d(float*** vec3d_o, float*** vec3d_i, sf_axis az, sf_axis ax, sf_axis ay,
         sf_axis azpad, sf_axis axpad, sf_axis aypad)
{
  int nz = sf_n(az);
  int nx = sf_n(ax);
  int ny = sf_n(ay);
  int nzpad = sf_n(azpad);
  int nxpad = sf_n(axpad);
  int nypad = sf_n(aypad);
  int nbz = (nzpad - nz) / 2;
  int nbx = (nxpad - nx) / 2;
  int nby = (nypad - ny) / 2;
#pragma omp parallel for schedule(dynamic, 1)
  for (int iy = 0; iy < ny; iy++) {
    for (int ix = 0; ix < nx; ix++) {
      for (int iz = 0; iz < nz; iz++) {
        vec3d_o[iy][ix][iz] = vec3d_i[iy + nby][ix + nbx][iz + nbz];
      }
    }
  }
  return;
}

void
wwin3d(float*** uo, float*** ui, int nzo, int nxo, int nyo, int nb)
{
#pragma omp parallel for schedule(dynamic, 1)
  for (int iy = 0; iy < nyo; iy++) {
    for (int ix = 0; ix < nxo; ix++) {
      for (int iz = 0; iz < nzo; iz++) {
        uo[iy][ix][iz] = ui[iy + nb][ix + nb][iz + nb];
      }
    }
  }
  return;
}

void
wwin2d(float** uo, float** ui, int nzo, int nxo, int nb)
{
#pragma omp parallel for schedule(dynamic, 1)
  for (int ix = 0; ix < nxo; ix++) {
    for (int iz = 0; iz < nzo; iz++) {
      uo[ix][iz] = ui[ix + nb][iz + nb];
    }
  }
  return;
}
