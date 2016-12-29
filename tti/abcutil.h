/* absorbing boundary utility*/
#ifndef _ABCUTIL_H
#define _ABCUTIL_H

float* damp_make(int nb);

void damp2d_apply(float **uu, float *damp, int nz, int nx, int nb);

void damp3d_apply(float ***uu, float *damp, int nz, int nx, int ny, int nb);
#endif
