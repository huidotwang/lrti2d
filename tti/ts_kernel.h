#ifndef _TS_KERNEL_H
#define _TS_KERNEL_H
#include <string.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <stdbool.h>

void lr_fft_stepforward(float** u0, float** u1, float* rwavem,
                        fftwf_complex* cwave, fftwf_complex* cwavem,
                        float** lft, float** rht, fftwf_plan forward_plan,
                        fftwf_plan inverse_plan, int nz, int nx, int nzpad,
                        int nxpad, int nkz, int nkx, int nrank, float wt,
                        bool adj);
#endif
