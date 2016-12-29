#ifndef _LRDECOMP_HH
#define _LRDECOMP_HH
/*
 * Interface for low-rank decomposition
 */
typedef struct lrpar_base_t lrpar_base;
struct lrpar_base_t
{
  size_t m;
  size_t n;
  int seed;
  int npk;
  float eps;
  float dt;
  float** kk;
};

typedef struct lrmat_t lrmat;
struct lrmat_t
{
  int nrank;
  float* lft_data;
  float* rht_data;
};

/*
 * Initialize lowrank decomposition construct
 */
void* lrdecomp_new(size_t m, size_t n);

void lrdecomp_init(void* lrpar, int seed, int npk, float eps, float dt_, ...);

/*
 * Compute lowrank decomposition, output lft, and rht
 * returns the rank of the matrix
 */
lrmat* lrdecomp_compute(void* lr);

/*
 * Clean up lowrank decomposition
 */
void lrdecomp_delete(void* lrpar, void* fltmat);

#endif
