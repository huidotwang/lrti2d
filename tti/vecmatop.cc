#include "vecmatop.hh"

/*
 * Matrix Pseudo inverse by Eigen SVD
 */
static int
pinv(const MatrixXf M, float eps, MatrixXf& R)
{
  size_t nrows = M.rows();
  size_t ncols = M.cols();
#if _JACOBISVD
  JacobiSVD<MatrixXf> m_svd(nrows, ncols, ComputeThinU | ComputeThinV);
#else
  BDCSVD<MatrixXf> m_svd(nrows, ncols, ComputeThinU | ComputeThinV);
#endif
  m_svd.setThreshold(eps);
  m_svd.compute(M);
  size_t m_rank = m_svd.rank();
  cerr << "svd rank = " << m_rank << endl;
  MatrixXf UT = m_svd.matrixU().leftCols(m_rank);
  UT.transposeInPlace();
  MatrixXf V = m_svd.matrixV().leftCols(m_rank);
  VectorXf S = m_svd.singularValues().head(m_rank);
#if 0
  for (size_t is = 0; is< m_rank; is ++)
    cerr << "S(" << is << ") = " << S(is) << endl;
#endif
  for (size_t j = 0; j < m_rank; j++) V.col(j) = V.col(j) / S[j];
  R = V * UT;
  return 0;
}

/*
 * QR decomposition with column (full) pivoting by Eigen
 */
static int
qr(const MatrixXf M, float eps, vector<size_t>& idx)
{
  ColPivHouseholderQR<MatrixXf> m_qr(M.rows(), M.cols());
  m_qr.setThreshold(eps);
  m_qr.compute(M);
  size_t m_rank = m_qr.rank();
  idx.resize(m_rank);
  for (size_t k = 0; k < m_rank; k++) idx[k] = m_qr.colsPermutation().indices()[k];
  return 0;
}

/*
 * Generic lowrank decomposition based on QR decomposition
 */
int
lowrank(size_t m, size_t n,
        int (*sample)(vector<size_t>&, vector<size_t>&, MatrixXf&, void*), float eps,
        size_t npk, vector<size_t>& cidx, vector<size_t>& ridx, MatrixXf& mid,
        void* ebuf)
{
  {
    size_t nc = std::min(npk, n);
    vector<size_t> cs(nc);
    for (size_t k = 0; k < nc; k++) cs[k] = size_t(floor(drand48() * (n - nc)));
    // for (size_t k=0; k<nc; k++)  cs[k] = size_t( floor(drand48()*(n-1)) );
    sort(cs.begin(), cs.end());
    for (size_t k = 0; k < nc; k++) cs[k] += k;
    vector<size_t> rs(m);
    for (size_t k = 0; k < m; k++) rs[k] = k;
    MatrixXf M2;
    (*sample)(rs, cs, M2, ebuf);
    M2.transposeInPlace();

    qr(M2, eps, ridx);

    cerr << "ROWS ";
    for (size_t k = 0; k < ridx.size(); k++) cerr << ridx[k] << " ";
    cerr << endl;
  }
  {
    size_t nr = std::min(npk, m);
    vector<size_t> rs(nr);
    for (size_t k = 0; k < nr; k++) rs[k] = size_t(floor(drand48() * (m - nr)));
    // for(size_t k=0; k<nr; k++) rs[k] = size_t( floor(drand48()*(m-1)) );
    sort(rs.begin(), rs.end());
    for (size_t k = 0; k < nr; k++) rs[k] += k;
    for (size_t k = 0; k < ridx.size(); k++) rs.push_back(ridx[k]);
    sort(rs.begin(), rs.end());
    vector<size_t>::iterator newend = unique(rs.begin(), rs.end());
    rs.resize(newend - rs.begin());
    vector<size_t> cs(n);
    for (size_t k = 0; k < n; k++) cs[k] = k;
    MatrixXf M1;
    (*sample)(rs, cs, M1, ebuf);

    qr(M1, eps, cidx);

    cerr << "COLS ";
    for (size_t k = 0; k < cidx.size(); k++) cerr << cidx[k] << " ";
    cerr << endl;
  }
  {
    size_t nc = std::min(npk, n);
    vector<size_t> cs(nc);
    for (size_t k = 0; k < nc; k++) cs[k] = size_t(floor(drand48() * (n - nc)));
    sort(cs.begin(), cs.end());
    for (size_t k = 0; k < nc; k++) cs[k] += k;
    for (size_t k = 0; k < cidx.size(); k++) cs.push_back(cidx[k]);
    sort(cs.begin(), cs.end());
    vector<size_t>::iterator csnewend = unique(cs.begin(), cs.end());
    cs.resize(csnewend - cs.begin());

    size_t nr = std::min(npk, m);
    vector<size_t> rs(nr);
    for (size_t k = 0; k < nr; k++) rs[k] = size_t(floor(drand48() * (m - nr)));
    sort(rs.begin(), rs.end());
    for (size_t k = 0; k < nr; k++) rs[k] += k;
    for (size_t k = 0; k < ridx.size(); k++) rs.push_back(ridx[k]);
    sort(rs.begin(), rs.end());
    vector<size_t>::iterator rsnewend = unique(rs.begin(), rs.end());
    rs.resize(rsnewend - rs.begin());

    MatrixXf M1;
    (*sample)(rs, cidx, M1, ebuf);
    MatrixXf M2;
    (*sample)(ridx, cs, M2, ebuf);
    MatrixXf M3;
    (*sample)(rs, cs, M3, ebuf);
    MatrixXf IM1, IM2;

    pinv(M1, 1e-7, IM1);
    pinv(M2, 1e-7, IM2);

    mid = IM1 * M3 * IM2;
  }
  if (1) {
    size_t nc = std::min(npk, n);
    vector<size_t> cs(nc);
    for (size_t k = 0; k < nc; k++) cs[k] = size_t(floor(drand48() * n));
    size_t nr = std::min(npk, m);
    vector<size_t> rs(nr);
    for (size_t k = 0; k < nr; k++) rs[k] = size_t(floor(drand48() * m));
    MatrixXf M1;
    (*sample)(rs, cidx, M1, ebuf);
    MatrixXf M2;
    (*sample)(ridx, cs, M2, ebuf);
    MatrixXf Mext;
    (*sample)(rs, cs, Mext, ebuf);
    MatrixXf Mapp = M1 * mid * M2;
    MatrixXf Merr = Mext - Mapp;
    cerr << "rel err = " << Merr.norm() / Mext.norm() << endl;
  }
  return 0;
}
