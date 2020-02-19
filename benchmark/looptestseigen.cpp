#include <Eigen/Dense>

using namespace Eigen;

extern "C" {
  void AmulB(double*, double*, double*, long, long, long);
  void AmulBt(double*, double*, double*, long, long, long);
  void AtmulB(double*, double*, double*, long, long, long);
  void AtmulBt(double*, double*, double*, long, long, long);
  void Amulvb(double*, double*, double*, long, long);
  void Atmulvb(double*, double*, double*, long, long);
  double dot(double*, double*, long);
  double selfdot(double*, long);
  double dot3(double*, double*, double*, long, long);
  void aplusBc(double*, double*, double*, double*, long, long);
  double OLSlp(double*, double*, double*, long, long);
  void AplusAt(double*, double*, long);
}

typedef Map<MatrixXd> mMatrix;
typedef Map<VectorXd> mVector;

void AmulB(double* pC, double* pA, double* pB, long M, long K, long N){
  mMatrix A(pA, M, K);
  mMatrix B(pB, K, N);
  mMatrix C(pC, M, N);
  C.noalias() = A * B;
  return;
}
void AmulBt(double* pC, double* pA, double* pBt, long M, long K, long N){
  mMatrix A(pA, M, K);
  mMatrix Bt(pBt, N, K);
  mMatrix C(pC, M, N);
  C.noalias() = A * Bt.transpose();
  return;
}
void AtmulB(double* pC, double* pAt, double* pB, long M, long K, long N){
  mMatrix At(pAt, K, M);
  mMatrix B(pB, K, N);
  mMatrix C(pC, M, N);
  C.noalias() = At.transpose() * B;
  return;
}
void AtmulBt(double* pC, double* pAt, double* pBt, long M, long K, long N){
  mMatrix At(pAt, K, M);
  mMatrix Bt(pBt, N, K);
  mMatrix C(pC, M, N);
  C.noalias() = At.transpose() * Bt.transpose();
  return;
}

void Amulvb(double* px, double* pA, double* py, long M, long N){
  mVector x(px, M);
  mMatrix A(pA, M, N);
  mVector y(py, N);
  x.noalias() = A * y;
  return;
}

void Atmulvb(double* px, double* pAt, double* py, long M, long N){
  mVector x(px, M);
  mMatrix At(pAt, M, N);
  mVector y(py, N);
  x.noalias() = At.transpose() * y;
  return;
}

double dot(double* pa, double* pb, long N){
  mVector a(pa, N);
  mVector b(pb, N);
  return a.dot(b);
}
 
double selfdot(double* pa, long N){
  mVector a(pa, N);
  return a.dot(a);
}

double dot3(double* px, double* pA, double* py, long M, long N){
  mVector x(px, M);
  mMatrix A(pA, M, N);
  mVector y(py, N);
  return x.dot(A * y);
}

void aplusBc(double* pD, double* pa, double* pB, double* pc, long M, long N){
  mMatrix D(pD, M, N);
  mVector a(pa, M);
  mMatrix B(pB, M, N);
  mVector c(pc, N);
  // D = (a + (B * c.asDiagonal()).colwise());
  D.colwise() = a;
  D.noalias() += B * c.asDiagonal();
  return;
}

double OLSlp(double* py, double* pA, double* px, long M, long N){
  mVector y(py, M);
  mMatrix A(pA, M, N);
  mVector x(px, N);
  return (y - A * x).squaredNorm();
}

void AplusAt(double* pB, double* pA, long N){
  mMatrix B(pB, N, N);
  mMatrix A(pA, N, N);
  B = A + A.transpose();
  return;
}

