#include<omp.h>

double dot(double* a, double* b, long N){
  double s = 0.0;
  #pragma omp parallel for reduction(+: s)
  for(long n = 0; n < N; n++){
    s += a[n]*b[n];
  }
  return s;
}

void cdot(double* c, double* a, double* b, long N){
  double r = 0.0, i = 0.0;
  #pragma omp parallel for reduction(+: r, i)
  for(long n = 0; n < N; n++){
    r += a[2*n] * b[2*n  ] + a[2*n+1] * b[2*n+1];
    i += a[2*n] * b[2*n+1] - a[2*n+1] * b[2*n  ];
  }
  c[0] = r;
  c[1] = i;
  return;
}

void cdot3(double* c, double* x, double* A, double* y, long M, long N){
  double sr = 0.0, si = 0.0;
#pragma omp parallel for reduction(+: sr, si)
  for (long n = 0; n < N; n++){
    double tr = 0.0, ti = 0.0;
    for(long m = 0; m < M; m++){
      tr += x[2*m] * A[2*m   + 2*n*N] + x[2*m+1] * A[2*m+1 + 2*n*N];
      ti += x[2*m] * A[2*m+1 + 2*n*N] - x[2*m+1] * A[2*m   + 2*n*N];
    }
    sr += tr * y[2*n  ] - ti * y[2*n+1];
    si += tr * y[2*n+1] + ti * y[2*n  ];
  }
  c[0] = sr;
  c[1] = si;
  return;
}

void conv(double* B, double* A, double* K, long M, long N){
  const long offset = 2;
  #pragma omp parallel for collapse(2)
  for (long i = offset; i < N-offset; i++){
    for (long j = offset; j < M-offset; j++){
      double tmp = 0.0;
      for (long k = -offset; k < offset + 1; k++){
        for (long l = -offset; l < offset + 1; l++){
          tmp += A[(j+l) + (i+k)*M] * K[(l+offset) + (k+offset)*(2*offset+1)];
        }
      }
      B[(j-offset) + (i-offset) * (M-2*offset)] = tmp;
    }
  }
  return;
}


