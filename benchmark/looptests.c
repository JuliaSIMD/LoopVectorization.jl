#include<math.h>

void gemm_mnk(double* restrict C, double* restrict A, double* restrict B, long M, long K, long N){
  for (long i = 0; i < M*N; i++){
    C[i] = 0.0;
  }
  for (long m = 0; m < M; m++){
    for (long n = 0; n < N; n++){
      for (long k = 0; k < K; k++){
	C[m + n*M] += A[m + k*M] * B[k + n*K];
      }
    }
  }
  return;
}
void gemm_mkn(double* restrict C, double* restrict A, double* restrict B, long M, long K, long N){
  for (long i = 0; i < M*N; i++){
    C[i] = 0.0;
  }
  for (long m = 0; m < M; m++){
    for (long k = 0; k < K; k++){
      for (long n = 0; n < N; n++){
	C[m + n*M] += A[m + k*M] * B[k + n*K];
      }
    }
  }
  return;
}
void gemm_nmk(double* restrict C, double* restrict A, double* restrict B, long M, long K, long N){
  for (long i = 0; i < M*N; i++){
    C[i] = 0.0;
  }
  for (long n = 0; n < N; n++){
    for (long m = 0; m < M; m++){
      for (long k = 0; k < K; k++){
	C[m + n*M] += A[m + k*M] * B[k + n*K];
      }
    }
  }
  return;
}
void gemm_nkm(double* restrict C, double* restrict A, double* restrict B, long M, long K, long N){
  for (long i = 0; i < M*N; i++){
    C[i] = 0.0;
  }
  for (long n = 0; n < N; n++){
    for (long k = 0; k < K; k++){
      for (long m = 0; m < M; m++){
	C[m + n*M] += A[m + k*M] * B[k + n*K];
      }
    }
  }
  return;
}
void gemm_kmn(double* restrict C, double* restrict A, double* restrict B, long M, long K, long N){
  for (long i = 0; i < M*N; i++){
    C[i] = 0.0;
  }
  for (long k = 0; k < K; k++){
    for (long m = 0; m < M; m++){
      for (long n = 0; n < N; n++){
	C[m + n*M] += A[m + k*M] * B[k + n*K];
      }
    }
  }
  return;
}
void gemm_knm(double* restrict C, double* restrict A, double* restrict B, long M, long K, long N){
  for (long i = 0; i < M*N; i++){
    C[i] = 0.0;
  }
  for (long k = 0; k < K; k++){
    for (long n = 0; n < N; n++){
      for (long m = 0; m < M; m++){
	C[m + n*M] += A[m + k*M] * B[k + n*K];
      }
    }
  }
  return;
}
void AtmulB(double* restrict C, double* restrict At, double* restrict B, long M, long K, long N){
  for (long i = 0; i < M*N; i++){
    C[i] = 0.0;
  }
  for (long n = 0; n < N; n++){
    for (long m = 0; m < M; m++){
      for (long k = 0; k < K; k++){
	C[m + n*M] += At[k + m*K] * B[k + n*K];
      }
    }
  }
  return;
}
void AmulBt(double* restrict C, double* restrict A, double* restrict Bt, long M, long K, long N){
  for (long i = 0; i < M*N; i++){
    C[i] = 0.0;
  }
  for (long k = 0; k < K; k++){
    for (long n = 0; n < N; n++){
      for (long m = 0; m < M; m++){
	C[m + n*M] += A[m + M*k] * Bt[n + N*k];
      }
    }
  }
  return;
}
double dot(double* restrict a, double* restrict b, long N){
  double s = 0.0;
  for (long n = 0; n < N; n++){
    s += a[n]*b[n];
  }
  return s;
}
double selfdot(double* restrict a, long N){
  double s = 0.0;
  for (long n = 0; n < N; n++){
    s += a[n]*a[n];
  }
  return s;
}
double dot3(double* restrict x, double* restrict A, double* restrict y, long M, long N){
  double s = 0.0;
  for (long n = 0; n < N; n++){
    for (long m = 0; m < M; m++){
      s += x[m] * A[m + n*M] * y[n];
    }
  }
  return s;
}
void gemv(double* restrict y, double* restrict  A, double* restrict x, long M, long K){
  for (long m = 0; m < M; m++){
    y[m] = 0.0;
  }
  for (long k = 0; k < K; k++){
    for (long m = 0; m < M; m++){
      y[m] += A[m + k*M] * x[k]; 
    }
  }
  return;
}
void Atmulvb(double* restrict y, double* restrict  A, double* restrict x, long M, long K){
  for (long m = 0; m < M; m++){
    double ym = 0.0;
    for (long k = 0; k < K; k++){
      ym += A[k + m*K] * x[k]; 
    }
    y[m] = ym;
  }
  return;
}
double svexp(double* restrict a, long N){
  double s = 0.0;
  for (long n = 0; n < N; n++){
    s += exp(a[n]);
  }
  return s;
}
void vexp(double* restrict b, double* restrict a, long N){
  for (long n = 0; n < N; n++){
    b[n] = exp(a[n]);
  }
  return;
}
void unscaledvar(double* restrict s, double* restrict A, double* restrict xb, long M, long N){
  for (long m = 0; m < M; m++){
    s[m] = 0.0;
  }
  for (long n = 0; n < N; n++){
    for (long m = 0; m < M; m++){
      double d = A[m + n*M] - xb[m];
      s[m] += d*d;
    }
  }
  return;
}
void aplusBc(double* restrict D, double* restrict a, double* restrict B, double* restrict c, long M, long N){
  for (long n = 0; n < N; n++){
    for (long m = 0; m < M; m++){
      D[m + n*M] = a[m] + B[m + n*M] * c[n];
    }
  }
  return;
}

double OLSlp(double* restrict y, double* restrict X, double* restrict b, long N, long P){
  double lp = 0.0;
  for (long n = 0; n < N; n++){
    double d = y[n];
    for (long p = 0; p < P; p++){
      d -= X[n + p*N] * b[p];
    }
    lp += d*d;
  }
  return lp;
}

void AplusAt(double* restrict B, double* restrict A, long N){
  for (long i = 0; i < N; i++){
    for (long j = 0; j < N; j++){
      B[j + i*N] = A[j + i*N] + A[i + j*N];
    }
  }
}
double randomaccess(double* restrict P, long* restrict basis, double* restrict coefs, long A, long C){
  double p = 0.0;
  for (long c = 0; c < C; c++){
    double pc = coefs[c];
    for (long a = 0; a < A; a++){
      pc *= P[a + (basis[a + c*A]-1)*A];
    }
    p += pc;
  }
  return p;
}


