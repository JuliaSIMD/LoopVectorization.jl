module looptests

  use ISO_C_BINDING

  implicit none

  contains

    subroutine gemm_mnk(C, A, B, M, K, N) BIND(C, name="gemm_mnk")
      integer(C_long), intent(in) :: M, K, N
      real(C_double), dimension(M, N), intent(out) :: C
      real(C_double), dimension(M, K), intent(in) :: A
      real(C_double), dimension(K, N), intent(in) :: B
      integer(C_long) :: mm, kk, nn
      C = 0.0
      do concurrent(mm = 1:M, nn = 1:N, kk = 1:K)
          C(mm,nn) = C(mm,nn) + A(mm,kk) * B(kk,nn)
      end do
    end subroutine gemm_mnk
    subroutine gemm_mkn(C, A, B, M, K, N) BIND(C, name="gemm_mkn")
      integer(C_long), intent(in) :: M, K, N
      real(C_double), dimension(M, N), intent(out) :: C
      real(C_double), dimension(M, K), intent(in) :: A
      real(C_double), dimension(K, N), intent(in) :: B
      integer(C_long) :: mm, kk, nn
      C = 0.0
      do concurrent(mm = 1:M, kk = 1:K, nn = 1:N)
          C(mm,nn) = C(mm,nn) + A(mm,kk) * B(kk,nn)
      end do
    end subroutine gemm_mkn
    subroutine gemm_nmk(C, A, B, M, K, N) BIND(C, name="gemm_nmk")
      integer(C_long), intent(in) :: M, K, N
      real(C_double), dimension(M, N), intent(out) :: C
      real(C_double), dimension(M, K), intent(in) :: A
      real(C_double), dimension(K, N), intent(in) :: B
      integer(C_long) :: mm, kk, nn
      C = 0.0
      do concurrent(nn = 1:N, mm = 1:M, kk = 1:K)
          C(mm,nn) = C(mm,nn) + A(mm,kk) * B(kk,nn)
      end do
    end subroutine gemm_nmk
    subroutine gemm_nkm(C, A, B, M, K, N) BIND(C, name="gemm_nkm")
      integer(C_long), intent(in) :: M, K, N
      real(C_double), dimension(M, N), intent(out) :: C
      real(C_double), dimension(M, K), intent(in) :: A
      real(C_double), dimension(K, N), intent(in) :: B
      integer(C_long) :: mm, kk, nn
      C = 0.0
      do concurrent(kk = 1:K, nn = 1:N, mm = 1:M)
          C(mm,nn) = C(mm,nn) + A(mm,kk) * B(kk,nn)
      end do
    end subroutine gemm_nkm
    subroutine gemm_kmn(C, A, B, M, K, N) BIND(C, name="gemm_kmn")
      integer(C_long), intent(in) :: M, K, N
      real(C_double), dimension(M, N), intent(out) :: C
      real(C_double), dimension(M, K), intent(in) :: A
      real(C_double), dimension(K, N), intent(in) :: B
      integer(C_long) :: mm, kk, nn
      C = 0.0
      do concurrent(kk = 1:K, mm = 1:M, nn = 1:N)
          C(mm,nn) = C(mm,nn) + A(mm,kk) * B(kk,nn)
      end do
    end subroutine gemm_kmn
    subroutine gemm_knm(C, A, B, M, K, N) BIND(C, name="gemm_knm")
      integer(C_long), intent(in) :: M, K, N
      real(C_double), dimension(M, N), intent(out) :: C
      real(C_double), dimension(M, K), intent(in) :: A
      real(C_double), dimension(K, N), intent(in) :: B
      integer(C_long) :: mm, kk, nn
      C = 0.0
      do concurrent(kk = 1:K, nn = 1:N, mm = 1:M)
         C(mm,nn) = C(mm,nn) + A(mm,kk) * B(kk,nn)
      end do
    end subroutine gemm_knm
    subroutine gemmbuiltin(C, A, B, M, K, N) BIND(C, name="gemmbuiltin")
      integer(C_long), intent(in) :: M, K, N
      real(C_double), dimension(M, N), intent(out) :: C
      real(C_double), dimension(M, K), intent(in) :: A
      real(C_double), dimension(K, N), intent(in) :: B
      C = matmul(A, B)
    end subroutine gemmbuiltin
    subroutine AtmulB(C, A, B, M, K, N) BIND(C, name="AtmulB")
      integer(C_long), intent(in) :: M, K, N
      real(C_double), dimension(M, N), intent(out) :: C
      real(C_double), dimension(K, M), intent(in) :: A
      real(C_double), dimension(K, N), intent(in) :: B
      integer(C_long) :: mm, kk, nn
      C = 0.0
      do concurrent(nn = 1:N, mm = 1:M, kk = 1:K)
          C(mm,nn) = C(mm,nn) + A(kk,mm) * B(kk,nn)
      end do
    end subroutine AtmulB
    subroutine AtmulBbuiltin(C, A, B, M, K, N) BIND(C, name="AtmulBbuiltin")
      integer(C_long), intent(in) :: M, K, N
      real(C_double), dimension(M, N), intent(out) :: C
      real(C_double), dimension(K, M), intent(in) :: A
      real(C_double), dimension(K, N), intent(in) :: B
      C = matmul(transpose(A), B)
    end subroutine AtmulBbuiltin
    subroutine AmulBt(C, A, B, M, K, N) BIND(C, name="AmulBt")
      integer(C_long), intent(in) :: M, K, N
      real(C_double), dimension(M, N), intent(out) :: C
      real(C_double), dimension(M, K), intent(in) :: A
      real(C_double), dimension(N, K), intent(in) :: B
      integer(C_long) :: mm, kk, nn
      C = 0.0d0
      do concurrent(kk = 1:K, nn = 1:N, mm = 1:M)
         C(mm,nn) = C(mm,nn) + A(mm,kk) * B(nn,kk)
      end do
    end subroutine AmulBt
    subroutine AmulBtbuiltin(C, A, B, M, K, N) BIND(C, name="AmulBtbuiltin")
      integer(C_long), intent(in) :: M, K, N
      real(C_double), dimension(M, N), intent(out) :: C
      real(C_double), dimension(M, K), intent(in) :: A
      real(C_double), dimension(N, K), intent(in) :: B
      C = matmul(A, transpose(B))
    end subroutine AmulBtbuiltin
    subroutine AtmulBt(C, A, B, M, K, N) BIND(C, name="AtmulBt")
      integer(C_long), intent(in) :: M, K, N
      real(C_double), dimension(M, N), intent(out) :: C
      real(C_double), dimension(K, M), intent(in) :: A
      real(C_double), dimension(N, K), intent(in) :: B
      integer(C_long) :: mm, kk, nn
      C = 0.0d0
      do concurrent(nn = 1:N, kk = 1:K, mm = 1:M)
         C(mm,nn) = C(mm,nn) + A(kk,mm) * B(nn,kk)
      end do
    end subroutine AtmulBt
    subroutine AtmulBtbuiltin(C, A, B, M, K, N) BIND(C, name="AtmulBtbuiltin")
      integer(C_long), intent(in) :: M, K, N
      real(C_double), dimension(M, N), intent(out) :: C
      real(C_double), dimension(K, M), intent(in) :: A
      real(C_double), dimension(N, K), intent(in) :: B
      C = transpose(matmul(B, A))
    end subroutine AtmulBtbuiltin
    real(C_double) function dot(a, b, N) BIND(C, name="dot")
      integer(C_long), intent(in) :: N
      real(C_double), dimension(N), intent(in) :: a, b
      integer(C_long) :: i
      dot = 0d0
      do concurrent(i = 1:N)
         dot = dot + a(i) * b(i)
      end do
    end function dot
    real(C_double) function selfdot(a, N) BIND(C, name="selfdot")
      integer(C_long), intent(in) :: N
      real(C_double), dimension(N), intent(in) :: a
      integer(C_long) :: i
      selfdot = 0d0
      do concurrent(i = 1:N)
         selfdot = selfdot + a(i) * a(i)
      end do
    end function selfdot
    real(C_double) function dot3v2(x, A, y, M, N) BIND(C, name="dot3v2")
      integer(C_long), intent(in) :: M, N
      real(C_double), intent(in) :: x(M), A(M,N), y(N)
      real(C_double) :: t
      integer(C_long) :: mm, nn
      dot3v2 = 0.0d0
      do concurrent(nn = 1:N, mm = 1:M)
         dot3v2 = dot3v2 + x(mm) * A(mm, nn) * y(nn)
      end do
    end function dot3v2
    real(C_double) function dot3(x, A, y, M, N) BIND(C, name="dot3")
      integer(C_long), intent(in) :: M, N
      real(C_double), intent(in) :: x(M), A(M,N), y(N)
      real(C_double) :: t
      integer(C_long) :: mm, nn
      dot3 = 0.0d0
      do concurrent(nn = 1:N)
         t = 0.0d0
         do concurrent(mm = 1:M)
            t = t + x(mm) * A(mm, nn)
         end do
         dot3 = dot3 + t * y(nn)
      end do
    end function dot3
    real(C_double) function dot3builtin(x, A, y, M, N) BIND(C, name="dot3builtin")
      integer(C_long), intent(in) :: M, N
      real(C_double), intent(in) :: x(M), A(M,N), y(N)
      dot3builtin = dot_product(x, matmul(A, y))
    end function dot3builtin
    !GCC$ builtin (exp) attributes simd (notinbranch) if('x86_64')
    subroutine vexp(b, a, N) BIND(C, name="vexp")
      integer(C_long), intent(in) :: N
      real(C_double), dimension(N), intent(in) :: a
      real(C_double), dimension(N), intent(out) :: b
      integer(C_long) :: i
      do concurrent(i = 1:N)
         b(i) = exp(a(i))
      end do
    end subroutine vexp
    real(C_double) function svexp(a, N) BIND(C, name="svexp")
      integer(C_long), intent(in) :: N
      real(C_double), dimension(N), intent(in) :: a
      integer(C_long) :: i
      svexp = 0
      do concurrent(i = 1:N)
         svexp = svexp + exp(a(i))
      end do
    end function svexp
    subroutine gemv(y, A, x, M, K) BIND(C, name="gemv")
      integer(C_long), intent(in) :: M, K
      real(C_double), intent(in) :: A(M,K), x(K)
      real(C_double), dimension(M), intent(out) :: y
      integer(C_long) :: mm, kk
      y = 0.0
      do concurrent(kk = 1:K, mm = 1:M)
         y(mm) = y(mm) + A(mm,kk) * x(kk)
      end do
    end subroutine gemv
    subroutine gemvbuiltin(y, A, x, M, K) BIND(C, name="gemvbuiltin")
      integer(C_long), intent(in) :: M, K
      real(C_double), intent(in) :: A(M,K), x(K)
      real(C_double), dimension(M), intent(out) :: y
      y = matmul(A, x)
    end subroutine gemvbuiltin
    subroutine Atmulvb(y, A, x, M, K) BIND(C, name="Atmulvb")
      integer(C_long), intent(in) :: M, K
      real(C_double), intent(in) :: A(K,M), x(K)
      real(C_double), dimension(M), intent(out) :: y
      integer(C_long) :: mm, kk
      real(C_double) :: ymm
      y = 0
      do concurrent(mm = 1:M, kk = 1:K)
          y(mm) = y(mm) + A(kk,mm) * x(kk)
      end do
    end subroutine Atmulvb
    subroutine Atmulvbbuiltin(y, A, x, M, K) BIND(C, name="Atmulvbbuiltin")
      integer(C_long), intent(in) :: M, K
      real(C_double), intent(in) :: A(K,M), x(1,K)
      real(C_double), dimension(1,M), intent(out) :: y
      ! y = matmul(transpose(A), x)
      y = matmul(x, A)
    end subroutine Atmulvbbuiltin
    subroutine unscaledvar(s, A, x, M, N) BIND(C, name="unscaledvar")
      integer(C_long), intent(in) :: M, N
      real(C_double), intent(in) :: A(M,N), x(M)
      real(C_double), dimension(M), intent(out) :: s
      integer(C_long) :: mm, nn
      real(C_double) :: d
      s = 0.0
      do concurrent(nn = 1:N, mm = 1:M)
         d = A(mm,nn) - x(mm)
         s(mm) = s(mm) + d * d
      end do
    end subroutine unscaledvar
    subroutine aplusBc(D, a, B, c, M, N) BIND(C, name="aplusBc")
      integer(C_long), intent(in) :: M, N
      real(C_double), intent(in) :: a(M), B(M,N), c(N)
      real(C_double), dimension(M,N), intent(out) :: D
      integer(C_long) :: mm, nn
      do concurrent(nn = 1:N, mm = 1:M)
         D(mm,nn) = a(mm) + B(mm,nn) * c(nn)
      end do
    end subroutine aplusBc
    real(C_double) function OLSlp(y, X, b, N, P) BIND(C, name="OLSlp")
      integer(C_long), intent(in) :: N, P
      real(C_double), intent(in) :: y(N), X(N, P), b(P)
      integer(C_long) :: nn, pp
      real(C_double) :: d
      OLSlp = 0
      do concurrent(nn = 1:N)
         d = y(nn)
         do concurrent(pp = 1:P)
            d = d - X(nn,pp) * b(pp)
         end do
         OLSlp = OLSlp + d*d
      end do
    end function OLSlp
    real(C_double) function OLSlpsplit(y, X, b, N, P) BIND(C, name="OLSlpsplit")
      integer(C_long), intent(in) :: N, P
      real(C_double), intent(in) :: y(N), X(N, P), b(P)
      integer(C_long) :: nn, pp
      real(C_double) :: d(N)
      d = y - matmul(X, b)
      OLSlpsplit = dot_product(d, d)
    end function OLSlpsplit
    subroutine AplusAt(B, A, N) BIND(C, name="AplusAt")
      integer(C_long), intent(in) :: N
      real(C_double), dimension(N,N), intent(out) :: B
      real(C_double), dimension(N,N), intent(in) :: A
      integer(C_long) :: i, j
      do concurrent(i = 1:N, j = 1:N)
         B(j,i) = A(j,i) + A(i,j)
      end do
    end subroutine AplusAt
    subroutine AplusAtbuiltin(B, A, N) BIND(C, name="AplusAtbuiltin")
      integer(C_long), intent(in) :: N
      real(C_double), dimension(N,N), intent(out) :: B
      real(C_double), dimension(N,N), intent(in) :: A
      B = A + transpose(A)
    end subroutine AplusAtbuiltin
    real(C_double) function randomaccess(P, basis, coefs, A, C) BIND(C, name="randomaccess")
      integer(C_long), intent(in) :: A, C, basis(A,C)
      real(C_double), intent(in) :: P(A,C), coefs(C)
      real(C_double) :: pc
      integer(C_long) :: aa, cc
      randomaccess = 0
      do cc = 1,C
         pc = coefs(cc)
         do aa = 1,A
            pc = pc * P(aa, basis(aa, cc))
         end do
         randomaccess = randomaccess + pc
      end do
    end function randomaccess
    !GCC$ builtin (log) attributes simd (notinbranch) if('x86_64')
    real(C_double) function logdettriangle(T, N) BIND(C, name="logdettriangle")
      integer(C_long), intent(in) :: N
      real(C_double), intent(in) :: T(N,N)
      integer(C_long) :: nn
      logdettriangle = 0
      do concurrent(nn = 1:N)
         logdettriangle = logdettriangle + log(T(nn,nn))
      end do
    end function logdettriangle

  subroutine filter2d(B, A, K, Ma, Na, offset) BIND(C, name="filter2d")
      integer(C_long), intent(in) :: Ma, Na, offset
      real(C_double), intent(in) :: A(Ma,Na), K(-offset:offset,-offset:offset)
      real(C_double), intent(out) :: B(1+offset:Ma-offset,1+offset:Na-offset)
      integer(C_long) :: mma, nna, mmk, nnk
      real(C_double) :: tmp
      do concurrent(mma = 1+offset:Ma-offset, nna = 1+offset:Na-offset)
         tmp = 0
         do concurrent(nnk = -offset:offset, mmk = -offset:offset)
            tmp = tmp + A(mma + mmk, nna + nnk) * K(mmk, nnk)
         end do
         B(mma,nna) = tmp
      end do
    end subroutine filter2d

    subroutine filter2d3x3(B, A, K, Ma, Na) BIND(C, name="filter2d3x3")
      integer(C_long), parameter :: offset = 1
      integer(C_long), intent(in) :: Ma, Na
      real(C_double), intent(in) :: A(Ma,Na), K(-offset:offset,-offset:offset)
      real(C_double), intent(out) :: B(1+offset:Ma-offset,1+offset:Na-offset)
      integer(C_long) :: mma, nna, mmk, nnk
      real(C_double) :: tmp
      do concurrent(mma = 1+offset:Ma-offset, nna = 1+offset:Na-offset)
         tmp = 0
         do concurrent(nnk = -offset:offset, mmk = -offset:offset)
            tmp = tmp + A(mma + mmk, nna + nnk) * K(mmk, nnk)
         end do
         B(mma,nna) = tmp
      end do
    end subroutine filter2d3x3

    subroutine filter2d3x3unrolled(B, A, K, Ma, Na) BIND(C, name="filter2d3x3unrolled")
      integer(C_long), intent(in) :: Ma, Na
      real(C_double), intent(in) :: A(Ma,Na), K(-1:1,-1:1)
      real(C_double), intent(out) :: B(2:Ma-1,2:Na-1)
      integer(C_long) :: mma, nna, mmk, nnk
      real(C_double) :: tmp
      do concurrent(mma = 2:Ma-1, nna = 2:Na-1)
         tmp =       A(mma - 1, nna - 1) * K(-1, -1)
         tmp = tmp + A(mma    , nna - 1) * K( 0, -1)
         tmp = tmp + A(mma + 1, nna - 1) * K( 1, -1)
         tmp = tmp + A(mma - 1, nna    ) * K(-1,  0)
         tmp = tmp + A(mma    , nna    ) * K( 0,  0)
         tmp = tmp + A(mma + 1, nna    ) * K( 1,  0)
         tmp = tmp + A(mma - 1, nna + 1) * K(-1,  1)
         tmp = tmp + A(mma    , nna + 1) * K( 0,  1)
         tmp = tmp + A(mma + 1, nna + 1) * K( 1,  1)
         B(mma,nna) = tmp
      end do
    end subroutine filter2d3x3unrolled




    
  end module looptests
    
