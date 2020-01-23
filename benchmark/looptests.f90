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
      do concurrent(mm = 1:M)
         do concurrent(nn = 1:N)
            do concurrent(kk = 1:K)
               C(mm,nn) = C(mm,nn) + A(mm,kk) * B(kk,nn)
            end do
         end do
      end do
    end subroutine gemm_mnk
    subroutine gemm_mkn(C, A, B, M, K, N) BIND(C, name="gemm_mkn")
      integer(C_long), intent(in) :: M, K, N
      real(C_double), dimension(M, N), intent(out) :: C
      real(C_double), dimension(M, K), intent(in) :: A
      real(C_double), dimension(K, N), intent(in) :: B
      integer(C_long) :: mm, kk, nn
      C = 0.0
      do concurrent(mm = 1:M)
         do concurrent(kk = 1:K)
            do concurrent(nn = 1:N)
               C(mm,nn) = C(mm,nn) + A(mm,kk) * B(kk,nn)
            end do
         end do
      end do
    end subroutine gemm_mkn
    subroutine gemm_nmk(C, A, B, M, K, N) BIND(C, name="gemm_nmk")
      integer(C_long), intent(in) :: M, K, N
      real(C_double), dimension(M, N), intent(out) :: C
      real(C_double), dimension(M, K), intent(in) :: A
      real(C_double), dimension(K, N), intent(in) :: B
      integer(C_long) :: mm, kk, nn
      C = 0.0
      do concurrent(nn = 1:N)
         do concurrent(mm = 1:M)
            do concurrent(kk = 1:K)
               C(mm,nn) = C(mm,nn) + A(mm,kk) * B(kk,nn)
            end do
         end do
      end do
    end subroutine gemm_nmk
    subroutine gemm_nkm(C, A, B, M, K, N) BIND(C, name="gemm_nkm")
      integer(C_long), intent(in) :: M, K, N
      real(C_double), dimension(M, N), intent(out) :: C
      real(C_double), dimension(M, K), intent(in) :: A
      real(C_double), dimension(K, N), intent(in) :: B
      integer(C_long) :: mm, kk, nn
      C = 0.0
      do concurrent(kk = 1:K)
         do concurrent(nn = 1:N)
            do concurrent(mm = 1:M)
               C(mm,nn) = C(mm,nn) + A(mm,kk) * B(kk,nn)
            end do
         end do
      end do
    end subroutine gemm_nkm
    subroutine gemm_kmn(C, A, B, M, K, N) BIND(C, name="gemm_kmn")
      integer(C_long), intent(in) :: M, K, N
      real(C_double), dimension(M, N), intent(out) :: C
      real(C_double), dimension(M, K), intent(in) :: A
      real(C_double), dimension(K, N), intent(in) :: B
      integer(C_long) :: mm, kk, nn
      C = 0.0
      do concurrent(kk = 1:K)
         do concurrent(mm = 1:M)
            do concurrent(nn = 1:N)
               C(mm,nn) = C(mm,nn) + A(mm,kk) * B(kk,nn)
            end do
         end do
      end do
    end subroutine gemm_kmn
    subroutine gemm_knm(C, A, B, M, K, N) BIND(C, name="gemm_knm")
      integer(C_long), intent(in) :: M, K, N
      real(C_double), dimension(M, N), intent(out) :: C
      real(C_double), dimension(M, K), intent(in) :: A
      real(C_double), dimension(K, N), intent(in) :: B
      integer(C_long) :: mm, kk, nn
      C = 0.0
      do concurrent(kk = 1:K)
         do concurrent(nn = 1:N)
            do concurrent(mm = 1:M)
               C(mm,nn) = C(mm,nn) + A(mm,kk) * B(kk,nn)
            end do
         end do
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
      do concurrent(nn = 1:N)
         do concurrent(mm = 1:M)
            do concurrent(kk = 1:K)
               C(mm,nn) = C(mm,nn) + A(kk,mm) * B(kk,nn)
            end do
         end do
      end do
    end subroutine AtmulB
    subroutine AtmulBbuiltin(C, A, B, M, K, N) BIND(C, name="AtmulBbuiltin")
      integer(C_long), intent(in) :: M, K, N
      real(C_double), dimension(M, N), intent(out) :: C
      real(C_double), dimension(K, M), intent(in) :: A
      real(C_double), dimension(K, N), intent(in) :: B
      integer(C_long) :: mm, kk, nn
      C = matmul(transpose(A), B)
    end subroutine AtmulBbuiltin
    subroutine dot(s, a, b, N) BIND(C, name="dot")
      integer(C_long), intent(in) :: N
      real(C_double), dimension(N), intent(in) :: a, b
      real(C_double), intent(out) :: s
      integer(C_long) :: i
      s = 0
      do concurrent(i = 1:N)
         s = s + a(i) * b(i)
      end do
    end subroutine dot
    subroutine selfdot(s, a, N) BIND(C, name="selfdot")
      integer(C_long), intent(in) :: N
      real(C_double), dimension(N), intent(in) :: a
      real(C_double), intent(out) :: s
      integer(C_long) :: i
      s = 0
      do concurrent(i = 1:N)
         s = s + a(i) * a(i)
      end do
    end subroutine selfdot
    subroutine dot3(s, x, A, y, M, N) BIND(C, name="dot3")
      integer(C_long), intent(in) :: M, N
      real(C_double), intent(in) :: x(M), A(M,N), y(N)
      real(C_double), intent(out) :: s
      integer(C_long) :: mm, nn
      do concurrent(nn = 1:N)
         do concurrent(mm = 1:M)
            s = s + x(mm) * A(mm, nn) * y(nn)
         end do
      end do
    end subroutine dot3
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
    subroutine svexp(s, a, N) BIND(C, name="svexp")
      integer(C_long), intent(in) :: N
      real(C_double), dimension(N), intent(in) :: a
      real(C_double), intent(out) :: s
      integer(C_long) :: i
      s = 0
      do concurrent(i = 1:N)
         s = s + exp(a(i))
      end do
    end subroutine svexp
    subroutine gemv(y, A, x, M, K) BIND(C, name="gemv")
      integer(C_long), intent(in) :: M, K
      real(C_double), intent(in) :: A(M,K), x(K)
      real(C_double), dimension(M), intent(out) :: y
      integer(C_long) :: mm, kk
      y = 0.0
      do concurrent(kk = 1:K)
         do concurrent(mm = 1:M)
            y(mm) = y(mm) + A(mm,kk) * x(kk)
         end do
      end do
    end subroutine gemv
    subroutine gemvbuiltin(y, A, x, M, K) BIND(C, name="gemv_builtin")
      integer(C_long), intent(in) :: M, K
      real(C_double), intent(in) :: A(M,K), x(K)
      real(C_double), dimension(M), intent(out) :: y
      integer(C_long) :: mm, kk
      y = matmul(A, x)
    end subroutine gemvbuiltin
    subroutine unscaledvar(s, A, x, M, N) BIND(C, name="unscaledvar")
      integer(C_long), intent(in) :: M, N
      real(C_double), intent(in) :: A(M,N), x(M)
      real(C_double), dimension(M), intent(out) :: s
      integer(C_long) :: mm, nn
      real(C_double) :: d
      s = 0.0
      do concurrent(nn = 1:N)
         do concurrent(mm = 1:M)
            d = A(mm,nn) - x(mm)
            s(mm) = s(mm) + d * d
         end do
      end do
    end subroutine unscaledvar
    subroutine aplusBc(D, a, B, c, M, N) BIND(C, name="aplusBc")
      integer(C_long), intent(in) :: M, N
      real(C_double), intent(in) :: a(M), B(M,N), c(N)
      real(C_double), dimension(M,N), intent(out) :: D
      integer(C_long) :: mm, nn
      do concurrent(nn = 1:N)
         do concurrent(mm = 1:M)
            D(mm,nn) = a(mm) + B(mm,nn) * c(nn)
         end do
      end do
    end subroutine aplusBc
    subroutine OLSlp(lp, y, X, b, N, P) BIND(C, name="OLSlp")
      integer(C_long), intent(in) :: N, P
      real(C_double), intent(in) :: y(N), X(N, P), b(P)
      real(C_double), intent(out) :: lp
      integer(C_long) :: nn, pp
      real(C_double) :: d
      lp = 0
      do concurrent(nn = 1:N)
         d = y(nn)
         do concurrent(pp = 1:P)
            d = d - X(nn,pp) * b(pp)
         end do
         lp = lp + d*d
      end do
    end subroutine OLSlp
    subroutine AplusAt(B, A, N) BIND(C, name="AplusAt")
      integer(C_long), intent(in) :: N
      real(C_double), dimension(N,N), intent(out) :: B
      real(C_double), dimension(N,N), intent(in) :: A
      integer(C_long) :: i, j
      do concurrent(i = 1:N)
         do concurrent(j = 1:N)
            B(j,i) = A(j,i) + A(i,j)
         end do
      end do
    end subroutine AplusAt
    subroutine AplusAtbuiltin(B, A, N) BIND(C, name="AplusAtbuiltin")
      integer(C_long), intent(in) :: N
      real(C_double), dimension(N,N), intent(out) :: B
      real(C_double), dimension(N,N), intent(in) :: A
      B = A + transpose(A)
    end subroutine AplusAtbuiltin
    subroutine randomaccess(pp, P, basis, coefs, A, C) BIND(C, name="randomaccess")
      integer(C_long), intent(in) :: A, C
      real(C_double), intent(in) :: P(A,C), coefs(C)
      integer(C_long), intent(in) :: basis(A,C)
      real(C_double), intent(out) :: pp
      real(C_double) :: pc
      integer(C_long) :: aa, cc
      pp = 0
      do cc = 1,C
         pc = coefs(cc)
         do aa = 1,A
            pc = pc * P(aa, basis(aa, cc))
         end do
         pp = pp + pc
      end do
    end subroutine randomaccess
  end module looptests
