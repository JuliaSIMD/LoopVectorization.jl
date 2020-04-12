module jitmul

use ISO_C_BINDING
use mkl_service
implicit none

! include "/opt/intel/mkl/include/mkl_direct_call.fi"
include "/home/chriselrod/intel/mkl/include/mkl_direct_call.fi"
! include "/home/chriselrod/intel/mkl/include/mkl_service.fi"
! include "/opt/intel/mkl/include/mkl_service.fi"
! include "/opt/intel/mkl/include/mkl.fi"

contains
  subroutine set_num_threads(N) bind(C, name = "set_num_threads")
    integer(C_int32_t) :: N
    call mkl_set_num_threads(N)
  end subroutine set_num_threads
  
    ! subroutine dgemmjit(C,A,B,M,K,N,alpha,beta) bind(C, name = "dgemmjit")
  subroutine sgemmjit(C,A,B,M,K,N,At,Bt) bind(C, name = "sgemmjit")
    integer(C_int32_t),  intent(in)  :: M, K, N
    integer(C_int8_t),   intent(in) :: At, Bt
    real(C_float), parameter :: alpha = 1.0e0, beta = 0.0e0
    ! real(C_float),                 intent(in)  :: alpha, beta
    real(C_float), dimension(M,K), intent(in)  :: A
    real(C_float), dimension(K,N), intent(in)  :: B
    real(C_float), dimension(M,N), intent(out) :: C
    character :: Atc, Btc
    ! call mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL)
    if (At == 1_C_int8_t) then
       Atc = 'T'
    else
       Atc = 'N'
    end if
    if (Bt == 1_C_int8_t) then
       Btc = 'T'
    else
       Btc = 'N'
    end if
    call sgemm(Atc, Btc, M, N, K, alpha, A, M, B, K, beta, C, M)
  end subroutine sgemmjit
  subroutine dgemmjit(C,A,B,M,K,N,At,Bt) bind(C, name = "dgemmjit")
    integer(C_int32_t),  intent(in)  :: M, K, N
    integer(C_int8_t),   intent(in) :: At, Bt
    real(C_double), parameter :: alpha = 1.0d0, beta = 0.0d0
    ! real(C_double),                 intent(in)  :: alpha, beta
    real(C_double), dimension(M,K), intent(in)  :: A
    real(C_double), dimension(K,N), intent(in)  :: B
    real(C_double), dimension(M,N), intent(out) :: C
    character :: Atc, Btc
    ! call mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL)
    if (At == 1_C_int8_t) then
       Atc = 'T'
    else
       Atc = 'N'
    end if
    if (Bt == 1_C_int8_t) then
       Btc = 'T'
    else
       Btc = 'N'
    end if
    call dgemm(Atc, Btc, M, N, K, alpha, A, M, B, K, beta, C, M)
  end subroutine dgemmjit
  subroutine dgemvjit(y,A,x,M,N,At) bind(C, name = "dgemvjit")
    integer(C_int32_t),  intent(in)  :: M, N
    integer(C_int8_t),   intent(in) :: At
    real(C_double), parameter :: alpha = 1.0d0, beta = 0.0d0
    ! real(C_double),                 intent(in)  :: alpha, beta
    real(C_double), dimension(M,N), intent(in)  :: A
    real(C_double), dimension(M), intent(in)  :: x
    real(C_double), dimension(N), intent(out) :: y
    character :: Atc
    ! call mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL)
    if (At == 1_C_int8_t) then
       Atc = 'T'
    else
       Atc = 'N'
    end if
    call dgemv(Atc, M, N, alpha, A, M, x, 1, beta, y, 1)
  end subroutine dgemvjit

end module jitmul
