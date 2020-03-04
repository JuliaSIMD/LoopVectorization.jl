module jitmul

include "/opt/intel/mkl/include/mkl_direct_call.fi"

use ISO_C_BINDING
implicit none

contains

    ! subroutine dgemmjit(C,A,B,M,K,N,alpha,beta) bind(C, name = "dgemmjit")
  subroutine dgemmjit(C,A,B,M,K,N,At,Bt) bind(C, name = "dgemmjit")
    integer(C_int32_t),  intent(in)  :: M, K, N
    integer(C_int8_t),   intent(in) :: At, Bt
    real(C_double), parameter :: alpha = 1.0D0, beta = 0.0D0
    ! real(C_double),                 intent(in)  :: alpha, beta
    real(C_double), dimension(M,K), intent(in)  :: A
    real(C_double), dimension(K,N), intent(in)  :: B
    real(C_double), dimension(M,N), intent(out) :: C
    character :: Atc, Btc
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

end module jitmul
