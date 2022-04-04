
function broutine2x2!(S::AbstractMatrix{Complex{T}}, U::AbstractMatrix, locs::Int) where {T}
  step_1 = 1 << (locs - 1)
  step_2 = 1 << locs
  U11_re = real(U[1, 1])
  U11_im = imag(U[1, 1])
  U12_re = real(U[1, 2])
  U12_im = imag(U[1, 2])
  U21_re = real(U[2, 1])
  U21_im = imag(U[2, 1])
  U22_re = real(U[2, 2])
  U22_im = imag(U[2, 2])

  Hmax = size(S, 2)
  Sr = reinterpret(reshape, T, S)

  if step_1 == 1
    for _j = 0:(Hmax>>>locs)-1, b in axes(S, 1)
      j = _j << locs

      ST1_re =
        U11_re * Sr[1, b, j+1] - U11_im * Sr[2, b, j+1] + U12_re * Sr[1, b, j+2] -
        U12_im * Sr[2, b, j+2]
      ST1_im =
        U11_re * Sr[2, b, j+1] +
        U11_im * Sr[1, b, j+1] +
        U12_re * Sr[2, b, j+2] +
        U12_im * Sr[1, b, j+2]
      ST2_re =
        U21_re * Sr[1, b, j+1] - U21_im * Sr[2, b, j+1] + U22_re * Sr[1, b, j+2] -
        U22_im * Sr[2, b, j+2]
      ST2_im =
        U21_re * Sr[2, b, j+1] +
        U21_im * Sr[1, b, j+1] +
        U22_re * Sr[2, b, j+2] +
        U22_im * Sr[1, b, j+2]


      Sr[1, b, j+1] = ST1_re
      Sr[2, b, j+1] = ST1_im
      Sr[1, b, j+2] = ST2_re
      Sr[2, b, j+2] = ST2_im
    end
    return S
  end
end

function broutine2x2_avx!(
  S::AbstractMatrix{Complex{T}},
  U::AbstractMatrix,
  locs::Int,
) where {T}
  step_1 = 1 << (locs - 1)
  step_2 = 1 << locs
  U11_re = real(U[1, 1])
  U11_im = imag(U[1, 1])
  U12_re = real(U[1, 2])
  U12_im = imag(U[1, 2])
  U21_re = real(U[2, 1])
  U21_im = imag(U[2, 1])
  U22_re = real(U[2, 2])
  U22_im = imag(U[2, 2])

  Hmax = size(S, 2)
  Sr = reinterpret(reshape, T, S)

  if step_1 == 1
    @turbo for _j = 0:(Hmax>>>locs)-1, b in axes(S, 1)
      j = _j << locs

      ST1_re =
        U11_re * Sr[1, b, j+1] - U11_im * Sr[2, b, j+1] + U12_re * Sr[1, b, j+2] -
        U12_im * Sr[2, b, j+2]
      ST1_im =
        U11_re * Sr[2, b, j+1] +
        U11_im * Sr[1, b, j+1] +
        U12_re * Sr[2, b, j+2] +
        U12_im * Sr[1, b, j+2]
      ST2_re =
        U21_re * Sr[1, b, j+1] - U21_im * Sr[2, b, j+1] + U22_re * Sr[1, b, j+2] -
        U22_im * Sr[2, b, j+2]
      ST2_im =
        U21_re * Sr[2, b, j+1] +
        U21_im * Sr[1, b, j+1] +
        U22_re * Sr[2, b, j+2] +
        U22_im * Sr[1, b, j+2]


      Sr[1, b, j+1] = ST1_re
      Sr[2, b, j+1] = ST1_im
      Sr[1, b, j+2] = ST2_re
      Sr[2, b, j+2] = ST2_im
    end
    return S
  end
end

@testset "Quantum" begin
  N = 10
  S = rand(ComplexF64, 100, 1 << N)
  U = rand(ComplexF64, 2, 2)

  S1 = broutine2x2!(copy(S), U, 1)
  S2 = broutine2x2_avx!(copy(S), U, 1)
  @test S1 ≈ S2 # 239
end
