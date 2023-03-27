import LinearAlgebra
function issue407!(Hdata::Matrix{Complex{T}}) where {T}
  Hview = reinterpret(reshape, T, Hdata)
  @tturbo for i = 1:size(Hdata, 1)
    Hview[1, i, i] = i
  end
end
function lv_turbo(r::Vector{UInt64}, mask::UInt64)
  @turbo for i = 1:length(r)
    r[i] &= mask
  end
  r
end

@testset "Simple Miscellaneous" begin
  @show @__LINE__
  r1 = rand(UInt, 239)
  mask = ~(one(eltype(r1)) << (2))
  r2 = r1 .& mask
  @test lv_turbo(r1, mask) == r2
end
@testset "issue 407" begin
  @show @__LINE__
  A = zeros(Complex{Float64}, 10, 10)
  issue407!(A)
  @test real.(A) == LinearAlgebra.Diagonal(1:10)
  @test all(iszero, imag.(A))
end
function issue480(x, y)
  z = false
  @turbo for i in eachindex(x)
    z |= x[i] > y[i]
  end
  z
end
@testset "issue 480" begin
  using LoopVectorization
  x = zeros(33)
  y = zeros(33)
  @test !issue480(x, y)
  for i in eachindex(x)
    x[i] = 1.0
    @test issue480(x, y)
    x[i] = 0.0
  end
end
