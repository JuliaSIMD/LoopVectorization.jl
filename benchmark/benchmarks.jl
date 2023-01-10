
using BenchmarkTools

const SUITE = BenchmarkGroup()
SUITE["linalg"] = BenchmarkGroup(["matmul", "dot"])

include(joinpath(@__DIR__, "looptests.jl"))

SUITE["linalg"]["matmul"] = BenchmarkGroup()
SUITE["linalg"]["dot"] = BenchmarkGroup()
for n ∈ 1:64
  A = rand(n, n)
  A′ = copy(A')
  B = rand(n, n)
  C = Matrix{Float64}(undef, n, n)
  SUITE["linalg"]["matmul"]["AmulB", n] = @benchmarkable gemmavx!($C, $A, $B)
  SUITE["linalg"]["matmul"]["A′mulB", n] =
    @benchmarkable jAtmulBavx!($C, $A′, $B)
  x = rand(n)
  y = rand(n)
  SUITE["linalg"]["dot"]["dot", n] = @benchmarkable jdotavx($x, $y)
  SUITE["linalg"]["dot"]["selfdot", n] = @benchmarkable jselfdotavx($x)
  SUITE["linalg"]["dot"]["dot3", n] = @benchmarkable jdot3avx($x, $A, $y)
end
