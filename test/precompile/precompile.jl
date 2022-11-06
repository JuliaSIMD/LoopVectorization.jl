using Test
using SnoopCompileCore

@testset "Invalidation and precompilation" begin
  invs = @snoopr using LVUser
  m = only(methods(LVUser.filter2davx))
  mi = m.specializations[1]
  @test mi ∉ invs
  A = rand(Float64, 512, 512)
  kern = [
    0.1 0.3 0.1
    0.3 0.5 0.3
    0.1 0.3 0.1
  ]
  B = filter2davx(A, kern)
  @test size(B) == size(A) .- size(kern) .+ 1
end
