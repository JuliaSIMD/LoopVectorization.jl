_f1(a) = SpecialFunctions.gamma(a)
_f2(a) = exp(a)
_f3(a, b) = a + SpecialFunctions.gamma(b)
_f4(a, b) = a + exp(b)
_f5(a, b) = a + SpecialFunctions.gamma(b)
_f6(a, b) = a + SpecialFunctions.gamma(b)

@testset "Safe @turbo" begin

  using LoopVectorization
  using Test
  import SpecialFunctions

  # All methods, both `can_avx` and `can_turbo`, should recognize that
  # `gamma` is not AVX-able

  @test !LoopVectorization.ArrayInterface.can_avx(SpecialFunctions.gamma)
  @test !LoopVectorization.can_turbo(SpecialFunctions.gamma, Val(1))
  @test !LoopVectorization.can_turbo(_f1, Val(1))

  # `can_avx` is not able to detect that a function `f` which is just
  # `gamma` can be AVX'd, but `can_turbo` can:

  @test LoopVectorization.ArrayInterface.can_avx(exp)
  @test !LoopVectorization.ArrayInterface.can_avx(_f2)
  @test LoopVectorization.can_turbo(exp, Val(1))
  @test LoopVectorization.can_turbo(_f2, Val(1))

  # Next, we test with multiple arguments:
  @test !LoopVectorization.can_turbo(_f3, Val(2))
  @test LoopVectorization.can_turbo(_f4, Val(2))

  x = Float32.(1.05:0.1:10)
  y = Float32.(0.55:0.1:10.5)
  z = similar(x)
  truth = similar(x)

  LoopVectorization.@turbo safe=true for i in indices(x)
      z[i] = SpecialFunctions.gamma(x[i])
  end
  for i in indices(x)
      truth[i] = SpecialFunctions.gamma(x[i])
  end
  @test z ≈ truth
  
  LoopVectorization.@turbo safe=true for i in indices(x)
      z[i] = _f5(x[i], y[i])
  end
  for i in indices(x)
      truth[i] = _f6(x[i], y[i])
  end
  @test z ≈ truth

end

