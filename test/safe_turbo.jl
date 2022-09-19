
@testset "Safe @turbo" begin

  import SpecialFunctions

  # All methods, both `can_avx` and `can_turbo`, should recognize that
  # `gamma` is not AVX-able
  f1(x) = SpecialFunctions.gamma(x)

  @test !LoopVectorization.ArrayInterface.can_avx(SpecialFunctions.gamma)
  @test !LoopVectorization.can_turbo(SpecialFunctions.gamma, Val(1))
  @test !LoopVectorization.can_turbo(f1, Val(1))

  # `can_avx` is not able to detect that a function `f` which is just
  # `gamma` can be AVX'd, but `can_turbo` can:
  f2(x) = exp(x)

  @test LoopVectorization.ArrayInterface.can_avx(exp)
  @test !LoopVectorization.ArrayInterface.can_avx(f2)
  @test LoopVectorization.can_turbo(exp, Val(1))
  @test LoopVectorization.can_turbo(f2, Val(1))

  # Next, we test with multiple arguments:
  g1(x, y) = x + SpecialFunctions.gamma(y)
  @test !LoopVectorization.can_turbo(g1, Val(2))
  g2(x, y) = x + exp(y)
  @test LoopVectorization.can_turbo(g2, Val(2))

  x = Float32.(1.05:0.1:10)
  y = Float32.(0.55:0.1:10.5)
  z = similar(x)
  truth = similar(x)

  @turbo safe=true for i in indices(x)
      z[i] = SpecialFunctions.gamma(x[i])
  end
  for i in indices(x)
      truth[i] = SpecialFunctions.gamma(x[i])
  end
  @test z ≈ truth
  
  f3(x, y) = x + SpecialFunctions.gamma(y)
  @turbo safe=true for i in indices(x)
      z[i] = f3(x[i], y[i])
  end
  f4(x, y) = x + SpecialFunctions.gamma(y)
  for i in indices(x)
      truth[i] = f4(x[i], y[i])
  end
  @test z ≈ truth

end
