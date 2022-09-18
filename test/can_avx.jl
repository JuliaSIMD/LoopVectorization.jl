


@testset "can_avx" begin

  import SpecialFunctions

  good_operators = [log, log1p, exp, +, -, Base.FastMath.add_fast, /, sqrt, tanh_fast, sigmoid_fast, LoopVectorization.relu]
  bad_operators = [clenshaw, println, SpecialFunctions.gamma]

  for op in good_operators
    @test LoopVectorization.can_avx(op)
  end
  for op in bad_operators
    @test !LoopVectorization.can_avx(op)
  end


  # Test safe @turbo
  x = Float32.(1:0.1:10)
  y = similar(x)
  truth = similar(x)

  @turbo safe=true for i in indices(x)
      y[i] = SpecialFunctions.gamma(x[i])
  end
  for i in indices(x)
      truth[i] = SpecialFunctions.gamma(x[i])
  end

  @test y ≈ truth

end
