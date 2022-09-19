


@testset "can_avx" begin

  using LoopVectorization

  good_operators = [log, log1p, exp, +, -, Base.FastMath.add_fast, /, sqrt, tanh_fast, sigmoid_fast, LoopVectorization.relu]
  bad_operators = [clenshaw, println]

  for op in good_operators
    @test LoopVectorization.can_avx(op)
  end
  for op in bad_operators
    @test !LoopVectorization.can_avx(op)
  end

end
