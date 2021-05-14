
@testset "Iteration Bound Tests" begin
  function masktest_incr1_none1start!(y,x)
    @vectorize for i ∈ 0:20
      y[i] = x[i] + 2
    end
  end
  function masktest_incr2_none1start!(y,x)
    @vectorize for i ∈ 0:2:20
      y[i] = x[i] + 2
    end
  end

  x = OffsetVector(rand(24),0:23);
  y = copy(x);
  masktest_incr1_none1start!(y,x)
  @test y == x .+ ifelse.(axes(x,1) .≤ 20, 2, 0)
  @vectorize y .= x;
  @test y == x
  masktest_incr2_none1start!(y,x)
  @test y == x .+ ifelse.((axes(x,1) .≤ 20) .& iseven.(axes(x,1)), 2, 0)
end
