using OffsetArrays, Test
@testset "UpperboundedIntegers" begin
  function ubsum(x)
    @assert firstindex(x) == 0
    r = LoopVectorization.CloseOpen(LoopVectorization.UpperBoundedInteger(length(x), LoopVectorization.StaticInt(15)))
    s = zero(eltype(x))
    @turbo for i ∈ r
      s += x[i]
    end
    s
  end
  function ubdouble!(y,x)
    @assert firstindex(x) == 0
    r = LoopVectorization.CloseOpen(LoopVectorization.UpperBoundedInteger(length(x), LoopVectorization.StaticInt(15)))
    @turbo for i ∈ r
      y[i] = 2*x[i]
    end
    y
  end
  for l ∈ 1:15
    x = OffsetVector(rand(l), -1)
    @test ubsum(x) ≈ sum(x)
    @test ubdouble!(similar(x), x) == x .* 2
  end
  for l ∈ 70:75
    # If the actual value is greater than the ubber bound, as is the case here,
    # (upper bound = 15, actual length = 70:75), then the result is undefined.
    # Currently, it'll evaluate to the actual value with heavily unrolled reductions
    x = OffsetVector(rand(l), -1)
    @test ubsum(x) ≈ ubsum(x)
    xs = similar(x)
    # while with other operations, it'll do at most `UF * W * cld(N, W)` iterations,
    # where `N` is the actual length, `W` the vector width, and `UF` the unroll factor
    @test @view(ubdouble!(xs, x)[begin:14]) == @view(x[begin:14]) .* 2;
    @test xs[end] ≠ 2x[end]
  end
end
