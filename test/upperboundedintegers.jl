using OffsetArrays, Test
@testset "UpperboundedIntegers" begin
  function ubsum(x)
    @assert firstindex(x) == 0
    r = LoopVectorization.CloseOpen(LoopVectorization.UpperBoundedInteger(length(x), StaticInt(15)))
    s = zero(eltype(x))
    @avx for i ∈ r
      s += x[i]
    end
    s
  end
  function ubdouble!(y,x)
    @assert firstindex(x) == 0
    r = LoopVectorization.CloseOpen(LoopVectorization.UpperBoundedInteger(length(x), StaticInt(15)))
    @avx for i ∈ r
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
    x = OffsetVector(rand(l), -1)
    # @test ubsum(x) ≈ sum(@view(x[begin:14]))
    @test ubsum(x) ≈ ubsum(x)
    xs = similar(x)
    @test @view(ubdouble!(xs, x)[begin:14]) == @view(x[begin:14]) .* 2;
    @test xs[end] ≠ 2x[end]
  end
end
