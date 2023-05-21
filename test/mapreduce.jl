

@testset "mapreduce" begin
  function maximum_avx(x)
    s = typemin(eltype(x))
    @turbo for i in eachindex(x)
      s = max(s, x[i])
    end
    s
  end
  function minimum_avx(x)
    s = typemax(eltype(x))
    @turbo for i in eachindex(x)
      s = min(s, x[i])
    end
    s
  end
  for T ∈ (Int32, Int64, Float32, Float64)
    @show T, @__LINE__
    if T <: Integer
      R = T(1):T(100)
      x7 = rand(R, 7)
      y7 = rand(R, 7)
      x = rand(R, 127, 7, 7)
      y = rand(R, 127, 7, 7)
      z = rand(R, 3, 3, 3, 3, 3, 3)
    else
      x7 = rand(T, 7)
      y7 = rand(T, 7)
      x = rand(T, 127, 7, 7)
      y = rand(T, 127, 7, 7)
      if VERSION ≥ v"1.4"
        @test vmapreduce(hypot, +, x, y) ≈ mapreduce(hypot, +, x, y)
        @test vmapreduce(^, (a, b) -> a + b, x7, y7) ≈ mapreduce(^, +, x7, y7)
      else
        @test vmapreduce(hypot, +, x, y) ≈ sum(hypot.(x, y))
        @test vmapreduce(^, (a, b) -> a + b, x7, y7) ≈ sum(x7 .^ y7)
      end
      z = rand(T, 3, 3, 3, 3, 3, 3)
    end
    @test vreduce(+, x7) ≈ sum(x7)
    @test vreduce(+, x) ≈ sum(x)
    if T === Int32
      @test vreduce(*, x7) == (prod(x7) % Int32)
      @test vreduce(*, x) == (prod(x) % Int32)
    else
      @test vreduce(*, x7) ≈ prod(x7)
      @test vreduce(*, x) ≈ prod(x)
    end
    @test vmapreduce(abs2, max, x) ≈ mapreduce(abs2, max, x)
    @test vmapreduce(abs2, min, x) ≈ mapreduce(abs2, min, x)
    @test vmapreduce(sqrt, *, x) ≈ mapreduce(sqrt, *, x)
    @test_throws AssertionError vmapreduce(hypot, +, x7, x)
    if VERSION ≥ v"1.4"
      @test vmapreduce(a -> 2a, *, x) ≈ mapreduce(a -> 2a, *, x)
      @test vmapreduce(sin, +, x7) ≈ mapreduce(sin, +, x7)
    else
      @test vmapreduce(a -> 2a, *, x) ≈ prod(2 .* x)
      @test vmapreduce(sin, +, x7) ≈ sum(sin.(x7))
    end
    @test vmapreduce(log, +, x) ≈ sum(log, x)
    @test vmapreduce(abs2, +, x) ≈ sum(abs2, x)
    @test vsum(log, x) ≈ sum(log, x)
    @test vsum(abs2, x) ≈ sum(abs2, x)
    @test vsum(x) ≈ sum(x)
    @test maximum(x) == vreduce(max, x) == maximum_avx(x)
    @test minimum(x) == vreduce(min, x) == minimum_avx(x)

    @test vreduce(max, vec(x); dims = 1) == maximum(vec(x); dims = 1)
    @test vreduce(min, vec(x); dims = 1) == minimum(vec(x); dims = 1)
    @test vreduce(+, vec(x); dims = 1) ≈ sum(vec(x); dims = 1)
    for d = 1:ndims(x)
      @test vreduce(max, x; dims = d) == maximum(x; dims = d)
      @test vreduce(min, x; dims = d) == minimum(x; dims = d)
      @test vreduce(+, x; dims = d) ≈ sum(x; dims = d)
    end
    for d = 1:ndims(z)
      vreduce(max, z, dims = d) == maximum(z, dims = d)
      vreduce(min, z, dims = d) == minimum(z, dims = d)
      vreduce(+, z, dims = d) ≈ sum(z, dims = d)
    end
  end

end
