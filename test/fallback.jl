
@testset "Fall back behavior" begin

  @show Float64, @__LINE__
  function msd(x)
    s = zero(eltype(x))
    for i in eachindex(x)
      s += x[i] * x[i]
    end
    s
  end
  function msdavx(x)
    s = zero(eltype(x))
    @turbo warn_check_args=true for i in eachindex(x)
      s = muladd(x[i], x[i], s) # Avoids fastmath in fallback loop.
    end
    s
  end

  x = fill(8.0, 128);
  x[1] = 1e9

  @test @inferred LoopVectorization.check_args(x)
  @test @inferred LoopVectorization.check_args(x, x)
  @test @inferred LoopVectorization.check_args(x, x, x)
  @test @inferred !LoopVectorization.check_args(view(x, [1,3,4,18]))
  @test @inferred !LoopVectorization.check_args(FallbackArrayWrapper(x))
  @test @inferred !LoopVectorization.check_args(FallbackArrayWrapper(x), x, x)
  @test @inferred !LoopVectorization.check_args(x, FallbackArrayWrapper(x))
  @test @inferred !LoopVectorization.check_args(x, FallbackArrayWrapper(x), x)
  @test @inferred !LoopVectorization.check_args(x, x, FallbackArrayWrapper(x))
  @test @inferred !LoopVectorization.check_args(x, x, FallbackArrayWrapper(x), FallbackArrayWrapper(x))
  @test @inferred !LoopVectorization.check_args(['a'])
  @test @inferred !LoopVectorization.check_args(Diagonal(x))

  @test_nowarn msdavx(x)
  lnn = LineNumberNode(14, joinpath(pkgdir(LoopVectorization),"test","fallback.jl"))
  warnstring = "$(lnn):\n`LoopVectorization.check_args` on your inputs failed; running fallback `@inbounds @fastmath` loop instead.\nUse `warn_check_args=false`, e.g. `@turbo warn_check_args=false ...`, to disable this warning."

  @test_logs (:warn,warnstring) msdavx(FallbackArrayWrapper(x))
  @test msdavx(FallbackArrayWrapper(x)) == 1e18
  @test msd(x) == msdavx(FallbackArrayWrapper(x))
  @test msdavx(x) != msdavx(FallbackArrayWrapper(x))

  x = rand(1000); # should be long enough to make zero differences incredibly unlikely
  @test exp.(x) != (@turbo exp.(x))
  @test exp.(x) == (@turbo exp.(FallbackArrayWrapper(x)))
end

