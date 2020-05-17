
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
        @avx for i in eachindex(x)
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

    @test msdavx(FallbackArrayWrapper(x)) == 1e18
    @test msd(x) == msdavx(FallbackArrayWrapper(x))
    @test msdavx(x) != msdavx(FallbackArrayWrapper(x))
end

