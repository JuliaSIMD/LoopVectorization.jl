
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
            s += x[i] * x[i]
        end
        s
    end

    x = fill(sqrt(63.0), 128);
    x[1] = 1e9

    @test LoopVectorization.check_args(x)
    @test LoopVectorization.check_args(x, x)
    @test LoopVectorization.check_args(x, x, x)
    @test !LoopVectorization.check_args(FallbackArrayWrapper(x))
    @test !LoopVectorization.check_args(FallbackArrayWrapper(x), x, x)
    @test !LoopVectorization.check_args(x, FallbackArrayWrapper(x))
    @test !LoopVectorization.check_args(x, FallbackArrayWrapper(x), x)
    @test !LoopVectorization.check_args(x, x, FallbackArrayWrapper(x))
    @test !LoopVectorization.check_args(x, x, FallbackArrayWrapper(x), FallbackArrayWrapper(x))

    @test msdavx(FallbackArrayWrapper(x)) == 1e18
    @test msd(x) == msdavx(FallbackArrayWrapper(x))
    @test msdavx(x) != msdavx(FallbackArrayWrapper(x))
end

