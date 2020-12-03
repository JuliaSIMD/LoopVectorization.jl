


@testset "can_avx" begin

    @test LoopVectorization.ArrayInterface.can_avx(log)
    @test LoopVectorization.ArrayInterface.can_avx(log1p)
    @test LoopVectorization.ArrayInterface.can_avx(exp)
    @test LoopVectorization.ArrayInterface.can_avx(+)
    @test LoopVectorization.ArrayInterface.can_avx(-)
    @test LoopVectorization.ArrayInterface.can_avx(Base.FastMath.add_fast)
    @test LoopVectorization.ArrayInterface.can_avx(/)
    @test LoopVectorization.ArrayInterface.can_avx(sqrt)
    @test !LoopVectorization.ArrayInterface.can_avx(clenshaw)

end


