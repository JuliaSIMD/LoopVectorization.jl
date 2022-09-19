
@testset "Safe @turbo" begin

  @testset "Test `can_turbo`" begin
    import SpecialFunctions
    using LoopVectorization

    # All methods, both `can_avx` and `can_turbo`, should recognize that
    # `gamma` is not AVX-able
    f(x) = SpecialFunctions.gamma(x)

    @test !LoopVectorization.ArrayInterface.can_avx(SpecialFunctions.gamma)
    @test !LoopVectorization.can_turbo(SpecialFunctions.gamma, Val(1))
    @test !LoopVectorization.can_turbo(f, Val(1))

    # `can_avx` is not able to detect that a function `f` which is just
    # `gamma` can be AVX'd, but `can_turbo` can:
    f(x) = exp(x)

    @test !LoopVectorization.ArrayInterface.can_avx(f)
    @test LoopVectorization.can_turbo(exp, Val(1))
    @test LoopVectorization.can_turbo(f, Val(1))

    # Next, we test with multiple arguments:
    g(x, y) = x + SpecialFunctions.gamma(y)
    @test !LoopVectorization.can_turbo(g, Val(2))
    g(x, y) = x + exp(y)
    @test LoopVectorization.can_turbo(g, Val(2))
  end

  @testset "Test `@turbo` with `safe=true`" begin
    import SpecialFunctions
    using LoopVectorization

    x = Float32.(1.05:0.1:10)
    y = Float32.(0.55:0.1:10.5)
    z = similar(x)
    truth = similar(x)

    @turbo safe=true for i in indices(x)
        z[i] = SpecialFunctions.gamma(x[i])
    end
    for i in indices(x)
        truth[i] = SpecialFunctions.gamma(x[i])
    end
    @test z ≈ truth
    
    f(x, y) = x + SpecialFunctions.gamma(y)
    @turbo safe=true for i in indices(x)
        z[i] = f(x[i], y[i])
    end
    for i in indices(x)
        truth[i] = f(x[i], y[i])
    end
    @test z ≈ truth
  end
end
