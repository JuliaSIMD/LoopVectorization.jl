@testset "filter" begin
    N = 347
    for T ∈ (Float32, Float64, Int32, Int64)
        @show T, @__LINE__
        if T <: Integer
            x = rand(T(-100):T(100), N);
        else
            x = rand(T, N);
        end
        y1 = filter(a -> a > 0.7, x);
        y2 = vfilter(a -> a > 0.7, x);
        @test y1 == y2
        y1 = filter(a -> a ≤ 0.7, x);
        y3 = vfilter(a -> a ≤ 0.7, x);
        @test y1 == y3
        @test length(y2) + length(y3) == N
    end
end
