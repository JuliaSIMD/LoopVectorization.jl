

@testset "mapreduce" begin
    function maximum_avx(x)
        s = typemin(eltype(x))
        @avx for i in eachindex(x)
            s = max(s, x[i])
        end
        s
    end
    for T ∈ (Int32, Int64, Float32, Float64)
        @show T, @__LINE__
        if T <: Integer
            R = T(1):T(100)
            x7 = rand(R, 7); y7 = rand(R, 7);
            x = rand(R, 127); y = rand(R, 127);
        else
            x7 = rand(T, 7); y7 = rand(T, 7);
            x = rand(T, 127); y = rand(T, 127);
            if VERSION ≥ v"1.4"
                @test vmapreduce(hypot, +, x, y) ≈ mapreduce(hypot, +, x, y)
                @test vmapreduce(^, (a,b) -> a + b, x7, y7) ≈ mapreduce(^, +, x7, y7)
            else
                @test vmapreduce(hypot, +, x, y) ≈ sum(hypot.(x, y))
                @test vmapreduce(^, (a,b) -> a + b, x7, y7) ≈ sum(x7 .^ y7)
            end
        end;
        @test vreduce(+, x7) ≈ sum(x7)
        @test vreduce(+, x) ≈ sum(x)
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
        @test maximum(x) == vreduce(max, x) == maximum_avx(x)
    end

end

