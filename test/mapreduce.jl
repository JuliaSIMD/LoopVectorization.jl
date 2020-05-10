
@testset "mapreduce" begin

    for T ∈ (Int32, Int64, Float32, Float64)
        if T <: Integer
            R = T(1):T(100)
            x7 = rand(R, 7); y7 = rand(R, 7);
            x = rand(R, 127); y = rand(R, 127);
        else
            x7 = rand(T, 7); y7 = rand(T, 7);
            x = rand(T, 127); y = rand(T, 127);
            @test vmapreduce(hypot, +, x, y) ≈ mapreduce(hypot, +, x, y)
            @test vmapreduce(^, (a,b) -> a + b, x7, y7) ≈ mapreduce(^, (a,b) -> a + b, x7, y7)
        end
        @test vreduce(+, x7) ≈ sum(x7)
        @test vreduce(+, x) ≈ sum(x)
        @test_throws AssertionError vmapreduce(hypot, +, x7, x)
        @test vmapreduce(a -> 2a, *, x) ≈ mapreduce(a -> 2a, *, x)
        @test vmapreduce(sin, +, x7) ≈ mapreduce(sin, +, x7)
        @test vmapreduce(log, +, x) ≈ mapreduce(log, +, x)
        @test vmapreduce(abs2, +, x) ≈ mapreduce(abs2, +, x)
    end

end

