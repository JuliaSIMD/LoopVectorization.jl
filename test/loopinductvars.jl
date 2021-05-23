
# testset for using them in loops
@testset "Loop Induction Variables" begin
	@show @__LINE__
    f(x) = cos(x) * log(x)
    function avxmax(v)
        max_x = -Inf
        @turbo for i ∈ eachindex(v)
            x = f(i)
            max_x = max(max_x, x)
        end
        max_x
    end
    function avxextrema(v)
        max_x = -Inf
        min_x = Inf
        @turbo for i ∈ eachindex(v)
            x = f(i)
            max_x = max(max_x, x)
            min_x = min(min_x, x)
        end
        min_x, max_x
    end

    v = 1:19
    minref, maxref = extrema(f, v)
    @test maxref ≈ avxmax(v)
    minavx, maxavx = avxextrema(v);
    @test minref ≈ minavx
    @test maxref ≈ maxavx
end

