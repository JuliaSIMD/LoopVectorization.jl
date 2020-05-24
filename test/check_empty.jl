
function mysum_checked(x)
    s = zero(eltype(x))
    @avx check_empty = true for i ∈ eachindex(x)
        s += x[i]
    end
    s
end
function mysum_unchecked(x)
    s = zero(eltype(x))
    @avx for i ∈ eachindex(x)
        s += x[i]
    end
    s
end

@testset "Check Empty" begin
    x = fill(9999, 100, 10, 10);
    xv = view(x, :, 1:0, :);
    @test mysum_checked(x) == mysum_unchecked(x) == sum(x)
    @test iszero(mysum_checked(xv))
    @test !iszero(mysum_unchecked(xv))
    
end

