@generated function sum_way_too_unrolled(A, ::Val{rows}, ::Val{cols}) where {rows, cols}
    terms = :( 0 )
    
    for i in 1:rows
        for j in 1:cols
            terms = :( $terms + A[$i, $j, k] )
        end
    end

    quote
        sum = 0.0
        @turbo for k in axes(A, 3)
            sum += $terms
        end
        sum
    end
end

@testset "Many Array References" begin
    A = rand(17, 16, 10)

    @test isapprox(sum_way_too_unrolled(A, Val(17), Val(16)), sum(A))
end
