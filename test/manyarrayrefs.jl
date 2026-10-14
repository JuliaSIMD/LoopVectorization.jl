@generated function sum_way_too_unrolled(A, ::Val{rows}, ::Val{cols}) where {rows,cols}
  terms = :(0)

  for i = 1:rows
    for j = 1:cols
      terms = :($terms + A[$i, $j, k])
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

@testset "Many Array References: type stability" begin
  # The element type of the loop is determined by `promote_type` over all array
  # references. With many references, a single `promote_type` call with many
  # arguments is not inferred on Julia >= 1.11, which made the generated code
  # type-unstable (dynamic dispatch and allocations in every iteration).
  A = rand(6, 7, 10)
  sum_way_too_unrolled(A, Val(6), Val(7)) # compile
  @test isapprox(sum_way_too_unrolled(A, Val(6), Val(7)), sum(A))
  @test (@allocated sum_way_too_unrolled(A, Val(6), Val(7))) == 0
end
