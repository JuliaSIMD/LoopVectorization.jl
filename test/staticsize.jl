
using LoopVectorization, StrideArraysCore, Test

function issue238_noavx!(output, matrix, input)
  for i in axes(output, 2), v in axes(output, 1)
    res = zero(eltype(output))
    for ii in axes(matrix, 2)
      res += matrix[i, ii] * input[v, ii]
    end
    output[v, i] = res
  end
  return nothing
end
function issue238!(output, matrix, input)
  @avx inline=true for i in axes(output, 2), v in axes(output, 1)
    res = zero(eltype(output))
    for ii in axes(matrix, 2)
      res += matrix[i, ii] * input[v, ii]
    end
    output[v, i] = res
  end
  return nothing
end

@testset "Statically Sized Arrays" begin
  for n1 ∈ 1:16, n3 ∈ 1:16
    output = StrideArray(undef, StaticInt(n1), StaticInt(n3))
    output_nonstatic0 = StrideArray(undef, n1, n3)
    output_nonstatic1 = StrideArray(undef, n1, n3)
    for n2 ∈ 1:16;
      input  = randn(n1, n2)
      matrix = randn(n3, n2)
      fill!(output, NaN)
      fill!(output_nonstatic0, NaN)
      fill!(output_nonstatic1, NaN)
      issue238_noavx!(output_nonstatic0, matrix, input)
      issue238!(output_nonstatic1, matrix, input)
      issue238!(output, matrix, input)
      if !(output ≈ output_nonstatic0 ≈ output_nonstatic1)
        @show n1, n2, n3
      end
      @test output_nonstatic0 ≈ output # static test
      @test output_nonstatic0 ≈ output_nonstatic1 # nonstatic test
    end
  end
end

