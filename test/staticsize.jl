
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
    output[v, i] += res
  end
  return nothing
end
function issue238_v2!(output, matrix, input)
  @avx inline=true for i in axes(output, 2), v in axes(output, 1)
    res = output[v, i]
    for ii in axes(matrix, 2)
      res += matrix[i, ii] * input[v, ii]
    end
    output[v, i] = res
  end
  return nothing
end
const MAXTESTSIXE = 10 # N^3 loop iterations, meaning N^3 different functions compiled
function n2testloop(output1,output2,output3,output_nonstatic0,output_nonstatic1)
  n1, n3 = size(output1)
  for n2 ∈ 1:MAXTESTSIXE;
    # @show n1, n2, n3
    input  = randn(n1, n2)
    matrix = randn(n3, n2)
    smatrix = StrideArray(matrix, (StaticInt(n1), StaticInt(n2)))
    issue238_noavx!(output_nonstatic0, matrix, input)
    fill!(output_nonstatic1, 0); issue238!(output_nonstatic1, matrix, input)
    fill!(output1,0); issue238!(output1, matrix, input)
    fill!(output2,0); issue238!(output2, smatrix, input)
    fill!(output3,0); issue238_v2!(output3, smatrix, input)
    if !(output1 ≈ output2 ≈ output3 ≈ output_nonstatic0 ≈ output_nonstatic1)
      @show n1, n2, n3
    end
    @test output_nonstatic0 ≈ output1 # static test
    @test output_nonstatic0 ≈ output2 # static test
    @test output_nonstatic0 ≈ output3 # static test
    @test output_nonstatic0 ≈ output_nonstatic1 # nonstatic test
  end
end

@testset "Statically Sized Arrays" begin
  for n1 ∈ 1:MAXTESTSIXE, n3 ∈ 1:MAXTESTSIXE
    output1 = StrideArray(undef, StaticInt(n1), StaticInt(n3))
    output2 = StrideArray(undef, StaticInt(n1), StaticInt(n3))
    output3 = StrideArray(undef, StaticInt(n1), StaticInt(n3))
    output_nonstatic0 = StrideArray(undef, n1, n3)
    output_nonstatic1 = StrideArray(undef, n1, n3)
    n2testloop(output1,output2,output3,output_nonstatic0,output_nonstatic1)
  end
end

