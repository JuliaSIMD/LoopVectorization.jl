
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
  @turbo inline = true for i in axes(output, 2), v in axes(output, 1)
    res = zero(eltype(output))
    for ii in axes(matrix, 2)
      res += matrix[i, ii] * input[v, ii]
    end
    output[v, i] += res
  end
  return nothing
end
function issue238_v2!(output, matrix, input)
  @turbo inline = true for i in axes(output, 2), v in axes(output, 1)
    res = output[v, i]
    for ii in axes(matrix, 2)
      res += matrix[i, ii] * input[v, ii]
    end
    output[v, i] = res
  end
  return nothing
end
const MAXTESTSIZE = 8 # N^3 loop iterations, meaning N^3 different functions compiled
function n2testloop(output1, output2, output3, output_nonstatic0, output_nonstatic1)
  n1, n3 = size(output1)
  for n2 ∈ 1:MAXTESTSIZE
    # @show n1, n2, n3
    input = randn(n1, n2)
    matrix = randn(n3, n2)
    smatrix = StrideArray(matrix, (StaticInt(n3), StaticInt(n2)))
    issue238_noavx!(output_nonstatic0, matrix, input)
    fill!(output_nonstatic1, 0)
    issue238!(output_nonstatic1, matrix, input)
    fill!(output1, 0)
    issue238!(output1, matrix, input)
    fill!(output2, 0)
    issue238!(output2, smatrix, input)
    fill!(output3, 0)
    issue238_v2!(output3, smatrix, input)
    if !(output1 ≈ output2 ≈ output3 ≈ output_nonstatic0 ≈ output_nonstatic1)
      @show n1, n2, n3
    end
    @test output_nonstatic0 ≈ output1 # static test
    @test output_nonstatic0 ≈ output2 # static test
    @test output_nonstatic0 ≈ output3 # static test
    @test output_nonstatic0 ≈ output_nonstatic1 # nonstatic test
  end
end
function update!(B⁻¹yₖ, B⁻¹, yₖ, sₖᵀyₖ⁻¹)
  yₖᵀB⁻¹yₖ = zero(eltype(B⁻¹))
  @inbounds @fastmath for c ∈ axes(B⁻¹, 2)
    t = zero(yₖᵀB⁻¹yₖ)
    for r ∈ axes(B⁻¹, 1)
      t += yₖ[r] * B⁻¹[r, c]
    end
    B⁻¹yₖ[c] = t * sₖᵀyₖ⁻¹
    yₖᵀB⁻¹yₖ += t * yₖ[c]
  end
  yₖᵀB⁻¹yₖ
end
function update_turbo!(B⁻¹yₖ, B⁻¹, yₖ, sₖᵀyₖ⁻¹)
  yₖᵀB⁻¹yₖ = zero(eltype(B⁻¹))
  @turbo for c ∈ axes(B⁻¹, 2)
    t = zero(yₖᵀB⁻¹yₖ)
    for r ∈ axes(B⁻¹, 1)
      t += yₖ[r] * B⁻¹[r, c]
    end
    B⁻¹yₖ[c] = t * sₖᵀyₖ⁻¹
    yₖᵀB⁻¹yₖ += t * yₖ[c]
  end
  yₖᵀB⁻¹yₖ
end

function maxabs(x)
  s = -Inf
  @turbo for i ∈ eachindex(x)
    s = max(s, abs(x[i]))
  end
  s
end
function sum_turbo(x)
  s = zero(eltype(x))
  @turbo for i ∈ eachindex(x)
    s += x[i]
  end
  s
end
function sum2_10turbo(x)
  s = zero(eltype(x))
  for i = 1:10, j = 1:2
    s += x[j, i]
  end
  s
end

@testset "Statically Sized Arrays" begin
  @show @__LINE__
  for n1 ∈ 1:MAXTESTSIZE, n3 ∈ 1:MAXTESTSIZE
    # @show n1, n3
    output1 = StrideArray(undef, StaticInt(n1), StaticInt(n3))
    output2 = StrideArray(undef, StaticInt(n1), StaticInt(n3))
    output3 = StrideArray(undef, StaticInt(n1), StaticInt(n3))
    output_nonstatic0 = StrideArray(undef, n1, n3)
    output_nonstatic1 = StrideArray(undef, n1, n3)
    n2testloop(output1, output2, output3, output_nonstatic0, output_nonstatic1)

    y = StrideArray(undef, StaticInt(max(n1, n3)))
    y .= rand.()
    By0 = StrideArray(undef, StaticInt(n3))
    By1 = StrideArray(undef, StaticInt(n3))
    GC.@preserve By0 By1 output1 y begin
      @test update_turbo!(By0, output1, y, 0.124) ≈ update!(By1, output1, y, 0.124)
      @test By0 ≈ By1
    end
  end
  for i = 1:65
    # @show i
    x = StrideArray(undef, StaticInt(i))
    x .= randn.()
    @test maxabs(x) == maximum(abs, x)
    @test sum_turbo(x) ≈ sum(x)
  end
  let A = rand(2, 10)
    @test sum2_10turbo(A) ≈ sum(A)
  end
end
