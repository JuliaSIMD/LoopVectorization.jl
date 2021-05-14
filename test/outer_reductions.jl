
function awmean_lv(x::Array{T1}, σ::Array{T2}) where {T1<:Number,T2<:Number}
  n = length(x)
  T3 = promote_type(T1,T2)
  T = sizeof(T3) ≤ 4 ? Float32 : Float64
  sum_of_values = sum_of_weights = χ2 = zero(T)
  @vectorize for i=1:n
    sum_of_values += x[i] / (σ[i]*σ[i])
    sum_of_weights += 1 / (σ[i]*σ[i])
  end
  wx = sum_of_values / sum_of_weights
  
  @vectorize for i=1:n
    χ2 += (x[i] - wx) * (x[i] - wx) / (σ[i] * σ[i])
  end
  mswd = χ2 / (n-1)
  wσ = sqrt(one(T) / sum_of_weights)
  return wx, wσ, mswd
end
function awmean_simd(x::Array{T1}, σ::Array{T2}) where {T1<:Number,T2<:Number}
  n = length(x)
  T3 = promote_type(T1,T2)
  T = sizeof(T3) ≤ 4 ? Float32 : Float64
  sum_of_values = sum_of_weights = χ2 = zero(T)
  @fastmath @inbounds @simd for i=1:n
    sum_of_values += x[i] / (σ[i]*σ[i])
    sum_of_weights += 1 / (σ[i]*σ[i])
  end
  wx = sum_of_values / sum_of_weights
  
  @fastmath @inbounds @simd for i=1:n
    χ2 += (x[i] - wx) * (x[i] - wx) / (σ[i] * σ[i])
  end
  mswd = χ2 / (n-1)
  wσ = sqrt(one(T) / sum_of_weights)
  return wx, wσ, mswd
end

function test_awmean(::Type{T}) where {T}
    for n ∈ 2:100
      if T <: Integer
        x = rand(T(-100):T(100), n)
        σ = rand(T(1):T(10), n)
      else
        x = randn(T, n)
        σ = rand(T, n)
      end
      wx, wσ, mswd = awmean_simd(x, σ)
      @test iszero(@allocated((wxlv, wσlv, mswdlv) = awmean_lv(x, σ)))
      wxlv, wσlv, mswdlv = awmean_lv(x, σ)
      @test wx ≈ wxlv
      @test wσ ≈ wσlv
      @test mswd ≈ mswdlv
    end
end

@testset "Outer Reductions" begin
  for T ∈ [Float32,Float64,Int32,Int64]
    test_awmean(T)
  end
end

