
function awmean_lv(x::AbstractArray{T1}, σ::AbstractArray{T2}) where {T1<:Number,T2<:Number}
  n = length(x)
  T3 = promote_type(T1,T2)
  T = sizeof(T3) ≤ 4 ? Float32 : Float64
  sum_of_values = sum_of_weights = χ2 = zero(T)
  @turbo for i=1:n
    sum_of_values += x[i] / (σ[i]*σ[i])
    sum_of_weights += 1 / (σ[i]*σ[i])
  end
  wx = sum_of_values / sum_of_weights
  
  @turbo for i=1:n
    χ2 += (x[i] - wx) * (x[i] - wx) / (σ[i] * σ[i])
  end
  mswd = χ2 / (n-1)
  wσ = sqrt(one(T) / sum_of_weights)
  return wx, wσ, mswd
end
function awmean_simd(x::AbstractArray{T1}, σ::AbstractArray{T2}) where {T1<:Number,T2<:Number}
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
        x = view(rand(T(-50):T(100), n + 32), 17:n+16)
        σ = view(rand(T(1):T(10), n + 32), 17:n+16)
      else
        x = view((randn(T, n + 32) .+= T(2)), 17:n+16)
        σ = view(rand(T, n + 32), 17:n+16)
      end
      wx, wσ, mswd = awmean_simd(x, σ)
      @test iszero(@allocated((wxlv, wσlv, mswdlv) = awmean_lv(x, σ)))
      wxlv, wσlv, mswdlv = awmean_lv(x, σ)
      isfinite(wx)   && @test wx ≈ wxlv
      isfinite(wσ)   && @test wσ ≈ wσlv
      isfinite(mswd) && @test mswd ≈ mswdlv atol=eps(typeof(mswdlv)) rtol=sqrt(eps(typeof(mswdlv)))
    end
end

function not_an_outer_reduct!(r, N::Int, x = 2.0, y= nothing) # there was a bug where this was classified as one
  @turbo for i ∈ eachindex(r)
    acc = y === nothing ? x : r[i]
    for n ∈ 1:N
      acc += 0
    end
    r[i] = acc
  end
  r
end
@testset "Outer Reductions" begin
  for T ∈ [Float32,Float64,Int32,Int64]
    test_awmean(T)
  end
  @test all(==(7.4), not_an_outer_reduct!(Vector{Float64}(undef, 5), 17, 7.4))
end

