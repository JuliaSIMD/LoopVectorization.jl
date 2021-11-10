using OffsetArrays

function reg_term(omega, B = size(omega,2); alpha=0.01)
  reg = 0.0
  for b in 1:B
    t = maximum(@views omega[:, b]) - omega[b,b]
    reg += t
  end
  return alpha*reg
end

function reg_term_turbo(omega, B = size(omega,2); alpha=0.01)
  reg = 0.0
  @turbo for b in 1:B
    m = -Inf
    for i in axes(omega,1)
      m = max(m, omega[i,b])
    end
    t = m - omega[b,b]
    reg += t
  end
  return alpha*reg
end

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

function logℒ_fast(α, β, t, c, x)
    eα = abs(α)
    n, k = size(x)
  
    (n == length(t) == length(c) && length(β) == k + 1) || throw(DimensionMismatch())
    s = zero(typeof(α))
    @inbounds for i in 1:n
      xb = sum(@inbounds(x[i, j] * β[j+1]) for j in 1:k) + β[1]
      ti = t[i]
      s += (1 - (c[i] == ti)) * (log(eα) + (eα - 1) * log(ti) + xb) - ti^eα * exp(xb)
    end
    return s
end
function logℒ_fast_turbo(α, β, t, c, x)
    eα = abs(α)
    n, k = size(x)
  
    (n == length(t) == length(c) && length(β) == k + 1) || throw(DimensionMismatch())
    s = zero(typeof(α))
    @turbo for i in 1:n
      xb0 = 0.0
      for j in 1:k
        xb0 += x[i,j] * β[j+1]
      end
      xb = xb0 + β[1]
      ti = t[i]
      s += (1 - (c[i] == ti)) * (log(eα) + (eα - 1) * log(ti) + xb) - ti^eα * exp(xb)
    end
    return s
end

function test_logℒ(n, k)
  t = rand(n)
  c = copy(t);
  b = rand(n) .> 0.5
  c[b] .= rand.();
  x = rand(n,k)
  α = 2.85
  β = rand(k+1)
  @test logℒ_fast(α, β, t, c, x) ≈ logℒ_fast_turbo(α, β, t, c, x)
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
function tk(v::Vector{T}, c_::Vector{T}, n::Int) where {T}
  l00 = l10 = l01 = l11 = zero(T)
  @turbo for j ∈ 1:n-1
    a00 =  v[3n-3j+1]
    a01 = -v[3n-3j+2]
    a10 =  v[3n-3j+2]
    a11 =  v[3n-3j+3]
    b00 =  c_[4j+1]
    b01 =  c_[4j+2]
    b10 =  c_[4j+3]
    b11 =  c_[4j+4]
    l00 += a00*b00 + a01*b10
    l01 += a00*b01 + a01*b11
    l10 += a10*b00 + a11*b10
    l11 += a10*b01 + a11*b11
  end
  l00 + l01 + l10 + l11  
end
function tk2(v::Vector{T}, c_::Vector{T}, n::Int) where {T}
  l00 = l10 = l01 = l11 = zero(T)
  @turbo for j ∈ 1:n-1
    i = n-j
    a00 =  v[3i+1]
    a01 = -v[3i+2]
    a10 =  v[3i+2]
    a11 =  v[3i+3]
    b00 =  c_[4j+1]
    b01 =  c_[4j+2]
    b10 =  c_[4j+3]
    b11 =  c_[4j+4]
    l00 += a00*b00 + a01*b10
    l01 += a00*b01 + a01*b11
    l10 += a10*b00 + a11*b10
    l11 += a10*b01 + a11*b11
  end
  l00 + l01 + l10 + l11  
end
function tk_base(v::Vector{T}, c_::Vector{T}, n::Int) where {T}
  l00 = l10 = l01 = l11 = zero(T)
  @inbounds @simd for j ∈ 1:n-1
    a00 =  v[3n-3j+1]
    a01 = -v[3n-3j+2]
    a10 =  v[3n-3j+2]
    a11 =  v[3n-3j+3]
    i = n-j
    a00 =  v[3i+1]
    a01 = -v[3i+2]
    a10 =  v[3i+2]
    a11 =  v[3i+3]
    b00 =  c_[4j+1]
    b01 =  c_[4j+2]
    b10 =  c_[4j+3]
    b11 =  c_[4j+4]
    l00 += a00*b00 + a01*b10
    l01 += a00*b01 + a01*b11
    l10 += a10*b00 + a11*b10
    l11 += a10*b01 + a11*b11
  end
  l00 + l01 + l10 + l11  
end


function noturbosum(x, kern)
  ks = zero(eltype(x))
  @inbounds @fastmath for i=2:size(x,1)-1
    for j in axes(kern,1), m in axes(kern,2)
      ks += x[i+j, i+m] * kern[j,m]
    end
  end
  ks
end
function outersum(x, kern)
  ks = zero(eltype(x))
  @turbo for i=2:size(x,1)-1
    for j in axes(kern,1), m in axes(kern,2)
      ks += x[i+j, i+m] * kern[j,m]
    end
  end
  ks
end
function insum(x, kern)
  ks = zero(eltype(x))
  @turbo for i=2:size(x,1)-1
    s1 = zero(eltype(x))
    for j in axes(kern,1), m in axes(kern,2)
      s1 += x[i+j, i+m] * kern[j,m]
    end
    ks += s1
  end
  ks
end
function in2sum(x, kern)
  ks = zero(eltype(x))
  @turbo for i=2:size(x,1)-1
    for j in axes(kern,1)
      s0 = zero(eltype(x))
      for m in axes(kern,2)
        s0 += x[i+j, i+m] * kern[j,m]
      end
      ks += s0
    end
  end
  ks
end


@testset "Outer Reductions" begin
  for T ∈ [Float32,Float64,Int32,Int64]
    test_awmean(T)
    v = rand(T, 4*127); c = rand(T, 4*127); 
    @test tk(v, c, 127) ≈ tk2(v, c, 127) ≈ tk_base(v, c, 127);
  end
  @test all(==(7.4), not_an_outer_reduct!(Vector{Float64}(undef, 5), 17, 7.4))
  for n ∈ 1:20, k ∈ 1:5
    test_logℒ(n,k)
  end
  omega = rand(87,87);
  @test reg_term(omega) ≈ reg_term_turbo(omega)

  x = rand(100,100);
  kern = OffsetArray(rand(3,3),-2,-2);
  s = noturbosum(x, kern)
  @test s ≈ outersum(x, kern)
  @test s ≈ insum(x, kern)
  @test s ≈ in2sum(x, kern)
end

