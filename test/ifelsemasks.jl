using LoopVectorization, Random, Test
T = Float32

@testset "ifelse (masks)" begin

  function promote_bool_store!(z, x, y)
    for i ∈ eachindex(x)
      z[i] = (x[i] * x[i] + y[i] * y[i]) < 1
    end
    z
  end
  function promote_bool_storeavx!(z, x, y)
    @turbo for i ∈ eachindex(x)
      z[i] = (x[i] * x[i] + y[i] * y[i]) < 1
    end
    z
  end
  function promote_bool_store_avx!(z, x, y)
    @_avx for i ∈ eachindex(x)
      z[i] = (x[i] * x[i] + y[i] * y[i]) < 1
    end
    z
  end
  # @macroexpand @_avx for i ∈ eachindex(x)
  # z[i] = (x[i]*x[i] + y[i]*y[i]) < 1
  # end
  function promote_bool_storeavx2!(z, x, y)
    @turbo for i ∈ eachindex(x)
      z[i] = (x[i] * x[i] + y[i] * y[i]) < 1 ? 1 : 0
    end
    z
  end
  function promote_bool_store_avx2!(z, x, y)
    @_avx for i ∈ eachindex(x)
      z[i] = (x[i] * x[i] + y[i] * y[i]) < 1 ? 1 : 0
    end
    z
  end

  function Bernoulli_logit(y, α::AbstractVector{T}) where {T}
    t = zero(promote_type(Float64, T))
    @inbounds for i ∈ eachindex(α)
      invOmP = 1 + exp(α[i])
      nlogOmP = log(invOmP)
      nlogP = nlogOmP - α[i]
      t -= y[i] ? nlogP : nlogOmP
    end
    t
  end
  function Bernoulli_logitavx(y, α::AbstractVector{T}) where {T}
    t = zero(T === Int32 ? Float32 : Float64)
    @turbo for i ∈ eachindex(α)
      invOmP = 1 + exp(α[i])
      nlogOmP = log(invOmP)
      t += ifelse(y[i], α[i], zero(T)) - nlogOmP
    end
    t
  end
  function Bernoulli_logit_avx(y, α::AbstractVector{T}) where {T}
    t = zero(T === Int32 ? Float32 : Float64)
    @_avx for i ∈ eachindex(α)
      invOmP = 1 + exp(α[i])
      nlogOmP = log(invOmP)
      t += ifelse(y[i], α[i], zero(T)) - nlogOmP
    end
    t
  end

  function addormul!(c, a, b)
    for i ∈ eachindex(c, a, b)
      c[i] = a[i] > b[i] ? a[i] + b[i] : a[i] * b[i]
    end
  end
  function addormul_avx!(c, a, b)
    @_avx for i ∈ eachindex(c, a, b)
      c[i] = a[i] > b[i] ? a[i] + b[i] : a[i] * b[i]
    end
  end
  function addormulavx!(c, a, b)
    @turbo for i ∈ eachindex(c, a, b)
      c[i] = if a[i] > b[i]
        a[i] + b[i]
      else
        a[i] * b[i]
      end
    end
  end
  function addormulp1!(c, a, b)
    for i ∈ eachindex(c, a, b)
      c[i] = 1 + (a[i] > b[i] ? a[i] + b[i] : a[i] * b[i])
    end
  end
  function addormulp1_avx!(c, a, b)
    @_avx for i ∈ eachindex(c, a, b)
      c[i] = 1 + (a[i] > b[i] ? a[i] + b[i] : a[i] * b[i])
    end
  end
  function addormulp1avx!(c, a, b)
    @turbo for i ∈ eachindex(c, a, b)
      a_greater_than_b = a[i] > b[i]
      c[i] = 1 + (a_greater_than_b ? a[i] + b[i] : a[i] * b[i])
    end
  end
  function addifelsemul_avx!(c, a, b)
    @_avx for i ∈ eachindex(c, a, b)
      c[i] = ifelse(a[i] > b[i], a[i] + b[i], a[i] * b[i])
    end
  end
  function addifelsemulavx!(c, a, b)
    @turbo for i ∈ eachindex(c, a, b)
      c[i] = ifelse(a[i] > b[i], a[i] + b[i], a[i] * b[i])
    end
  end
  function addifelsemulp1_avx!(c, a, b)
    @_avx for i ∈ eachindex(c, a, b)
      c[i] = 1 + ifelse(a[i] > b[i], a[i] + b[i], a[i] * b[i])
    end
  end
  function addifelsemulp1avx!(c, a, b)
    @turbo for i ∈ eachindex(c, a, b)
      c[i] = 1 + ifelse(a[i] > b[i], a[i] + b[i], a[i] * b[i])
    end
  end
  function ifelseoverwrite!(p)
    for i ∈ eachindex(p)
      p[i] = p[i] < 0.5 ? p[i]^2 : p[i]^3
    end
  end
  function ifelseoverwriteavx!(p)
    @turbo for i ∈ eachindex(p)
      p[i] = p[i] < 0.5 ? p[i]^2 : p[i]^3
    end
  end



  function maybewriteand!(c, a, b)
    @inbounds for i ∈ eachindex(c, a, b)
      a[i] > b[i] && (c[i] = a[i] + b[i])
    end
  end
  function maybewriteand_avx!(c, a, b)
    @_avx for i ∈ eachindex(c, a, b)
      a[i] > b[i] && (c[i] = a[i] + b[i])
    end
  end
  function maybewriteandavx!(c, a, b)
    @turbo for i ∈ eachindex(c, a, b)
      a[i] > b[i] && (c[i] = a[i] + b[i])
    end
  end
  function maybewriteor!(c, a, b)
    @inbounds for i ∈ eachindex(c, a, b)
      a[i] > b[i] || (c[i] = a[i]^b[i])
    end
  end
  function maybewriteor_avx!(c, a, b)
    @_avx for i ∈ eachindex(c, a, b)
      a[i] > b[i] || (c[i] = a[i]^b[i])
    end
  end
  function maybewriteoravx!(c, a, b)
    @turbo for i ∈ eachindex(c, a, b)
      a[i] > b[i] || (c[i] = a[i]^b[i])
    end
  end
  function maybewriteor!(c::AbstractVector{<:Integer}, a, b)
    @inbounds for i ∈ eachindex(c, a, b)
      a[i] > b[i] || (c[i] = a[i] & b[i])
    end
  end
  function maybewriteor_avx!(c::AbstractVector{<:Integer}, a, b)
    @_avx for i ∈ eachindex(c, a, b)
      a[i] > b[i] || (c[i] = a[i] & b[i])
    end
  end
  function maybewriteoravx!(c::AbstractVector{<:Integer}, a, b)
    @turbo for i ∈ eachindex(c, a, b)
      a[i] > b[i] || (c[i] = a[i] & b[i])
    end
  end

  function notacondload!(C, A, b)
    @inbounds for n ∈ 1:size(C, 2), m ∈ 1:size(C, 1)
      C[m, n] = A[m, n] * (b[n] > 0 ? b[n] : -b[n])
    end
  end
  function notacondloadavx!(C, A, b)
    @turbo for n ∈ 1:size(C, 2), m ∈ 1:size(C, 1)
      C[m, n] = A[m, n] * (b[n] > 0 ? b[n] : -b[n])
    end
  end
  function condloadscalar!(C, A, c, b)
    @inbounds for n ∈ 1:size(C, 2), m ∈ 1:size(C, 1)
      C[m, n] = A[m, n] * (c[n] > 0 ? b[n] : 1) + c[n]
    end
  end
  function condloadscalaravx!(C, A, c, b)
    @turbo for n ∈ 1:size(C, 2), m ∈ 1:size(C, 1)
      C[m, n] = A[m, n] * (c[n] > 0 ? b[n] : 1) + c[n]
    end
  end
  function maskedloadscalar!(C, A, b)
    @inbounds for n ∈ 1:size(C, 2), m ∈ 1:size(C, 1)
      C[m, n] = A[m, n] * (A[m, n] > 0 ? b[n] : 1)
    end
  end
  function maskedloadscalaravx!(C, A, b)
    @turbo for n ∈ 1:size(C, 2), m ∈ 1:size(C, 1)
      C[m, n] = A[m, n] * (A[m, n] > 0 ? b[n] : 1)
    end
  end
  function AtmulBpos!(C, A, B)
    @inbounds for n ∈ 1:size(C, 2), m ∈ 1:size(C, 1)
      Cₘₙ = zero(eltype(C))
      @simd ivdep for k ∈ 1:size(A, 1)
        Cₘₙ += A[k, m] * B[k, n]
      end
      C[m, n] > 0 && (C[m, n] = Cₘₙ)
    end
  end
  function AtmulBposavx!(C, A, B)
    @turbo for n ∈ 1:size(C, 2), m ∈ 1:size(C, 1)
      Cₘₙ = zero(eltype(C))
      for k ∈ 1:size(A, 1)
        Cₘₙ += A[k, m] * B[k, n]
      end
      C[m, n] > 0 && (C[m, n] = Cₘₙ)
    end
  end
  function AtmulBpos_avx!(C, A, B)
    @_avx for n ∈ 1:size(C, 2), m ∈ 1:size(C, 1)
      Cₘₙ = zero(eltype(C))
      for k ∈ 1:size(A, 1)
        Cₘₙ += A[k, m] * B[k, n]
      end
      C[m, n] > 0 && (C[m, n] = Cₘₙ)
    end
  end
  function condstore!(x)
    @inbounds for i ∈ eachindex(x)
      x1 = 2 * x[i] - 100
      x2 = x1 * x1
      x3 = x2 + x1
      x4 = x3
      x[i] = x1
      (x1 < -50) && (x[i] = x2)
      (x1 < 60) || (x[i] = x4)
    end
  end
  function condstore1avx!(x)
    @turbo for i ∈ eachindex(x)
      x1 = 2 * x[i] - 100
      x2 = x1 * x1
      x3 = x2 + x1
      x[i] = x1
      (x1 < -50) && (x[i] = x2)
      (x1 < 60) || (x[i] = x3)
    end
  end
  function condstore1_avx!(x)
    @_avx for i ∈ eachindex(x)
      x1 = 2 * x[i] - 100
      x2 = x1 * x1
      x3 = x2 + x1
      x[i] = x1
      (x1 < -50) && (x[i] = x2)
      (x1 < 60) || (x[i] = x3)
    end
  end
  function condstore2avx!(x)
    @turbo for i ∈ eachindex(x)
      x1 = 2 * getindex(x, i) - 100
      x2 = x1 * x1
      x3 = x2 + x1
      setindex!(x, x1, i)
      (x1 < -50) && setindex!(x, x2, i)
      (x1 < 60) || setindex!(x, x3, i)
    end
  end
  function condstore2_avx!(x)
    @_avx for i ∈ eachindex(x)
      x1 = 2 * getindex(x, i) - 100
      x2 = x1 * x1
      x3 = x2 + x1
      setindex!(x, x1, i)
      (x1 < -50) && setindex!(x, x2, i)
      (x1 < 60) || setindex!(x, x3, i)
    end
  end
  function andorassignment!(x, y, z)
    @inbounds for i ∈ eachindex(x, y, z)
      yᵢ = y[i]
      zᵢ = z[i]
      (yᵢ > 0.5) || (yᵢ *= 2)
      (zᵢ < 0.5) && (zᵢ *= 2)
      x[i] = yᵢ * zᵢ
    end
  end
  function andorassignmentavx!(x, y, z)
    @turbo for i ∈ eachindex(x, y, z)
      yᵢ = y[i]
      zᵢ = z[i]
      (yᵢ > 0.5) || (yᵢ *= 2)
      (zᵢ < 0.5) && (zᵢ *= 2)
      x[i] = yᵢ * zᵢ
    end
  end
  function andorassignment_avx!(x, y, z)
    @turbo for i ∈ eachindex(x, y, z)
      yᵢ = y[i]
      zᵢ = z[i]
      (yᵢ > 0.5) || (yᵢ *= 2)
      (zᵢ < 0.5) && (zᵢ *= 2)
      x[i] = yᵢ * zᵢ
    end
  end

  function twoifelses!(res, half, m, keep = nothing, final = true)
    local 𝒶𝓍j = axes(half, 1)
    for j in 𝒶𝓍j
      𝓇𝒽𝓈 = if isnothing(keep)
        log(half[j]) + m[j]
      else
        res[j] + (log(half[j]) + m[j])
      end
      res[j] = isnothing(final) ? 𝓇𝒽𝓈 : exp(𝓇𝒽𝓈)
    end
    res
  end
  function twoifelses_avx!(res, half, m, keep = nothing, final = true)
    local 𝒶𝓍j = axes(half, 1)
    @turbo for j in 𝒶𝓍j
      𝓇𝒽𝓈 = if isnothing(keep)
        log(half[j]) + m[j]
      else
        res[j] + (log(half[j]) + m[j])
      end
      res[j] = if isnothing(final)
        𝓇𝒽𝓈
      else
        exp(𝓇𝒽𝓈)
      end
    end
    res
  end
  function testfunction!(f::Matrix{<:AbstractFloat}, v, d, g, s, θ)
    @inbounds @simd for j = 1:size(f, 1)
      x = v[j, s] + v[j, d] - v[j, g] + f[j, g] + θ
      _x = ifelse(isnan(x), typemin(eltype(f)), x)
      f[j, d] = _x
    end
  end
  function testfunctionavx!(f::Matrix{<:AbstractFloat}, v, d, g, s, θ)
    @turbo for j = 1:size(f, 1)
      x = v[j, s] + v[j, d] - v[j, g] + f[j, g] + θ
      _x = ifelse(isnan(x), typemin(eltype(f)), x)
      f[j, d] = _x
    end
  end

  chebpts(m) = (n = m - 1; [cos(k * pi / n) for k = 0:n])
  function barycentric_weight0(X)
    local T = eltype(X)
    local n = length(X) - 1
    local w = zero(X)
    @inbounds @fastmath for j = 0:n
      tmp = one(T)
      for k = 0:n
        tmp = k == j ? tmp : tmp * (X[j+1] - X[k+1])
      end
      w[j+1] = inv(tmp)
    end
    return w
  end
  function barycentric_weight1(X)
    local T = eltype(X)
    local n = length(X) - 1
    local w = zero(X)
    @turbo for j = 0:n
      tmp = one(T)
      for k = 0:n
        tmp = k != j ? tmp * (X[j+1] - X[k+1]) : tmp
      end
      w[j+1] = inv(tmp)
    end
    return w
  end
  function barycentric_weight2(X)
    local T = eltype(X)
    local n = length(X) - 1
    local w = zero(X)
    @turbo inline = true for j = 0:n
      tmp = one(T)
      for k = 0:n
        tmp = k == j ? tmp : tmp * (X[j+1] - X[k+1])
      end
      w[j+1] = inv(tmp)
    end
    return w
  end
  function barycentric_weight3(X)
    local T = eltype(X)
    local n = length(X) - 1
    local w = zero(X)
    @turbo inline = true for j = 0:n
      tmp = one(T)
      for k = 0:n
        tmp = ifelse(k != j, tmp * (X[j+1] - X[k+1]), tmp)
      end
      w[j+1] = inv(tmp)
    end
    return w
  end
  function barycentric_weight4(X)
    local T = eltype(X)
    local n = length(X) - 1
    local w = zero(X)
    @turbo for j = 0:n
      tmp = one(T)
      for k = 0:n
        tmp = ifelse(k == j, tmp, tmp * (X[j+1] - X[k+1]))
      end
      w[j+1] = inv(tmp)
    end
    return w
  end


  function mwe_v(dest, src, lut)
    # for i in eachindex(src)
    LoopVectorization.@turbo for i in eachindex(src)
      v = src[i]
      s = 128
      s2 = s | 64
      s = ifelse(lut[s2] <= v, s2, s)
      s2 = s | 32
      s = ifelse(lut[s2] <= v, s2, s)
      dest[i] = s
    end
    return dest
  end

  function mwe_s(dest, src, lut)
    for i in eachindex(src)
      # LoopVectorization.@turbo for i in eachindex(src)
      v = src[i]
      s = 128
      s2 = s | 64
      s = ifelse(lut[s2] <= v, s2, s)
      s2 = s | 32
      s = ifelse(lut[s2] <= v, s2, s)
      dest[i] = s
    end
    return dest
  end

  function turbocomparison!(m)
    @turbo for i in eachindex(m)
      m[i] = ifelse(0 < m[i] < 0.5, 0.0, m[i])
    end
  end
  function turbocomparison!(m, y)
    @turbo for i in eachindex(m)
      m[i] = ifelse(0 < m[i] < y[i] < 0.5, 0.0, m[i])
    end
  end

  function absmax_tturbo(a) # LV threaded
    local result = zero(eltype(a))
    @tturbo for i = 1:length(a)
      abs(a[i]) > result && (result = abs(a[i]))
    end
    result
  end

  function findminturbo(x)
    local indmin = 0
    local minval = typemax(eltype(x))
    @turbo for i ∈ eachindex(x)
      newmin = x[i] < minval
      minval = newmin ? x[i] : minval
      indmin = newmin ? i : indmin
    end
    minval, indmin
  end
  function findmintturbo(x)
    local indmin = 0
    local minval = typemax(eltype(x))
    @tturbo for i ∈ eachindex(x)
      newmin = x[i] < minval
      minval = newmin ? x[i] : minval
      indmin = newmin ? i : indmin
    end
    minval, indmin
  end
  function findminturbo_u4(x)
    local indmin = 0
    local minval = typemax(eltype(x))
    @turbo unroll = 4 for i ∈ eachindex(x)
      newmin = x[i] < minval
      minval = newmin ? x[i] : minval
      indmin = newmin ? i : indmin
    end
    minval, indmin
  end
  function extrema_turbo(x)
    local a = first(x)
    local b = a
    @turbo for i in eachindex(x)
      local e = x[i]
      b = max(b, e)
      a = min(a, e)
    end
    a, b
  end

  N = 117
  for T ∈ (Float32, Float64, Int32, Int64)
    @show T, @__LINE__
    if T <: Integer
      a = rand(-T(100):T(100), N)
      b = rand(-T(100):T(100), N)
      mv, mi = findminturbo(a)
      mv2, mi2 = findminturbo_u4(a)
      @test mv == a[mi] == minimum(a) == mv2 == a[mi2]
      for n = 1000:1000:10_000
        x = rand(-T(100):T(100), n)
        @test absmax_tturbo(x) == mapreduce(abs, max, x)
        mv, mi = findmintturbo(x)
        @test mv == x[mi] == minimum(x)
      end
    else
      a = rand(T, N)
      b = rand(T, N)
      @test findmin(a) == findminturbo(a) == findminturbo_u4(a)
      for n = 1000:1000:10_000
        x = randn(T, n)
        @test absmax_tturbo(x) == mapreduce(abs, max, x)
        mv, mi = findmintturbo(x)
        @test mv == x[mi] == minimum(x)
      end
    end
    c1 = similar(a)
    c2 = similar(a)

    @test extrema(a) == extrema_turbo(a)
    promote_bool_store!(c1, a, b)
    promote_bool_storeavx!(c2, a, b)
    @test c1 == c2
    fill!(c2, -999999999)
    promote_bool_store_avx!(c2, a, b)
    @test c1 == c2
    fill!(c2, -999999999)
    promote_bool_storeavx2!(c2, a, b)
    @test c1 == c2
    fill!(c2, -999999999)
    promote_bool_store_avx2!(c2, a, b)
    @test c1 == c2

    fill!(c1, 999999999)
    addormul!(c1, a, b)
    fill!(c2, -999999999)
    addormul_avx!(c2, a, b)
    @test c1 ≈ c2
    fill!(c2, -999999999)
    addormulavx!(c2, a, b)
    @test c1 ≈ c2
    fill!(c2, -999999999)
    addifelsemul_avx!(c2, a, b)
    @test c1 ≈ c2
    fill!(c2, -999999999)
    addifelsemulavx!(c2, a, b)
    @test c1 ≈ c2
    addormulp1!(c1, a, b)
    addormulp1_avx!(c2, a, b)
    @test c1 ≈ c2
    fill!(c2, -999999999)
    addormulp1avx!(c2, a, b)
    @test c1 ≈ c2
    fill!(c2, -999999999)
    addifelsemulp1_avx!(c2, a, b)
    @test c1 ≈ c2
    fill!(c2, -999999999)
    addifelsemulp1avx!(c2, a, b)
    @test c1 ≈ c2

    fill!(c1, -999999999)
    maybewriteand!(c1, a, b)
    fill!(c2, -999999999)
    maybewriteand_avx!(c2, a, b)
    @test c1 ≈ c2
    fill!(c2, -999999999)
    maybewriteandavx!(c2, a, b)
    @test c1 ≈ c2

    fill!(c1, -999999999)
    maybewriteor!(c1, a, b)
    fill!(c2, -999999999)
    maybewriteor_avx!(c2, a, b)
    @test c1 ≈ c2
    fill!(c2, -999999999)
    maybewriteoravx!(c2, a, b)
    @test c1 ≈ c2

    andorassignment!(c1, a, b)
    andorassignmentavx!(c2, a, b)
    @test c1 ≈ c2
    fill!(c2, -999999999)
    andorassignment_avx!(c2, a, b)
    @test c1 ≈ c2

    a1 = copy(a)
    a2 = copy(a)
    ifelseoverwrite!(a1)
    ifelseoverwriteavx!(a2)
    @test a1 ≈ a2

    if T <: Union{Float32,Float64}
      a .*= 100
    end
    b1 = copy(a)
    b2 = copy(a)
    # This is broken on Apple ARM CPUs (Apple M series)
    # for some reason.
    # TODO: Fix the underlying issue!
    if (Sys.ARCH === :aarch64) && Sys.isapple() && T <: AbstractFloat
      condstore!(b1)
      condstore1avx!(b2)
      @test b1 ≈ b2
      copyto!(b2, a)
      condstore1_avx!(b2)
      @test b1 ≈ b2
      copyto!(b2, a)
      condstore2avx!(b2)
      @test b1 ≈ b2
      copyto!(b2, a)
      condstore2_avx!(b2)
      @test b1 ≈ b2
    else
      condstore!(b1)
      condstore1avx!(b2)
      @test b1 == b2
      copyto!(b2, a)
      condstore1_avx!(b2)
      @test b1 == b2
      copyto!(b2, a)
      condstore2avx!(b2)
      @test b1 == b2
      copyto!(b2, a)
      condstore2_avx!(b2)
      @test b1 == b2
    end

    M, K, N = 83, 85, 79
    if T <: Integer
      A = rand(T(-100):T(100), K, M)
      B = rand(T(-100):T(100), K, N)
      C1 = rand(T(-100):T(100), M, N)
      b = rand(T(-100):T(100), N)
      d = rand(T(-100):T(100), N)
    else
      A = randn(T, K, M)
      B = randn(T, K, N)
      C1 = randn(T, M, N)
      b = randn(T, N)
      d = randn(T, N)
    end
    C2 = copy(C1)
    C3 = copy(C1)
    AtmulBpos!(C1, A, B)
    AtmulBposavx!(C2, A, B)
    AtmulBpos_avx!(C3, A, B)
    @test C1 ≈ C2
    @test C1 ≈ C3
    C1 = similar(B)
    C2 = similar(B)
    notacondload!(C1, B, b)
    notacondloadavx!(C2, B, b)
    @test C1 ≈ C2
    maskedloadscalar!(C1, B, b)
    maskedloadscalaravx!(C2, B, b)
    @test C1 ≈ C2
    condloadscalar!(C1, B, b, d)
    condloadscalaravx!(C2, B, b, d)
    @test C1 ≈ C2

    if T <: Integer
      half = rand(T(1):T(100), 7)
      m = rand(T(-10):T(10), 7)
    else
      half = rand(T, 7)
      m = rand(T, 7)
    end
    if sizeof(T) == 4
      res1 = Vector{Float32}(undef, 7)
      res2 = Vector{Float32}(undef, 7)
    else
      res1 = Vector{Float64}(undef, 7)
      res2 = Vector{Float64}(undef, 7)
    end

    for keep ∈ (nothing, true), final ∈ (nothing, true)
      @test twoifelses!(res1, half, m) ≈ twoifelses_avx!(res2, half, m)
    end

  end


  a = rand(-10:10, 43)
  bit = a .> 0.5
  bool = copyto!(Vector{Bool}(undef, length(bit)), bit)
  t = Bernoulli_logit(bit, a)
  # This is broken on Apple ARM CPUs (Apple M series)
  # for some reason.
  if (Sys.ARCH === :aarch64) && Sys.isapple()
    # This test fails on some systems but works on other systems (CI)
    @test isapprox(t, Bernoulli_logitavx(bit, a), atol = ifelse(Int === Int32, 0.1, 1e-5))
  else
    @test isapprox(t, Bernoulli_logitavx(bit, a), atol = ifelse(Int === Int32, 0.1, 0.0))
  end
  if LoopVectorization.pick_vector_width(eltype(a)) ≥ 4
    # @_avx isn't really expected to work with bits if you don't have AVX512
    # but it happens to work with AVX2 for this anyway, so may as well keep testing.
    # am ruling out non-avx2 with the `VectorizationBase.pick_vector_width(eltype(a)) ≥ 4` check
    @test isapprox(t, Bernoulli_logit_avx(bit, a), atol = ifelse(Int === Int32, 0.1, 0.0))
  end
  # This is broken on Apple ARM CPUs (Apple M series)
  # for some reason.
  if (Sys.ARCH === :aarch64) && Sys.isapple()
    # This test fails on some systems but works on other systems (CI)
    @test isapprox(t, Bernoulli_logitavx(bool, a), atol = ifelse(Int === Int32, 0.1, 1e-5))
  else
    @test isapprox(t, Bernoulli_logitavx(bool, a), atol = ifelse(Int === Int32, 0.1, 0.0))
  end
  @test isapprox(t, Bernoulli_logit_avx(bool, a), atol = ifelse(Int === Int32, 0.1, 0.0))
  a = rand(43)
  bit = a .> 0.5
  bool = copyto!(Vector{Bool}(undef, length(bit)), bit)
  t = Bernoulli_logit(bit, a)
  # This is broken on Apple ARM CPUs (Apple M series)
  # for some reason.
  # TODO: Fix the underlying issue!
  if (Sys.ARCH === :aarch64) && Sys.isapple()
    @test isapprox(t, Bernoulli_logitavx(bit, a), atol=1e-5)
    @test isapprox(t, Bernoulli_logit_avx(bit, a), atol=1e-5)
  else
    @test t ≈ Bernoulli_logitavx(bit, a)
    @test t ≈ Bernoulli_logit_avx(bit, a)
  end
  @test t ≈ Bernoulli_logitavx(bool, a)
  @test t ≈ Bernoulli_logit_avx(bool, a)

  ai = [rand(Bool) for _ = 1:71]
  bi = [rand(Bool) for _ = 1:71]
  # if LoopVectorization.VectorizationBase.AVX2 || Base.libllvm_version ≥ v"8" #FIXME Why doesn't this work on Travis Ivy Bridge Julia 1.1?
  @test (ai .& bi) == (@turbo ai .& bi)
  @test (ai .| bi) == (@turbo ai .| bi)
  @test (ai .⊻ bi) == (@turbo ai .⊻ bi)
  # else
  #     @test_broken (ai .& bi) == (@turbo ai .& bi)
  #     @test_broken (ai .| bi) == (@turbo ai .| bi)
  #     @test_broken (ai .⊻ bi) == (@turbo ai .⊻ bi)
  # end
  a = bitrand(127)
  b = bitrand(127)
  @test (a .& b) == (@turbo a .& b)
  @test (a .| b) == (@turbo a .| b)
  @test (a .⊻ b) == (@turbo a .⊻ b)

  s, d, g = 3, 1, 2
  f = rand(N, 2)
  v = rand(N, 3)
  θ = 0.78
  v[rand(eachindex(v), length(v) >> 3)] .= NaN
  fc = copy(f)
  testfunction!(fc, v, d, g, s, θ)
  # fc2 = copy(f);
  testfunctionavx!(f, v, d, g, s, θ)
  finite_subset = isfinite.(fc)
  @test finite_subset == isfinite.(f)
  @test f[finite_subset] ≈ fc[finite_subset]

  for i ∈ 1:50
    X = chebpts(i)
    bX = barycentric_weight0(X)
    @test barycentric_weight1(X) ≈ bX
    @test barycentric_weight2(X) ≈ bX
    @test barycentric_weight3(X) ≈ bX
    @test barycentric_weight4(X) ≈ bX
  end

  let
    lut = let x = cumsum(rand(Float32, 256) ./ 128)
      x[end] = Inf
      x
    end
    src = rand(Float32, N)

    @test mwe_v(Vector{Int}(undef, N), src, lut) == mwe_s(Vector{Int}(undef, N), src, lut)
    @test mwe_v(Vector{Int32}(undef, N), src, lut) ==
          mwe_s(Vector{Int32}(undef, N), src, lut)
  end

  let m = rand(25, 25),
    y = rand(25, 25),
    baseline5 = (@. ifelse(0 < m < 0.5, 0.0, m)),
    baseline7 = @. ifelse(0 < y < m < 0.5, 0.0, y)

    turbocomparison!(y, m)
    @test y == baseline7
    turbocomparison!(m)
    @test m == baseline5
  end

end
