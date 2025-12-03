using LoopVectorization: vpermilps177, vmovshdup, vfmsubadd, vfmaddsub, vmovsldup
function dot_simd(a::AbstractVector, b::AbstractVector)
  s = zero(eltype(a))
  @fastmath @inbounds @simd for i ∈ eachindex(a)
    s += a[i]' * b[i]
  end
  s
end
function cdot_mat(ca::AbstractVector{Complex{T}}, cb::AbstractVector{Complex{T}}) where {T}
  a = reinterpret(reshape, T, ca)
  b = reinterpret(reshape, T, cb)
  re = zero(T)
  im = zero(T)
  @turbo for i ∈ axes(a, 2)
    re += a[1, i] * b[1, i] + a[2, i] * b[2, i]
    im += a[1, i] * b[2, i] - a[2, i] * b[1, i]
  end
  Complex(re, im)
end
function cdot_swizzle(
  ca::AbstractVector{Complex{T}},
  cb::AbstractVector{Complex{T}},
) where {T}
  a = reinterpret(T, ca)
  b = reinterpret(T, cb)
  reim = Vec(zero(T), zero(T))
  @turbo for i ∈ eachindex(a)
    reim =
      vfmsubadd(vmovsldup(a[i]), b[i], vfmsubadd(vmovshdup(a[i]), vpermilps177(b[i]), reim))
  end
  Complex(reim(1), reim(2))
end
function cdot_affine(
  ca::AbstractVector{Complex{T}},
  cb::AbstractVector{Complex{T}},
) where {T}
  a = reinterpret(T, ca)
  b = reinterpret(T, cb)
  re = zero(T)
  im = zero(T)
  # with a multiplier, we go from `i = 1 -> 2i = 2` to `i = 0 -> 2i = 0
  # 2(i+1-1) = 2i + 2 - 2, so....
  @turbo for i ∈ 1:length(a)>>>1
    re += a[2i-1] * b[2i-1] + a[2i] * b[2i]
    im += a[2i-1] * b[2i] - a[2i] * b[2i-1]
  end
  Complex(re, im)
end
function cdot_stride(
  ca::AbstractVector{Complex{T}},
  cb::AbstractVector{Complex{T}},
) where {T}
  a = reinterpret(T, ca)
  b = reinterpret(T, cb)
  re = zero(T)
  im = zero(T)
  @turbo for i ∈ 1:2:length(a)
    re += a[i] * b[i] + a[i+1] * b[i+1]
    im += a[i] * b[i+1] - a[i+1] * b[i]
  end
  Complex(re, im)
end
function qdot_simd(x::AbstractVector{NTuple{4,T}}, y::AbstractVector{NTuple{4,T}}) where {T}
  a = zero(T)
  b = zero(T)
  c = zero(T)
  d = zero(T)
  @fastmath @inbounds @simd for i ∈ eachindex(x)
    a₁, b₁, c₁, d₁ = x[i]
    a₂, b₂, c₂, d₂ = y[i]
    a += a₁ * a₂ + b₁ * b₂ + c₁ * c₂ + d₁ * d₂
    b += a₁ * b₂ - b₁ * a₂ - c₁ * d₂ + d₁ * c₂
    c += a₁ * c₂ + b₁ * d₂ - c₁ * a₂ - d₁ * b₂
    d += a₁ * d₂ - b₁ * c₂ + c₁ * b₂ - d₁ * a₂
  end
  (a, b, c, d)
end
function qdot_mat(x::AbstractMatrix, y::AbstractMatrix)
  a = zero(eltype(x))
  b = zero(eltype(x))
  c = zero(eltype(x))
  d = zero(eltype(x))
  @turbo for i ∈ axes(x, 2)
    a₁ = x[1, i]
    b₁ = x[2, i]
    c₁ = x[3, i]
    d₁ = x[4, i]
    a₂ = y[1, i]
    b₂ = y[2, i]
    c₂ = y[3, i]
    d₂ = y[4, i]
    a += a₁ * a₂ + b₁ * b₂ + c₁ * c₂ + d₁ * d₂
    b += a₁ * b₂ - b₁ * a₂ - c₁ * d₂ + d₁ * c₂
    c += a₁ * c₂ + b₁ * d₂ - c₁ * a₂ - d₁ * b₂
    d += a₁ * d₂ - b₁ * c₂ + c₁ * b₂ - d₁ * a₂
  end
  (a, b, c, d)
end
function qdot_affine(x::AbstractVector, y::AbstractVector)
  a = zero(eltype(x))
  b = zero(eltype(x))
  c = zero(eltype(x))
  d = zero(eltype(x))
  @turbo for i ∈ 1:length(x)>>2
    a₁ = x[4i-3]
    b₁ = x[4i-2]
    c₁ = x[4i-1]
    d₁ = x[4i]
    a₂ = y[4i-3]
    b₂ = y[4i-2]
    c₂ = y[4i-1]
    d₂ = y[4i]
    a += a₁ * a₂ + b₁ * b₂ + c₁ * c₂ + d₁ * d₂
    b += a₁ * b₂ - b₁ * a₂ - c₁ * d₂ + d₁ * c₂
    c += a₁ * c₂ + b₁ * d₂ - c₁ * a₂ - d₁ * b₂
    d += a₁ * d₂ - b₁ * c₂ + c₁ * b₂ - d₁ * a₂
  end
  (a, b, c, d)
end
function qdot_stride(x::AbstractVector, y::AbstractVector)
  a = zero(eltype(x))
  b = zero(eltype(x))
  c = zero(eltype(x))
  d = zero(eltype(x))
  @turbo for i ∈ 1:4:length(x)
    a₁ = x[i]
    b₁ = x[i+1]
    c₁ = x[i+2]
    d₁ = x[i+3]
    a₂ = y[i]
    b₂ = y[i+1]
    c₂ = y[i+2]
    d₂ = y[i+3]
    a += a₁ * a₂ + b₁ * b₂ + c₁ * c₂ + d₁ * d₂
    b += a₁ * b₂ - b₁ * a₂ - c₁ * d₂ + d₁ * c₂
    c += a₁ * c₂ + b₁ * d₂ - c₁ * a₂ - d₁ * b₂
    d += a₁ * d₂ - b₁ * c₂ + c₁ * b₂ - d₁ * a₂
  end
  (a, b, c, d)
end
function cmatmul_array!(
  Cc::AbstractMatrix{Complex{T}},
  Ac::AbstractMatrix{Complex{T}},
  Bc::AbstractMatrix{Complex{T}},
) where {T}
  C = reinterpret(reshape, Float64, Cc)
  A = reinterpret(reshape, Float64, Ac)
  B = reinterpret(reshape, Float64, Bc)
  @turbo for n ∈ indices((C, B), 3), m ∈ indices((C, A), 2)
    Cre = zero(T)
    Cim = zero(T)
    for k ∈ indices((A, B), (3, 2))
      Cre += A[1, m, k] * B[1, k, n] - A[2, m, k] * B[2, k, n]
      Cim += A[1, m, k] * B[2, k, n] + A[2, m, k] * B[1, k, n]
    end
    C[1, m, n] = Cre
    C[2, m, n] = Cim
  end
  return Cc
end
function cmatmul_array_v2!(
  Cc::AbstractMatrix{Complex{T}},
  Ac::AbstractMatrix{Complex{T}},
  Bc::AbstractMatrix{Complex{T}},
) where {T}
  C = reinterpret(Float64, Cc)
  A = reinterpret(Float64, Ac)
  B = reinterpret(reshape, Float64, Bc)
  @turbo vectorize = 2 for n ∈ indices((C, B), (2, 3)), m ∈ indices((C, A), 1)
    Cmn = zero(T)
    for k ∈ indices((A, B), (2, 2))
      Amk = A[m, k]
      Aperm = vpermilps177(Amk)
      Cmn = vfmaddsub(Amk, B[1, k, n], vfmaddsub(Aperm, B[2, k, n], Cmn))
    end
    C[m, n] = Cmn
  end
  return Cc
end

function issue209(M, G, J, H, B, ϕ)
  # tmp = similar(ϕ, G-1, (2*J+1)*(H + 1));
  tmp = view(
    fill(eltype(ϕ)(123456789), G + 15, (2 * J + 1) * (H + 1) + 16),
    9:G+7,
    9:(2*J+1)*(H+1)+8,
  )
  Bf = reinterpret(reshape, Float64, B)
  ϕf = reinterpret(reshape, Float64, ϕ)
  tmpf = reinterpret(reshape, Float64, tmp)
  jmax = 2 * J + 1
  # B is being indexed via ptr offsetting
  # thus B's initial `gesp`ing must set it up for this
  # currently, it isn't because  `jj` and `hh` are loop induct vars
  for mm = 1:M
    m_idx = M + 2 - mm
    @turbo for hh = 1:H+1
      h_idx = (hh - 1) * jmax
      for jj = 1:jmax, gg = 1:G-1
        tmpf[1, gg, jj+h_idx] = ϕf[1, jj, gg+1, hh, m_idx] + Bf[1, jj, gg, hh, m_idx]
        tmpf[2, gg, jj+h_idx] = ϕf[2, jj, gg+1, hh, m_idx] + Bf[2, jj, gg, hh, m_idx]
      end
    end
  end
  parent(tmp)
end
function issue209_noavx(M, G, J, H, B, ϕ)
  tmp = view(
    fill(eltype(ϕ)(123456789), G + 15, (2 * J + 1) * (H + 1) + 16),
    9:G+7,
    9:(2*J+1)*(H+1)+8,
  )
  Bf = reinterpret(reshape, Float64, B)
  ϕf = reinterpret(reshape, Float64, ϕ)
  tmpf = reinterpret(reshape, Float64, tmp)
  jmax = 2 * J + 1
  for mm = 1:M
    m_idx = M + 2 - mm
    for hh = 1:H+1
      h_idx = (hh - 1) * jmax
      for jj = 1:jmax, gg = 1:G-1
        tmpf[1, gg, jj+h_idx] = ϕf[1, jj, gg+1, hh, m_idx] + Bf[1, jj, gg, hh, m_idx]
        tmpf[2, gg, jj+h_idx] = ϕf[2, jj, gg+1, hh, m_idx] + Bf[2, jj, gg, hh, m_idx]
      end
    end
  end
  parent(tmp)
end
using LoopVectorization

function sumdim2_turbo!(r1, r2)
  @turbo thread = true for j in indices((r1, r2), (3, 4)), i ∈ indices((r1, r2), (2, 3))
    r1[1, i, j] = r2[1, 1, i, j] + r2[1, 2, i, j]
    r1[2, i, j] = r2[2, 1, i, j] - r2[2, 2, i, j]
    r1[3, i, j] = r2[3, 1, i, j] * r2[3, 2, i, j]
    r1[4, i, j] = r2[4, 1, i, j] / r2[4, 2, i, j]
  end
  r1
end
function sumdim2!(r1, r2)
  @inbounds @fastmath for j in indices((r1, r2), (3, 4)), i ∈ indices((r1, r2), (2, 3))
    r1[1, i, j] = r2[1, 1, i, j] + r2[1, 2, i, j]
    r1[2, i, j] = r2[2, 1, i, j] - r2[2, 2, i, j]
    r1[3, i, j] = r2[3, 1, i, j] * r2[3, 2, i, j]
    r1[4, i, j] = r2[4, 1, i, j] / r2[4, 2, i, j]
  end
  r1
end

# Issue 287
function my_gemm_noturbo!(out, s::Matrix{UInt8}, V)
  Vcols = size(V, 2)
  srows = size(s, 1)
  scols = size(s, 2)
  k = srows >> 2
  rem = srows & 3
  @inbounds @fastmath for c = 1:Vcols
    for j = 1:scols
      for l = 1:k
        block = s[l, j]
        for p = 1:4
          Aij = (block >> (2 * (p - 1))) & 3
          out[4*(l-1)+p, c] += ((Aij >= 2) + (Aij == 3)) * V[j, c]
        end
      end
    end
  end
  # TODO handle rem
end
function my_gemm_unroll(out, s::Matrix{UInt8}, V)
  Vcols = size(V, 2)
  srows = size(s, 1)
  scols = size(s, 2)
  k = srows >> 2
  rem = srows & 3
  @avx for c = 1:Vcols
    for j = 1:scols
      for l = 1:k
        block = s[l, j]
        for p = 1:4
          Aij = (block >> (2 * (p - 1))) & 3
          out[4*(l-1)+p, c] += ((Aij >= 2) + (Aij == 3)) * V[j, c]
        end
      end
    end
  end
  # TODO handle rem
end
function my_gemm_manual_unroll(out, s::Matrix{UInt8}, V)
  Vcols = size(V, 2)
  srows = size(s, 1)
  scols = size(s, 2)
  k = srows >> 2
  rem = srows & 3
  @avx for c = 1:Vcols
    for j = 1:scols
      for l = 1:k
        block = s[l, j]
        # unrolled loop
        thisiszero = 0
        p = 1
        Aij = (block >> (2 * (p - 1))) & 3
        out[4*(l-1)+p, c+thisiszero] += ((Aij >= 2) + (Aij == 3)) * V[j, c]
        p = 2
        Aij = (block >> (2 * (p - 1))) & 3
        out[4*(l-1)+p, c] += ((Aij >= 2) + (Aij == 3)) * V[j, c]
        p = 3
        Aij = (block >> (2 * (p - 1))) & 3
        out[4*(l-1)+p+thisiszero, c] += ((Aij >= 2) + (Aij == 3)) * V[j, c]
        p = 4
        Aij = (block >> (2 * (p - 1))) & 3
        out[4*(l-1)+p, c] += ((Aij >= 2) + (Aij == 3)) * V[j, c]
      end
    end
  end
  # TODO handle rem
end
function my_gemm_nexpr_unroll(out, s::Matrix{UInt8}, V)
  Vcols = size(V, 2)
  srows = size(s, 1)
  scols = size(s, 2)
  k = srows >> 2
  rem = srows & 3
  @turbo for c = 1:Vcols
    for j = 1:scols
      for l = 1:k
        block = s[l, j]
        # unrolled loop
        Base.Cartesian.@nexprs 4 p -> begin
          Aij = (block >> (2 * (p - 1))) & 3
          out[4*(l-1)+p, c] += ((Aij >= 2) + (Aij == 3)) * V[j, c]
        end
      end
    end
  end
  # TODO handle rem
end

function readraw_turbo!(img, raw)
  npack = length(raw) ÷ 3
  @turbo for i = 0:npack-1
    img[1+4i] = raw[2+3i] << 4
    img[2+4i] = raw[1+3i]
    img[3+4i] = raw[2+3i]
    img[4+4i] = raw[3+3i]
  end
  img
end
function readraw!(img, raw)
  npack = length(raw) ÷ 3
  @inbounds @simd for i = 0:npack-1
    img[1+4i] = raw[2+3i] << 4
    img[2+4i] = raw[1+3i]
    img[3+4i] = raw[2+3i]
    img[4+4i] = raw[3+3i]
  end
  img
end

function issue348_ref!(hi, lo)
  @inbounds @fastmath for j = 0:(size(hi, 2)-3)÷3 # This tturbo
    for i = 0:(size(hi, 1)-3)÷3
      hi[3i+2, 3j+2] = lo[i+2, j+2]
      hi[3i+3, 3j+2] = lo[i+2, j+2]
      hi[3i+4, 3j+2] = lo[i+2, j+2]
      hi[3i+2, 3j+3] = lo[i+2, j+2]
      hi[3i+3, 3j+3] = lo[i+2, j+2]
      hi[3i+4, 3j+3] = lo[i+2, j+2]
      hi[3i+2, 3j+4] = lo[i+2, j+2]
      hi[3i+3, 3j+4] = lo[i+2, j+2]
      hi[3i+4, 3j+4] = lo[i+2, j+2]
    end
  end
end
function issue348_v0!(hi, lo)
  @turbo for j = 0:(size(hi, 2)-3)÷3 # This tturbo
    for i = 0:(size(hi, 1)-3)÷3
      hi[3i+2, 3j+2] = lo[i+2, j+2]
      hi[3i+3, 3j+2] = lo[i+2, j+2]
      hi[3i+4, 3j+2] = lo[i+2, j+2]
      hi[3i+2, 3j+3] = lo[i+2, j+2]
      hi[3i+3, 3j+3] = lo[i+2, j+2]
      hi[3i+4, 3j+3] = lo[i+2, j+2]
      hi[3i+2, 3j+4] = lo[i+2, j+2]
      hi[3i+3, 3j+4] = lo[i+2, j+2]
      hi[3i+4, 3j+4] = lo[i+2, j+2]
    end
  end
end
function issue348_v1!(hi, lo)
  @turbo for j = 0:3:size(hi, 2)-3 # This tturbo
    for i = 0:3:size(hi, 1)-3
      i_lo = i ÷ 3 + 2
      j_lo = j ÷ 3 + 2
      i_hi = i + 2
      j_hi = j + 2
      hi[i_hi, j_hi] = lo[i_lo, j_lo]
      hi[i_hi+1, j_hi] = lo[i_lo, j_lo]
      hi[i_hi+2, j_hi] = lo[i_lo, j_lo]
      hi[i_hi, j_hi+1] = lo[i_lo, j_lo]
      hi[i_hi+1, j_hi+1] = lo[i_lo, j_lo]
      hi[i_hi+2, j_hi+1] = lo[i_lo, j_lo]
      hi[i_hi, j_hi+2] = lo[i_lo, j_lo]
      hi[i_hi+1, j_hi+2] = lo[i_lo, j_lo]
      hi[i_hi+2, j_hi+2] = lo[i_lo, j_lo]
    end
  end
end
function reverse_part(n1, n2)
  A = zeros(n1, n2)
  @turbo for i = 1:n1÷2, j = 1:n2
    c = 1.0
    A[i, j] = c
    r = n1 + 1 - i
    A[r, j] = c
  end
  return A
end
function reverse_part_ref(n1, n2)
  A = zeros(n1, n2)
  @inbounds for i = 1:n1÷2
    @simd for j = 1:n2
      c = 1.0
      A[i, j] = c
      r = n1 + 1 - i
      A[r, j] = c
    end
  end
  return A
end


function tullio_issue_131_ref(arr)
  M, N = size(arr)
  out = zeros(M >>> 1, N >>> 1)
  @inbounds @fastmath for j in axes(out, 2)
    for i in axes(out, 1)
      out[i, j] = arr[2i, 2j] + arr[2i-1, 2j] + arr[2i-1, 2j-1] + arr[2i, 2j-1]
    end
  end
  out
end


function tullio_issue_131(arr)
  M, N = size(arr)
  out = zeros(M >>> 1, N >>> 1)
  @turbo for j in axes(out, 2)
    for i in axes(out, 1)
      out[i, j] = arr[2i, 2j] + arr[2i-1, 2j] + arr[2i-1, 2j-1] + arr[2i, 2j-1]
    end
  end
  out
end


@testset "shuffles load/stores" begin
  @show @__LINE__
  for i ∈ 1:128
    ac = rand(Complex{Float64}, i)
    bc = rand(Complex{Float64}, i)
    dsimd = dot_simd(ac, bc)
    if VERSION ≥ v"1.6.0-rc1"
      @test dsimd ≈ cdot_mat(ac, bc)
    end
    @test dsimd ≈ cdot_affine(ac, bc) ≈ cdot_stride(ac, bc)


    xq = [ntuple(_ -> rand(), Val(4)) for _ ∈ 1:i]
    yq = [ntuple(_ -> rand(), Val(4)) for _ ∈ 1:i]
    xqv = reinterpret(Float64, xq)
    yqv = reinterpret(Float64, yq)
    qsimd = Base.vect(qdot_simd(xq, yq)...)
    if VERSION ≥ v"1.6.0-rc1"
      xqm = reinterpret(reshape, Float64, xq)
      yqm = reinterpret(reshape, Float64, yq)
      @test qsimd ≈ Base.vect(qdot_mat(xqm, yqm)...)
    end
    @test qsimd ≈ Base.vect(qdot_affine(xqv, yqv)...) ≈ Base.vect(qdot_stride(xqv, yqv)...)

    # TODO: This should likely be
    #   for j ∈ max(1, i - 5):(i + 5), k ∈ max(1, i - 5):(i + 5)
    # but this leads to segfaults on some systems (e.g., x64 Linux).
    for j ∈ max(1, i - 5):(i + 5), k ∈ max(1, i - 5, i + 5)
      A = rand(j + 1, k)
      # This is broken on Apple ARM CPUs (Apple M series)
      # for some reason. This is likely related to the register size
      # differences (128 vs 256 bit) and the smaller vector width
      # for Float64 (2 vs 4) compared to many x64 CPUs.
      # TODO: Fix the underlying issue!
      pattern_for_failing_tests = (j + 1 >= 6) &&
        (k >= 2) &&
        (((j + 1) % 4) == 2 || ((j + 1) % 4) == 3)
      if pattern_for_failing_tests && (Sys.ARCH === :aarch64) &&
                                      Sys.isapple()
        @test_broken tullio_issue_131(A) ≈ tullio_issue_131_ref(A)
      else
        @test tullio_issue_131(A) ≈ tullio_issue_131_ref(A)
      end
      if VERSION ≥ v"1.6.0-rc1"
        Ac = rand(Complex{Float64}, j, i)
        Bc = rand(Complex{Float64}, i, k)
        Cc1 = Ac * Bc
        Cc2 = similar(Cc1)
        Cc3 = similar(Cc1)
        @test Cc1 ≈ cmatmul_array!(Cc2, Ac, Bc)
        Cc2 .= NaN
        @test Cc1 ≈ cmatmul_array_v2!(Cc2, Ac, Bc)
      end
    end
  end
  @show @__LINE__
  if VERSION ≥ v"1.6.0-rc1"
    M = 10
    G = 50
    J = 50
    H = 30

    # B = rand(Complex{Float64}, 2*J+1, G-1, H+1, M+1);
    # ϕ = rand(Complex{Float64}, 2*J+1, G+1, H+1, M+1);
    rbc = let rb = 1.0:((2*J+17)*(G+15)*(H+17)*(M+17)), rbr = reverse(rb)
      Complex{Float64}[rb[i] + im * rbr[i] for i ∈ eachindex(rb)]
    end
    B =
      view(
        reshape(rbc, (2 * J + 17, G + 15, H + 17, M + 17)),
        9:2*J+9,
        9:G+9,
        9:H+9,
        9:M+9,
      ) .= rand.() .+ rand.() .* im
    ϕ =
      view(
        fill(1e5 + 1e7im, 2 * J + 17, G + 17, H + 17, M + 17),
        9:2*J+9,
        9:G+9,
        9:H+9,
        9:M+9,
      ) .= rand.() .+ rand.() .* im
    @test issue209(M, G, J, H, B, ϕ) ≈ issue209_noavx(M, G, J, H, B, ϕ)
  end

  s = Array{Float64}(undef, 4, 128, 128)
  s2 = rand(4, 2, 128, 128)
  @test sumdim2_turbo!(s, s2) ≈ sumdim2!(similar(s), s2)

  # issue 287
  out_test = zeros(100, 10)
  out_test1 = zeros(100, 10)
  s = rand(UInt8, 25, 100)
  V = rand(100, 10)
  my_gemm_noturbo!(out_test, s, V)
  my_gemm_unroll(out_test1, s, V)
  @test out_test ≈ out_test1
  my_gemm_manual_unroll(fill!(out_test1, 0), s, V)
  @test out_test ≈ out_test1
  my_gemm_nexpr_unroll(fill!(out_test1, 0), s, V)
  @test out_test ≈ out_test1


  w = 2048
  raw = rand(UInt8, (3w * w) ÷ 4)
  img1 = Matrix{UInt8}(undef, w, w)
  img2 = Matrix{UInt8}(undef, w, w)
  @test readraw!(img1, raw) == readraw_turbo!(img2, raw)

  for n_hi ∈ 9:100
    iszero((n_hi - 1) % 3) && continue
    n_lo = n_hi ÷ 3
    a_lo_gc = rand(n_lo + 2, n_lo + 2)
    a_hi_tmp_ref = zeros(n_hi + 2, n_hi + 2)
    a_hi_tmp0 = zeros(n_hi + 2, n_hi + 2)

    issue348_ref!(a_hi_tmp_ref, a_lo_gc)
    issue348_v0!(a_hi_tmp0, a_lo_gc)
    @test a_hi_tmp_ref == a_hi_tmp0
    a_hi_tmp1 = view(
      zeros(size(a_hi_tmp0) .* 9),
      map((x, y) -> x .+ (4y), axes(a_hi_tmp0), size(a_hi_tmp0))...,
    )
    issue348_v1!(a_hi_tmp1, a_lo_gc)
    @test a_hi_tmp_ref == a_hi_tmp1
    @turbo a_hi_tmp1 .= 0
    @test all(iszero, parent(a_hi_tmp1))

    @test reverse_part(n_hi, 4) == reverse_part_ref(n_hi, 4)
  end
end
