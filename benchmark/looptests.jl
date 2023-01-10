using LoopVectorization, LinearAlgebra, OffsetArrays, ArrayInterface
BLAS.set_num_threads(1)

using LoopVectorization: Static
# TODO: remove this once this PR merges: https://github.com/JuliaArrays/OffsetArrays.jl/pull/170
@inline Base.unsafe_convert(::Type{Ptr{T}}, A::OffsetArray{T}) where {T} =
  pointer(parent(A))

struct SizedOffsetMatrix{T,LR,UR,LC,UC} <: DenseMatrix{T}
  data::Matrix{T}
end
Base.size(::SizedOffsetMatrix{<:Any,LR,UR,LC,UC}) where {LR,UR,LC,UC} =
  (UR - LR + 1, UC - LC + 1)
Base.axes(::SizedOffsetMatrix{T,LR,UR,LC,UC}) where {T,LR,UR,LC,UC} =
  (StaticInt{LR}():StaticInt{UR}(), StaticInt{LC}():StaticInt{UC}())
Base.parent(A::SizedOffsetMatrix) = A.data
Base.unsafe_convert(::Type{Ptr{T}}, A::SizedOffsetMatrix{T}) where {T} =
  pointer(A.data)
ArrayInterface.contiguous_axis(::Type{<:SizedOffsetMatrix}) = StaticInt(1)
ArrayInterface.contiguous_batch_size(::Type{<:SizedOffsetMatrix}) = StaticInt(0)
ArrayInterface.stride_rank(::Type{<:SizedOffsetMatrix}) =
  (StaticInt(1), StaticInt(2))
function ArrayInterface.strides(
  A::SizedOffsetMatrix{T,LR,UR,LC,UC}
) where {T,LR,UR,LC,UC}
  (StaticInt{1}(), (StaticInt{UR}() - StaticInt{LR}() + StaticInt{1}()))
end
ArrayInterface.offsets(
  A::SizedOffsetMatrix{T,LR,UR,LC,UC}
) where {T,LR,UR,LC,UC} = (StaticInt{LR}(), StaticInt{LC}())
ArrayInterface.parent_type(::Type{<:SizedOffsetMatrix{T}}) where {T} = Matrix{T}
Base.getindex(A::SizedOffsetMatrix, i, j) =
  LoopVectorization.vload(LoopVectorization.stridedpointer(A), (i, j))

function jgemm!(𝐂, 𝐀, 𝐁)
  𝐂 .= 0
  M, N = size(𝐂)
  K = size(𝐁, 1)
  @inbounds for n ∈ 1:N, k ∈ 1:K
    @simd ivdep for m ∈ 1:M
      @fastmath 𝐂[m, n] += 𝐀[m, k] * 𝐁[k, n]
    end
  end
end
function jgemm!(𝐂, 𝐀ᵀ::Adjoint, 𝐁)
  𝐀 = parent(𝐀ᵀ)
  @inbounds for n ∈ 1:size(𝐂, 2), m ∈ 1:size(𝐂, 1)
    𝐂ₘₙ = zero(eltype(𝐂))
    @simd ivdep for k ∈ 1:size(𝐀, 1)
      @fastmath 𝐂ₘₙ += 𝐀[k, m] * 𝐁[k, n]
    end
    𝐂[m, n] = 𝐂ₘₙ
  end
end
function jgemm!(𝐂, 𝐀, 𝐁ᵀ::Adjoint)
  𝐂 .= 0
  𝐁 = parent(𝐁ᵀ)
  M, N = size(𝐂)
  K = size(𝐁ᵀ, 1)
  @inbounds for k ∈ 1:K, n ∈ 1:N
    @simd ivdep for m ∈ 1:M
      @fastmath 𝐂[m, n] += 𝐀[m, k] * 𝐁[n, k]
    end
  end
end
function jgemm!(𝐂, 𝐀ᵀ::Adjoint, 𝐁ᵀ::Adjoint)
  𝐂 .= 0
  𝐀 = parent(𝐀ᵀ)
  𝐁 = parent(𝐁ᵀ)
  M, N = size(𝐂)
  K = size(𝐁ᵀ, 1)
  @inbounds for n ∈ 1:N, k ∈ 1:K
    @simd ivdep for m ∈ 1:M
      @fastmath 𝐂[m, n] += 𝐀[k, m] * 𝐁[n, k]
    end
  end
end
gemmavx!(𝐂, 𝐀, 𝐁) = @turbo for m ∈ indices((𝐀, 𝐂), 1), n ∈ indices((𝐁, 𝐂), 2)
    𝐂ₘₙ = zero(eltype(𝐂))
    for k ∈ indices((𝐀, 𝐁), (2, 1))
      𝐂ₘₙ += 𝐀[m, k] * 𝐁[k, n]
    end
    𝐂[m, n] = 𝐂ₘₙ
  end
function gemmavx!(
  Cc::AbstractMatrix{Complex{T}},
  Ac::AbstractMatrix{Complex{T}},
  Bc::AbstractMatrix{Complex{T}}
) where {T}
  A = reinterpret(reshape, T, Ac)
  B = reinterpret(reshape, T, Bc)
  C = reinterpret(reshape, T, Cc)
  @turbo for m ∈ indices((A, C), 2), n ∈ indices((B, C), 3)
    Cre = zero(T)
    Cim = zero(T)
    for k ∈ indices((A, B), (3, 2))
      Cre += A[1, m, k] * B[1, k, n] - A[2, m, k] * B[2, k, n]
      Cim += A[1, m, k] * B[2, k, n] + A[2, m, k] * B[1, k, n]
    end
    C[1, m, n] = Cre
    C[2, m, n] = Cim
  end
end
gemmavxt!(𝐂, 𝐀, 𝐁) = @tturbo for m ∈ indices((𝐀, 𝐂), 1), n ∈ indices((𝐁, 𝐂), 2)
    𝐂ₘₙ = zero(eltype(𝐂))
    for k ∈ indices((𝐀, 𝐁), (2, 1))
      𝐂ₘₙ += 𝐀[m, k] * 𝐁[k, n]
    end
    𝐂[m, n] = 𝐂ₘₙ
  end
function gemmavxt!(
  Cc::AbstractMatrix{Complex{T}},
  Ac::AbstractMatrix{Complex{T}},
  Bc::AbstractMatrix{Complex{T}}
) where {T}
  A = reinterpret(reshape, T, Ac)
  B = reinterpret(reshape, T, Bc)
  C = reinterpret(reshape, T, Cc)
  @tturbo for m ∈ indices((A, C), 2), n ∈ indices((B, C), 3)
    Cre = zero(T)
    Cim = zero(T)
    for k ∈ indices((A, B), (3, 2))
      Cre += A[1, m, k] * B[1, k, n] - A[2, m, k] * B[2, k, n]
      Cim += A[1, m, k] * B[2, k, n] + A[2, m, k] * B[1, k, n]
    end
    C[1, m, n] = Cre
    C[2, m, n] = Cim
  end
end
function jdot(a, b)
  s = zero(eltype(a))
  # @inbounds @simd ivdep for i ∈ eachindex(a,b)
  @inbounds @simd ivdep for i ∈ eachindex(a)
    s += a[i] * b[i]
  end
  s
end
function jdotavx(a, b)
  s = zero(eltype(a))
  # @turbo for i ∈ eachindex(a,b)
  @turbo for i ∈ eachindex(a)
    s += a[i] * b[i]
  end
  s
end
function jdotavxt(a, b)
  s = zero(eltype(a))
  # @turbo for i ∈ eachindex(a,b)
  @tturbo for i ∈ eachindex(a)
    s += a[i] * b[i]
  end
  s
end
function jselfdot(a)
  s = zero(eltype(a))
  @inbounds @simd ivdep for i ∈ eachindex(a)
    s += a[i] * a[i]
  end
  s
end
function jselfdotavx(a)
  s = zero(eltype(a))
  @turbo for i ∈ eachindex(a)
    s += a[i] * a[i]
  end
  s
end
function jdot3v2(x, A, y)
  M, N = size(A)
  s = zero(promote_type(eltype(x), eltype(A), eltype(y)))
  @inbounds @fastmath for n ∈ 1:N, m ∈ 1:M
    s += x[m] * A[m, n] * y[n]
  end
  s
end
function jdot3v2avx(x, A, y)
  M, N = size(A)
  s = zero(promote_type(eltype(x), eltype(A), eltype(y)))
  @turbo for n ∈ 1:N, m ∈ 1:M
    s += x[m] * A[m, n] * y[n]
  end
  s
end
function jdot3(x, A, y)
  s = zero(promote_type(eltype(x), eltype(A), eltype(y)))
  @inbounds @fastmath for n ∈ axes(A, 2)
    t = zero(s)
    @simd ivdep for m ∈ axes(A, 1)
      t += x[m] * A[m, n]
    end
    s += t * y[n]
  end
  s
end
function jdot3avx(x, A, y)
  s = zero(promote_type(eltype(x), eltype(A), eltype(y)))
  @turbo for n ∈ axes(A, 2)
    t = zero(s)
    for m ∈ axes(A, 1)
      t += x[m] * A[m, n]
    end
    s += t * y[n]
  end
  s
end
jvexp!(b, a) = @inbounds for i ∈ eachindex(a)
    b[i] = exp(a[i])
  end
jvexpavx!(b, a) = @turbo for i ∈ eachindex(a)
    b[i] = exp(a[i])
  end
function jsvexp(a)
  s = zero(eltype(a))
  @inbounds for i ∈ eachindex(a)
    s += exp(a[i])
  end
  s
end
function jsvexpavx(a)
  s = zero(eltype(a))
  @turbo for i ∈ eachindex(a)
    s += exp(a[i])
  end
  s
end
function jgemv!(y, 𝐀, x)
  y .= zero(eltype(y))
  @inbounds for j ∈ eachindex(x)
    @simd ivdep for i ∈ eachindex(y)
      @fastmath y[i] += 𝐀[i, j] * x[j]
    end
  end
end
function jgemv!(𝐲, 𝐀ᵀ::Adjoint, 𝐱)
  𝐀 = parent(𝐀ᵀ)
  @inbounds for i ∈ eachindex(𝐲)
    𝐲ᵢ = zero(eltype(𝐲))
    @simd ivdep for j ∈ eachindex(𝐱)
      @fastmath 𝐲ᵢ += 𝐀[j, i] * 𝐱[j]
    end
    𝐲[i] = 𝐲ᵢ
  end
end
jgemvavx!(𝐲, 𝐀, 𝐱) = @turbo for i ∈ eachindex(𝐲)
    𝐲ᵢ = zero(eltype(𝐲))
    for j ∈ eachindex(𝐱)
      𝐲ᵢ += 𝐀[i, j] * 𝐱[j]
    end
    𝐲[i] = 𝐲ᵢ
  end
function jvar!(𝐬², 𝐀, x̄)
  @. s² = zero(eltype(𝐬²))
  @inbounds @fastmath for i ∈ 1:size(𝐀, 2)
    @simd for j ∈ eachindex(𝐬²)
      δ = 𝐀[j, i] - x̄[j]
      𝐬²[j] += δ * δ
    end
  end
end
jvaravx!(𝐬², 𝐀, x̄) = @turbo for j ∈ eachindex(𝐬²)
    𝐬²ⱼ = zero(eltype(𝐬²))
    x̄ⱼ = x̄[j]
    for i ∈ 1:size(𝐀, 2)
      δ = 𝐀[j, i] - x̄ⱼ
      𝐬²ⱼ += δ * δ
    end
    𝐬²[j] = 𝐬²ⱼ
  end
japlucBc!(D, a, B, c) = @. D = a + B * c';
japlucBcavx!(D, a, B, c) = @turbo @. D = a + B * c';

function jOLSlp(y, X, β)
  lp = zero(eltype(y))
  @inbounds @fastmath for i ∈ eachindex(y)
    δ = y[i]
    @simd for j ∈ eachindex(β)
      δ -= X[i, j] * β[j]
    end
    lp += δ * δ
  end
  lp
end
function jOLSlp_avx(y, X, β)
  lp = zero(eltype(y))
  @turbo for i ∈ eachindex(y)
    δ = y[i]
    for j ∈ eachindex(β)
      δ -= X[i, j] * β[j]
    end
    lp += δ * δ
  end
  lp
end
function randomaccess(P, basis, coeffs::Vector{T}) where {T}
  C = length(coeffs)
  A = size(P, 1)
  p = zero(T)
  @fastmath @inbounds for c ∈ 1:C
    pc = coeffs[c]
    for a = 1:A
      pc *= P[a, basis[a, c]]
    end
    p += pc
  end
  return p
end
function randomaccessavx(P, basis, coeffs::Vector{T}) where {T}
  C = length(coeffs)
  A = size(P, 1)
  p = zero(T)
  @turbo for c ∈ 1:C
    pc = coeffs[c]
    for a = 1:A
      pc *= P[a, basis[a, c]]
    end
    p += pc
  end
  return p
end
function jlogdettriangle(B::Union{LowerTriangular,UpperTriangular})
  ld = 0.0
  @inbounds @fastmath for n ∈ 1:size(B, 1)
    ld += log(B[n, n])
  end
  ld
end
function jlogdettriangleavx(B::Union{LowerTriangular,UpperTriangular})
  A = parent(B) # No longer supported
  ld = zero(eltype(A))
  @turbo for n ∈ axes(A, 1)
    ld += log(A[n, n])
  end
  ld
end

function filter2d!(out::AbstractMatrix, A::AbstractMatrix, kern)
  @inbounds @fastmath for J in CartesianIndices(out)
    tmp = zero(eltype(out))
    for I ∈ CartesianIndices(kern)
      tmp += A[I+J] * kern[I]
    end
    out[J] = tmp
  end
  out
end
function filter2davx!(out::AbstractMatrix, A::AbstractMatrix, kern)
  @turbo for J in CartesianIndices(out)
    tmp = zero(eltype(out))
    for I ∈ CartesianIndices(kern)
      tmp += A[I+J] * kern[I]
    end
    out[J] = tmp
  end
  out
end

function filter2dunrolled!(
  out::AbstractMatrix,
  A::AbstractMatrix,
  kern::SizedOffsetMatrix{T,-1,1,-1,1}
) where {T}
  rng1, rng2 = axes(out)
  Base.Cartesian.@nexprs 3 jk ->
    Base.Cartesian.@nexprs 3 ik -> kern_ik_jk = kern[ik-2, jk-2]
  @inbounds for j in rng2
    @simd ivdep for i in rng1
      tmp_0 = zero(eltype(out))
      Base.Cartesian.@nexprs 3 jk -> Base.Cartesian.@nexprs 3 ik ->
        tmp_{ik + (jk - 1) * 3} = Base.FastMath.add_fast(
          Base.FastMath.mul_fast(A[i+(ik-2), j+(jk-2)], kern_ik_jk),
          tmp_{ik + (jk - 1) * 3 - 1}
        )
      out[i, j] = tmp_9
    end
  end
  out
end
function filter2dunrolledavx!(
  out::AbstractMatrix,
  A::AbstractMatrix,
  kern::SizedOffsetMatrix{T,-1,1,-1,1}
) where {T}
  rng1, rng2 = axes(out)
  Base.Cartesian.@nexprs 3 jk ->
    Base.Cartesian.@nexprs 3 ik -> kern_ik_jk = kern[ik-2, jk-2]
  @turbo for j in rng2, i in rng1
    tmp_0 = zero(eltype(out))
    Base.Cartesian.@nexprs 3 jk -> Base.Cartesian.@nexprs 3 ik ->
      tmp_{ik + (jk - 1) * 3} =
        A[i+(ik-2), j+(jk-2)] * kern_ik_jk + tmp_{ik + (jk - 1) * 3 - 1}
    out[i, j] = tmp_9
  end
  out
end

# function smooth_line!(sl,nrm1,j,i1,rl,ih2,denom)
#     @fastmath @inbounds @simd ivdep for i=i1:2:nrm1
#         sl[i,j]=denom*(rl[i,j]+ih2*(sl[i,j-1]+sl[i-1,j]+sl[i+1,j]+sl[i,j+1]))
#     end
# end
# function smooth_line_avx!(sl,nrm1,j,i1,sl,rl,ih2,denom)
#     @turbo for i=i1:2:nrm1
#         sl[i,j]=denom*(rl[i,j]+ih2*(sl[i,j-1]+sl[i-1,j]+sl[i+1,j]+sl[i,j+1]))
#     end
# end
