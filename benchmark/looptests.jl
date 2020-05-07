using LoopVectorization, LinearAlgebra, OffsetArrays
BLAS.set_num_threads(1)

struct SizedOffsetMatrix{T,LR,UR,LC,RC} <: DenseMatrix{T}
    data::Matrix{T}
end
using LoopVectorization.VectorizationBase: StaticUnitRange
Base.axes(::SizedOffsetMatrix{T,LR,UR,LC,UC}) where {T,LR,UR,LC,UC} = (StaticUnitRange{LR,UR}(),StaticUnitRange{LC,UC}())
@generated function LoopVectorization.stridedpointer(A::SizedOffsetMatrix{T,LR,UR,LC,RC}) where {T,LR,UR,LC,RC}
    quote
        $(Expr(:meta,:inline))
        LoopVectorization.OffsetStridedPointer(
            LoopVectorization.StaticStridedPointer{$T,Tuple{1,$(UR-LR+1)}}(pointer(A.data)),
            ($(LR-2), $(LC-2))
        )
    end
end
Base.size(A::SizedOffsetMatrix{T,LR,UR,LC,UC}) where {T,LR,UR,LC,UC} = (1 + UR-LR, 1 + UC-LC)
Base.getindex(A::SizedOffsetMatrix, i, j) = LoopVectorization.vload(LoopVectorization.stridedpointer(A), (i,j)) # only needed to print
Base.unsafe_convert(::Type{Ptr{Float64}}, A::SizedOffsetMatrix) = Base.unsafe_convert(Ptr{Float64}, A.data)


function jgemm!(𝐂, 𝐀, 𝐁)
    𝐂 .= 0
    M, N = size(𝐂); K = size(𝐁,1)
    @inbounds for n ∈ 1:N, k ∈ 1:K
        @simd ivdep for m ∈ 1:M
            @fastmath 𝐂[m,n] += 𝐀[m,k] * 𝐁[k,n]
        end
    end
end
function jgemm!(𝐂, 𝐀ᵀ::Adjoint, 𝐁)
    𝐀 = parent(𝐀ᵀ)
    @inbounds for n ∈ 1:size(𝐂,2), m ∈ 1:size(𝐂,1)
        𝐂ₘₙ = zero(eltype(𝐂))
        @simd ivdep for k ∈ 1:size(𝐀,1)
            @fastmath 𝐂ₘₙ += 𝐀[k,m] * 𝐁[k,n]
        end
        𝐂[m,n] = 𝐂ₘₙ
    end
end
function jgemm!(𝐂, 𝐀, 𝐁ᵀ::Adjoint)
    𝐂 .= 0
    𝐁 = parent(𝐁ᵀ)
    M, N = size(𝐂); K = size(𝐁ᵀ,1)
    @inbounds for k ∈ 1:K, n ∈ 1:N
        @simd ivdep for m ∈ 1:M
            @fastmath 𝐂[m,n] += 𝐀[m,k] * 𝐁[n,k]
        end
    end
end
function jgemm!(𝐂, 𝐀ᵀ::Adjoint, 𝐁ᵀ::Adjoint)
    𝐂 .= 0
    𝐀 = parent(𝐀ᵀ)
    𝐁 = parent(𝐁ᵀ)
    M, N = size(𝐂); K = size(𝐁ᵀ,1)
    @inbounds for n ∈ 1:N, k ∈ 1:K
        @simd ivdep for m ∈ 1:M
            @fastmath 𝐂[m,n] += 𝐀[k,m] * 𝐁[n,k]
        end
    end
end
function gemmavx!(𝐂, 𝐀, 𝐁)
    @avx for m ∈ 1:size(𝐀,1), n ∈ 1:size(𝐁,2)
        𝐂ₘₙ = zero(eltype(𝐂))
        for k ∈ 1:size(𝐀,2)
            𝐂ₘₙ += 𝐀[m,k] * 𝐁[k,n]
        end
        𝐂[m,n] = 𝐂ₘₙ
    end
end
function jdot(a, b)
    s = zero(eltype(a))
    @inbounds @simd ivdep for i ∈ eachindex(a, b)
        s += a[i] * b[i]
    end
    s
end
function jdotavx(a, b)
    s = zero(eltype(a))
    @avx for i ∈ eachindex(a, b)
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
    @avx for i ∈ eachindex(a)
        s += a[i] * a[i]
    end
    s
end
function jdot3(x, A, y)
    M, N = size(A)
    s = zero(promote_type(eltype(x), eltype(A), eltype(y)))
    @inbounds @fastmath for n ∈ 1:N, m ∈ 1:M
        s += x[m] * A[m,n] * y[n]
    end
    s
end
function jdot3avx(x, A, y)
    M, N = size(A)
    s = zero(promote_type(eltype(x), eltype(A), eltype(y)))
    @avx for n ∈ 1:N, m ∈ 1:M
        s += x[m] * A[m,n] * y[n]
    end
    s
end
function jdot3v2(x, A, y)
    M, N = size(A)
    s = zero(promote_type(eltype(x), eltype(A), eltype(y)))
    @inbounds @fastmath for n ∈ 1:N
        t = zero(s)
        @simd ivdep for m ∈ 1:M
            t += x[m] * A[m,n]
        end
        s += t * y[n]
    end
    s
end
function jdot3v2avx(x, A, y)
    M, N = size(A)
    s = zero(promote_type(eltype(x), eltype(A), eltype(y)))
    @avx for n ∈ 1:N
        t = zero(s)
        for m ∈ 1:M
            t += x[m] * A[m,n]
        end
        s += t * y[n]
    end
    s
end
function jvexp!(b, a)
    @inbounds for i ∈ eachindex(a)
        b[i] = exp(a[i])
    end
end
function jvexpavx!(b, a)
    @avx for i ∈ eachindex(a)
        b[i] = exp(a[i])
    end
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
    @avx for i ∈ eachindex(a)
        s += exp(a[i])
    end
    s
end
function jgemv!(y, 𝐀, x)
    y .= zero(eltype(y))
    @inbounds for j ∈ eachindex(x)
        @simd ivdep for i ∈ eachindex(y)
            @fastmath y[i] += 𝐀[i,j] * x[j]
        end
    end
end
function jgemv!(𝐲, 𝐀ᵀ::Adjoint, 𝐱)
    𝐀 = parent(𝐀ᵀ)
    @inbounds for i ∈ eachindex(𝐲)
        𝐲ᵢ = zero(eltype(𝐲))
        @simd ivdep for j ∈ eachindex(𝐱)
            @fastmath 𝐲ᵢ += 𝐀[j,i] * 𝐱[j]
        end
        𝐲[i] = 𝐲ᵢ
    end
end
function jgemvavx!(𝐲, 𝐀, 𝐱)
    @avx for i ∈ eachindex(𝐲)
        𝐲ᵢ = zero(eltype(𝐲))
        for j ∈ eachindex(𝐱)
            𝐲ᵢ += 𝐀[i,j] * 𝐱[j]
        end
        𝐲[i] = 𝐲ᵢ
    end
end
function jvar!(𝐬², 𝐀, x̄)
    @. s² = zero(eltype(𝐬²))
    @inbounds @fastmath for i ∈ 1:size(𝐀,2)
        @simd for j ∈ eachindex(𝐬²)
            δ = 𝐀[j,i] - x̄[j]
            𝐬²[j] += δ*δ
        end
    end
end
function jvaravx!(𝐬², 𝐀, x̄)
    @avx for j ∈ eachindex(𝐬²)
        𝐬²ⱼ = zero(eltype(𝐬²))
        x̄ⱼ = x̄[j]
        for i ∈ 1:size(𝐀,2)
            δ = 𝐀[j,i] - x̄ⱼ
            𝐬²ⱼ += δ*δ
        end
        𝐬²[j] = 𝐬²ⱼ
    end
end
japlucBc!(d, a, B, c) =      @. d = a + B * c';
japlucBcavx!(d, a, B, c) = @avx @. d = a + B * c';

function jOLSlp(y, X, β)
    lp = zero(eltype(y))
    @inbounds @fastmath for i ∈ eachindex(y)
        δ = y[i]
        @simd for j ∈ eachindex(β)
            δ -= X[i,j] * β[j]
        end
        lp += δ * δ
    end
    lp
end
function jOLSlp_avx(y, X, β)
    lp = zero(eltype(y))
    @avx for i ∈ eachindex(y)
        δ = y[i]
        for j ∈ eachindex(β)
            δ -= X[i,j] * β[j]
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
    @avx for c ∈ 1:C
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
    @inbounds @fastmath for n ∈ 1:size(B,1)
        ld += log(B[n,n])
    end
    ld
end
function jlogdettriangleavx(B::Union{LowerTriangular,UpperTriangular})
    A = parent(B) # No longer supported
    ld = zero(eltype(A))
    @avx for n ∈ axes(A,1)
        ld += log(A[n,n])
    end
    ld
end




function filter2d!(out::AbstractMatrix, A::AbstractMatrix, kern)
    @inbounds @fastmath for J in CartesianIndices(out)
        tmp = zero(eltype(out))
        for I ∈ CartesianIndices(kern)
            tmp += A[I + J] * kern[I]
        end
        out[J] = tmp
    end
    out
end
function filter2davx!(out::AbstractMatrix, A::AbstractMatrix, kern)
    @avx for J in CartesianIndices(out)
        tmp = zero(eltype(out))
        for I ∈ CartesianIndices(kern)
            tmp += A[I + J] * kern[I]
        end
        out[J] = tmp
    end
    out
end

function filter2dunrolled!(out::AbstractMatrix, A::AbstractMatrix, kern::SizedOffsetMatrix{T,-1,1,-1,1}) where {T}
    rng1,  rng2  = axes(out)
    Base.Cartesian.@nexprs 3 jk -> Base.Cartesian.@nexprs 3 ik -> kern_ik_jk = kern[ik-2,jk-2]
    @inbounds for j in rng2
        @simd ivdep for i in rng1
            tmp_0 = zero(eltype(out))
            Base.Cartesian.@nexprs 3 jk -> Base.Cartesian.@nexprs 3 ik -> tmp_{ik+(jk-1)*3} =  Base.FastMath.add_fast(Base.FastMath.mul_fast(A[i+(ik-2),j+(jk-2)], kern_ik_jk), tmp_{ik+(jk-1)*3-1})
            out[i,j] = tmp_9
        end
    end
    out
end
function filter2dunrolledavx!(out::AbstractMatrix, A::AbstractMatrix, kern::SizedOffsetMatrix{T,-1,1,-1,1}) where {T}
    rng1,  rng2  = axes(out)
    Base.Cartesian.@nexprs 3 jk -> Base.Cartesian.@nexprs 3 ik -> kern_ik_jk = kern[ik-2,jk-2]
    @avx for j in rng2, i in rng1
        tmp_0 = zero(eltype(out))
        Base.Cartesian.@nexprs 3 jk -> Base.Cartesian.@nexprs 3 ik -> tmp_{ik+(jk-1)*3} = A[i+(ik-2),j+(jk-2)] * kern_ik_jk + tmp_{ik+(jk-1)*3-1}
        out[i,j] = tmp_9
    end
    out
end
