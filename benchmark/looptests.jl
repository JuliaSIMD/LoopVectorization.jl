using LoopVectorization, LinearAlgebra
BLAS.set_num_threads(1)

function jgemm!(𝐂, 𝐀, 𝐁)
    𝐂 .= 0
    M, N = size(𝐂); K = size(𝐁,1)
    @inbounds for n ∈ 1:N, k ∈ 1:K
        @simd ivdep for m ∈ 1:M
            @fastmath 𝐂[m,n] += 𝐀[m,k] * 𝐁[k,n]
        end
    end
end
@inline function jgemm!(𝐂, 𝐀ᵀ::Adjoint, 𝐁)
    𝐀 = parent(𝐀ᵀ)
    @inbounds for n ∈ 1:size(𝐂,2), m ∈ 1:size(𝐂,1)
        𝐂ₘₙ = zero(eltype(𝐂))
        @simd ivdep for k ∈ 1:size(𝐀,1)
            @fastmath 𝐂ₘₙ += 𝐀[k,m] * 𝐁[k,n]
        end
        𝐂[m,n] = 𝐂ₘₙ
    end
end
@inline function jgemm!(𝐂, 𝐀, 𝐁ᵀ::Adjoint)
    𝐂 .= 0
    𝐁 = parent(𝐁ᵀ)
    M, N = size(𝐂); K = size(𝐁ᵀ,1)
    @inbounds for k ∈ 1:K, n ∈ 1:N
        @simd ivdep for m ∈ 1:M
            @fastmath 𝐂[m,n] += 𝐀[m,k] * 𝐁[n,k]
        end
    end
end
@inline function jgemm!(𝐂, 𝐀ᵀ::Adjoint, 𝐁ᵀ::Adjoint)
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
@inline function gemmavx!(𝐂, 𝐀, 𝐁)
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
    @inbounds for n ∈ 1:N
        @simd ivdep for m ∈ 1:M
            @fastmath s += x[m] * A[m,n] * y[n]
        end
    end
    s
end
function jdot3avx(x, A, y)
    M, N = size(A)
    s = zero(promote_type(eltype(x), eltype(A), eltype(y)))
    @avx for m ∈ 1:M, n ∈ 1:N
        s += x[m] * A[m,n] * y[n]
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
@inline function jgemv!(𝐲, 𝐀ᵀ::Adjoint, 𝐱)
    𝐀 = parent(𝐀ᵀ)
    @inbounds for i ∈ eachindex(𝐲)
        𝐲ᵢ = zero(eltype(𝐲))
        @simd ivdep for j ∈ eachindex(𝐱)
            @fastmath 𝐲ᵢ += 𝐀[j,i] * 𝐱[j]
        end
        𝐲[i] = 𝐲ᵢ
    end
end
@inline function jgemvavx!(𝐲, 𝐀, 𝐱)
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
function jlogdettriangle(T::Union{LowerTriangular,UpperTriangular})
    ld = 0.0
    @inbounds for n ∈ 1:size(T,1)
        ld += log(T[n,n])
    end
    ld
end
function jlogdettriangleavx(T::Union{LowerTriangular,UpperTriangular})
    ld = 0.0
    @avx for n ∈ 1:size(T,1)
        ld += log(T[n,n])
    end
    ld
end


