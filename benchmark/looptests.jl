using LoopVectorization, LinearAlgebra
BLAS.set_num_threads(1)

function jgemm!(C, A, B)
    C .= 0
    M, N = size(C); K = size(B,1)
    @inbounds for n ∈ 1:N, k ∈ 1:K
        @simd ivdep for m ∈ 1:M
            C[m,n] += A[m,k] * B[k,n]
        end
    end
end
@inline function jgemm!(C, Aᵀ::Adjoint, B)
    A = parent(Aᵀ)
    @inbounds for n ∈ 1:size(C,2), m ∈ 1:size(C,1)
        Cₘₙ = zero(eltype(C))
        @simd ivdep for k ∈ 1:size(A,1)
            Cₘₙ += A[k,m] * B[k,n]
        end
        C[m,n] = Cₘₙ
    end
end
@inline function jgemm!(C, A, Bᵀ::Adjoint)
    C .= 0
    B = parent(Bᵀ)
    M, N = size(C); K = size(B,1)
    @inbounds for k ∈ 1:K, n ∈ 1:N
        @simd ivdep for m ∈ 1:M
            C[m,n] += A[m,k] * B[n,k]
        end
    end
end
@inline function gemmavx!(C, A, B)
    @avx for i ∈ 1:size(A,1), j ∈ 1:size(B,2)
        Cᵢⱼ = zero(eltype(C))
        for k ∈ 1:size(A,2)
            Cᵢⱼ += A[i,k] * B[k,j]
        end
        C[i,j] = Cᵢⱼ
    end
end
function jdot(a, b)
    s = 0.0
    @inbounds @simd ivdep for i ∈ eachindex(a, b)
        s += a[i] * b[i]
    end
    s
end
function jdotavx(a, b)
    s = 0.0
    @avx for i ∈ eachindex(a, b)
        s += a[i] * b[i]
    end
    s
end
function jselfdot(a)
    s = 0.0
    @inbounds @simd ivdep for i ∈ eachindex(a)
        s += a[i] * a[i]
    end
    s
end
function jselfdotavx(a)
    s = 0.0
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
            s += x[m] * A[m,n] * y[n]
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
    s = 0.0
    @inbounds for i ∈ eachindex(a)
        s += exp(a[i])
    end
    s
end
function jsvexpavx(a)
    s = 0.0
    @avx for i ∈ eachindex(a)
        s += exp(a[i])
    end
    s
end
function jgemv!(y, A, x)
    y .= 0.0
    @inbounds for j ∈ eachindex(x)
        @simd ivdep for i ∈ eachindex(y)
            y[i] += A[i,j] * x[j]
        end
    end
end
@inline function jgemv!(y, Aᵀ::Adjoint, x)
    A = parent(Aᵀ)
    @inbounds for i ∈ eachindex(y)
        yᵢ = 0.0
        @simd ivdep for j ∈ eachindex(x)
            yᵢ += A[j,i] * x[j]
        end
        y[i] = yᵢ
    end
end
@inline function jgemvavx!(y, A, x)
    @avx for i ∈ eachindex(y)
        yᵢ = 0.0
        for j ∈ eachindex(x)
            yᵢ += A[i,j] * x[j]
        end
        y[i] = yᵢ
    end
end
function jvar!(s², A, x̄)
    @. s² = 0
    @inbounds for i ∈ 1:size(A,2)
        @simd for j ∈ eachindex(s²)
            δ = A[j,i] - x̄[j]
            s²[j] += δ*δ
        end
    end
end
function jvaravx!(s², A, x̄)
    @avx for j ∈ eachindex(s²)
        s²ⱼ = 0.0
        x̄ⱼ = x̄[j]
        for i ∈ 1:size(A,2)
            δ = A[j,i] - x̄ⱼ
            s²ⱼ += δ*δ
        end
        s²[j] = s²ⱼ
    end
end
japlucBc!(d, a, B, c) =      @. d = a + B * c';
japlucBcavx!(d, a, B, c) = @avx @. d = a + B * c';

function jOLSlp(y, X, β)
    lp = 0.0
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
    lp = 0.0
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



