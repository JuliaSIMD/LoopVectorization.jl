using OffsetArrays, LinearAlgebra
function mydotavx(a, b)
    s = zero(eltype(a))
    @avxt for i ∈ eachindex(a,b)
        s += a[i]*b[i]
    end
    s
end
function AmulB!(C,A,B)
    @avxt for n in indices((B,C),2), m in indices((A,C),1)
        Cmn = zero(eltype(C))
        for k in indices((A,B),(2,1))
            Cmn += A[m,k] * B[k,n]
        end
        C[m,n] = Cmn
    end
    C
end
function dot3(x::AbstractVector{Complex{T}}, A::AbstractMatrix{Complex{T}}, y::AbstractVector{Complex{T}}) where {T}
    xr = reinterpret(reshape, T, x);
    yr = reinterpret(reshape, T, y);
    Ar = reinterpret(reshape, T, A);
    sre = zero(T)
    sim = zero(T)
    @avxt for n in axes(Ar,3)
        tre = zero(T)
        tim = zero(T)
        for m in axes(Ar,2)
            tre += xr[1,m] * Ar[1,m,n] + xr[2,m] * Ar[2,m,n]
            tim += xr[1,m] * Ar[2,m,n] - xr[2,m] * Ar[1,m,n]
        end
        sre += tre * yr[1,n] - tim * yr[2,n]
        sim += tre * yr[2,n] + tim * yr[1,n]
    end
    sre + im*sim
end
function conv!(out, A, kern)
    @avxt for n ∈ axes(out,2), m ∈ axes(out,1)
        tmp = zero(eltype(out))
        for nᵢ ∈ -1:1, mᵢ ∈ -1:1
            tmp += A[mᵢ+m, nᵢ+n]*kern[mᵢ, nᵢ]
        end
        out[m,n] = tmp
    end
    out
end
function conv_baseline!(out, A, kern)
    for n ∈ axes(out,2), m ∈ axes(out,1)
        tmp = zero(eltype(out))
        for nᵢ ∈ -1:1, mᵢ ∈ -1:1
            tmp += A[mᵢ+m, nᵢ+n]*kern[mᵢ, nᵢ]
        end
        out[m,n] = tmp
    end
    out
end

@testset "Threading" begin
    for M ∈ 17:399
        # @show M
        K = M; N = M;
        A = rand(M,K); B = rand(K,N);
        @test dot(A,B) ≈ mydotavx(A,B)

        C1 = A * B; C0 = similar(C1);
        @test AmulB!(C0, A, B) ≈ C1

        x = randn(Complex{Float64}, 3M-1);
        W = randn(Complex{Float64}, 3M-1, 3M+1);
        y = randn(Complex{Float64}, 3M+1);
        @test dot(x,W,y) ≈ dot3(x,W,y)

        kern = OffsetArray(randn(3,3),-2,-2)
        out1 = OffsetArray(randn(size(A) .- 2), 1, 1)
        out2 = similar(out1);
        @test conv!(out1, A, kern) ≈ conv_baseline!(out2, A, kern)
    end
end


