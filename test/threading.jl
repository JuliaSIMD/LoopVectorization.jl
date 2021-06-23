using OffsetArrays, LinearAlgebra, LoopVectorization, Test

let nt = min(Threads.nthreads(), (Sys.CPU_THREADS)::Int) - 1
  if (LoopVectorization.num_cores() < 4) && (nt ≥ 4)
    @eval LoopVectorization.num_cores() = LoopVectorization.StaticInt{$nt}()
  end
end

function mydotavx(a, b)
    s = zero(eltype(a))
    @tturbo for i ∈ eachindex(a,b)
        s += a[i]*b[i]
    end
    s
end
function AmulB!(C,A,B)
    @tturbo for n in indices((B,C),2), m in indices((A,C),1)
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
    @tturbo for n in axes(Ar,3)
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
    @tturbo for n ∈ axes(out,2), m ∈ axes(out,1)
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


struct DenseConvDims{N,K,C_in,C_out} end

function kernaxes(::DenseConvDims{2,K,C_in, C_out}) where {K,C_in, C_out}
    K₁ =  LoopVectorization.StaticInt(1):LoopVectorization.StaticInt(K[1])
    K₂ =  LoopVectorization.StaticInt(1):LoopVectorization.StaticInt(K[2])
    Cᵢₙ =  LoopVectorization.StaticInt(1):LoopVectorization.StaticInt(C_in)
    Cₒᵤₜ = LoopVectorization.StaticInt(1):LoopVectorization.StaticInt(C_out)
    (K₁, K₂, Cᵢₙ, Cₒᵤₜ)
end

function convlayer!(
    out::AbstractArray{<:Any,4}, img, kern,
    dcd::DenseConvDims{2, <:Any, <:Any, <:Any}
)
    (K₁, K₂, Cᵢₙ, Cₒᵤₜ) = kernaxes(dcd)
    @tturbo for j₁ ∈ axes(out,1), j₂ ∈ axes(out,2), d ∈ axes(out,4), o ∈ Cₒᵤₜ
        s = zero(eltype(out))
        for k₁ ∈ K₁, k₂ ∈ K₂, i ∈ Cᵢₙ
            s += img[j₁ + k₁ - 1, j₂ + k₂ - 1, i, d] * kern[k₁, k₂, i, o]
        end
        out[j₁, j₂, o, d] = s
    end
    out
end
function convlayer_direct!(
    out::AbstractArray{<:Any,4}, img, kern,
    dcd::DenseConvDims{2, <:Any, <:Any, <:Any}
)
    (K₁, K₂, Cᵢₙ, Cₒᵤₜ) = kernaxes(dcd)
    @inbounds @fastmath for j₁ ∈ axes(out,1), j₂ ∈ axes(out,2), d ∈ axes(out,4), o ∈ Cₒᵤₜ
        s = zero(eltype(out))
        for k₁ ∈ K₁, k₂ ∈ K₂, i ∈ Cᵢₙ
            s += img[j₁ + k₁ - 1, j₂ + k₂ - 1, i, d] * kern[k₁, k₂, i, o]
        end
        out[j₁, j₂, o, d] = s
    end
    out
end

@testset "Threading" begin
  @show @__LINE__
  dcd = DenseConvDims{2,(5,5),3,6}()
  kern4 = rand(Float32, 5, 5, 3, 6);
  @time for M ∈ 17:50:267
    img = rand(Float32, M, M, 3, 100);
    outimage1 = Array{Float32}(undef, size(img,1)+1-size(kern4,1), size(img,2)+1-size(kern4,2), size(kern4,4), size(img,4));
    outimage2 = similar(outimage1);

    convlayer!(outimage1, img, kern4, dcd);
    convlayer_direct!(outimage2, img, kern4, dcd);
    @test outimage1 ≈ outimage2
  end

  @time for M ∈ 17:399
    # @show M
    K = M; N = M;
    A = rand(M,K); B = rand(K,N); b = rand(K);
    @test dot(A,B) ≈ mydotavx(A,B)

    C1 = A * B;
    @test AmulB!(similar(C1), A, B) ≈ C1
    c1 = A * b;
    @test AmulB!(similar(c1), A, b) ≈ c1

    if VERSION ≥ v"1.6"
      x = randn(Complex{Float64}, 3M-1);
      W = randn(Complex{Float64}, 3M-1, 3M+1);
      y = randn(Complex{Float64}, 3M+1);
      @test dot(x,W,y) ≈ dot3(x,W,y)
    end

    kern = OffsetArray(randn(3,3),-2,-2)
    out1 = OffsetArray(randn(size(A) .- 2), 1, 1)
    out2 = similar(out1);
    @test conv!(out1, A, kern) ≈ conv_baseline!(out2, A, kern)
  end
end


