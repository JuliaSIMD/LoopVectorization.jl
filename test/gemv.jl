using LoopVectorization
using Test

@testset "GEMV" begin
    gemvq = :(for i ∈ eachindex(y)
              yᵢ = 0.0
              for j ∈ eachindex(x)
              yᵢ += A[i,j] * x[j]
              end
              y[i] = yᵢ
              end)
    lsgemv = LoopVectorization.LoopSet(gemvq);
    @test LoopVectorization.choose_order(lsgemv) == (Symbol[:i, :j], :i, 4, -1)

    function mygemv!(y, A, x)
        @inbounds for i ∈ eachindex(y)
            yᵢ = zero(eltype(y))
            @simd for j ∈ eachindex(x)
                yᵢ += A[i,j] * x[j]
            end
            y[i] = yᵢ
        end
    end
    function mygemvavx!(y, A, x)
        @avx for i ∈ eachindex(y)
            yᵢ = zero(eltype(y))
            for j ∈ eachindex(x)
                yᵢ += A[i,j] * x[j]
            end
            y[i] = yᵢ
        end
    end
    function mygemvavx_range!(y, A, x)
        rng1, rng2 = axes(A)
        @avx for i ∈ rng1
            yᵢ = zero(eltype(y))
            for j ∈ rng2
                yᵢ += A[i,j] * x[j]
            end
            y[i] = yᵢ
        end
    end
    q = :(for i ∈ eachindex(y)
          yᵢ = zero(eltype(y))
          for j ∈ eachindex(x)
          yᵢ += A[i,j] * x[j]
          end
          y[i] = yᵢ
          end)
    function mygemv_avx!(y, A, x)
        # Need to test 0s somewhere!
        @_avx for i ∈ eachindex(y)
            yᵢ = 0.0#zero(eltype(y))
            for j ∈ eachindex(x)
                yᵢ += A[i,j] * x[j]
            end
            y[i] = yᵢ
        end
    end


    function AtmulvB!(G, B,κ)
        d = size(G,1)
        @inbounds for d1=1:d
            G[d1,κ] = B[1,d1]*B[1,κ]
            for d2=2:d
                G[d1,κ] += B[d2,d1]*B[d2,κ]
            end
        end
    end
    function AtmulvBavx1!(G, B,κ)
        d = size(G,1)
        z = zero(eltype(G))
        @avx for d1=1:d
            G[d1,κ] = z
            for d2=1:d
                G[d1,κ] += B[d2,d1]*B[d2,κ]
            end
        end
    end
    function AtmulvBavx2!(G, B,κ)
        d = size(G,1)
        @avx for d1=1:d
            z = zero(eltype(G))
            for d2=1:d
                z += B[d2,d1]*B[d2,κ]
            end
            G[d1,κ] = z
        end
    end
    function AtmulvBavx3!(G, B,κ)
        d = size(G,1)
        @avx for d1=1:d
            G[d1,κ] = B[1,d1]*B[1,κ]
            for d2=2:d
                G[d1,κ] += B[d2,d1]*B[d2,κ]
            end
        end
    end
    function AtmulvB_avx1!(G, B,κ)
        d = size(G,1)
        z = zero(eltype(G))
        @_avx for d1=1:d
            G[d1,κ] = z
            for d2=1:d
                G[d1,κ] += B[d2,d1]*B[d2,κ]
            end
        end
    end
    function AtmulvB_avx2!(G, B,κ)
        d = size(G,1)
        @_avx for d1=1:d
            z = zero(eltype(G))
            for d2=1:d
                z += B[d2,d1]*B[d2,κ]
            end
            G[d1,κ] = z
        end
    end
    gemvq = :(for d1=1:d
              z = zero(eltype(G))
              for d2=1:d
              z += B[d2,d1]*B[d2,κ]
              end
              G[d1,κ] = z
              end)
    # if LoopVectorization.VectorizationBase.REGISTER_COUNT == 16
    # else
    # end
    Unum, Tnum = LoopVectorization.VectorizationBase.REGISTER_COUNT == 16 ? (3, 4) : (4, 4)
    lsgemv = LoopVectorization.LoopSet(gemvq);
    @test LoopVectorization.choose_order(lsgemv) == ([:d1,:d2], :d2, Unum, Tnum)
    function AtmulvB_avx3!(G, B,κ)
        d = size(G,1)
        @_avx for d1=1:d
            G[d1,κ] = B[1,d1]*B[1,κ]
            for d2=2:d
                G[d1,κ] += B[d2,d1]*B[d2,κ]
            end
        end
    end
    pq = :(for d1=1:d
           G[d1,κ] = B[1,d1]*B[1,κ]
           for d2=2:d
           G[d1,κ] += B[d2,d1]*B[d2,κ]
           end
           end)
    lsp = LoopVectorization.LoopSet(pq);
    @test LoopVectorization.choose_order(lsp) == ([:d1, :d2], :d2, Unum, Tnum)
    # lsp.preamble_symsym

    M, K, N = 51, 49, 61
    for T ∈ (Float32, Float64, Int32, Int64)
        @show T, @__LINE__
        TC = sizeof(T) == 4 ? Float32 : Float64
        R = T <: Integer ? (T(1):T(1000)) : T

        A = rand(R, M, K);
        x = rand(R, K);
        y1 = Vector{TC}(undef, M); y2 = similar(y1);
        mygemv!(y1, A, x)
        mygemvavx!(y2, A, x)
        @test y1 ≈ y2
        fill!(y2, -999.9); mygemv_avx!(y2, A, x)
        @test y1 ≈ y2
        fill!(y2, -999.9)
        mygemvavx_range!(y2, A, x)
        @test y1 ≈ y2

        B = rand(R, N, N);
        G1 = Matrix{TC}(undef, N, 1);
        G2 = similar(G1);
        # G3 = similar(G1);
        AtmulvB!(G1,B,1)
        AtmulvBavx1!(G2,B,1)
        @test G1 ≈ G2
        fill!(G2, TC(NaN)); AtmulvBavx2!(G2,B,1);

        @test G1 ≈ G2
        fill!(G2, TC(NaN)); AtmulvBavx3!(G2,B,1);
        @test G1 ≈ G2
        fill!(G2, TC(NaN)); AtmulvB_avx1!(G2,B,1);
        @test G1 ≈ G2
        fill!(G2, TC(NaN)); AtmulvB_avx2!(G2,B,1);
        @test G1 ≈ G2
        fill!(G2, TC(NaN)); AtmulvB_avx3!(G2,B,1);
        @test G1 ≈ G2
    end
end
