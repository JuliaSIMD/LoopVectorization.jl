using LoopVectorization, OffsetArrays
using Test

@testset "OffsetArrays" begin

    function old2d!(out::AbstractMatrix, A::AbstractMatrix, kern, R=CartesianIndices(out), z=zero(eltype(out)))
        rng1k, rng2k = axes(kern)
        rng1,  rng2  = R.indices
        for j in rng2, i in rng1
            tmp = z
            @inbounds for jk in rng2k, ik in rng1k
                tmp += oftype(tmp, A[i+ik,j+jk])*kern[ik,jk]
            end
            @inbounds out[i,j] = tmp
        end
        out
    end
    function avx2d!(out::AbstractMatrix, A::AbstractMatrix, kern::OffsetArray, R=CartesianIndices(out), z=zero(eltype(out)))
        rng1k, rng2k = axes(kern)
        rng1,  rng2  = R.indices
        # Manually unpack the OffsetArray
        kernA = parent(kern)
        o1, o2 = kern.offsets
        for j in rng2, i in rng1
            tmp = z
            @avx for jk in rng2k, ik in rng1k
                tmp += A[i+ik,j+jk]*kernA[ik-o1,jk-o2]
            end
            out[i,j] = tmp
        end
        out
    end
    function avx2douter!(out::AbstractMatrix, A::AbstractMatrix, kern::OffsetArray, R=CartesianIndices(out), z=zero(eltype(out)))
        rng1k, rng2k = axes(kern)
        rng1,  rng2  = R.indices
        # Manually unpack the OffsetArray
        kernA = parent(kern)
        o1, o2 = kern.offsets
        @avx for j in rng2, i in rng1
            tmp = z
            for jk in rng2k, ik in rng1k
                tmp += A[i+ik,j+jk]*kernA[ik-o1,jk-o2]
                1
            end
            out[i,j] = tmp
        end
        out
    end
    
    for T ∈ (Float32, Float64)
        @show T, @__LINE__
        A = rand(T, 100, 100);
        kern = OffsetArray(rand(T, 3, 3), -1:1, -1:1);
        out1 = OffsetArray(similar(A, size(A).-2), 1, 1);   # stay away from the edges of A
        out2 = similar(out1); out3 = similar(out1);

        old2d!(out1, A, kern);
        avx2d!(out2, A, kern);
        @test out1 ≈ out2
        avx2douter!(out3, A, kern);
        @test out1 ≈ out3
    end

    
end

