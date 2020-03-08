using LoopVectorization, OffsetArrays
using LoopVectorization.VectorizationBase: StaticUnitRange
using Test
# T = Float32

@testset "OffsetArrays" begin

    function old2d!(out::AbstractMatrix, A::AbstractMatrix, kern)
        rng1k, rng2k = axes(kern)
        rng1,  rng2  = axes(out)
        for j in rng2, i in rng1
            tmp = zero(eltype(out))
            @inbounds for jk in rng2k, ik in rng1k
                tmp += oftype(tmp, A[i+ik,j+jk])*kern[ik,jk]
            end
            @inbounds out[i,j] = tmp
        end
        out
    end

    # out = out1;
    # R=CartesianIndices(out);
    # z=zero(eltype(out));
    # rng1k, rng2k = axes(skern);
    # rng1,  rng2  = R.indices;
    # tmp = z; i = 2; j = 2;
    # ls1 = LoopVectorization.@avx_debug for jk in rng2k, ik in rng1k
    #     tmp += A[i+ik,j+jk]*skern[ik,jk]
    # end;
    # ls1
    # rng1,  rng2  = CartesianIndices(out1).indices;
    # rng1k, rng2k = axes(skern);
    # ls2dstatic = LoopVectorization.@avx_debug for j in rng2, i in rng1
    #         tmp = zero(eltype(out))
    #         for jk in rng2k, ik in rng1k
    #             tmp += A[i+ik,j+jk]*skern[ik,jk]
    #         end
    #         out1[i,j] = tmp
    # end;
    # LoopVectorization.choose_order(ls2dstatic)
    # q2d = :(for j in rng2, i in rng1
    #         tmp = zero(eltype(out))
    #         for jk in rng2k, ik in rng1k
    #             tmp += A[i+ik,j+jk]*kern[ik,jk]
    #         end
    #         out[i,j] = tmp
    #        end);
    # lsq2d = LoopVectorization.LoopSet(q2d); LoopVectorization.choose_order(lsq2d)

    # oq2 = :(for j in rng2, i in rng1
    #         tmp = zero(eltype(out))
    #         for jk in -1:1, ik in -1:1
    #             tmp += A[i+ik,j+jk]*kern[ik,jk]
    #         end
    #         out[i,j] = tmp
    #        end);
    # lsoq = LoopVectorization.LoopSet(oq2);
    # LoopVectorization.choose_order(lsoq)

    function avx2d!(out::AbstractMatrix, A::AbstractMatrix, kern)
        rng1k, rng2k = axes(kern)
        rng1,  rng2  = axes(out)
        for j in rng2, i in rng1
            tmp = zero(eltype(out))
            @avx for jk in rng2k, ik in rng1k
                tmp += A[i+ik,j+jk]*kern[ik,jk]
            end
            out[i,j] = tmp
        end
        out
    end
    function avx2douter!(out::AbstractMatrix, A::AbstractMatrix, kern)
        rng1k, rng2k = axes(kern)
        rng1,  rng2  = axes(out)
        @avx for j in rng2, i in rng1
            tmp = zero(eltype(out))
            for jk in rng2k, ik in rng1k
                tmp += A[i+ik,j+jk]*kern[ik,jk]
            end
            out[i,j] = tmp
        end
        out
    end



    struct SizedOffsetMatrix{T,LR,UR,LC,RC} <: AbstractMatrix{T}
        data::Matrix{T}
    end
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
    # Base.size(A::SizedOffsetMatrix{T,LR,UR,LC,UC}) where {T,LR,UR,LC,UC} = (1 + UR-LR, 1 + UC-LC)
    # Base.CartesianIndices(::SizedOffsetMatrix{T,LR,UR,LC,UC}) where {T,LR,UR,LC,UC} = CartesianIndices((LR:UR,LC:UC))
    Base.getindex(A::SizedOffsetMatrix, i, j) = LoopVectorization.vload(LoopVectorization.stridedpointer(A), (i,j)) # only needed to print
    function avx2dunrolled!(out::AbstractMatrix, A::AbstractMatrix, kern::SizedOffsetMatrix{T,-1,1,-1,1}) where {T}
        rng1,  rng2  = axes(out)
        Base.Cartesian.@nexprs 3 jk -> Base.Cartesian.@nexprs 3 ik -> kern_ik_jk = kern[ik-2,jk-2]
        # Manually unpack the OffsetArray
        @avx for j in rng2, i in rng1
            tmp_0 = zero(eltype(out))
            Base.Cartesian.@nexprs 3 jk -> Base.Cartesian.@nexprs 3 ik -> tmp_{ik+(jk-1)*3} = A[i+(ik-2),j+(jk-2)] * kern_ik_jk + tmp_{ik+(jk-1)*3-1}
            out[i,j] = tmp_9
        end
        out
    end
    function avx2dunrolled2x2!(out::AbstractMatrix, A::AbstractMatrix, kern::SizedOffsetMatrix{T,-1,1,-1,1}) where {T}
        rng1,  rng2  = axes(out)
        Base.Cartesian.@nexprs 3 jk -> Base.Cartesian.@nexprs 3 ik -> kern_ik_jk = kern[ik-2,jk-2]
        # Manually unpack the OffsetArray
        @avx tile=(2,2) for j in rng2, i in rng1
            tmp_0 = zero(eltype(out))
            Base.Cartesian.@nexprs 3 jk -> Base.Cartesian.@nexprs 3 ik -> tmp_{ik+(jk-1)*3} = A[i+(ik-2),j+(jk-2)] * kern_ik_jk + tmp_{ik+(jk-1)*3-1}
            out[i,j] = tmp_9
        end
        out
    end
    function avx2dunrolled3x3!(out::AbstractMatrix, A::AbstractMatrix, kern::SizedOffsetMatrix{T,-1,1,-1,1}) where {T}
        rng1,  rng2  = axes(out)
        Base.Cartesian.@nexprs 3 jk -> Base.Cartesian.@nexprs 3 ik -> kern_ik_jk = kern[ik-2,jk-2]
        # Manually unpack the OffsetArray
        @avx tile=(3,3) for j in rng2, i in rng1
            tmp_0 = zero(eltype(out))
            Base.Cartesian.@nexprs 3 jk -> Base.Cartesian.@nexprs 3 ik -> tmp_{ik+(jk-1)*3} = A[i+(ik-2),j+(jk-2)] * kern_ik_jk + tmp_{ik+(jk-1)*3-1}
            out[i,j] = tmp_9
        end
        out
    end
    # uq = :(for j in rng2, i in rng1
    #         tmp_0 = zero(eltype(out))
    #         Base.Cartesian.@nexprs 3 jk -> Base.Cartesian.@nexprs 3 ik -> tmp_{ik+(jk-1)*3} = A[i+(ik-2),j+(jk-2)] * kern_ik_jk + tmp_{ik+(jk-1)*3-1}
    #         out[i,j] = tmp_9
    #        end);
    # lsuq = LoopVectorization.LoopSet(macroexpand(Base, uq));
    # LoopVectorization.choose_order(lsuq)


    function avxgeneric!(out, A, kern, R=CartesianIndices(out), z=zero(eltype(out)))
       Rk = CartesianIndices(kern)
       @avx for I in R
           tmp = z
           for J in Rk
               tmp += A[I+J]*kern[J]
           end
           out[I] = tmp
       end
       out
   end

    for T ∈ (Float32, Float64)
        @show T, @__LINE__
        A = rand(T, 100, 100);
        kern = OffsetArray(rand(T, 3, 3), -1:1, -1:1);
        skern = SizedOffsetMatrix{T,-1,1,-1,1}(parent(kern));
        out1 = OffsetArray(similar(A, size(A).-2), 1, 1);   # stay away from the edges of A
        out2 = similar(out1); out3 = similar(out1); out4 = similar(out1)

        old2d!(out1, A, kern);
        avx2d!(out2, A, kern);
        @test out1 ≈ out2

        avx2douter!(out3, A, kern);
        @test out1 ≈ out3

        fill!(out2, NaN); avx2d!(out2, A, skern);
        @test out1 ≈ out2

        fill!(out3, NaN); avx2douter!(out3, A, skern);
        @test out1 ≈ out3

        fill!(out3, NaN); avx2dunrolled!(out3, A, skern);
        @test out1 ≈ out3

        fill!(out3, NaN); avx2dunrolled2x2!(out3, A, skern);
        @test out1 ≈ out3

        fill!(out3, NaN); avx2dunrolled3x3!(out3, A, skern);
        @test out1 ≈ out3

        @test_broken avxgeneric!(out4, A, kern) ≈ out1
    end


end
