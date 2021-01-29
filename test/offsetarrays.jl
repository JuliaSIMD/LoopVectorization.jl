using LoopVectorization, ArrayInterface, OffsetArrays, Test
using LoopVectorization: Static
# T = Float64; r = -1:1;
# T = Float32; r = -1:1;


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
    # # rng1k, rng2k = axes(skern);
    # rng1,  rng2  = R.indices;
    # tmp = z; i = 2; j = 2;
    # ls1 = LoopVectorization.@avx_debug for jk in rng2, ik in rng1
    #     tmp += A[i+ik,j+jk]*kern[ik,jk]
    # end;
    # ls1

    # out = out1;
    # C = At';
    # ls2d = LoopVectorization.@avx_debug for j in axes(out1,2), i in axes(out1,1)
    #         tmp = zero(eltype(out1))
    #         for jk in axes(kern,2), ik in axes(kern,1)
    #             tmp += A[i+ik,j+jk]*kern[ik,jk]
    #         end
    #         out1[i,j] = tmp
    # end;
    # LoopVectorization.choose_order(ls2d)
    # ls2ds = LoopVectorization.@avx_debug for j in axes(out1,2), i in axes(out1,1)
    #         tmp = zero(eltype(out1))
    #         for jk in axes(skern,2), ik in axes(skern,1)
    #             tmp += A[i+ik,j+jk]*skern[ik,jk]
    #         end
    #         out1[i,j] = tmp
    # end;
    # LoopVectorization.choose_order(ls2ds)

    
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


    struct SizedOffsetMatrix{T,LR,UR,LC,UC} <: DenseMatrix{T}
        data::Matrix{T}
    end
    Base.size(::SizedOffsetMatrix{<:Any,LR,UR,LC,UC}) where {LR,UR,LC,UC} = (UR-LR+1,UC-LC+1)
    Base.axes(::SizedOffsetMatrix{T,LR,UR,LC,UC}) where {T,LR,UR,LC,UC} = (Static{LR}():Static{UR}(),Static{LC}():Static{UC}())
    Base.parent(A::SizedOffsetMatrix) = A.data
    Base.unsafe_convert(::Type{Ptr{T}}, A::SizedOffsetMatrix{T}) where {T} = pointer(A.data)
    ArrayInterface.contiguous_axis(::Type{<:SizedOffsetMatrix}) = ArrayInterface.One()
    ArrayInterface.contiguous_batch_size(::Type{<:SizedOffsetMatrix}) = ArrayInterface.Zero()
    ArrayInterface.stride_rank(::Type{<:SizedOffsetMatrix}) = (ArrayInterface.StaticInt(1), ArrayInterface.StaticInt(2))
    function ArrayInterface.strides(A::SizedOffsetMatrix{T,LR,UR,LC,UC}) where {T,LR,UR,LC,UC}
        (Static{1}(), (Static{UR}() - Static{LR}() + Static{1}()))
    end
    ArrayInterface.offsets(A::SizedOffsetMatrix{T,LR,UR,LC,UC}) where {T,LR,UR,LC,UC} = (Static{LR}(), Static{LC}())
    Base.getindex(A::SizedOffsetMatrix, i, j) = LoopVectorization.vload(LoopVectorization.stridedpointer(A), (i,j))
    function avx2dunrolled!(out::AbstractMatrix, A::AbstractMatrix, kern::SizedOffsetMatrix{T,-1,1,-1,1}) where {T}
        Base.Cartesian.@nexprs 3 jk -> Base.Cartesian.@nexprs 3 ik -> kern_ik_jk = kern[ik-2,jk-2]
        # Manually unpack the OffsetArray
        @avx for j in axes(out,2), i in axes(out,1)
            tmp_0 = zero(eltype(out))
            Base.Cartesian.@nexprs 3 jk -> Base.Cartesian.@nexprs 3 ik -> tmp_{ik+(jk-1)*3} = A[(ik-2)+i,(jk-2) + j*1] * kern_ik_jk + tmp_{ik+(jk-1)*3-1}
            out[i,j] = tmp_9
        end
        out
    end
    function avx2dunrolled2x2!(out::AbstractMatrix, A::AbstractMatrix, kern::SizedOffsetMatrix{T,-1,1,-1,1}) where {T}
        # rng1,  rng2  = axes(out)
        # Manually unpack the OffsetArray
        # @avx for j in rng2, i in rng1
        @avx unroll=(2,2) for j in axes(out,2), i in axes(out,1)
            Base.Cartesian.@nexprs 3 jk -> Base.Cartesian.@nexprs 3 ik -> kern_ik_jk = kern[ik - 2, jk + (-2)]
            tmp_0 = zero(eltype(out))
            j1 = j * 1
            Base.Cartesian.@nexprs 3 jk -> Base.Cartesian.@nexprs 3 ik -> tmp_{ik+(jk-1)*3} = A[i + (ik-2), (jk-2) + j1] * kern_ik_jk + tmp_{ik+(jk-1)*3-1}
            out[i,j] = tmp_9
        end
        out
    end
    function avx2dunrolled3x3!(out::AbstractMatrix, A::AbstractMatrix, kern::SizedOffsetMatrix{T,-1,1,-1,1}) where {T}
        rng1,  rng2  = axes(out)
        Base.Cartesian.@nexprs 3 jk -> Base.Cartesian.@nexprs 3 ik -> kern_ik_jk = kern[ik-2,jk-2]
        # Manually unpack the OffsetArray
        @avx unroll=(3,3) for j in rng2, i in rng1
            tmp_0 = zero(eltype(out))
            Base.Cartesian.@nexprs 3 jk -> Base.Cartesian.@nexprs 3 ik -> tmp_{ik+(jk-1)*3} = A[(ik-2) + i, j*1 + (jk-2)] * kern_ik_jk + tmp_{ik+(jk-1)*3-1}
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
    # lsuq
    
    # using LoopVectorization, OffsetArrays
    # T = Float64
    # A = rand(T, 100, 100);
    # kern = OffsetArray(rand(T, 3, 3), -1:1, -1:1);
    # out = OffsetArray(similar(A, size(A).-2), 1, 1);   # stay away from the edges of A
    # lsgeneric = LoopVectorization.@avx_debug for I in CartesianIndices(out)
    #        tmp = zero(eltype(out))
    #        for J in CartesianIndices(kern)
    #            tmp += A[I+J]*kern[J]
    #        end
    #        out[I] = tmp
    #    end;
    # LoopVectorization.choose_order(lsgeneric)
    # # out = out1;
#     lsgenerics = LoopVectorization.@avx_debug for I in CartesianIndices(out)
#            tmp = zero(eltype(out))
#            for J in CartesianIndices(skern)
#                tmp += A[I+J]*kern[J]
#            end
#            out[I] = tmp
#        end;
#     LoopVectorization.choose_order(lsgenerics)
# @macroexpand @avx for I in R
#            tmp = z
#            for J in Rk
#                tmp += A[I+J]*kern[J]
#            end
#            out[I] = tmp
#        end

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
    function avxgeneric2!(out, A, kern)
        @avx for I in CartesianIndices(out)
            tmp = zero(eltype(out))
            for J in CartesianIndices(kern)
                tmp += A[I+J]*kern[J]
            end
            out[I] = tmp
        end
        out
    end

    for T ∈ (Float32, Float64)
        @show T, @__LINE__
        Abase = fill(T(NaN), 200, 200);
        # out of bounds reads load NaNs, poisoning results leading to test failure.
        A = view(Abase, 51:152, 51:152);
        A .= rand.();
        Atbase = copy(Abase');
        At = view(Atbase, 51:152, 51:152);
        for r ∈ (-1:1, -2:2)
            @show r
            fr = first(r); lr = last(r);
            kern = OffsetArray(rand(T, length(r), length(r)), r, r);
            # We test parent equality so that an accidental write out of bounds leading to test failure.
            out1 = OffsetArray(view(fill(T(-123456.789), size(A) .+ 32), (1+lr:100-lr) .+ 32, (1+lr:100-lr) .+ 32), lr, lr);   # stay away from the edges of A
            # out1 = OffsetArray(similar(A, size(A).-2), 1, 1);   # stay away from the edges of A
            out2 = deepcopy(out1); out3 = deepcopy(out1); out4 = deepcopy(out1);
            skern = SizedOffsetMatrix{T,fr,lr,fr,lr}(parent(kern));

            old2d!(out1, A, kern);
            avx2d!(out2, A, kern);
            @test parent(out1) ≈ parent(out2)

            avx2douter!(out3, A, kern);
            @test parent(out1) ≈ parent(out3)

            fill!(out2, NaN); avx2d!(out2, A, skern);
            @test parent(out1) ≈ parent(out2)

            fill!(out2, NaN); avx2douter!(out2, At', kern);
            @test parent(out1) ≈ parent(out2)

            fill!(out2, NaN); avx2douter!(out2', A, kern);
            @test parent(out1) ≈ parent(out2)'

            fill!(out2, NaN); avx2douter!(out2', At', kern);
            @test parent(out1) ≈ parent(out2)'

            fill!(out3, NaN); avx2douter!(out3, A, skern);
            @test parent(out1) ≈ parent(out3)

            if r == -1:1
                fill!(out3, NaN); avx2dunrolled!(out3, A, skern);
                @test parent(out1) ≈ parent(out3)

                fill!(out3, NaN); avx2dunrolled2x2!(out3, A, skern);
                @test parent(out1) ≈ parent(out3)

                fill!(out3, NaN); avx2dunrolled3x3!(out3, A, skern);
                @test parent(out1) ≈ parent(out3)
            end
            
            @test parent(avxgeneric!(out4, A, kern)) ≈ parent(out1)
            fill!(out4, NaN);
            @test parent(avxgeneric!(out4, A, skern)) ≈ parent(out1)

            fill!(out4, NaN); @test parent(avxgeneric2!(out4, A, kern)) ≈ parent(out1)
            fill!(out4, NaN); @test parent(avxgeneric2!(out4, A, skern)) ≈ parent(out1)
            fill!(out4, NaN); @test parent(avxgeneric2!(out4, At', kern)) ≈ parent(out1)
            fill!(out4, NaN); @test parent(avxgeneric2!(out4, At', skern)) ≈ parent(out1)
            fill!(out4, NaN); @test parent(avxgeneric2!(out4', A, kern)')' ≈ parent(out1)
            fill!(out4, NaN); @test parent(avxgeneric2!(out4', A, skern)')' ≈ parent(out1)
            fill!(out4, NaN); @test parent(avxgeneric2!(out4', At', kern)')' ≈ parent(out1)
            fill!(out4, NaN); @test parent(avxgeneric2!(out4', At', skern)')' ≈ parent(out1)
        end
    end


end
