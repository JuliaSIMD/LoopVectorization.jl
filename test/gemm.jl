@testset "GEMM" begin
    # using LoopVectorization, LinearAlgebra, Test; T = Float64
    if LoopVectorization.cache_linesize() == 64
        Unum, Tnum = LoopVectorization.register_count() == 16 ? (2, 6) : (3, 9)
    else
        Unum, Tnum = LoopVectorization.register_count() == 16 ? (2, 6) : (4, 6)
    end
    Unumt, Tnumt = LoopVectorization.register_count() == 16 ? (2, 6) : (5, 5)
    if LoopVectorization.register_count() != 8
        @test @inferred(LoopVectorization.matmul_params()) == (Unum, Tnum)
    end
    AmulBtq1 = :(for m ∈ axes(A,1), n ∈ axes(B,2)
                 C[m,n] = zeroB
                 for k ∈ axes(A,2)
                 C[m,n] += A[m,k] * B[n,k]
                 end
                 end);
    lsAmulBt1 = LoopVectorization.loopset(AmulBtq1);
    if LoopVectorization.register_count() != 8
        @test LoopVectorization.choose_order(lsAmulBt1) == (Symbol[:n,:m,:k], :m, :n, :m, Unum, Tnum)
    end
    AmulBq1 = :(for n ∈ axes(B,2), m ∈ axes(A,1)
                C[m,n] = 0.0
                for k ∈ axes(A,2)
                C[m,n] += A[m,k] * B[k,n]
                end
                end);
    lsAmulB1 = LoopVectorization.loopset(AmulBq1);
    if LoopVectorization.register_count() != 8
        @test LoopVectorization.choose_order(lsAmulB1) == (Symbol[:n,:m,:k], :m, :n, :m, Unum, Tnum)
    end
    AmulBq2 = :(for m ∈ 1:M, n ∈ 1:N
                C[m,n] = zero(eltype(B))
                for k ∈ 1:K
                C[m,n] += A[m,k] * B[k,n]
                end
                end)
    lsAmulB2 = LoopVectorization.loopset(AmulBq2);
    if LoopVectorization.register_count() != 8
        @test LoopVectorization.choose_order(lsAmulB2) == (Symbol[:n,:m,:k], :m, :n, :m, Unum, Tnum)
    end
    AmulBq3 = :(for m ∈ axes(A,1), n ∈ axes(B,2)
                ΔCₘₙ = zero(eltype(C))
                for k ∈ axes(A,2)
                ΔCₘₙ += A[m,k] * B[k,n]
                end
                C[m,n] += ΔCₘₙ
                end)
    lsAmulB3 = LoopVectorization.loopset(AmulBq3);
    if LoopVectorization.register_count() != 8
        @test LoopVectorization.choose_order(lsAmulB3) == (Symbol[:n,:m,:k], :m, :n, :m, Unum, Tnum)
    end
    if LoopVectorization.register_count() != 8
        for (fA,fB,v,Un,Tn) ∈ [(identity,identity,:m,Unum,Tnum),(adjoint,identity,:k,Unumt,Tnumt),(identity,adjoint,:m,Unum,Tnum),(adjoint,adjoint,:n,Unum,Tnum)]
            A = fA(rand(2,2))
            B = fB(rand(2,2))
            C = similar(A)
            ls = LoopVectorization.@turbo_debug for m ∈ axes(A,1), n ∈ axes(B,2)
                ΔCₘₙ = zero(eltype(C))
                for k ∈ axes(A,2)
                    ΔCₘₙ += A[m,k] * B[k,n]
                end
                C[m,n] += ΔCₘₙ
            end
            (m, n) = v === :n ? (:n, :m) : (:m, :n)
            @test LoopVectorization.choose_order(ls) == (Symbol[:n,:m,:k], m, n, v, Un, Tn)
        end
    end
    function AmulB!(C, A, B)
        C .= 0
        for k ∈ axes(A,2), j ∈ axes(B,2)
            @simd ivdep for i ∈ axes(A,1)
                @inbounds C[i,j] += A[i,k] * B[k,j]
            end
        end
    end
    function AmulBavx1!(C, A, B)
        dM, rM = divrem(size(C,1), 3)
        dN, rN = divrem(size(C,2), 3)
        dK, rK = divrem(size(B,1), 3)
        @turbo for m ∈ 1:3*dM + rM, n ∈ 1:3*dN + rN
            Cₘₙ = zero(eltype(C))
            for k ∈ 1:3*dK + rK
                Cₘₙ += A[m,k] * B[k,n]
            end
            C[m,n] = Cₘₙ
        end
    end
    function AmulBavx2!(C, A, B)
        z = zero(eltype(C))
        @turbo unroll=(2,1) for m ∈ axes(A,1), n ∈ axes(B,2)
            C[m,n] = z
            for k ∈ axes(A,2)
                C[m,n] += A[m,k] * B[k,n]
            end
        end
    end
    function AmulBavx3!(C, A, B)
        @turbo unroll=(2,2) for m ∈ axes(A,1), n ∈ axes(B,2)
            C[m,n] = zero(eltype(C))
            for k ∈ axes(A,2)
                C[m,n] += A[m,k] * B[k,n]
            end
        end
    end
    myzero(A) = zero(eltype(A))
    # function AmulBavx4!(C, A, B)
    #     @turbo for m ∈ axes(A,1), n ∈ axes(B,2)
    #         C[m,n] = myzero(C)
    #         for k ∈ axes(A,2)
    #             C[m,n] += A[m,k] * B[k,n]
    #         end
    #     end
    # end
    # C = Cs; A = Ats'; B = Bs; factor = 1;
    # ls = LoopVectorization.@turbo_debug for m ∈ axes(A,1), n ∈ axes(B,2)
    #         ΔCₘₙ = zero(eltype(C))
    #         for k ∈ axes(A,2)
    #             ΔCₘₙ += A[m,k] * B[k,n]
    #         end
    #         C[m,n] += ΔCₘₙ * factor
    #     end;
    function AmuladdBavx!(C, A, B, α = one(eltype(C)))
        @turbo unroll=(2,2) for m ∈ indices((A,C),1), n ∈ indices((B,C),2)
            ΔCₘₙ = zero(eltype(C))
            for k ∈ indices((A,B),(2,1))
                ΔCₘₙ += A[m,k] * B[k,n]
            end
            C[m,n] += α * ΔCₘₙ
        end
    end
    function AmuladdBavx!(C, A, B, α, β)# = zero(eltype(C)))
        @turbo unroll=(1,1) for m ∈ axes(A,1), n ∈ axes(B,2)
            ΔCₘₙ = zero(eltype(C))
            for k ∈ axes(A,2)
                ΔCₘₙ += A[m,k] * B[k,n]
            end
            C[m,n] = α * ΔCₘₙ + β * C[m,n]
        end
    end
    Amuladdq = :(for m ∈ axes(A,1), n ∈ axes(B,2)
                 ΔCₘₙ = zero(eltype(C))
                 for k ∈ axes(A,2)
                 ΔCₘₙ += A[m,k] * B[k,n]
                 end
                 C[m,n] = α * ΔCₘₙ + β * C[m,n]
                 end);
    lsAmuladd = LoopVectorization.loopset(Amuladdq);
    if LoopVectorization.register_count() != 8
        @test LoopVectorization.choose_order(lsAmuladd) == (Symbol[:n,:m,:k], :m, :n, :m, Unum, Tnum)
    end
    Atmuladdq = :(for m ∈ axes(A,2), n ∈ axes(B,2)
                 ΔCₘₙ = zero(eltype(C))
                 for k ∈ axes(A,1)
                 ΔCₘₙ += A[k,m] * B[k,n]
                  end
                  C[m,n] += α * ΔCₘₙ
                 end);
    lsAtmuladd = LoopVectorization.loopset(Atmuladdq);
    # LoopVectorization.lower(lsAtmuladd, 2, 2)
    # lsAmuladd.operations
    # LoopVectorization.loopdependencies.(lsAmuladd.operations)
    # LoopVectorization.reduceddependencies.(lsAmuladd.operations)
    if LoopVectorization.register_count() != 8
        @test LoopVectorization.choose_order(lsAtmuladd) == (Symbol[:n,:m,:k], :m, :n, :k, Unumt, Tnumt)
    end

    function AmulB_avx1!(C, A, B)
        @_avx unroll=(2,2) for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
            Cₘₙ = zero(eltype(C))
            for k ∈ axes(A,2)
                Cₘₙ += A[m,k] * B[k,n]
            end
            C[m,n] = Cₘₙ
        end
    end
    fq = :(for m ∈ axes(A,1), n ∈ axes(B,2)
           Cₘₙ = zero(eltype(C))
           for k ∈ axes(A,2)
           Cₘₙ += A[m,k] * B[k,n]
           end
           C[m,n] = Cₘₙ
           end);
    # exit()
    #         using LoopVectorization, Test
    #         T = Float64
    #         M = 77
    #         A = rand(M, M); B = rand(M, M); C = similar(A);
    function AmulB_avx2!(C, A, B)
        z = zero(eltype(C))
        @_avx unroll=(2,2) for m ∈ axes(A,1), n ∈ axes(B,2)
            C[m,n] = z
            for k ∈ axes(A,2)
                C[m,n] += A[m,k] * B[k,n]
            end
        end
    end
    # AmulB_avx2!(C, A, B)
    # gq = :(for m ∈ axes(A,1), n ∈ axes(B,2)
    # C[m,n] = z
    # for k ∈ axes(A,2)
    # C[m,n] += A[m,k] * B[k,n]
    # end
    # end);
    # ls = LoopVectorization.loopset(gq);
    # # ls.preamble_symsym
    # ls.operations[1]
    function AmulB_avx3!(C, A, B)
        Kmin = first(axes(A,2)); Kmax = last(axes(A,2))
        @_avx unroll=(2,2) for m ∈ axes(A,1), n ∈ axes(B,2)
            C[m,n] = zero(eltype(C))
            for k ∈ Kmin:Kmax
                C[m,n] += A[m,k] * B[k,n]
            end
        end
    end
    # function AmulB_avx4!(C, A, B)
    #     @_avx for m ∈ axes(A,1), n ∈ axes(B,2)
    #         C[m,n] = myzero(C)
    #         for k ∈ axes(A,2)
    #             C[m,n] += A[m,k] * B[k,n]
    #         end
    #     end
    # end
    # q = :(for m ∈ axes(A,1), n ∈ axes(B,2)
    #       C[m,n] = myzero(C)
    #       for k ∈ axes(A,2)
    #       C[m,n] += A[m,k] * B[k,n]
    #       end
    #       end)
    # ls = LoopVectorization.loopset(q);
    function AmuladdB_avx!(C, A, B, factor = 1)
        @_avx unroll=(2,2) for m ∈ axes(A,1), n ∈ axes(B,2)
            ΔCₘₙ = zero(eltype(C))
            for k ∈ axes(A,2)
                ΔCₘₙ += A[m,k] * B[k,n]
            end
            C[m,n] += ΔCₘₙ * factor
        end
    end

    function AmulB2x2avx!(C, A, B)
        @turbo unroll=(2,2) for m ∈ axes(A,1), n ∈ axes(B,2)
            ΔCₘₙ = zero(eltype(C))
            for k ∈ axes(A,2)
                ΔCₘₙ += A[m,k] * B[k,n]
            end
            C[m,n] = ΔCₘₙ
        end
    end
    function AmulB2x2_avx!(C, A, B)
        @_avx unroll=(2,2) for m ∈ axes(A,1), n ∈ axes(B,2)
            ΔCₘₙ = zero(eltype(C))
            for k ∈ axes(A,2)
                ΔCₘₙ += A[m,k] * B[k,n]
            end
            C[m,n] = ΔCₘₙ
        end
    end

    # function AtmulB!(C, A, B)
    #     for j ∈ axes(C,2), i ∈ axes(C,1)
    #         Cᵢⱼ = zero(eltype(C))
    #         @simd ivdep for k ∈ axes(A,1)
    #             @inbounds Cᵢⱼ += A[k,i] * B[k,j]
    #         end
    #         C[i,j] = Cᵢⱼ
    #     end
    # end
    AtmulBq = :(for n ∈ axes(C,2), m ∈ axes(C,1)
                Cₘₙ = zero(eltype(C))
                for k ∈ axes(A,1)
                Cₘₙ += A[k,m] * B[k,n]
                end
                C[m,n] = Cₘₙ
                end)
    lsAtmulB = LoopVectorization.loopset(AtmulBq);
    if LoopVectorization.register_count() != 8
        @test LoopVectorization.choose_order(lsAtmulB) == (Symbol[:n,:m,:k], :n, :m, :k, Unumt, Tnumt)
    end
    function AtmulBavx1!(C, A, B)
        @turbo for n ∈ axes(C,2), m ∈ axes(C,1)
            Cₘₙ = zero(eltype(C))
            for k ∈ axes(A,1)
                Cₘₙ += A[k,m] * B[k,n]
            end
            C[m,n] = Cₘₙ
        end
    end
    Atq = :(for n ∈ axes(C,2), m ∈ axes(C,1)
            Cₘₙ = zero(eltype(C))
            for k ∈ axes(A,1)
            Cₘₙ += A[k,m] * B[k,n]
            end
            C[m,n] += Cₘₙ * factor
            end);
    atls = LoopVectorization.loopset(Atq);
    # LoopVectorization.operations(atls)
    # LoopVectorization.loopdependencies.(operations(atls))
    # LoopVectorization.reduceddependencies.(operations(atls))
    function AtmulB_avx1!(C, A, B)
        @_avx unroll=(2,2) for n ∈ axes(C,2), m ∈ axes(C,1)
            Cₘₙ = zero(eltype(C))
            for k ∈ axes(A,1)
                Cₘₙ += A[k,m] * B[k,n]
            end
            C[m,n] = Cₘₙ
        end
    end
    function AtmulBavx2!(C, A, B)
        M, N = size(C); K = size(B,1)
        @assert size(C, 1) == size(A, 2)
        @assert size(C, 2) == size(B, 2)
        @assert size(A, 1) == size(B, 1)
        # When the @turbo macro is available, this code is faster:
        z = zero(eltype(C))
        @turbo unroll=(2,2) for n in axes(C,2), m in axes(C,1)
            Cmn = z
            for k in axes(A,1)
                Cmn += A[k,m] * B[k,n]
            end
            C[m,n] = Cmn
        end
        return C
    end
    function AtmulB_avx2!(C, A, B)
        M, N = size(C); K = size(B,1)
        @assert size(C, 1) == size(A, 2)
        @assert size(C, 2) == size(B, 2)
        @assert size(A, 1) == size(B, 1)
        # When the @turbo macro is available, this code is faster:
        z = zero(eltype(C))
        @_avx for n in axes(C,2), m in axes(C,1)
            Cmn = z
            for k in axes(A,1)
                Cmn += A[k,m] * B[k,n]
            end
            C[m,n] = Cmn
        end
        return C
    end
    function rank2AmulB!(C, Aₘ, Aₖ, B)
        @inbounds for m ∈ axes(C,1), n ∈ axes(C,2)
            Cₘₙ = zero(eltype(C))
            @fastmath for k ∈ axes(B,1)
                Cₘₙ += (Aₘ[m,1]*Aₖ[1,k]+Aₘ[m,2]*Aₖ[2,k]) * B[k,n]
            end
            C[m,n] = Cₘₙ
        end
    end
    r2ambq = :(for m ∈ axes(C,1), n ∈ axes(C,2)
               Cₘₙ = zero(eltype(C))
               for k ∈ axes(B,1)
               Cₘₙ += (Aₘ[m,1]*Aₖ[1,k]+Aₘ[m,2]*Aₖ[2,k]) * B[k,n]
               end
               C[m,n] = Cₘₙ
               end);
    lsr2amb = LoopVectorization.loopset(r2ambq);
    if LoopVectorization.register_count() == 32
        if LoopVectorization.cache_linesize() == LoopVectorization.register_size()
            @test LoopVectorization.choose_order(lsr2amb) == ([:n, :m, :k], :m, :n, :m, 3, 7)
        else
            @test LoopVectorization.choose_order(lsr2amb) == ([:m, :n, :k], :m, :n, :m, 3, 7)
        end
    elseif LoopVectorization.register_count() == 16
        # @test LoopVectorization.choose_order(lsr2amb) == ([:m, :n, :k], :m, :n, :m, 1, 6)
        # @test LoopVectorization.choose_order(lsr2amb) == ([:m, :n, :k], :m, :n, :m, 2, 4)
        @test LoopVectorization.choose_order(lsr2amb) == ([:m, :n, :k], :n, :m, :m, 3, 3)
    end
    function rank2AmulBavx!(C, Aₘ, Aₖ, B)
        @turbo for m ∈ axes(C,1), n ∈ axes(C,2)
            Cₘₙ = zero(eltype(C))
            for k ∈ axes(B,1)
                Cₘₙ += (Aₘ[m,1]*Aₖ[1,k]+Aₘ[m,2]*Aₖ[2,k]) * B[k,n]
            end
            C[m,n] = Cₘₙ
        end
    end
    function rank2AmulB_avx!(C, Aₘ, Aₖ, B)
        @_avx for m ∈ axes(C,1), n ∈ axes(C,2)
            Cₘₙ = zero(eltype(C))
            for k ∈ axes(B,1)
                Cₘₙ += (Aₘ[m,1]*Aₖ[1,k]+Aₘ[m,2]*Aₖ[2,k]) * B[k,n]
            end
            C[m,n] = Cₘₙ
        end
    end
    function rank2AmulBavx_noinline!(C, Aₘ, Aₖ, B)
        @turbo inline=false for m ∈ axes(C,1), n ∈ axes(C,2)
            Cₘₙ = zero(eltype(C))
            for k ∈ axes(B,1)
                Cₘₙ += (Aₘ[m,1]*Aₖ[1,k]+Aₘ[m,2]*Aₖ[2,k]) * B[k,n]
            end
            C[m,n] = Cₘₙ
        end
    end

    function mulCAtB_2x2blockavx!(C, A, B)
        M, N = size(C); K = size(B,1)
        @assert size(C, 1) == size(A, 2)
        @assert size(C, 2) == size(B, 2)
        @assert size(A, 1) == size(B, 1)
        T = eltype(C)
        for m ∈ 1:2:(M & -2)
            m1 = m + 1
            for n ∈ 1:2:(N & -2)
                n1 = n + 1
                C11, C21, C12, C22 = zero(T), zero(T), zero(T), zero(T)
                @turbo inline=true for k ∈ 1:K
                    C11 += A[k,m] * B[k,n]
                    C21 += A[k,m1] * B[k,n]
                    C12 += A[k,m] * B[k,n1]
                    C22 += A[k,m1] * B[k,n1]
                end
                C[m,n] = C11
                C[m1,n] = C21
                C[m,n1] = C12
                C[m1,n1] = C22
            end
            if isodd(N)
                C1n = 0.0
                C2n = 0.0
                @turbo inline=true for k ∈ 1:K
                    C1n += A[k,m] * B[k,N]
                    C2n += A[k,m1] * B[k,N]
                end
                C[m,N] = C1n
                C[m1,N] = C2n
            end
        end
        if isodd(M)
            for n ∈ 1:2:(N & -2)
                n1 = n + 1
                Cm1, Cm2 = zero(T), zero(T)
                @turbo inline=true for k ∈ 1:K
                    Cm1 += A[k,M] * B[k,n] 
                    Cm2 += A[k,M] * B[k,n1] 
                end
                C[M,n] = Cm1
                C[M,n1] = Cm2
            end
            if isodd(N)
                Cmn = 0.0
                @turbo inline=true for k ∈ 1:K
                    Cmn += A[k,M] * B[k,N]
                end
                C[M,N] = Cmn
            end
        end
        return C
    end
    function mulCAtB_2x2block_avx!(C, A, B)
        M, N = size(C); K = size(B,1)
        @assert size(C, 1) == size(A, 2)
        @assert size(C, 2) == size(B, 2)
        @assert size(A, 1) == size(B, 1)
        T = eltype(C)
        for m ∈ 1:2:(M & -2)
            m1 = m + 1
            for n ∈ 1:2:(N & -2)
                n1 = n + 1
                C11, C21, C12, C22 = zero(T), zero(T), zero(T), zero(T)
                @_avx for k ∈ 1:K
                    C11 += A[k,m] * B[k,n] 
                    C21 += A[k,m1] * B[k,n] 
                    C12 += A[k,m] * B[k,n1] 
                    C22 += A[k,m1] * B[k,n1]
                end
                C[m,n] = C11
                C[m1,n] = C21
                C[m,n1] = C12
                C[m1,n1] = C22
            end
            if isodd(N)
                C1n = 0.0
                C2n = 0.0
                @_avx for k ∈ 1:K
                    C1n += A[k,m] * B[k,N]
                    C2n += A[k,m1] * B[k,N]
                end
                C[m,N] = C1n
                C[m1,N] = C2n
            end
        end
        if isodd(M)
            for n ∈ 1:2:(N & -2)
                n1 = n + 1
                Cm1, Cm2 = zero(T), zero(T)
                @_avx for k ∈ 1:K
                    Cm1 += A[k,M] * B[k,n] 
                    Cm2 += A[k,M] * B[k,n1] 
                end
                C[M,n] = Cm1
                C[M,n1] = Cm2
            end
            if isodd(N)
                Cmn = 0.0
                @_avx for k ∈ 1:K
                    Cmn += A[k,M] * B[k,N]
                end
                C[M,N] = Cmn
            end
        end
        return C
    end
    
    function mulCAtB_2x2blockavx_noinline!(C, A, B)
        M, N = size(C); K = size(B,1)
        @assert size(C, 1) == size(A, 2)
        @assert size(C, 2) == size(B, 2)
        @assert size(A, 1) == size(B, 1)
        T = eltype(C)
        for m ∈ 1:2:(M & -2)
            m1 = m + 1
            for n ∈ 1:2:(N & -2)
                n1 = n + 1
                C11, C21, C12, C22 = zero(T), zero(T), zero(T), zero(T)
                @turbo inline=false for k ∈ 1:K
                    C11 += A[k,m] * B[k,n] 
                    C21 += A[k,m1] * B[k,n] 
                    C12 += A[k,m] * B[k,n1] 
                    C22 += A[k,m1] * B[k,n1]
                end
                C[m,n] = C11
                C[m1,n] = C21
                C[m,n1] = C12
                C[m1,n1] = C22
            end
            if isodd(N)
                C1n = 0.0
                C2n = 0.0
                @turbo inline=false for k ∈ 1:K
                    C1n += A[k,m] * B[k,N]
                    C2n += A[k,m1] * B[k,N]
                end
                C[m,N] = C1n                    
                C[m1,N] = C2n                    
            end
        end
        if isodd(M)
            for n ∈ 1:2:(N & -2)
                n1 = n + 1
                Cm1, Cm2 = zero(T), zero(T)
                @turbo inline=false for k ∈ 1:K
                    Cm1 += A[k,M] * B[k,n] 
                    Cm2 += A[k,M] * B[k,n1] 
                end
                C[M,n] = Cm1
                C[M,n1] = Cm2
            end
            if isodd(N)
                Cmn = 0.0
                @turbo inline=false for k ∈ 1:K
                    Cmn += A[k,M] * B[k,N]
                end
                C[M,N] = Cmn
            end
        end
        return C
    end
  function dense!(f::F, C, A, B) where {F}
    Kp1 = LoopVectorization.size(A, LoopVectorization.StaticInt(2))
    K = Kp1 - LoopVectorization.StaticInt(1)
    @turbo for n ∈ indices((B,C),2), m ∈ indices((A,C),1)
      Cmn = zero(eltype(C))
      for k ∈ 1:K
        Cmn += A[m,k] * B[k,n]
      end
      C[m,n] = f(Cmn + A[m,Kp1])
    end
  end

    # TODO: add fast=false option to `@turbo`
    # function gemm_accurate!(C, A, B)
    #     @turbo for n in axes(C,2), m in axes(C,1)
    #         Cmn_hi = zero(eltype(C))
    #         Cmn_lo = zero(eltype(C))
    #         for k in axes(B,1)
    #             hiprod = vmul(A[m,k], B[k,n])
    #             loprod = vfmsub(A[m,k], B[k,n], hiprod)
    #             hi_ts = vadd(hiprod, Cmn_hi)
    #             a1_ts = vsub(hi_ts, Cmn_hi)
    #             b1_ts = vsub(hi_ts, a1_ts)
    #             lo_ts = vadd(vsub(hiprod, a1_ts), vsub(Cmn_hi, b1_ts))
    #             thi = vadd(loprod, Cmn_lo)
    #             a1_t = vsub(thi, Cmn_lo)
    #             b1_t = vsub(thi, a1_t)
    #             tlo = vadd(vsub(loprod, a1_t), vsub(Cmn_lo, b1_t))
    #             c1 = vadd(lo_ts, thi)
    #             hi_ths = vadd(hi_ts, c1)
    #             lo_ths = vsub(c1, vsub(hi_ths, hi_ts))
    #             c2 = vadd(tlo, lo_ths)
    #             Cmn_hi = vadd(hi_ths, c2)
    #             Cmn_lo = vsub(c2, vsub(Cmn_hi, hi_ths))
    #         end
    #         C[m,n] = Cmn_hi
    #     end
    # end
    function AB_plus_BA!(du, u, mat)
        @assert size(u, 1) == size(u, 2) == size(mat, 1) == size(mat, 2)
        for i2 in 1:size(u, 2), i1 in 1:size(u, 1)
            for sum_idx in 1:size(u, 1)
                du[i1, i2] += mat[i1, sum_idx] * u[sum_idx, i2] + mat[i2, sum_idx] * u[i1, sum_idx]
            end
        end
        return nothing
    end

    function AB_plus_BA_avx!(du, u, mat)
        @assert size(u, 1) == size(u, 2) == size(mat, 1) == size(mat, 2)
        @turbo for i2 in 1:size(u, 2), i1 in 1:size(u, 1)
            for sum_idx in 1:size(u, 1)
                du[i1, i2] += mat[i1, sum_idx] * u[sum_idx, i2] + mat[i2, sum_idx] * u[i1, sum_idx]
            end
        end
        return nothing
    end

    function threegemms!(Ab, Bb, Cb, A, B, C)
        M, N = size(Cb); K = size(B,1)
        @turbo for m in 1:M, k in 1:K, n in 1:N
            Ab[m,k] += C[m,n] * B[k,n]
            Bb[k,n] += A[m,k] * C[m,n]
            Cb[m,n] += A[m,k] * B[k,n]
        end
    end
    # M = 77;
    # A = rand(M,M); B = rand(M,M); C = similar(A);
    # mulCAtB_2x2block_avx!(C,A,B)
    # using LoopVectorization
    # mul2x2q = :(for k ∈ 1:K
    # C11 += A[k,m] * B[k,n] 
    # C21 += A[k,m1] * B[k,n] 
    # C12 += A[k,m] * B[k,n1] 
    # C22 += A[k,m1] * B[k,n1]
    # end)
    # lsmul2x2q = LoopVectorization.loopset(mul2x2q)

    lsAtmulBt8 = :(for m ∈ 1:8, n ∈ 1:8
    ΔCₘₙ = zero(eltype(C))
    for k ∈ 1:8
    ΔCₘₙ += A[k,m] * B[n,k]
    end
    C[m,n] = ΔCₘₙ
    end) |> LoopVectorization.loopset;
    if LoopVectorization.register_count() == 32
      if LoopVectorization.register_size() == 64
        @test LoopVectorization.choose_order(lsAtmulBt8) == ([:n, :m, :k], :m, :n, :m, 1, 8)
        # @test LoopVectorization.choose_order(lsAtmulBt8) == ([:n, :m, :k], :k, :n, :m, 1, 8)
      elseif LoopVectorization.register_size() == 16
        @test LoopVectorization.choose_order(lsAtmulBt8) == ([:n, :m, :k], :m, :n, :m, 2, 8)
      end            
    elseif LoopVectorization.register_count() == 16
      # vectorizing `n` is better, as we unroll `m`, neaning `C` can use shuffle stores
      # as we don't unroll `k`, we can't use shuffle loads from `C`
      @test LoopVectorization.choose_order(lsAtmulBt8) == ([:n, :m, :k], :n, :m, :n, 2, 4)
    elseif LoopVectorization.register_count() == 8
      @test LoopVectorization.choose_order(lsAtmulBt8) == ([:n, :m, :k], :m, :n, :m, 1, 4)
    end
    
    struct SizedMatrix{M,N,T} <: DenseMatrix{T}
        data::Matrix{T}
        function SizedMatrix{M,N}(data::Matrix{T}) where {M,N,T}
            @assert (M,N) === size(data)
            new{M,N,T}(data)
        end
    end
    Base.parent(A::SizedMatrix) = A.data
    Base.IndexStyle(::Type{<:SizedMatrix}) = Base.IndexLinear()
    Base.@propagate_inbounds Base.getindex(A::SizedMatrix, i::Int) = getindex(parent(A), i)
    Base.@propagate_inbounds Base.setindex!(A::SizedMatrix, v, i::Int) = setindex!(parent(A), v, i)
    Base.@propagate_inbounds Base.getindex(A::SizedMatrix, i::CartesianIndex) = getindex(parent(A), i + oneunit(i))
    Base.@propagate_inbounds Base.setindex!(A::SizedMatrix, v, i::CartesianIndex) = setindex!(parent(A), v, i + oneunit(i))
    Base.@propagate_inbounds Base.getindex(A::SizedMatrix, i::Int, j::Int) = getindex(parent(A), i+1, j+1)
    Base.@propagate_inbounds Base.setindex!(A::SizedMatrix, v, i::Int, j::Int) = setindex!(parent(A), v, i+1, j+1)
    Base.size(::SizedMatrix{M,N}) where {M,N} = (M,N)
    LoopVectorization.ArrayInterface.size(::SizedMatrix{M,N}) where {M,N} = (LoopVectorization.Static{M}(),LoopVectorization.Static{N}())
    function Base.axes(::SizedMatrix{M,N}) where {M,N}
        (LoopVectorization.CloseOpen(LoopVectorization.Static{M}()), LoopVectorization.CloseOpen(LoopVectorization.Static{N}()))
    end
    function LoopVectorization.ArrayInterface.axes_types(::Type{SizedMatrix{M,N,T}}) where {M,N,T}
        Tuple{LoopVectorization.CloseOpen{LoopVectorization.Static{0},LoopVectorization.Static{M}}, LoopVectorization.CloseOpen{LoopVectorization.Static{0},LoopVectorization.Static{N}}}
    end
    Base.unsafe_convert(::Type{Ptr{T}}, A::SizedMatrix{M,N,T}) where {M,N,T} = pointer(A.data)
    LoopVectorization.ArrayInterface.strides(::SizedMatrix{M}) where {M} = (LoopVectorization.Static{1}(),LoopVectorization.Static{M}())
    LoopVectorization.ArrayInterface.contiguous_axis(::Type{<:SizedMatrix}) = LoopVectorization.One()
    LoopVectorization.ArrayInterface.contiguous_batch_size(::Type{<:SizedMatrix}) = LoopVectorization.Zero()
    LoopVectorization.ArrayInterface.stride_rank(::Type{<:SizedMatrix}) = (LoopVectorization.Static(1), LoopVectorization.Static(2))
    # LoopVectorization.ArrayInterface.offsets(::Type{SizedMatrix{M,N,T}}) where {M,N,T}  = (LoopVectorization.Static{0}(), LoopVectorization.Static{0}())
    LoopVectorization.ArrayInterface.offsets(::SizedMatrix) = (LoopVectorization.Static{0}(), LoopVectorization.Static{0}())
    LoopVectorization.ArrayInterface.dense_dims(::Type{SizedMatrix{M,N,T}}) where {M,N,T} = LoopVectorization.ArrayInterface.dense_dims(Matrix{T})
# struct ZeroInitializedArray{T,N,A<:DenseArray{T,N}} <: DenseArray{T,N}
#     data::A
# end
# Base.size(A::ZeroInitializedArray) = size(A.data)
# Base.length(A::ZeroInitializedArray) = length(A.data)
# Base.axes(A::ZeroInitializedArray, i) = axes(A.data, i)
# @inline Base.getindex(A::ZeroInitializedArray{T}) where {T} = zero(T)
# Base.@propagate_inbounds Base.setindex!(A::ZeroInitializedArray, v, i...) = setindex!(A.data, v, i...)
# function LoopVectorization.VectorizationBase.stridedpointer(A::ZeroInitializedArray)
#     LoopVectorization.VectorizationBase.ZeroInitializedStridedPointer(LoopVectorization.VectorizationBase.stridedpointer(A.data))
# end

    for T ∈ (Float32, Float64, Int32, Int64)
        TC = sizeof(T) == 4 ? Float32 : Float64
        R = T <: Integer ? (T(-1000):T(1000)) : T
        for M ∈ 48:54
            C0 = zeros(TC, M, M); C1 = zeros(TC, M, M);
            A = rand(R, M, M); B = rand(R, M, M);
            AB_plus_BA!(C0, A, B)
            AB_plus_BA_avx!(C1, A, B)
            @test C0 ≈ C1
        end
    # let T = Int32
        # exceeds_time_limit() && break
        @show T, @__LINE__
        # M, K, N = 128, 128, 128;
        N = 69;
        @time for M ∈ 72:80, K ∈ 72:80
            # @show M,K
        # M, K, N = 73, 75, 69;
            C = Matrix{TC}(undef, M, N);
            A = rand(R, M, K); B = rand(R, K, N);
            At = copy(A');
            Bt = copy(B');
            C2 = similar(C);
            A2 = rand(R, M, K+1);
            dense!(LoopVectorization.relu, C, A2, B);
            @test C ≈ LoopVectorization.relu.(@view(A2[:,begin:end-1]) * B .+ @view(A2[:,end]))
            @testset "avx $T dynamc gemm" begin
                AmulB!(C2, A, B)
                AmulBavx1!(C, A, B)
                @test C ≈ C2
                fill!(C, 999.99); AmulBavx1!(C, At', B)
                @test C ≈ C2
                fill!(C, 999.99); AmulBavx2!(C, A, B)
                @test C ≈ C2
                fill!(C, 999.99); AmulBavx2!(C, At', B)
                @test C ≈ C2
                fill!(C, 999.99); AmulBavx2!(C, A, Bt')
                @test C ≈ C2
                fill!(C, 999.99); AmulBavx2!(C, At', Bt')
                @test C ≈ C2
                fill!(C, 999.99); AmulBavx3!(C, A, B)
                @test C ≈ C2
                fill!(C, 999.99); AmulBavx3!(C, At', B)
                @test C ≈ C2
                fill!(C, 0.0); AmuladdBavx!(C, A, B)
                @test C ≈ C2
                AmuladdBavx!(C, At', B)
                @test C ≈ 2C2
                AmuladdBavx!(C, A, B, -1)
                @test C ≈ C2
                AmuladdBavx!(C, At', Bt', -2)
                @test C ≈ -C2
                AmuladdBavx!(C, At', B, 3, 2)
                @test C ≈ C2
                # How much of this can I do before rounding errors are likely to cause test failures?
                # Setting back to zero here...
                AmuladdBavx!(C, At', B, 1, 0) 
                @test C ≈ C2
                AmuladdBavx!(C, At', Bt', 2, -1)
                @test C ≈ C2
                # TODO: Reimplement the ZeroInitialized wrappers
                # fill!(C, 9999.999); AmuladdBavx!(ZeroInitializedArray(C), At', Bt')
                # @test C ≈ C2
                fill!(C, 9999.999); AmulB2x2avx!(C, A, B);
                @test C ≈ C2
                fill!(C, 9999.999); AmulB2x2avx!(C, At', B);
                @test C ≈ C2
                fill!(C, 9999.999); AtmulBavx1!(C, At, B)
                @test C ≈ C2
                fill!(C, 9999.999); AtmulBavx1!(C, A', B)
                @test C ≈ C2
                fill!(C, 9999.999); AtmulBavx2!(C, At, B);
                @test C ≈ C2
                fill!(C, 9999.999); AtmulBavx2!(C, A', B);
                @test C ≈ C2
                fill!(C, 9999.999); mulCAtB_2x2blockavx!(C, At, B);
                @test C ≈ C2
                fill!(C, 9999.999); mulCAtB_2x2blockavx!(C, A', B);
                @test C ≈ C2
                fill!(C, 9999.999); mulCAtB_2x2blockavx_noinline!(C, At, B);
                @test C ≈ C2
                fill!(C, 9999.999); mulCAtB_2x2blockavx_noinline!(C, A', B);
                @test C ≈ C2
                if RUN_SLOW_TESTS
                    # fill!(C, 9999.999); gemm_accurate!(C, A, B);
                    # @test C ≈ C2
                    # fill!(C, 9999.999); gemm_accurate!(C, At', B);
                    # @test C ≈ C2
                    # fill!(C, 9999.999); gemm_accurate!(C, A, Bt');
                    # @test C ≈ C2
                    # fill!(C, 9999.999); gemm_accurate!(C, At', Bt');
                    # @test C ≈ C2
                    Ab = zeros(eltype(C), size(A)); Bb = zeros(eltype(C), size(B)); Cb = zero(C);
                    threegemms!(Ab, Bb, Cb, A, B, C)
                    @test Ab ≈ C * B'
                    @test Bb ≈ A' * C
                    @test Cb ≈ A * B
                end
                if iszero(size(A,1) % 8)
                    Abit = A .> 0.5;
                    fill!(C, 9999.999); AmulBavx1!(C, Abit, B);
                    @test C ≈ Abit * B
                end
                if iszero(size(B,1) % 8)
                    Bbit = B .> 0.5;
                    fill!(C, 9999.999); AmulBavx1!(C, A, Bbit);
                    @test C ≈ A * Bbit
                end
            end
            # exceeds_time_limit() && break
            @testset "_avx $T dynamic gemm" begin
                AmulB_avx1!(C, A, B)
                @test C ≈ C2
                fill!(C, 999.99); AmulB_avx1!(C, At', B)
                @test C ≈ C2
                fill!(C, 999.99); AmulB_avx2!(C, A, B)
                @test C ≈ C2
                fill!(C, 999.99); AmulB_avx2!(C, At', B)
                @test C ≈ C2
                fill!(C, 999.99); AmulB_avx2!(C, A, Bt')
                @test C ≈ C2
                fill!(C, 999.99); AmulB_avx2!(C, At', Bt')
                @test C ≈ C2
                fill!(C, 999.99); AmulB_avx3!(C, A, B)
                @test C ≈ C2
                fill!(C, 999.99); AmulB_avx3!(C, At', B)
                @test C ≈ C2
                fill!(C, 0.0); AmuladdB_avx!(C, A, B)
                @test C ≈ C2
                AmuladdB_avx!(C, At', B)
                @test C ≈ 2C2
                AmuladdB_avx!(C, A, B, -1)
                @test C ≈ C2
                AmuladdB_avx!(C, At', B, -2)
                @test C ≈ -C2
                fill!(C, 9999.999); AmulB2x2_avx!(C, A, B);
                @test C ≈ C2
                fill!(C, 9999.999); AmulB2x2_avx!(C, At', B);
                @test C ≈ C2
                fill!(C, 9999.999); AtmulB_avx1!(C, At, B);
                @test C ≈ C2
                fill!(C, 9999.999); AtmulB_avx1!(C, A', B);
                @test C ≈ C2
                fill!(C, 9999.999); AtmulB_avx2!(C, At, B);
                @test C ≈ C2
                fill!(C, 9999.999); AtmulB_avx2!(C, A', B);
                @test C ≈ C2
                fill!(C, 9999.999); mulCAtB_2x2block_avx!(C, At, B);
                @test C ≈ C2
                fill!(C, 9999.999); mulCAtB_2x2block_avx!(C, A', B);
                @test C ≈ C2
            end
            # exceeds_time_limit() && break
            if (M,K) === (73,77) # pick a random size, we only want to compile once
                As = SizedMatrix{M,K}(A);
                Ats = SizedMatrix{K,M}(At);
                Bs = SizedMatrix{K,N}(B);
                Bts = SizedMatrix{N,K}(Bt);
                Cs = SizedMatrix{M,N}(C);
                C2z = LoopVectorization.OffsetArray(C2, -1, -1);
                @testset "avx $T static gemm" begin
                    # AmulBavx1!(Cs, As, Bs)
                    # @test Cs ≈ C2
                    # fill!(Cs, 999.99); AmulBavx1!(Cs, Ats', Bs)
                    # @test Cs ≈ C2
                    fill!(Cs, 999.99); AmulBavx2!(Cs, As, Bs)
                    @test Cs ≈ C2z
                    fill!(Cs, 999.99); AmulBavx2!(Cs, Ats', Bs)
                    @test Cs ≈ C2z
                    fill!(Cs, 999.99); AmulBavx2!(Cs, As, Bts')
                    @test Cs ≈ C2z
                    fill!(Cs, 999.99); AmulBavx2!(Cs, Ats', Bts')
                    @test Cs ≈ C2z
                    fill!(Cs, 999.99); AmulBavx3!(Cs, As, Bs)
                    @test Cs ≈ C2z
                    fill!(Cs, 999.99); AmulBavx3!(Cs, Ats', Bs)
                    @test Cs ≈ C2z
                    fill!(Cs, 0.0); AmuladdBavx!(Cs, As, Bs)
                    @test Cs ≈ C2z
                    AmuladdBavx!(Cs, Ats', Bs)
                    @test Cs ≈ 2C2z
                    AmuladdBavx!(Cs, As, Bs, -1)
                    @test Cs ≈ C2z
                    AmuladdBavx!(Cs, Ats', Bs, -2)
                    @test Cs ≈ -C2z
                    fill!(Cs, 9999.999); AmulB2x2avx!(Cs, As, Bs)
                    @test Cs ≈ C2z
                    fill!(Cs, 9999.999); AmulB2x2avx!(Cs, Ats', Bs)
                    @test Cs ≈ C2z
                    fill!(Cs, 9999.999); AtmulBavx1!(Cs, Ats, Bs);
                    @test Cs ≈ C2z
                    fill!(Cs, 9999.999); AtmulBavx1!(Cs, As', Bs);
                    @test Cs ≈ C2z
                    fill!(Cs, 9999.999); AtmulBavx2!(Cs, Ats, Bs);
                    @test Cs ≈ C2z
                    fill!(Cs, 9999.999); AtmulBavx2!(Cs, As', Bs);
                    @test Cs ≈ C2z
                    # fill!(Cs, 9999.999); mulCAtB_2x2blockavx!(Cs, Ats, Bs);
                    # @test Cs ≈ C2
                    # fill!(Cs, 9999.999); mulCAtB_2x2blockavx!(Cs, As', Bs);
                    # @test Cs ≈ C2
                    # fill!(Cs, 9999.999); mulCAtB_2x2blockavx_noinline!(Cs, Ats, Bs);
                    # @test Cs ≈ C2
                    # fill!(Cs, 9999.999); mulCAtB_2x2blockavx_noinline!(Cs, As', Bs);
                    # @test Cs ≈ C2
                end
                # exceeds_time_limit() && break
                @testset "_avx $T static gemm" begin
                    # AmulB_avx1!(Cs, As, Bs)
                    # @test Cs ≈ C2
                    # fill!(Cs, 999.99); AmulB_avx1!(Cs, Ats', Bs)
                    # @test Cs ≈ C2
                    fill!(Cs, 999.99); AmulB_avx2!(Cs, As, Bs)
                    @test Cs ≈ C2z
                    fill!(Cs, 999.99); AmulB_avx2!(Cs, Ats', Bs)
                    @test Cs ≈ C2z
                    fill!(Cs, 999.99); AmulB_avx2!(Cs, As, Bts')
                    @test Cs ≈ C2z
                    fill!(Cs, 999.99); AmulB_avx2!(Cs, Ats', Bts')
                    @test Cs ≈ C2z
                    fill!(Cs, 999.99); AmulB_avx3!(Cs, As, Bs)
                    @test Cs ≈ C2z
                    fill!(Cs, 999.99); AmulB_avx3!(Cs, Ats', Bs)
                    @test Cs ≈ C2z
                    fill!(Cs, 0.0); AmuladdB_avx!(Cs, As, Bs)
                    @test Cs ≈ C2z
                    AmuladdB_avx!(Cs, Ats', Bs)
                    @test Cs ≈ 2C2z
                    AmuladdB_avx!(Cs, As, Bs, -1)
                    @test Cs ≈ C2z
                    AmuladdB_avx!(Cs, Ats', Bs, -2)
                    @test Cs ≈ -C2z
                    fill!(Cs, 9999.999); AmulB2x2_avx!(Cs, As, Bs)
                    @test Cs ≈ C2z
                    fill!(Cs, 9999.999); AmulB2x2_avx!(Cs, Ats', Bs)
                    @test Cs ≈ C2z
                    fill!(Cs, 9999.999); AtmulB_avx1!(Cs, Ats, Bs)
                    @test Cs ≈ C2z
                    fill!(Cs, 9999.999); AtmulB_avx1!(Cs, As', Bs)
                    @test Cs ≈ C2z
                    fill!(Cs, 9999.999); AtmulB_avx2!(Cs, Ats, Bs);
                    @test Cs ≈ C2z
                    fill!(Cs, 9999.999); AtmulB_avx2!(Cs, As', Bs);
                    @test Cs ≈ C2z
                    # fill!(Cs, 9999.999); mulCAtB_2x2block_avx!(Cs, Ats, Bs);
                    # @test Cs ≈ C2
                    # fill!(Cs, 9999.999); mulCAtB_2x2block_avx!(Cs, As', Bs);
                    # @test Cs ≈ C2
                end
            end
            # exceeds_time_limit() && break
            @testset "$T rank2mul" begin
                Aₘ= rand(R, M, 2); Aₖ = rand(R, 2, K);
                Aₖ′ = copy(Aₖ');
                rank2AmulB!(C2, Aₘ, Aₖ, B)
                rank2AmulBavx!(C, Aₘ, Aₖ, B)
                @test C ≈ C2
                fill!(C, 9999.999); rank2AmulB_avx!(C, Aₘ, Aₖ, B)
                @test C ≈ C2
                fill!(C, 9999.999); rank2AmulBavx_noinline!(C, Aₘ, Aₖ, B)
                @test C ≈ C2
                fill!(C, 9999.999); rank2AmulBavx!(C, Aₘ, Aₖ′', B)
                @test C ≈ C2
                fill!(C, 9999.999); rank2AmulB_avx!(C, Aₘ, Aₖ′', B)
                @test C ≈ C2
                fill!(C, 9999.999); rank2AmulBavx_noinline!(C, Aₘ, Aₖ′', B)
                @test C ≈ C2
            end
        end
    end
end

