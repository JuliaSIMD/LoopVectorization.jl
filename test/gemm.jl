@testset "GEMM" begin
 # using LoopVectorization, LinearAlgebra, Test; T = Float64
    Unum, Tnum = LoopVectorization.VectorizationBase.REGISTER_COUNT == 16 ? (3, 4) : (5, 5)
    @test LoopVectorization.mᵣ == Unum
    @test LoopVectorization.nᵣ == Tnum
    AmulBtq1 = :(for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
                 C[m,n] = zeroB
                 for k ∈ 1:size(A,2)
                 C[m,n] += A[m,k] * B[n,k]
                 end
                 end);
    lsAmulBt1 = LoopVectorization.LoopSet(AmulBtq1);
    # @test LoopVectorization.choose_order(lsAmulBt1) == (Symbol[:n,:m,:k], :n, :m, :m, Unum, Tnum)
    @test LoopVectorization.choose_order(lsAmulBt1) == (Symbol[:n,:m,:k], :m, :n, :m, Unum, Tnum)

    AmulBq1 = :(for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
                C[m,n] = zeroB
                for k ∈ 1:size(A,2)
                C[m,n] += A[m,k] * B[k,n]
                end
                end)
    lsAmulB1 = LoopVectorization.LoopSet(AmulBq1);
    # @test LoopVectorization.choose_order(lsAmulB1) == (Symbol[:n,:m,:k], :n, :m, :m, Unum, Tnum)
    @test LoopVectorization.choose_order(lsAmulB1) == (Symbol[:n,:m,:k], :m, :n, :m, Unum, Tnum)
    AmulBq2 = :(for m ∈ 1:M, n ∈ 1:N
                C[m,n] = zero(eltype(B))
                for k ∈ 1:K
                C[m,n] += A[m,k] * B[k,n]
                end
                end)
    lsAmulB2 = LoopVectorization.LoopSet(AmulBq2);
    # @test LoopVectorization.choose_order(lsAmulB2) == (Symbol[:n,:m,:k], :n, :m, :m, Unum, Tnum)
    @test LoopVectorization.choose_order(lsAmulB2) == (Symbol[:n,:m,:k], :m, :n, :m, Unum, Tnum)
    AmulBq3 = :(for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
                ΔCₘₙ = zero(eltype(C))
                for k ∈ 1:size(A,2)
                ΔCₘₙ += A[m,k] * B[k,n]
                end
                C[m,n] += ΔCₘₙ
                end)
    lsAmulB3 = LoopVectorization.LoopSet(AmulBq3);
    @test LoopVectorization.choose_order(lsAmulB3) == (Symbol[:n,:m,:k], :m, :n, :m, Unum, Tnum)

    function AmulB!(C, A, B)
        C .= 0
        for k ∈ 1:size(A,2), j ∈ 1:size(B,2)
            @simd ivdep for i ∈ 1:size(A,1)
                @inbounds C[i,j] += A[i,k] * B[k,j]
            end
        end
    end
    function AmulBavx1!(C, A, B)
        @avx for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
            Cₘₙ = zero(eltype(C))
            for k ∈ 1:size(A,2)
                Cₘₙ += A[m,k] * B[k,n]
            end
            C[m,n] = Cₘₙ
        end
    end
    function AmulBavx2!(C, A, B)
        z = zero(eltype(C))
        @avx for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
            C[m,n] = z
            for k ∈ 1:size(A,2)
                C[m,n] += A[m,k] * B[k,n]
            end
        end
    end
    function AmulBavx3!(C, A, B)
        @avx for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
            C[m,n] = zero(eltype(C))
            for k ∈ 1:size(A,2)
                C[m,n] += A[m,k] * B[k,n]
            end
        end
    end
    myzero(A) = zero(eltype(A))
    # function AmulBavx4!(C, A, B)
    #     @avx for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
    #         C[m,n] = myzero(C)
    #         for k ∈ 1:size(A,2)
    #             C[m,n] += A[m,k] * B[k,n]
    #         end
    #     end
    # end
    # C = Cs; A = Ats'; B = Bs; factor = 1;
    # ls = LoopVectorization.@avx_debug for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
    #         ΔCₘₙ = zero(eltype(C))
    #         for k ∈ 1:size(A,2)
    #             ΔCₘₙ += A[m,k] * B[k,n]
    #         end
    #         C[m,n] += ΔCₘₙ * factor
    #     end;
    function AmuladdBavx!(C, A, B, α = one(eltype(C)))
        @avx for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
            ΔCₘₙ = zero(eltype(C))
            for k ∈ 1:size(A,2)
                ΔCₘₙ += A[m,k] * B[k,n]
            end
            C[m,n] += α * ΔCₘₙ
        end
    end
    function AmuladdBavx!(C, A, B, α, β)# = zero(eltype(C)))
        @avx for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
            ΔCₘₙ = zero(eltype(C))
            for k ∈ 1:size(A,2)
                ΔCₘₙ += A[m,k] * B[k,n]
            end
            C[m,n] = α * ΔCₘₙ + β * C[m,n]
        end
    end
    Amuladdq = :(for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
                 ΔCₘₙ = zero(eltype(C))
                 for k ∈ 1:size(A,2)
                 ΔCₘₙ += A[m,k] * B[k,n]
                 end
                 C[m,n] = α * ΔCₘₙ + β * C[m,n]
                 end);
    lsAmuladd = LoopVectorization.LoopSet(Amuladdq);
    # @test LoopVectorization.choose_order(lsAmuladd) == (Symbol[:n,:m,:k], :n, :m, :m, Unum, Tnum)
    @test LoopVectorization.choose_order(lsAmuladd) == (Symbol[:n,:m,:k], :m, :n, :m, Unum, Tnum)
    Atmuladdq = :(for m ∈ 1:size(A,2), n ∈ 1:size(B,2)
                 ΔCₘₙ = zero(eltype(C))
                 for k ∈ 1:size(A,1)
                 ΔCₘₙ += A[k,m] * B[k,n]
                  end
                  C[m,n] += α * ΔCₘₙ
                 end);
    lsAtmuladd = LoopVectorization.LoopSet(Atmuladdq);
    # LoopVectorization.lower(lsAtmuladd, 2, 2)
    # lsAmuladd.operations
    # LoopVectorization.loopdependencies.(lsAmuladd.operations)
    # LoopVectorization.reduceddependencies.(lsAmuladd.operations)
    # @test LoopVectorization.choose_order(lsAtmuladd) == (Symbol[:n,:m,:k], :n, :m, :k, Unum, Tnum)
    @test LoopVectorization.choose_order(lsAtmuladd) == (Symbol[:n,:m,:k], :m, :n, :k, Unum, Tnum)

    function AmulB_avx1!(C, A, B)
        @_avx for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
            Cₘₙ = zero(eltype(C))
            for k ∈ 1:size(A,2)
                Cₘₙ += A[m,k] * B[k,n]
            end
            C[m,n] = Cₘₙ
        end
    end
    fq = :(for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
           Cₘₙ = zero(eltype(C))
           for k ∈ 1:size(A,2)
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
        @_avx for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
            C[m,n] = z
            for k ∈ 1:size(A,2)
                C[m,n] += A[m,k] * B[k,n]
            end
        end
    end
    # AmulB_avx2!(C, A, B)
    # gq = :(for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
    # C[m,n] = z
    # for k ∈ 1:size(A,2)
    # C[m,n] += A[m,k] * B[k,n]
    # end
    # end);
    # ls = LoopVectorization.LoopSet(gq);
    # ls.preamble_symsym
    # ls.operations[1]
    function AmulB_avx3!(C, A, B)
        @_avx for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
            C[m,n] = zero(eltype(C))
            for k ∈ 1:size(A,2)
                C[m,n] += A[m,k] * B[k,n]
            end
        end
    end
    # function AmulB_avx4!(C, A, B)
    #     @_avx for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
    #         C[m,n] = myzero(C)
    #         for k ∈ 1:size(A,2)
    #             C[m,n] += A[m,k] * B[k,n]
    #         end
    #     end
    # end
    # q = :(for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
    #       C[m,n] = myzero(C)
    #       for k ∈ 1:size(A,2)
    #       C[m,n] += A[m,k] * B[k,n]
    #       end
    #       end)
    # ls = LoopVectorization.LoopSet(q);
    function AmuladdB_avx!(C, A, B, factor = 1)
        @_avx for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
            ΔCₘₙ = zero(eltype(C))
            for k ∈ 1:size(A,2)
                ΔCₘₙ += A[m,k] * B[k,n]
            end
            C[m,n] += ΔCₘₙ * factor
        end
    end

    function AmulB2x2avx!(C, A, B)
        @avx unroll=(2,2) for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
            ΔCₘₙ = zero(eltype(C))
            for k ∈ 1:size(A,2)
                ΔCₘₙ += A[m,k] * B[k,n]
            end
            C[m,n] = ΔCₘₙ
        end
    end
    function AmulB2x2_avx!(C, A, B)
        @_avx unroll=(2,2) for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
            ΔCₘₙ = zero(eltype(C))
            for k ∈ 1:size(A,2)
                ΔCₘₙ += A[m,k] * B[k,n]
            end
            C[m,n] = ΔCₘₙ
        end
    end

    # function AtmulB!(C, A, B)
    #     for j ∈ 1:size(C,2), i ∈ 1:size(C,1)
    #         Cᵢⱼ = zero(eltype(C))
    #         @simd ivdep for k ∈ 1:size(A,1)
    #             @inbounds Cᵢⱼ += A[k,i] * B[k,j]
    #         end
    #         C[i,j] = Cᵢⱼ
    #     end
    # end
    AtmulBq = :(for n ∈ 1:size(C,2), m ∈ 1:size(C,1)
                Cₘₙ = zero(eltype(C))
                for k ∈ 1:size(A,1)
                Cₘₙ += A[k,m] * B[k,n]
                end
                C[m,n] = Cₘₙ
                end)
    lsAtmulB = LoopVectorization.LoopSet(AtmulBq);
    # LoopVectorization.choose_order(lsAtmulB)
    # @test LoopVectorization.choose_order(lsAtmulB) == (Symbol[:n,:m,:k], :m, :n, :k, Unum, Tnum)
    @test LoopVectorization.choose_order(lsAtmulB) == (Symbol[:n,:m,:k], :n, :m, :k, Unum, Tnum)
    
    function AtmulBavx1!(C, A, B)
        @avx for n ∈ 1:size(C,2), m ∈ 1:size(C,1)
            Cₘₙ = zero(eltype(C))
            for k ∈ 1:size(A,1)
                Cₘₙ += A[k,m] * B[k,n]
            end
            C[m,n] = Cₘₙ
        end
    end
    Atq = :(for n ∈ 1:size(C,2), m ∈ 1:size(C,1)
            Cₘₙ = zero(eltype(C))
            for k ∈ 1:size(A,1)
            Cₘₙ += A[k,m] * B[k,n]
            end
            C[m,n] += Cₘₙ * factor
            end);
    atls = LoopVectorization.LoopSet(Atq);
    # LoopVectorization.operations(atls)
    # LoopVectorization.loopdependencies.(operations(atls))
    # LoopVectorization.reduceddependencies.(operations(atls))
    function AtmulB_avx1!(C, A, B)
        @_avx for n ∈ 1:size(C,2), m ∈ 1:size(C,1)
            Cₘₙ = zero(eltype(C))
            for k ∈ 1:size(A,1)
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
        # When the @avx macro is available, this code is faster:
        z = zero(eltype(C))
        @avx for n in 1:size(C,2), m in 1:size(C,1)
            Cmn = z
            for k in 1:size(A,1)
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
        # When the @avx macro is available, this code is faster:
        z = zero(eltype(C))
        @_avx for n in 1:size(C,2), m in 1:size(C,1)
            Cmn = z
            for k in 1:size(A,1)
                Cmn += A[k,m] * B[k,n]
            end
            C[m,n] = Cmn
        end
        return C
    end
    function rank2AmulB!(C, Aₘ, Aₖ, B)
        @inbounds for m ∈ 1:size(C,1), n ∈ 1:size(C,2)
            Cₘₙ = zero(eltype(C))
            @fastmath for k ∈ 1:size(B,1)
                Cₘₙ += (Aₘ[m,1]*Aₖ[1,k]+Aₘ[m,2]*Aₖ[2,k]) * B[k,n]
            end
            C[m,n] = Cₘₙ
        end
    end
    r2ambq = :(for m ∈ 1:size(C,1), n ∈ 1:size(C,2)
               Cₘₙ = zero(eltype(C))
               for k ∈ 1:size(B,1)
               Cₘₙ += (Aₘ[m,1]*Aₖ[1,k]+Aₘ[m,2]*Aₖ[2,k]) * B[k,n]
               end
               C[m,n] = Cₘₙ
               end)
    lsr2amb = LoopVectorization.LoopSet(r2ambq);
    if LoopVectorization.VectorizationBase.REGISTER_COUNT == 32
        # @test LoopVectorization.choose_order(lsr2amb) == ([:n, :m, :k], :n, :m, :m, 3, 3)
        @test LoopVectorization.choose_order(lsr2amb) == ([:n, :m, :k], :m, :n, :m, 3, 6)
    else
        # @test LoopVectorization.choose_order(lsr2amb) == ([:n, :m, :k], :n, :m, :m, 2, 2)
        @test LoopVectorization.choose_order(lsr2amb) == ([:n, :m, :k], :m, :n, :m, 2, 4)
    end
    function rank2AmulBavx!(C, Aₘ, Aₖ, B)
        @avx for m ∈ 1:size(C,1), n ∈ 1:size(C,2)
            Cₘₙ = zero(eltype(C))
            for k ∈ 1:size(B,1)
                Cₘₙ += (Aₘ[m,1]*Aₖ[1,k]+Aₘ[m,2]*Aₖ[2,k]) * B[k,n]
            end
            C[m,n] = Cₘₙ
        end
    end
    function rank2AmulB_avx!(C, Aₘ, Aₖ, B)
        @_avx for m ∈ 1:size(C,1), n ∈ 1:size(C,2)
            Cₘₙ = zero(eltype(C))
            for k ∈ 1:size(B,1)
                Cₘₙ += (Aₘ[m,1]*Aₖ[1,k]+Aₘ[m,2]*Aₖ[2,k]) * B[k,n]
            end
            C[m,n] = Cₘₙ
        end
    end
    function rank2AmulBavx_noinline!(C, Aₘ, Aₖ, B)
        @avx inline=false for m ∈ 1:size(C,1), n ∈ 1:size(C,2)
            Cₘₙ = zero(eltype(C))
            for k ∈ 1:size(B,1)
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
		@avx for k ∈ 1:K
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
	    	@avx for k ∈ 1:K
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
		@avx for k ∈ 1:K
		    Cm1 += A[k,M] * B[k,n] 
		    Cm2 += A[k,M] * B[k,n1] 
		end
		C[M,n] = Cm1
		C[M,n1] = Cm2
	    end
            if isodd(N)
	    	Cmn = 0.0
	    	@avx for k ∈ 1:K
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
		@avx inline=false for k ∈ 1:K
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
	    	@avx inline=false for k ∈ 1:K
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
		@avx inline=false for k ∈ 1:K
		    Cm1 += A[k,M] * B[k,n] 
		    Cm2 += A[k,M] * B[k,n1] 
		end
		C[M,n] = Cm1
		C[M,n1] = Cm2
	    end
            if isodd(N)
	    	Cmn = 0.0
	    	@avx inline=false for k ∈ 1:K
	    	    Cmn += A[k,M] * B[k,N]
	    	end
	    	C[M,N] = Cmn
            end
        end
        return C
    end

    function gemm_accurate!(C, A, B)
        @avx for n in 1:size(C,2), m in 1:size(C,1)
            Cmn_hi = zero(eltype(C))
            Cmn_lo = zero(eltype(C))
            for k in 1:size(B,1)
                hiprod = evmul(A[m,k], B[k,n])
                loprod = vfmsub(A[m,k], B[k,n], hiprod)
                hi_ts = evadd(hiprod, Cmn_hi)
                a1_ts = evsub(hi_ts, Cmn_hi)
                b1_ts = evsub(hi_ts, a1_ts)
                lo_ts = evadd(evsub(hiprod, a1_ts), evsub(Cmn_hi, b1_ts))
                thi = evadd(loprod, Cmn_lo)
                a1_t = evsub(thi, Cmn_lo)
                b1_t = evsub(thi, a1_t)
                tlo = evadd(evsub(loprod, a1_t), evsub(Cmn_lo, b1_t))
                c1 = evadd(lo_ts, thi)
                hi_ths = evadd(hi_ts, c1)
                lo_ths = evsub(c1, evsub(hi_ths, hi_ts))
                c2 = evadd(tlo, lo_ths)
                Cmn_hi = evadd(hi_ths, c2)
                Cmn_lo = evsub(c2, evsub(Cmn_hi, hi_ths))
            end
            C[m,n] = Cmn_hi
        end
    end
    
    function threegemms!(Ab, Bb, Cb, A, B, C)
        M, N = size(Cb); K = size(B,1)
        @avx for m in 1:M, k in 1:K, n in 1:N
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
    # lsmul2x2q = LoopVectorization.LoopSet(mul2x2q)

    struct SizedMatrix{M,N,T} <: DenseMatrix{T}
        data::Matrix{T}
    end
    Base.parent(A::SizedMatrix) = A.data
    SizedMatrix{M,N}(A::Matrix{T}) where {M,N,T} = SizedMatrix{M,N,T}(A)
    Base.@propagate_inbounds Base.getindex(A::SizedMatrix, i...) = getindex(parent(A), i...)
    Base.@propagate_inbounds Base.setindex!(A::SizedMatrix, v, i...) = setindex!(parent(A), v, i...)
    Base.size(::SizedMatrix{M,N}) where {M,N} = (M,N)
    @inline function LoopVectorization.stridedpointer(A::SizedMatrix{M,N,T}) where {M,N,T}
        LoopVectorization.StaticStridedPointer{T,Tuple{1,M}}(pointer(parent(A)))
    end
    @inline function LoopVectorization.stridedpointer(A::LinearAlgebra.Adjoint{T,SizedMatrix{M,N,T}}) where {M,N,T}
        LoopVectorization.StaticStridedPointer{T,Tuple{M,1}}(pointer(parent(A).data))
    end
    @inline function LoopVectorization.stridedpointer(A::LinearAlgebra.Transpose{T,SizedMatrix{M,N,T}}) where {M,N,T}
        LoopVectorization.StaticStridedPointer{T,Tuple{M,1}}(pointer(parent(A).data))
    end
    LoopVectorization.maybestaticsize(::SizedMatrix{M,N}, ::Val{1}) where {M,N} = LoopVectorization.Static{M}()
    LoopVectorization.maybestaticsize(::SizedMatrix{M,N}, ::Val{2}) where {M,N} = LoopVectorization.Static{N}()
    LoopVectorization.maybestaticsize(::LinearAlgebra.Adjoint{T,SizedMatrix{M,N,T}}, ::Val{1}) where {M,N,T} = LoopVectorization.Static{N}()
    LoopVectorization.maybestaticsize(::LinearAlgebra.Adjoint{T,SizedMatrix{M,N,T}}, ::Val{2}) where {M,N,T} = LoopVectorization.Static{M}()
    LoopVectorization.maybestaticsize(::LinearAlgebra.Transpose{T,SizedMatrix{M,N,T}}, ::Val{1}) where {M,N,T} = LoopVectorization.Static{N}()
    LoopVectorization.maybestaticsize(::LinearAlgebra.Transpose{T,SizedMatrix{M,N,T}}, ::Val{2}) where {M,N,T} = LoopVectorization.Static{M}()
    
    for T ∈ (Float32, Float64, Int32, Int64)
        @show T, @__LINE__
        # M, K, N = 128, 128, 128;
        M, K, N = 73, 75, 69;
        TC = sizeof(T) == 4 ? Float32 : Float64
        R = T <: Integer ? (T(-1000):T(1000)) : T
        C = Matrix{TC}(undef, M, N);
        A = rand(R, M, K); B = rand(R, K, N);
        At = copy(A');
        Bt = copy(B');
        C2 = similar(C);
        As = SizedMatrix{M,K}(A);
        Ats = SizedMatrix{K,M}(At);
        Bs = SizedMatrix{K,N}(B);
        Bts = SizedMatrix{N,K}(Bt);
        Cs = SizedMatrix{M,N}(C);
        @time @testset "avx $T dynamc gemm" begin
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
            AmuladdBavx!(C, At', B, -2)
            @test C ≈ -C2
            AmuladdBavx!(C, At', B, 3, 2)
            @test C ≈ C2
            # How much of this can I do before rounding errors are likely to cause test failures?
            # Setting back to zero here...
            AmuladdBavx!(C, At', B, 1, 0) 
            @test C ≈ C2
            AmuladdBavx!(C, At', Bt', 2, -1)
            @test C ≈ C2
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
            fill!(C, 9999.999); gemm_accurate!(C, A, B);
            @test C ≈ C2
            fill!(C, 9999.999); gemm_accurate!(C, At', B);
            @test C ≈ C2
            fill!(C, 9999.999); gemm_accurate!(C, A, Bt');
            @test C ≈ C2
            fill!(C, 9999.999); gemm_accurate!(C, At', Bt');
            @test C ≈ C2
            Abit = A .> 0.5
            fill!(C, 9999.999); AmulBavx1!(C, Abit, B)
            @test C ≈ Abit * B
            Bbit = B .> 0.5
            fill!(C, 9999.999); AmulBavx1!(C, A, Bbit)
            @test C ≈ A * Bbit
            Ab = zero(A); Bb = zero(B); Cb = zero(C);
            threegemms!(Ab, Bb, Cb, A, B, C)
            @test Ab ≈ C * B'
            @test Bb ≈ A' * C
            @test Cb ≈ A * B
        end
        @time @testset "_avx $T dynamic gemm" begin
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
        @time @testset "avx $T static gemm" begin
            AmulBavx1!(Cs, As, Bs)
            @test Cs ≈ C2
            fill!(Cs, 999.99); AmulBavx1!(Cs, Ats', Bs)
            @test Cs ≈ C2
            fill!(Cs, 999.99); AmulBavx2!(Cs, As, Bs)
            @test Cs ≈ C2
            fill!(Cs, 999.99); AmulBavx2!(Cs, Ats', Bs)
            @test Cs ≈ C2
            fill!(Cs, 999.99); AmulBavx2!(Cs, As, Bts')
            @test Cs ≈ C2
            fill!(Cs, 999.99); AmulBavx2!(Cs, Ats', Bts')
            @test Cs ≈ C2
            fill!(Cs, 999.99); AmulBavx3!(Cs, As, Bs)
            @test Cs ≈ C2
            fill!(Cs, 999.99); AmulBavx3!(Cs, Ats', Bs)
            @test Cs ≈ C2
            fill!(Cs, 0.0); AmuladdBavx!(Cs, As, Bs)
            @test Cs ≈ C2
            AmuladdBavx!(Cs, Ats', Bs)
            @test Cs ≈ 2C2
            AmuladdBavx!(Cs, As, Bs, -1)
            @test Cs ≈ C2
            AmuladdBavx!(Cs, Ats', Bs, -2)
            @test Cs ≈ -C2
            fill!(Cs, 9999.999); AmulB2x2avx!(Cs, As, Bs)
            @test Cs ≈ C2
            fill!(Cs, 9999.999); AmulB2x2avx!(Cs, Ats', Bs)
            @test Cs ≈ C2
            fill!(Cs, 9999.999); AtmulBavx1!(Cs, Ats, Bs);
            @test Cs ≈ C2
            fill!(Cs, 9999.999); AtmulBavx1!(Cs, As', Bs);
            @test Cs ≈ C2
            fill!(Cs, 9999.999); AtmulBavx2!(Cs, Ats, Bs);
            @test Cs ≈ C2
            fill!(Cs, 9999.999); AtmulBavx2!(Cs, As', Bs);
            @test Cs ≈ C2
            fill!(Cs, 9999.999); mulCAtB_2x2blockavx!(Cs, Ats, Bs);
            @test Cs ≈ C2
            fill!(Cs, 9999.999); mulCAtB_2x2blockavx!(Cs, As', Bs);
            @test Cs ≈ C2
            fill!(Cs, 9999.999); mulCAtB_2x2blockavx_noinline!(Cs, Ats, Bs);
            @test Cs ≈ C2
            fill!(Cs, 9999.999); mulCAtB_2x2blockavx_noinline!(Cs, As', Bs);
            @test Cs ≈ C2
        end
        @time @testset "_avx $T static gemm" begin
            AmulB_avx1!(Cs, As, Bs)
            @test Cs ≈ C2
            fill!(Cs, 999.99); AmulB_avx1!(Cs, Ats', Bs)
            @test Cs ≈ C2
            fill!(Cs, 999.99); AmulB_avx2!(Cs, As, Bs)
            @test Cs ≈ C2
            fill!(Cs, 999.99); AmulB_avx2!(Cs, Ats', Bs)
            @test Cs ≈ C2
            fill!(Cs, 999.99); AmulB_avx2!(Cs, As, Bts')
            @test Cs ≈ C2
            fill!(Cs, 999.99); AmulB_avx2!(Cs, Ats', Bts')
            @test Cs ≈ C2
            fill!(Cs, 999.99); AmulB_avx3!(Cs, As, Bs)
            @test Cs ≈ C2
            fill!(Cs, 999.99); AmulB_avx3!(Cs, Ats', Bs)
            @test Cs ≈ C2
            fill!(Cs, 0.0); AmuladdB_avx!(Cs, As, Bs)
            @test Cs ≈ C2
            AmuladdB_avx!(Cs, Ats', Bs)
            @test Cs ≈ 2C2
            AmuladdB_avx!(Cs, As, Bs, -1)
            @test Cs ≈ C2
            AmuladdB_avx!(Cs, Ats', Bs, -2)
            @test Cs ≈ -C2
            fill!(Cs, 9999.999); AmulB2x2_avx!(Cs, As, Bs)
            @test Cs ≈ C2
            fill!(Cs, 9999.999); AmulB2x2_avx!(Cs, Ats', Bs)
            @test Cs ≈ C2
            fill!(Cs, 9999.999); AtmulB_avx1!(Cs, Ats, Bs)
            @test Cs ≈ C2
            fill!(Cs, 9999.999); AtmulB_avx1!(Cs, As', Bs)
            @test Cs ≈ C2
            fill!(Cs, 9999.999); AtmulB_avx2!(Cs, Ats, Bs);
            @test Cs ≈ C2
            fill!(Cs, 9999.999); AtmulB_avx2!(Cs, As', Bs);
            @test Cs ≈ C2
            fill!(Cs, 9999.999); mulCAtB_2x2block_avx!(Cs, Ats, Bs);
            @test Cs ≈ C2
            fill!(Cs, 9999.999); mulCAtB_2x2block_avx!(Cs, As', Bs);
            @test Cs ≈ C2
        end

        @time @testset "$T rank2mul" begin
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

