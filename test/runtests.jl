using Test
using LoopVectorization
using LinearAlgebra


@testset "LoopVectorization.jl" begin

    
    @testset "GEMM" begin
        # using LoopVectorization, Test; T = Float64
        Unum, Tnum = LoopVectorization.VectorizationBase.REGISTER_COUNT == 16 ? (3, 4) : (4, 4)
        AmulBq1 = :(for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
                    C[m,n] = zeroB
               for k ∈ 1:size(A,2)
                   C[m,n] += A[m,k] * B[k,n]
               end
               end)
        lsAmulB1 = LoopVectorization.LoopSet(AmulBq1);
        @test LoopVectorization.choose_order(lsAmulB1) == (Symbol[:n,:m,:k], :m, Unum, Tnum)
        AmulBq2 = :(for m ∈ 1:M, n ∈ 1:N
               C[m,n] = zero(eltype(B))
               for k ∈ 1:K
                    C[m,n] += A[m,k] * B[k,n]
               end
                end)
        lsAmulB2 = LoopVectorization.LoopSet(AmulBq2);
        @test LoopVectorization.choose_order(lsAmulB2) == (Symbol[:n,:m,:k], :m, Unum, Tnum)
        AmulBq3 = :(for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
                ΔCₘₙ = zero(eltype(C))
                for k ∈ 1:size(A,2)
                    ΔCₘₙ += A[m,k] * B[k,n]
                end
                C[m,n] += ΔCₘₙ
           end)
        lsAmulB3 = LoopVectorization.LoopSet(AmulBq3);
        @test LoopVectorization.choose_order(lsAmulB3) == (Symbol[:n,:m,:k], :m, Unum, Tnum)

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
        function AmuladdBavx!(C, A, B, factor = 1)
            @avx for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
                ΔCₘₙ = zero(eltype(C))
                for k ∈ 1:size(A,2)
                    ΔCₘₙ += A[m,k] * B[k,n]
                end
                C[m,n] += ΔCₘₙ * factor
            end
        end
        function AmulB_avx1!(C, A, B)
            @_avx for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
                Cₘₙ = zero(eltype(C))
                for k ∈ 1:size(A,2)
                    Cₘₙ += A[m,k] * B[k,n]
                end
                C[m,n] = Cₘₙ
            end
        end
        function AmulB_avx2!(C, A, B)
            z = zero(eltype(C))
            @_avx for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
                C[m,n] = z
                for k ∈ 1:size(A,2)
                    C[m,n] += A[m,k] * B[k,n]
                end
            end
        end
        function AmulB_avx3!(C, A, B)
            @_avx for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
                C[m,n] = zero(eltype(C))
                for k ∈ 1:size(A,2)
                    C[m,n] += A[m,k] * B[k,n]
                end
            end
        end
        function AmuladdB_avx!(C, A, B, factor = 1)
            @_avx for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
                ΔCₘₙ = zero(eltype(C))
                for k ∈ 1:size(A,2)
                    ΔCₘₙ += A[m,k] * B[k,n]
                end
                C[m,n] += ΔCₘₙ * factor
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
        @test LoopVectorization.choose_order(lsAtmulB) == (Symbol[:m,:n,:k], :k, Unum, Tnum)
        
        function AtmulBavx!(C, A, B)
            @avx for n ∈ 1:size(C,2), m ∈ 1:size(C,1)
                Cₘₙ = zero(eltype(C))
                for k ∈ 1:size(A,1)
                    Cₘₙ += A[k,m] * B[k,n]
                end
                C[m,n] = Cₘₙ
            end
        end
        function AtmulB_avx!(C, A, B)
            @_avx for n ∈ 1:size(C,2), m ∈ 1:size(C,1)
                Cₘₙ = zero(eltype(C))
                for k ∈ 1:size(A,1)
                    Cₘₙ += A[k,m] * B[k,n]
                end
                C[m,n] = Cₘₙ
            end
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
        lsr2amb = LoopVectorization.LoopSet(r2ambq)
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
        mul2x2q = :(for k ∈ 1:K
	    C11 += A[k,m] * B[k,n] 
	    C21 += A[k,m1] * B[k,n] 
	    C12 += A[k,m] * B[k,n1] 
	    C22 += A[k,m1] * B[k,n1]
	    end)
        # lsmul2x2q = LoopVectorization.LoopSet(mul2x2q)


        for T ∈ (Float32, Float64, Int32, Int64)
            @show T, @__LINE__
            M, K, N = 73, 75, 69;
            TC = sizeof(T) == 4 ? Float32 : Float64
            R = T <: Integer ? (T(1):T(1000)) : T
            C = Matrix{TC}(undef, M, N);
            A = rand(R, M, K); B = rand(R, K, N);
            At = copy(A');
            C2 = similar(C);
            @testset "avx $T gemm" begin
                AmulB!(C2, A, B)
                AmulBavx1!(C, A, B)
                @test C ≈ C2
                fill!(C, 999.99); AmulBavx2!(C, A, B)
                @test C ≈ C2
                fill!(C, 999.99); AmulBavx3!(C, A, B)
                @test C ≈ C2
                fill!(C, 0.0); AmuladdBavx!(C, A, B)
                @test C ≈ C2
                AmuladdBavx!(C, A, B)
                @test C ≈ 2C2
                AmuladdBavx!(C, A, B, -1)
                @test C ≈ C2
                fill!(C, 9999.999); AtmulBavx!(C, At, B)
                @test C ≈ C2
                fill!(C, 9999.999); mulCAtB_2x2blockavx!(C, At, B);
                @test C ≈ C2
            end
            @testset "_avx $T gemm" begin
                fill!(C, 999.99); AmulB_avx1!(C, A, B)
                @test C ≈ C2
                fill!(C, 999.99); AmulB_avx2!(C, A, B)
                @test C ≈ C2
                fill!(C, 999.99); AmulB_avx3!(C, A, B)
                @test C ≈ C2
                fill!(C, 0.0); AmuladdB_avx!(C, A, B)
                @test C ≈ C2
                AmuladdB_avx!(C, A, B)
                @test C ≈ 2C2
                AmuladdB_avx!(C, A, B, -1)
                @test C ≈ C2
                fill!(C, 9999.999); AtmulB_avx!(C, At, B)
                @test C ≈ C2
                fill!(C, 9999.999); mulCAtB_2x2block_avx!(C, At, B);
                @test C ≈ C2
            end

            @testset "$T rank2mul" begin
                Aₘ= rand(R, M, 2); Aₖ = rand(R, 2, K);
                rank2AmulB!(C2, Aₘ, Aₖ, B)
                rank2AmulBavx!(C, Aₘ, Aₖ, B)
                @test C ≈ C2
                fill!(C, 9999.999); rank2AmulB_avx!(C, Aₘ, Aₖ, B)
                @test C ≈ C2
            end

        end
    end

@testset "dot" begin
    using LoopVectorization, Test
        dotq = :(for i ∈ eachindex(a,b)
                 s += a[i]*b[i]
                 end)
        lsdot = LoopVectorization.LoopSet(dotq);
        @test LoopVectorization.choose_order(lsdot) == (Symbol[:i], :i, 4, -1)

        function mydot(a, b)
            s = zero(eltype(a))
            @inbounds @simd for i ∈ eachindex(a,b)
                s += a[i]*b[i]
            end
            s
        end
        function mydotavx(a, b)
            s = zero(eltype(a))
            @avx for i ∈ eachindex(a,b)
                s += a[i]*b[i]
            end
            s
        end
        function mydot_avx(a, b)
            s = zero(eltype(a))
            @_avx for i ∈ eachindex(a,b)
                s += a[i]*b[i]
            end
            s
        end

        selfdotq = :(for i ∈ eachindex(a)
                     s += a[i]*a[i]
                     end)
        lsselfdot = LoopVectorization.LoopSet(selfdotq);
        @test LoopVectorization.choose_order(lsselfdot) == (Symbol[:i], :i, 8, -1)

        function myselfdot(a)
            s = zero(eltype(a))
            @inbounds @simd for i ∈ eachindex(a)
                s += a[i]*a[i]
            end
            s
        end
        function myselfdotavx(a)
            s = zero(eltype(a))
            @avx for i ∈ eachindex(a)
                s += a[i]*a[i]
            end
            s
        end
        function myselfdot_avx(a)
            s = zero(eltype(a))
            @_avx for i ∈ eachindex(a)
                s += a[i]*a[i]
            end
            s
        end

        # a = rand(400);
        for T ∈ (Float32, Float64)
            @show T, @__LINE__
            a = rand(T, 100); b = rand(T, 100);
            s = mydot(a,b)
            @test mydotavx(a,b) ≈ s
            @test mydot_avx(a,b) ≈ s
            s = myselfdot(a)
            @test myselfdotavx(a) ≈ s
            @test myselfdot_avx(a) ≈ s
        end
    end

    @testset "Special Functions" begin
        vexpq = :(for i ∈ eachindex(a)
                  b[i] = exp(a[i])
                  end)
        lsvexp = LoopVectorization.LoopSet(vexpq);
        @test LoopVectorization.choose_order(lsvexp) == (Symbol[:i], :i, 1, -1)

        function myvexp!(b, a)
            @inbounds for i ∈ eachindex(a)
                b[i] = exp(a[i])
            end
        end
        function myvexpavx!(b, a)
            @avx for i ∈ eachindex(a)
                b[i] = exp(a[i])
            end
        end
        function myvexp_avx!(b, a)
            @_avx for i ∈ eachindex(a)
                b[i] = exp(a[i])
            end
        end

        vexpsq = :(for i ∈ eachindex(a)
                   s += exp(a[i])
                   end)
        lsvexps = LoopVectorization.LoopSet(vexpsq);
        @test LoopVectorization.choose_order(lsvexps) == (Symbol[:i], :i, 1, -1)

        function myvexp(a)
            s = zero(eltype(a))
            @inbounds for i ∈ eachindex(a)
                s += exp(a[i])
            end
            s
        end
        function myvexpavx(a)
            s = zero(eltype(a))
            @avx for i ∈ eachindex(a)
                s += exp(a[i])
            end
            s
        end
        function myvexp_avx(a)
            s = zero(eltype(a))
            @_avx for i ∈ eachindex(a)
                s += exp(a[i])
            end
            s
        end
        function trianglelogdetavx(L)
            ld = zero(eltype(L))
            @avx for i ∈ 1:size(L,1)
                ld += log(L[i,i])
            end
            ld
        end
        function trianglelogdet_avx(L)
            ld = zero(eltype(L))
            @_avx for i ∈ 1:size(L,1)
                ld += log(L[i,i])
            end
            ld
        end
        ldq = :(for i ∈ 1:size(L,1)
                ld += log(L[i,i])
                end)
        lsld = LoopVectorization.LoopSet(ldq)
        @test LoopVectorization.choose_order(lsld) == (Symbol[:i], :i, 1, -1)

        function logsumexp!(r::AbstractArray{T}, x::AbstractArray{T}) where {T}
            n = length(x)
            length(r) == n || throw(DimensionMismatch())
            isempty(x) && return -T(Inf)
            1 == stride(r,1) == stride(x,1) || throw(error("Arrays not strided"))

            u = maximum(x)                                       # max value used to re-center
            abs(u) == Inf && return any(isnan, x) ? T(NaN) : u   # check for non-finite values
            s = zero(T)
            @inbounds for i = 1:n
                tmp = exp(x[i] - u)
                r[i] = tmp
                s += tmp
            end

            invs = inv(s)
            r .*= invs

            return log1p(s-1) + u
        end
        function logsumexpavx!(r::AbstractArray{T}, x::AbstractArray{T}) where {T}
            n = length(x)
            length(r) == n || throw(DimensionMismatch())
            isempty(x) && return -T(Inf)
            1 == stride(r,1) == stride(x,1) || throw(error("Arrays not strided"))

            u = maximum(x)                                       # max value used to re-center
            abs(u) == Inf && return any(isnan, x) ? T(NaN) : u   # check for non-finite values
            s = zero(T)
            @avx for i = 1:n
                tmp = exp(x[i] - u)
                r[i] = tmp
                s += tmp
            end

            invs = inv(s)
            r .*= invs

            return log1p(s-1) + u
        end
        function logsumexp_avx!(r::AbstractArray{T}, x::AbstractArray{T}) where {T}
            n = length(x)
            length(r) == n || throw(DimensionMismatch())
            isempty(x) && return -T(Inf)
            1 == stride(r,1) == stride(x,1) || throw(error("Arrays not strided"))

            u = maximum(x)                                       # max value used to re-center
            abs(u) == Inf && return any(isnan, x) ? T(NaN) : u   # check for non-finite values
            s = zero(T)
            @_avx for i = 1:n
                tmp = exp(x[i] - u)
                r[i] = tmp
                s += tmp
            end

            invs = inv(s)
            r .*= invs

            return log1p(s-1) + u
        end
        feq = :(for i = 1:n
                tmp = exp(x[i] - u)
                r[i] = tmp
                s += tmp
                end)
        lsfeq = LoopVectorization.LoopSet(feq);
        lsfeq.operations
        
        
        for T ∈ (Float32, Float64)
            @show T, @__LINE__
            a = randn(T, 127);
            b1 = similar(a);
            b2 = similar(a);

            myvexp!(b1, a)
            myvexpavx!(b2, a)
            @test b1 ≈ b2
            fill!(b2, -999.9); myvexp_avx!(b2, a)
            @test b1 ≈ b2
            s = myvexp(a)
            @test s ≈ myvexpavx(a)
            @test s ≈ myvexp_avx(a)
            @test b1 ≈ @avx exp.(a)

            A = rand(T, 73, 73);
            ld = logdet(UpperTriangular(A))
            @test ld ≈ trianglelogdetavx(A)
            @test ld ≈ trianglelogdet_avx(A)

            x = rand(1000);
            r1 = similar(x);
            r2 = similar(x);
            lse = logsumexp!(r1, x);
            @test logsumexpavx!(r2, x) ≈ lse
            @test r1 ≈ r2;
            fill!(r2, T(NaN));
            @test logsumexp_avx!(r2, x) ≈ lse
            @test r1 ≈ r2;
        end
    end

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
        function mygemv_avx!(y, A, x)
            @_avx for i ∈ eachindex(y)
                yᵢ = zero(eltype(y))
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
        function AtmulvB_avx3!(G, B,κ)
            d = size(G,1)
            @_avx for d1=1:d
                G[d1,κ] = B[1,d1]*B[1,κ]
                for d2=2:d
                    G[d1,κ] += B[d2,d1]*B[d2,κ]
                end
            end
        end


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



@testset "Miscellaneous" begin
    using LoopVectorization
    dot3q = :(for m ∈ 1:M, n ∈ 1:N
            s += x[m] * A[m,n] * y[n]
              end)
    lsdot3 = LoopVectorization.LoopSet(dot3q);
    LoopVectorization.choose_order(lsdot3)

    @static if VERSION < v"1.4"
        dot3(x, A, y) = dot(x, A * y)
    else
        dot3(x, A, y) = dot(x, A, y)
    end
    function dot3avx(x, A, y)
        M, N = size(A)
        s = zero(promote_type(eltype(x), eltype(A), eltype(y)))
        @avx for m ∈ 1:M, n ∈ 1:N
            s += x[m] * A[m,n] * y[n]
        end
        s
    end
    function dot3_avx(x, A, y)
        M, N = size(A)
        s = zero(promote_type(eltype(x), eltype(A), eltype(y)))
        @_avx for m ∈ 1:M, n ∈ 1:N
            s += x[m] * A[m,n] * y[n]
        end
        s
    end
    
    subcolq = :(for i ∈ 1:size(A,2), j ∈ eachindex(x)
                B[j,i] = A[j,i] - x[j]
                end)
    lssubcol = LoopVectorization.LoopSet(subcolq);
    @test LoopVectorization.choose_order(lssubcol) == (Symbol[:j,:i], :j, 4, -1)
    ## @avx is SLOWER!!!!
    ## need to fix!
    function mysubcol!(B, A, x)
        @inbounds for i ∈ 1:size(A,2)
            @simd for j ∈ eachindex(x)
                B[j,i] = A[j,i] - x[j]
            end
        end
    end
    function mysubcolavx!(B, A, x)
        @avx for i ∈ 1:size(A,2), j ∈ eachindex(x)
            B[j,i] = A[j,i] - x[j]
        end
    end
    function mysubcol_avx!(B, A, x)
        @_avx for i ∈ 1:size(A,2), j ∈ eachindex(x)
            B[j,i] = A[j,i] - x[j]
        end
    end

    colsumq = :(for i ∈ 1:size(A,2), j ∈ eachindex(x)
                x[j] += A[j,i] - 0.25
                end)
    lscolsum = LoopVectorization.LoopSet(colsumq);
    @test LoopVectorization.choose_order(lscolsum) == (Symbol[:j,:i], :j, 4, -1)

    # my colsum is wrong (by 0.25), but slightly more interesting
    function mycolsum!(x, A)
        @. x = 0
        @inbounds for i ∈ 1:size(A,2)
            @simd for j ∈ eachindex(x)
                x[j] += A[j,i] - 0.25
            end
        end
    end
    function mycolsumavx!(x, A)
        @avx for j ∈ eachindex(x)
            xⱼ = zero(eltype(x))
            for i ∈ 1:size(A,2)
                xⱼ += A[j,i] - 0.25
            end
            x[j] = xⱼ
        end
    end
    function mycolsum_avx!(x, A)
        @_avx for j ∈ eachindex(x)
            xⱼ = zero(eltype(x))
            for i ∈ 1:size(A,2)
                xⱼ += A[j,i] - 0.25
            end
            x[j] = xⱼ
        end
    end

    varq = :(for j ∈ eachindex(s²), i ∈ 1:size(A,2)
             δ = A[j,i] - x̄[j]
             s²[j] += δ*δ
             end)
    lsvar = LoopVectorization.LoopSet(varq);
    LoopVectorization.choose_order(lsvar)
    @test LoopVectorization.choose_order(lsvar) == (Symbol[:j,:i], :j, 4, -1)

    function myvar!(s², A, x̄)
        @. s² = 0
        @inbounds for i ∈ 1:size(A,2)
            @simd for j ∈ eachindex(s²)
                δ = A[j,i] - x̄[j]
                s²[j] += δ*δ
            end
        end
    end
    function myvaravx!(s², A, x̄)
        @avx for j ∈ eachindex(s²)
            s²ⱼ = zero(eltype(s²))
            x̄ⱼ = x̄[j]
            for i ∈ 1:size(A,2)
                δ = A[j,i] - x̄ⱼ
                s²ⱼ += δ*δ
            end
            s²[j] = s²ⱼ
        end
    end
    function myvar_avx!(s², A, x̄)
        @_avx for j ∈ eachindex(s²)
            s²ⱼ = zero(eltype(s²))
            x̄ⱼ = x̄[j]
            for i ∈ 1:size(A,2)
                δ = A[j,i] - x̄ⱼ
                s²ⱼ += δ*δ
            end
            s²[j] = s²ⱼ
        end
    end


    function mvp(P, basis, coeffs::Vector{T}) where {T}
        len_c = length(coeffs)
        len_P = size(P, 1)
        p = zero(T)
        for n = 1:len_c
            pn = coeffs[n]
            for a = 1:len_P
                pn *= P[a, basis[a, n]]
            end
            p += pn
        end
        return p
    end
    function mvpavx(P, basis, coeffs::Vector{T}) where {T}
        len_c = length(coeffs)
        len_P = size(P, 1)
        p = zero(T)
        @avx for n = 1:len_c
            pn = coeffs[n]
            for a = 1:len_P
                pn *= P[a, basis[a, n]]
            end
            p += pn
        end
        return p
    end
    function mvp_avx(P, basis, coeffs::Vector{T}) where {T}
        len_c = length(coeffs)
        len_P = size(P, 1)
        p = zero(T)
        @_avx for n = 1:len_c
            pn = coeffs[n]
            for a = 1:len_P
                pn *= P[a, basis[a, n]]
            end
            p += pn
        end
        return p
    end
    bq = :(for n = 1:len_c
           pn = coeffs[n]
           for a = 1:len_P
                pn *= P[a, basis[a, n]]
            end
            p += pn
           end)
    lsb = LoopVectorization.LoopSet(bq);

    
    for T ∈ (Float32, Float64)
        @show T, @__LINE__
        A = randn(T, 199, 498);
        x = randn(T, size(A,1));
        B1 = similar(A); B2 = similar(A);

        mysubcol!(B1, A, x)
        mysubcolavx!(B2, A, x)
        @test B1 ≈ B2
        fill!(B2, T(NaN)); mysubcolavx!(B2, A, x)
        @test B1 ≈ B2

        x1 = similar(x); x2 = similar(x);
        mycolsum!(x1, A)
        mycolsumavx!(x2, A)
        @test x1 ≈ x2
        fill!(x2, T(NaN)); mycolsum_avx!(x2, A)
        @test x1 ≈ x2

        x̄ = x1 ./ size(A,2);
        myvar!(x1, A, x̄)
        myvaravx!(x2, A, x̄)
        @test x1 ≈ x2
        fill!(x2, T(NaN)); myvar_avx!(x2, A, x̄)
        @test x1 ≈ x2

        M, N = 47, 73;
        x = rand(T, M); A = rand(T, M, N); y = rand(T, N);
        d3 = dot3(x, A, y)
        @test dot3avx(x, A, y) ≈ d3
        @test dot3_avx(x, A, y) ≈ d3

        maxdeg = 20; nbasis = 1_000; dim = 15;
        r = T == Float32 ? (Int32(1):Int32(maxdeg+1)) : (1:maxdeg+1)
        basis = rand(r, (dim, nbasis));
        coeffs = rand(T, nbasis);
        P = rand(T, dim, maxdeg+1);
        mvp(P, basis, coeffs)
        mvpavx(P, basis, coeffs)
        mvpv = mvp(P, basis, coeffs)
        @test mvpv ≈ mvpavx(P, basis, coeffs)
        @test mvpv ≈ mvp_avx(P, basis, coeffs)
    end
end

@testset "broadcast" begin
    M, N = 37, 47
    # M = 77;
    for T ∈ (Float32, Float64)
        @show T, @__LINE__
        a = rand(T,100,100,100);
        b = rand(T,100,100,1);
        bl = LowDimArray{(true,true,false)}(b);
        br = reshape(b, (100,100));
        c1 = a .+ b;
        c2 = @avx a .+ bl;
        @test c1 ≈ c2
        fill!(c2, 99999.9);
        @avx c2 .= a .+ br;
        @test c1 ≈ c2
        br = reshape(b, (100,1,100));
        bl = LowDimArray{(true,false,true)}(br);
        @. c1 = a + br;
        fill!(c2, 99999.9);
        @avx @. c2 = a + bl;
        @test c1 ≈ c2
        br = reshape(b, (1,100,100));
        bl = LowDimArray{(false,true,true)}(br);
        @. c1 = a + br;
        fill!(c2, 99999.9);
        @avx @. c2 = a + bl;
        @test c1 ≈ c2
        
        a = rand(T, M); B = rand(T, M, N); c = rand(T, N); c′ = c';
        d1 =      @. a + B * c′;
        d2 = @avx @. a + B * c′;
        @test d1 ≈ d2
        
        @.      d1 = a + B * c′;
        @avx @. d2 = a + B * c′;
        @test d1 ≈ d2

        d3 = a .+ B * c;
        d4 = @avx a .+ B *ˡ c;
        @test d3 ≈ d4

        fill!(d3, -1000.0);
        fill!(d4, 91000.0);

        d3 .= a .+ B * c;
        @avx d4 .= a .+ B *ˡ c;
        @test d3 ≈ d4

        fill!(d4, 91000.0);
        @avx @. d4 = a + B *ˡ c;
        @test d3 ≈ d4

        M, K, N = 77, 83, 57;
        A = rand(T,M,K); B = rand(T,K,N); C = rand(T,M,N);

        D1 = C .+ A * B;
        D2 = @avx C .+ A *ˡ B;
        @test D1 ≈ D2

        D3 = exp.(B');
        D4 = @avx exp.(B');
        @test D3 ≈ D4

        fill!(D3, -1e3); fill!(D4, 9e9);
        Bt = Transpose(B);
        @. D3 = exp(Bt);
        @avx @. D4 = exp(Bt);
        @test D3 ≈ D4

        D1 = similar(B); D2 = similar(B);
        D1t = Transpose(D1);
        D2t = Transpose(D2);
        @. D1t = exp(Bt);
        @avx @. D2t = exp(Bt);
        @test D1t ≈ D2t

        fill!(D1, -1e3);
        fill!(D2, 9e9);
        @. D1' = exp(Bt);
        lset = @avx @. D2' = exp(Bt);
        
        @test D1 ≈ D2

        a = rand(137);
        b1 = @avx @. 3*a + sin(a) + sqrt(a);
        b2 =      @. 3*a + sin(a) + sqrt(a);
        @test b1 ≈ b2
        three = 3; fill!(b1, -9999.999);
        @avx @. b1 = three*a + sin(a) + sqrt(a);
        @test b1 ≈ b2

        C = rand(100,10,10);
        D1 = C .^ 0.3;
        D2 = @avx C .^ 0.3;
        @test D1 ≈ D2

    end
end

@testset "map" begin
    @inline foo(x, y) = exp(x) - sin(y)
    N = 37
    for T ∈ (Float32,Float64)
        @show T, @__LINE__
        a = rand(T, N); b = rand(T, N);
        c1 = map(foo, a, b);
        c2 = vmap(foo, a, b);
        @test c1 ≈ c2
    end
end

end
