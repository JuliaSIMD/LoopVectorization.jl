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
    function offset_exp!(A, B)
        @avx for i=1:size(A,1), j=1:size(B,2)
	    A[i,j] = exp(B[i,j])
        end
    end
    function offset_expavx!(A, B)
        @avx for i=1:size(A,1), j=1:size(B,2)
	    A[i,j] = exp(B[i,j])
        end
    end
    function offset_exp_avx!(A, B)
        @_avx for i=1:size(A,1), j=1:size(B,2)
	    A[i,j] = exp(B[i,j])
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

    function calc_sins!(res::AbstractArray{T}) where {T}
        code_phase_delta = T(0.01)
        @inbounds for i ∈ eachindex(res)
            res[i] = sin(i * code_phase_delta)
        end
    end
    function calc_sinsavx!(res::AbstractArray{T}) where {T}
        code_phase_delta = T(0.01)
        @avx for i ∈ eachindex(res)
            res[i] = sin(i * code_phase_delta)
        end
    end
    function calc_sins_avx!(res::AbstractArray{T}) where {T}
        code_phase_delta = T(0.01)
        @_avx for i ∈ eachindex(res)
            res[i] = sin(i * code_phase_delta)
        end
    end
    
    function logsumexp!(r::AbstractArray{T}, x::AbstractArray{T}) where {T}
        n = length(x)
        length(r) == n || throw(DimensionMismatch())
        isempty(x) && return -T(Inf)
        1 == LinearAlgebra.stride1(r) == LinearAlgebra.stride1(x) || throw(error("Arrays not strided"))

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
        1 == LinearAlgebra.stride1(r) == LinearAlgebra.stride1(x) || throw(error("Arrays not strided"))

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
        1 == LinearAlgebra.stride1(r) == LinearAlgebra.stride1(x) || throw(error("Arrays not strided"))

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
    # lsfeq.operations
    
    
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

        x = rand(T, 1000);
        r1 = similar(x);
        r2 = similar(x);
        lse = logsumexp!(r1, x);
        @test logsumexpavx!(r2, x) ≈ lse
        @test r1 ≈ r2
        fill!(r2, T(NaN));
        @test logsumexp_avx!(r2, x) ≈ lse
        @test r1 ≈ r2

        calc_sins!(r1)
        calc_sinsavx!(r2)
        @test r1 ≈ r2
        fill!(r2, NaN); calc_sins_avx!(r2)
        @test r1 ≈ r2

        N,M = 47,53
        B = reshape(cumsum(ones(T, 3N)),N,:)
        A1 = zeros(T, N, M)
        A2 = zeros(T, N, M)
        offset_exp!(A1, B)
        offset_expavx!(A2, B)
        @test A1 ≈ A2
        fill!(A2, 0); offset_exp_avx!(A2, B)
        @test A1 ≈ A2
    end
end
