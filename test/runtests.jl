using Test
using LoopVectorization
using LinearAlgebra


@testset "LoopVectorization.jl" begin

    
    @generated function logsumexp!(r::AbstractArray{T}, x::AbstractArray{T}) where {T}
        quote
            n = length(x)
            length(r) == n || throw(DimensionMismatch())
            isempty(x) && return -T(Inf)
            1 == stride(r,1) == stride(x,1) || throw(error("Arrays not strided"))

            u = maximum(x)                                       # max value used to re-center
            abs(u) == Inf && return any(isnan, x) ? T(NaN) : u   # check for non-finite values

            s = zero(T)
            @vectorize $T for i = 1:n
                tmp = exp(x[i] - u)
                r[i] = tmp
                s += tmp
            end

            invs = inv(s)
            r .*= invs

            return log1p(s-1) + u
        end
    end

    x = collect(1:1_000) ./ 10;
    r = similar(x);

    @test logsumexp!(r, x) ≈ 102.35216846104409

    @testset "GEMM" begin
        gemmq = :(for i ∈ 1:size(A,1), j ∈ 1:size(B,2)
                  Cᵢⱼ = zero(eltype(C))
                  for k ∈ 1:size(A,2)
                  Cᵢⱼ += A[i,k] * B[k,j]
                  end
                  C[i,j] = Cᵢⱼ
                  end)

        lsgemm = LoopVectorization.LoopSet(gemmq);
        U, T = LoopVectorization.VectorizationBase.REGISTER_COUNT == 16 ? (3,4) : (6, 4)
        @test LoopVectorization.choose_order(lsgemm) == (Symbol[:j,:i,:k], :i, U, T)

        function mygemm!(C, A, B)
            @inbounds for i ∈ 1:size(A,1), j ∈ 1:size(B,2)
                Cᵢⱼ = zero(eltype(C))
                @simd ivdep for k ∈ 1:size(A,2)
                    Cᵢⱼ += A[i,k] * B[k,j]
                end
                C[i,j] = Cᵢⱼ
            end
        end
        function mygemmavx!(C, A, B)
            @avx for i ∈ 1:size(A,1), j ∈ 1:size(B,2)
                Cᵢⱼ = zero(eltype(C))
                for k ∈ 1:size(A,2)
                    Cᵢⱼ += A[i,k] * B[k,j]
                end
                C[i,j] = Cᵢⱼ
            end
        end


        for T ∈ (Float32, Float64)
            M, K, N = 72, 75, 71;
            C = Matrix{T}(undef, M, N); A = randn(T, M, K); B = randn(T, K, N);
            C2 = similar(C);
            mygemmavx!(C, A, B)
            mygemm!(C2, A, B)
            @test C ≈ C2
        end
    end

    @testset "dot" begin
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

        # a = rand(400);
        for T ∈ (Float32, Float64)
            a = rand(T, 100); b = rand(T, 100);
            @test mydotavx(a,b) ≈ mydot(a,b)
            @test myselfdotavx(a) ≈ myselfdot(a)
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

        for T ∈ (Float32, Float64)
            a = randn(T, 127);
            b1 = similar(a);
            b2 = similar(a);

            myvexp!(b1, a)
            myvexpavx!(b2, a)
            @test b1 ≈ b2
            @test myvexp(a) ≈ myvexpavx(a)
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
        @test LoopVectorization.choose_order(lsgemv) == (Symbol[:i, :j], :i, 8, -1)

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
        M, K = 51, 49
        for T ∈ (Float32, Float64)
            A = randn(T, M, K);
            x = randn(T, K);
            y1 = Vector{T}(undef, M); y2 = similar(y1);
            mygemv!(y1, A, x)
            mygemvavx!(y2, A, x)

            @test y1 ≈ y2
        end
    end



@testset "Miscellaneous" begin
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

    colsumq = :(for i ∈ 1:size(A,2), j ∈ eachindex(x)
                x[j] += A[j,i]
                end)
    lscolsum = LoopVectorization.LoopSet(colsumq);
    @test LoopVectorization.choose_order(lscolsum) == (Symbol[:j,:i], :j, 4, -1)

    function mycolsum!(x, A)
        @. x = 0
        @inbounds for i ∈ 1:size(A,2)
            @simd for j ∈ eachindex(x)
                x[j] += A[j,i]
            end
        end
    end

    function mycolsumavx!(x, A)
        @avx for j ∈ eachindex(x)
            xⱼ = zero(eltype(x))
            for i ∈ 1:size(A,2)
                xⱼ += A[j,i]
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
    @test LoopVectorization.choose_order(lsvar) == (Symbol[:j,:i], :j, 5, -1)

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

    for T ∈ (Float32, Float64)
        A = randn(T, 199, 498)
        x = randn(T, size(A,1))
        B1 = similar(A); B2 = similar(A);

        mysubcol!(B1, A, x)
        mysubcolavx!(B2, A, x)

        @test B1 ≈ B2
        x1 = similar(x); x2 = similar(x);
        mycolsum!(x1, A)
        mycolsumavx!(x2, A)

        @test x1 ≈ x2

        x̄ = x1 ./ size(A,2);
        myvar!(x1, A, x̄)
        myvaravx!(x2, A, x̄)
        @test x1 ≈ x2
    end
end

@testset "broadcast" begin
    M, N = 37, 47
    # M = 77;
    for T ∈ (Float32, Float64)
        a = rand(T, M); B = rand(T, M, N); c = rand(T, N); c′ = c';

        d1 =      @. a + B * c′;
        d2 = @avx @. a + B * c′;

        @test d1 ≈ d2
        @.      d1 = a + B * c′;
        @avx @. d2 = a + B * c′;
        @test d1 ≈ d2

        d3 = a .+ B * c;
        # no method matching _similar_for(::UnitRange{Int64}, ::Type{Any}, ::Product)
        d4 = @avx a .+ B ∗ c;
        @test d3 ≈ d4

        fill!(d3, -1000.0);
        fill!(d4, 91000.0);

        d3 .= a .+ B * c;
        @avx d4 .= a .+ B ∗ c;
        @test d3 ≈ d4

        fill!(d4, 91000.0);
        @avx @. d4 = a + B ∗ c;
        @test d3 ≈ d4

        # T = Float64
        # T = Float32
        M, K, N = 77, 83, 57;
        A = rand(T,M,K); B = rand(T,K,N); C = rand(T,M,N);

        D1 = C .+ A * B;
        D2 = @avx C .+ A ∗ B;
        @test D1 ≈ D2

        B = rand(T,K,N);
        D3 = exp.(B');
        D4 = @avx exp.(B');
        @test D3 ≈ D4

        d4q = @avx exp.(B')
        
        D3c = copy(D3); D4c = copy(D4);
        D3[1:21,1:10]
        D4[1:21,1:10]
        
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
    end
end

@testset "map" begin
    foo(x, y) = exp(x) - sin(y)
    N = 37
    for T ∈ (Float32,Float64)
        a = rand(T, N); b = rand(T, N)
        c1 = map(foo, a, b)
        c2 = vmap(foo, a, b)
        @test c1 ≈ c2
    end
end

end
