using LoopVectorization, Random, Test
T = Float32

@testset "ifelse (masks)" begin

    function promote_bool_store!(z, x, y)
        for i ∈ eachindex(x)
            z[i] = (x[i]*x[i] + y[i]*y[i]) < 1
        end
        z
    end
    function promote_bool_storeavx!(z, x, y)
        @avx for i ∈ eachindex(x)
            z[i] = (x[i]*x[i] + y[i]*y[i]) < 1
        end
        z
    end
    function promote_bool_store_avx!(z, x, y)
        @_avx for i ∈ eachindex(x)
            z[i] = (x[i]*x[i] + y[i]*y[i]) < 1
        end
        z
    end
    # @macroexpand @_avx for i ∈ eachindex(x)
            # z[i] = (x[i]*x[i] + y[i]*y[i]) < 1
        # end
    function promote_bool_storeavx2!(z, x, y)
        @avx for i ∈ eachindex(x)
            z[i] = (x[i]*x[i] + y[i]*y[i]) < 1 ? 1 : 0
        end
        z
    end
    function promote_bool_store_avx2!(z, x, y)
        @_avx for i ∈ eachindex(x)
            z[i] = (x[i]*x[i] + y[i]*y[i]) < 1 ? 1 : 0
        end
        z
    end

    function Bernoulli_logit(y::BitVector, α::AbstractVector{T}) where {T}
        t = zero(promote_type(Float64,T))
        @inbounds for i ∈ eachindex(α)
            invOmP = 1 + exp(α[i])
            nlogOmP = log(invOmP)
            nlogP = nlogOmP - α[i]
            t -= y[i] ? nlogP : nlogOmP
        end
        t
    end
    function Bernoulli_logitavx(y::BitVector, α::AbstractVector{T}) where {T}
        t = zero(T === Int32 ? Float32 : Float64)
        @avx for i ∈ eachindex(α)
            invOmP = 1 + exp(α[i])
            nlogOmP = log(invOmP)
            nlogP = nlogOmP - α[i]
            t -= y[i] ? nlogP : nlogOmP
        end
        t
    end
    function Bernoulli_logit_avx(y::BitVector, α::AbstractVector{T}) where {T}
        t = zero(T === Int32 ? Float32 : Float64)
        @_avx for i ∈ eachindex(α)
            invOmP = 1 + exp(α[i])
            nlogOmP = log(invOmP)
            nlogP = nlogOmP - α[i]
            t -= y[i] ? nlogP : nlogOmP
        end
        t
    end
    
    function addormul!(c, a, b)
        for i ∈ eachindex(c,a,b)
            c[i] = a[i] > b[i] ? a[i] + b[i] : a[i] * b[i]
        end
    end
    function addormul_avx!(c, a, b)
        @_avx for i ∈ eachindex(c,a,b)
            c[i] = a[i] > b[i] ? a[i] + b[i] : a[i] * b[i]
        end
    end
    function addormulavx!(c, a, b)
        @avx for i ∈ eachindex(c,a,b)
            c[i] = if a[i] > b[i]
                a[i] + b[i]
            else
                a[i] * b[i]
            end
        end
    end
    function addormulp1!(c, a, b)
        for i ∈ eachindex(c,a,b)
            c[i] = 1 + (a[i] > b[i] ? a[i] + b[i] : a[i] * b[i])
        end
    end
    function addormulp1_avx!(c, a, b)
        @_avx for i ∈ eachindex(c,a,b)
            c[i] = 1 + (a[i] > b[i] ? a[i] + b[i] : a[i] * b[i])
        end
    end
    function addormulp1avx!(c, a, b)
        @avx for i ∈ eachindex(c,a,b)
            a_greater_than_b = a[i] > b[i]
            c[i] = 1 + (a_greater_than_b ? a[i] + b[i] : a[i] * b[i])
        end
    end
    function addifelsemul_avx!(c, a, b)
        @_avx for i ∈ eachindex(c,a,b)
            c[i] = ifelse(a[i] > b[i], a[i] + b[i], a[i] * b[i])
        end
    end
    function addifelsemulavx!(c, a, b)
        @avx for i ∈ eachindex(c,a,b)
            c[i] = ifelse(a[i] > b[i], a[i] + b[i], a[i] * b[i])
        end
    end
    function addifelsemulp1_avx!(c, a, b)
        @_avx for i ∈ eachindex(c,a,b)
            c[i] = 1 + ifelse(a[i] > b[i], a[i] + b[i], a[i] * b[i])
        end
    end
    function addifelsemulp1avx!(c, a, b)
        @avx for i ∈ eachindex(c,a,b)
            c[i] = 1 + ifelse(a[i] > b[i], a[i] + b[i], a[i] * b[i])
        end
    end
    function ifelseoverwrite!(p)
        for i ∈ eachindex(p)
            p[i] = p[i] < 0.5 ? p[i]^2 : p[i]^3
        end
    end
    function ifelseoverwriteavx!(p)
        @avx for i ∈ eachindex(p)
            p[i] = p[i] < 0.5 ? p[i]^2 : p[i]^3
        end
    end
    


    function maybewriteand!(c, a, b)
        @inbounds for i ∈ eachindex(c,a,b)
            a[i] > b[i] && (c[i] = a[i] + b[i])
        end
    end
    function maybewriteand_avx!(c, a, b)
        @_avx for i ∈ eachindex(c,a,b)
            a[i] > b[i] && (c[i] = a[i] + b[i])
        end
    end
    function maybewriteandavx!(c, a, b)
        @avx for i ∈ eachindex(c,a,b)
            a[i] > b[i] && (c[i] = a[i] + b[i])
        end
    end
    function maybewriteor!(c, a, b)
        @inbounds for i ∈ eachindex(c,a,b)
            a[i] > b[i] || (c[i] = a[i] ^ b[i])
        end
    end
    function maybewriteor_avx!(c, a, b)
        @_avx for i ∈ eachindex(c,a,b)
            a[i] > b[i] || (c[i] = a[i] ^ b[i])
        end
    end
    function maybewriteoravx!(c, a, b)
        @avx for i ∈ eachindex(c,a,b)
            a[i] > b[i] || (c[i] = a[i] ^ b[i])
        end
    end
    function maybewriteor!(c::AbstractVector{<:Integer}, a, b)
        @inbounds for i ∈ eachindex(c,a,b)
            a[i] > b[i] || (c[i] = a[i] & b[i])
        end
    end
    function maybewriteor_avx!(c::AbstractVector{<:Integer}, a, b)
        @_avx for i ∈ eachindex(c,a,b)
            a[i] > b[i] || (c[i] = a[i] & b[i])
        end
    end
    function maybewriteoravx!(c::AbstractVector{<:Integer}, a, b)
        @avx for i ∈ eachindex(c,a,b)
            a[i] > b[i] || (c[i] = a[i] & b[i])
        end
    end

    function notacondload!(C, A, b)
        @inbounds for n ∈ 1:size(C,2), m ∈ 1:size(C,1)
            C[m,n] = A[m,n] * (b[n] > 0 ? b[n] : -b[n])
        end
    end
    function notacondloadavx!(C, A, b)
        @avx for n ∈ 1:size(C,2), m ∈ 1:size(C,1)
            C[m,n] = A[m,n] * (b[n] > 0 ? b[n] : -b[n])
        end
    end
    function condloadscalar!(C, A, c, b)
        @inbounds for n ∈ 1:size(C,2), m ∈ 1:size(C,1)
            C[m,n] = A[m,n] * (c[n] > 0 ? b[n] : 1) + c[n]
        end
    end
    function condloadscalaravx!(C, A, c, b)
        @avx for n ∈ 1:size(C,2), m ∈ 1:size(C,1)
            C[m,n] = A[m,n] * (c[n] > 0 ? b[n] : 1) + c[n]
        end
    end
    function maskedloadscalar!(C, A, b)
        @inbounds for n ∈ 1:size(C,2), m ∈ 1:size(C,1)
            C[m,n] = A[m,n] * (A[m,n] > 0 ? b[n] : 1)
        end
    end
    function maskedloadscalaravx!(C, A, b)
        @avx for n ∈ 1:size(C,2), m ∈ 1:size(C,1)
            C[m,n] = A[m,n] * (A[m,n] > 0 ? b[n] : 1)
        end
    end
    function AtmulBpos!(C, A, B)
        @inbounds for n ∈ 1:size(C,2), m ∈ 1:size(C,1)
            Cₘₙ = zero(eltype(C))
            @simd ivdep for k ∈ 1:size(A,1)
                Cₘₙ += A[k,m] * B[k,n]
            end
            C[m,n] > 0 && (C[m,n] = Cₘₙ)
        end
    end
    function AtmulBposavx!(C, A, B)
        @avx for n ∈ 1:size(C,2), m ∈ 1:size(C,1)
            Cₘₙ = zero(eltype(C))
            for k ∈ 1:size(A,1)
                Cₘₙ += A[k,m] * B[k,n]
            end
            C[m,n] > 0 && (C[m,n] = Cₘₙ)
        end
    end
    function AtmulBpos_avx!(C, A, B)
        @_avx for n ∈ 1:size(C,2), m ∈ 1:size(C,1)
            Cₘₙ = zero(eltype(C))
            for k ∈ 1:size(A,1)
                Cₘₙ += A[k,m] * B[k,n]
            end
            C[m,n] > 0 && (C[m,n] = Cₘₙ)
        end
    end
    function condstore!(x)
        @inbounds for i ∈ eachindex(x)
            x1 = 2*x[i]-100
            x2 = x1*x1
            x3 = x2 + x1
            x4 = x3
            x[i] = x1
            (x1 < -50) && (x[i] = x2)
            (x1 < 60) || (x[i] = x4)
        end
    end
    function condstore1avx!(x)
        @avx for i ∈ eachindex(x)
            x1 = 2*x[i]-100
            x2 = x1*x1
            x3 = x2 + x1
            x[i] = x1
            (x1 < -50) && (x[i] = x2)
            (x1 < 60) || (x[i] = x3)
        end
    end
    function condstore1_avx!(x)
        @_avx for i ∈ eachindex(x)
            x1 = 2*x[i]-100
            x2 = x1*x1
            x3 = x2 + x1
            x[i] = x1
            (x1 < -50) && (x[i] = x2)
            (x1 < 60) || (x[i] = x3)
        end
    end
    function condstore2avx!(x)
        @avx for i ∈ eachindex(x)
            x1 = 2*getindex(x, i)-100
            x2 = x1*x1
            x3 = x2 + x1
            setindex!(x, x1, i)
            (x1 < -50) && setindex!(x, x2, i)
            (x1 < 60) || setindex!(x, x3, i)
        end
    end
    function condstore2_avx!(x)
        @_avx for i ∈ eachindex(x)
            x1 = 2*getindex(x, i)-100
            x2 = x1*x1
            x3 = x2 + x1
            setindex!(x, x1, i)
            (x1 < -50) && setindex!(x, x2, i)
            (x1 < 60) || setindex!(x, x3, i)
        end
    end
    function andorassignment!(x, y, z)
        @inbounds for i ∈ eachindex(x, y, z)
            yᵢ = y[i]
            zᵢ = z[i]
            (yᵢ > 0.5) || (yᵢ *= 2)
            (zᵢ < 0.5) && (zᵢ *= 2)
            x[i] = yᵢ * zᵢ
        end
    end
    function andorassignmentavx!(x, y, z)
        @avx for i ∈ eachindex(x, y, z)
            yᵢ = y[i]
            zᵢ = z[i]
            (yᵢ > 0.5) || (yᵢ *= 2)
            (zᵢ < 0.5) && (zᵢ *= 2)
            x[i] = yᵢ * zᵢ
        end
    end
    function andorassignment_avx!(x, y, z)
        @avx for i ∈ eachindex(x, y, z)
            yᵢ = y[i]
            zᵢ = z[i]
            (yᵢ > 0.5) || (yᵢ *= 2)
            (zᵢ < 0.5) && (zᵢ *= 2)
            x[i] = yᵢ * zᵢ
        end
    end
    
    function twoifelses!(res, half, m, keep=nothing, final=true)
        𝒶𝓍j=axes(half,1)
        for j in 𝒶𝓍j
            𝓇𝒽𝓈 = if isnothing(keep)
                log(half[j]) + m[j]
            else
                res[j] + (log(half[j]) + m[j])
            end
            res[j] = isnothing(final) ? 𝓇𝒽𝓈 : exp(𝓇𝒽𝓈)
        end
        res
    end
    function twoifelses_avx!(res, half, m, keep=nothing, final=true)
        𝒶𝓍j=axes(half,1)
        @avx for j in 𝒶𝓍j
            𝓇𝒽𝓈 = if isnothing(keep)
                log(half[j]) + m[j]
            else
                res[j] + (log(half[j]) + m[j])
            end
            res[j] = if isnothing(final)
                𝓇𝒽𝓈
            else
                exp(𝓇𝒽𝓈)
            end
        end
        res
    end
    
    N = 117
    for T ∈ (Float32, Float64, Int32, Int64)
        @show T, @__LINE__
        if T <: Integer
            a = rand(-T(100):T(100), N); b = rand(-T(100):T(100), N);
        else
            a = rand(T, N); b = rand(T, N);
        end;
        c1 = similar(a); c2 = similar(a);

        promote_bool_store!(c1, a, b);
        promote_bool_storeavx!(c2, a, b);
        @test c1 == c2
        fill!(c2, -999999999); promote_bool_store_avx!(c2, a, b);
        @test c1 == c2
        fill!(c2, -999999999); promote_bool_storeavx2!(c2, a, b);
        @test c1 == c2
        fill!(c2, -999999999); promote_bool_store_avx2!(c2, a, b);
        @test c1 == c2

        fill!(c1,  999999999); addormul!(c1, a, b)
        fill!(c2, -999999999); addormul_avx!(c2, a, b)
        @test c1 ≈ c2
        fill!(c2, -999999999); addormulavx!(c2, a, b)
        @test c1 ≈ c2
        fill!(c2, -999999999); addifelsemul_avx!(c2, a, b)
        @test c1 ≈ c2
        fill!(c2, -999999999); addifelsemulavx!(c2, a, b)
        @test c1 ≈ c2
        addormulp1!(c1, a, b)
        addormulp1_avx!(c2, a, b)
        @test c1 ≈ c2
        fill!(c2, -999999999); addormulp1avx!(c2, a, b)
        @test c1 ≈ c2
        fill!(c2, -999999999); addifelsemulp1_avx!(c2, a, b)
        @test c1 ≈ c2
        fill!(c2, -999999999); addifelsemulp1avx!(c2, a, b)
        @test c1 ≈ c2

        fill!(c1, -999999999); maybewriteand!(c1, a, b)
        fill!(c2, -999999999); maybewriteand_avx!(c2, a, b)
        @test c1 ≈ c2
        fill!(c2, -999999999); maybewriteandavx!(c2, a, b)
        @test c1 ≈ c2

        fill!(c1, -999999999); maybewriteor!(c1, a, b)
        fill!(c2, -999999999); maybewriteor_avx!(c2, a, b)
        @test c1 ≈ c2
        fill!(c2, -999999999); maybewriteoravx!(c2, a, b)
        @test c1 ≈ c2

        andorassignment!(c1, a, b);
        andorassignmentavx!(c2, a, b);
        @test c1 ≈ c2
        fill!(c2, -999999999); andorassignment_avx!(c2, a, b);
        @test c1 ≈ c2

        a1 = copy(a); a2 = copy(a);
        ifelseoverwrite!(a1)
        ifelseoverwriteavx!(a2)
        @test a1 ≈ a2
        
        if T <: Union{Float32,Float64}
            a .*= 100;
        end;
        b1 = copy(a);
        b2 = copy(a);
        condstore!(b1)
        condstore1avx!(b2)
        @test b1 == b2
        copyto!(b2, a); condstore1_avx!(b2)
        @test b1 == b2
        copyto!(b2, a); condstore2avx!(b2)
        @test b1 == b2
        copyto!(b2, a); condstore2_avx!(b2)
        @test b1 == b2

        M, K, N = 83, 85, 79;
        if T <: Integer
            A = rand(T(-100):T(100), K, M);
            B = rand(T(-100):T(100), K, N);
            C1 = rand(T(-100):T(100), M, N);
            b = rand(T(-100):T(100), N);
            d = rand(T(-100):T(100), N);
        else
            A = randn(T, K, M);
            B = randn(T, K, N);
            C1 = randn(T, M, N);
            b = randn(T, N);
            d = randn(T, N);
        end;
        C2 = copy(C1); C3 = copy(C1);
        AtmulBpos!(C1, A, B)
        AtmulBposavx!(C2, A, B)
        AtmulBpos_avx!(C3, A, B)
        @test C1 ≈ C2
        @test C1 ≈ C3
        C1 = similar(B);
        C2 = similar(B);
        notacondload!(C1, B, b)
        notacondloadavx!(C2, B, b)
        @test C1 ≈ C2
        maskedloadscalar!(C1, B, b)
        maskedloadscalaravx!(C2, B, b)
        @test C1 ≈ C2
        condloadscalar!(C1, B, b, d)
        condloadscalaravx!(C2, B, b, d)
        @test C1 ≈ C2

        if T <: Integer
            half = rand(T(1):T(100), 7);
            m = rand(T(-10):T(10), 7);
        else
            half = rand(T, 7); m = rand(T, 7);
        end;
        if sizeof(T) == 4
            res1 = Vector{Float32}(undef, 7);
            res2 = Vector{Float32}(undef, 7);
        else
            res1 = Vector{Float64}(undef, 7);
            res2 = Vector{Float64}(undef, 7);
        end

        for keep ∈ (nothing,true), final ∈ (nothing,true)
            @test twoifelses!(res1, half, m) ≈ twoifelses_avx!(res2, half, m)
        end

    end
    
    
    a = rand(-10:10, 43);
    bit = a .> 0.5;
    t = Bernoulli_logit(bit, a);
    @test isapprox(t, Bernoulli_logitavx(bit, a), atol = Int === Int32 ? 0.1 : 0)
    @test isapprox(t, Bernoulli_logit_avx(bit, a), atol = Int === Int32 ? 0.1 : 0)
    a = rand(43);
    bit = a .> 0.5;
    t = Bernoulli_logit(bit, a);
    @test t ≈ Bernoulli_logitavx(bit, a)
    @test t ≈ Bernoulli_logit_avx(bit, a)

    ai = [rand(Bool) for _ in 1:71];
    bi = [rand(Bool) for _ in 1:71];
    # if LoopVectorization.VectorizationBase.AVX2 || Base.libllvm_version ≥ v"8" #FIXME Why doesn't this work on Travis Ivy Bridge Julia 1.1?
        @test (ai .& bi) == (@avx ai .& bi)
        @test (ai .| bi) == (@avx ai .| bi)
        @test (ai .⊻ bi) == (@avx ai .⊻ bi)
    # else
    #     @test_broken (ai .& bi) == (@avx ai .& bi)
    #     @test_broken (ai .| bi) == (@avx ai .| bi)
    #     @test_broken (ai .⊻ bi) == (@avx ai .⊻ bi)
    # end
    a = bitrand(127); b = bitrand(127);
    @test (a .& b) == (@avx a .& b)
    @test (a .| b) == (@avx a .| b)
    @test (a .⊻ b) == (@avx a .⊻ b)
end
