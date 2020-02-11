@testset "ifelse (masks)" begin
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
            c[i] = a[i] > b[i] ? a[i] + b[i] : a[i] * b[i]
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

    N = 117
    for T ∈ (Float32, Float64, Int32, Int64)
        @show T, @__LINE__
        if T <: Integer
            a = rand(-T(100):T(100), N); b = rand(-T(100):T(100), N);
        else
            a = rand(T, N); b = rand(T, N);
        end
        c1 = similar(a); c2 = similar(a);
        addormul!(c1, a, b)
        addormul_avx!(c2, a, b)
        @test c1 ≈ c2
        fill!(c2, -999999999); addormulavx!(c2, a, b)
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

        if T <: Union{Float32,Float64}
            a .*= 100;
        end
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
        else
            A = randn(T, K, M);
            B = randn(T, K, N);
            C1 = randn(T, M, N);
        end
        C2 = copy(C1); C3 = copy(C1);
        AtmulBpos!(C1, A, B)
        AtmulBposavx!(C2, A, B)
        AtmulBpos_avx!(C3, A, B)
        @test C1 ≈ C2
        @test C1 ≈ C3
    end
end
