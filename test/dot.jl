using LoopVectorization, OffsetArrays
using Test

@testset "dot" begin
    dotq = :(for i ∈ eachindex(a,b)
             s += a[i]*b[i]
             end)
    lsdot = LoopVectorization.LoopSet(dotq);
    @test LoopVectorization.choose_order(lsdot) == (Symbol[:i], :i, Symbol("##undefined##"), :i, 4, -1)
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
    @test LoopVectorization.choose_order(lsselfdot) == (Symbol[:i], :i, Symbol("##undefined##"), :i, 8, -1)

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
    function myselfdotavx_range(a)
        s = zero(eltype(a))
        rng = axes(a, 1)
        @avx for i ∈ rng
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
    function dot_unroll2avx(x::Vector{T}, y::Vector{T}) where {T<:Number}
        z = zero(T)
        @avx unroll=2 for i ∈ 1:length(x)
            z += x[i]*y[i]
        end
        return z
    end
    function dot_unroll3avx(x::Vector{T}, y::Vector{T}) where {T<:Number}
        z = zero(T)
        @avx unroll=3 for i ∈ 1:length(x)
            z += x[i]*y[i]
        end
        return z
    end
    function dot_unroll2avx_noinline(x::Vector{T}, y::Vector{T}) where {T<:Number}
        z = zero(T)
        @avx inline=false unroll=2 for i ∈ 1:length(x)
            z += x[i]*y[i]
        end
        return z
    end
    function dot_unroll3avx_inline(x::Vector{T}, y::Vector{T}) where {T<:Number}
        z = zero(T)
        @avx unroll=3 inline=true for i ∈ 1:length(x)
            z += x[i]*y[i]
        end
        return z
    end
    function dot_unroll2_avx(x::Vector{T}, y::Vector{T}) where {T<:Number}
        z = zero(T)
        @_avx unroll=2 for i ∈ 1:length(x)
            z += x[i]*y[i]
        end
        return z
    end
    function dot_unroll3_avx(x::Vector{T}, y::Vector{T}) where {T<:Number}
        z = zero(T)
        @_avx unroll=3 for i ∈ 1:length(x)
            z += x[i]*y[i]
        end
        return z
    end
    function complex_dot_soa(
        xre::AbstractVector{T}, xim::AbstractVector{T},
        yre::AbstractVector{T}, yim::AbstractVector{T}
    ) where {T}
        zre = zero(T)
        zim = zero(T)
        @avx for i ∈ 1:length(xre)
            zre += xre[i]*yre[i] - xim[i]*yim[i]
            zim += xre[i]*yim[i] + xim[i]*yre[i]
        end
        return Complex{T}(zre,zim)
    end
    qc = :(for i ∈ 1:length(xre)
           zre += xre[i]*yre[i] - xim[i]*yim[i]
           zim += xre[i]*yim[i] + xim[i]*yre[i]
           end);
    lsc = LoopVectorization.LoopSet(qc)
    function complex_mul_with_index_offset!(c_re, c_im, a_re, a_im, b_re, b_im)
        @inbounds @simd ivdep for i = 1:length(a_re) - 1
            c_re[i] = b_re[i] * a_re[i + 1] - b_im[i] * a_im[i + 1]
            c_im[i] = b_re[i] * a_im[i + 1] + b_im[i] * a_re[i + 1]
        end
    end
    function complex_mul_with_index_offsetavx!(c_re, c_im, a_re, a_im, b_re, b_im)
        @avx for i = 1:length(a_re) - 1
            c_re[i] = b_re[i] * a_re[i + 1] - b_im[i] * a_im[i + 1]
            c_im[i] = b_re[i] * a_im[i + 1] + b_im[i] * a_re[i + 1]
        end
    end
    function complex_mul_with_index_offset_avx!(c_re, c_im, a_re, a_im, b_re, b_im)
        @_avx for i = 1:length(a_re) - 1
            c_re[i] = b_re[i] * a_re[i + 1] - b_im[i] * a_im[i + 1]
            c_im[i] = b_re[i] * a_im[i + 1] + b_im[i] * a_re[i + 1]
        end
    end


    function pi(x, y)
        acc = 0
        @inbounds @simd for i ∈ eachindex(x)
            acc += (x[i]*x[i] + y[i]*y[i]) < 1.0
        end
        4acc/length(x)
    end
    function piavx(x, y)
        acc = 0
        @avx for i ∈ eachindex(x)
            acc += (x[i]*x[i] + y[i]*y[i]) < 1.0
        end
        4acc/length(x)
    end
    function piavx_u4(x, y)
        acc = 0
        @avx unroll=4 for i ∈ eachindex(x)
            acc += (x[i]*x[i] + y[i]*y[i]) < 1.0
        end
        4acc/length(x)
    end
    function pi_avx(x, y)
        acc = 0
        @_avx for i ∈ eachindex(x)
            acc += (x[i]*x[i] + y[i]*y[i]) < 1.0
        end
        4acc/length(x)
    end
    function pi_avx_u4(x, y)
        acc = 0
        @_avx unroll=4 for i ∈ eachindex(x)
            acc += (x[i]*x[i] + y[i]*y[i]) < 1.0
        end
        4acc/length(x)
    end

    # @macroexpand @_avx for i = 1:length(a_re) - 1
    #     c_re[i] = b_re[i] * a_re[i + 1] - b_im[i] * a_im[i + 1]
    #     c_im[i] = b_re[i] * a_im[i + 1] + b_im[i] * a_re[i + 1]
    # end

    # a = rand(400);
    for T ∈ (Float32, Float64, Int32, Int64)
        @show T, @__LINE__
        N = 143
        R = T <: Integer ? (T(-100):T(100)) : T
        a = rand(T, N); b = rand(R, N);
        ao = OffsetArray(a, -60:N-61); bo = OffsetArray(b, -60:N-61);
        s = mydot(a, b)
        @test mydotavx(a,b) ≈ s
        @test mydot_avx(a,b) ≈ s
        @test mydotavx(ao,bo) ≈ s
        @test mydot_avx(ao,bo) ≈ s
        @test dot_unroll2avx(a,b) ≈ s
        @test dot_unroll3avx(a,b) ≈ s
        @test dot_unroll2_avx(a,b) ≈ s
        @test dot_unroll3_avx(a,b) ≈ s
        @test dot_unroll2avx_noinline(a,b) ≈ s
        @test dot_unroll3avx_inline(a,b) ≈ s
        s = myselfdot(a)
        @test myselfdotavx(a) ≈ s
        @test myselfdotavx_range(a) ≈ s
        @test myselfdot_avx(a) ≈ s
        @test myselfdotavx(a) ≈ s

        if T <: Union{Float32,Float64}
            πest = pi(a, b)
            @test πest == piavx(a, b)
            @test πest == piavx_u4(a, b)
            @test πest == pi_avx(a, b)
            @test πest == pi_avx_u4(a, b)
        end

        a_re = rand(R, N); a_im = rand(R, N);
        b_re = rand(R, N); b_im = rand(R, N);
        ac = Complex.(a_re, a_im);
        bc = Complex.(b_re, b_im);

        @test mydot(ac, bc) ≈ complex_dot_soa(a_re, a_im, b_re, b_im)

        c_re1 = similar(a_re); c_im1 = similar(a_im);
        c_re2 = similar(a_re); c_im2 = similar(a_im);
        # b_re = rand(R, length(a_re) + 1); b_im = rand(R, length(a_im) + 1);
        complex_mul_with_index_offset!(c_re1, c_im1, a_re, a_im, b_re, b_im)
        complex_mul_with_index_offsetavx!(c_re2, c_im2, a_re, a_im, b_re, b_im)
        c_re1v, c_im1v, c_re2v, c_im2v = @views c_re1[1:end-1], c_im1[1:end-1], c_re2[1:end-1], c_im2[1:end-1];
        @test c_re1v ≈ c_re2v
        @test c_im1v ≈ c_im2v
        fill!(c_re2, -999999); fill!(c_im2, 99999999);
        complex_mul_with_index_offset_avx!(c_re2, c_im2, a_re, a_im, b_re, b_im)
        @test c_re1v ≈ c_re2v
        @test c_im1v ≈ c_im2v
    end
end
