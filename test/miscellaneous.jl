using LoopVectorization
using LinearAlgebra
using Test

@testset "Miscellaneous" begin

    Unum, Tnum = LoopVectorization.REGISTER_COUNT == 16 ? (2, 6) : (2, 8)
    dot3q = :(for m ∈ 1:M, n ∈ 1:N
              s += x[m] * A[m,n] * y[n]
              end);
    lsdot3 = LoopVectorization.LoopSet(dot3q);
    if LoopVectorization.REGISTER_COUNT != 8
        @test LoopVectorization.choose_order(lsdot3) == ([:n, :m], :m, :n, :m, Unum, Tnum)#&-2
    end

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
    function dot3v2avx(x, A, y)
        M, N = size(A)
        s = zero(promote_type(eltype(x), eltype(A), eltype(y)))
        @avx for n ∈ 1:N
            t = zero(s)
            for m ∈ 1:M
                t += x[m] * A[m,n]
            end
            s += t * y[n]
        end
        s
    end
    q = :( for n ∈ 1:N
            t = zero(s)
            for m ∈ 1:M
                t += x[m] * A[m,n]
            end
            s += t * y[n]
           end);
    ls = LoopVectorization.LoopSet(q);
    
    function dot3avx24(x, A, y)
        M, N = size(A)
        s = zero(promote_type(eltype(x), eltype(A), eltype(y)))
        @avx unroll=(2,4) for m ∈ 1:M, n ∈ 1:N
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
    # if LoopVectorization.REGISTER_COUNT != 8
    #     # @test LoopVectorization.choose_order(lssubcol) == (Symbol[:j,:i], :j, :i, :j, Unum, Tnum)
    #     @test LoopVectorization.choose_order(lssubcol) == (Symbol[:j,:i], :j, :i, :j, 1, 1)
    # end
    @test LoopVectorization.choose_order(lssubcol) == (Symbol[:i,:j], :i, Symbol("##undefined##"), :j, 1, -1)
    # @test LoopVectorization.choose_order(lssubcol) == (Symbol[:j,:i], :i, Symbol("##undefined##"), :j, 4, -1)
    # if LoopVectorization.REGISTER_COUNT == 32
    #     @test LoopVectorization.choose_order(lssubcol) == (Symbol[:i,:j], :j, :i, :j, 2, 10)
    # elseif LoopVectorization.REGISTER_COUNT == 16
    #     @test LoopVectorization.choose_order(lssubcol) == (Symbol[:i,:j], :j, :i, :j, 2, 6)
    # end
    # @test LoopVectorization.choose_order(lssubcol) == (Symbol[:j,:i], :j, Symbol("##undefined##"), :j, 4, -1)
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
    # if LoopVectorization.REGISTER_COUNT != 8
    #     # @test LoopVectorization.choose_order(lscolsum) == (Symbol[:j,:i], :j, :i, :j, Unum, Tnum)
    #     @test LoopVectorization.choose_order(lscolsum) == (Symbol[:j,:i], :j, :i, :j, 1, 1)
    # end
    @test LoopVectorization.choose_order(lscolsum) == (Symbol[:j,:i], :j, Symbol("##undefined##"), :j, 4, -1)
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
    # LoopVectorization.choose_order(lsvar)
    # @test LoopVectorization.choose_order(lsvar) == (Symbol[:j,:i], :j, :i, :j, Unum, Tnum)
    @test LoopVectorization.choose_order(lsvar) == (Symbol[:j,:i], :j, Symbol("##undefined##"), :j, 4, -1)
    # if LoopVectorization.REGISTER_COUNT == 32
    #     @test LoopVectorization.choose_order(lsvar) == (Symbol[:j,:i], :j, :i, :j, 2, 10)
    # elseif LoopVectorization.REGISTER_COUNT == 16
    #     @test LoopVectorization.choose_order(lsvar) == (Symbol[:j,:i], :j, :i, :j, 2, 6)
    # end
    
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

    function setcolumstovectorplus100!(Z::AbstractArray{T}, A) where {T}
        for i = axes(A,1), j = axes(Z,2)
            acc = zero(T)
            acc = acc + A[i] + 100
            Z[i, j] = acc
        end
    end
    function setcolumstovectorplus100avx!(Z::AbstractArray{T}, A) where {T} 
        @avx for i = axes(A,1), j = axes(Z,2)
            acc = zero(T)
            acc = acc + A[i] + 100
            Z[i, j] = acc
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
        C = length(coeffs)
        A = size(P, 1)
        p = zero(T)
        @avx for c ∈ 1:C
            pc = coeffs[c]
            for a = 1:A
                pc *= P[a, basis[a, c]]
            end
            p += pc
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

    function clenshaw!(ret,x,coeff)
        @inbounds for j in 1:length(ret)
            ret[j] = clenshaw(x[j], coeff)
        end
    end
    function clenshaw_avx!(ret,x,coeff)
        @_avx for j in 1:length(ret)
            ret[j] = clenshaw(x[j], coeff)
        end
    end
    function clenshawavx!(ret,x,coeff)
        @avx for j in 1:length(ret)
            ret[j] = clenshaw(x[j], coeff)
        end
    end
# ret = y2; coeff = c;
#     LoopVectorization.@avx_debug for j in 1:length(ret)
#             ret[j] = clenshaw(x[j], coeff)
#     end
#     t = β₁ = β₂ = ρ = s = 0.0; weights = rand(1); nodes = rand(1); lomnibus(args...) = +(args...)
# LoopVectorization.@avx_debug for i ∈ eachindex(weights, nodes)
#         s += weights[i] * lomnibus(nodes[i], t, β₁, β₂, ρ)
#     end
# @macroexpand @avx for i ∈ eachindex(weights, nodes)
#         s += weights[i] * lomnibus(nodes[i], t, β₁, β₂, ρ)
#     end
    function softmax3_core!(lse, qq, xx, tmpmax, maxk, nk)
        for k in Base.OneTo(maxk)
            @inbounds for i in eachindex(lse)
                tmp = exp(xx[i,k] - tmpmax[i])
                lse[i] += tmp
                qq[i,k] = tmp
            end
        end
        for k in maxk+1:nk
            @inbounds for i in eachindex(lse)
                tmp = exp(xx[i,k] - tmpmax[i])
                lse[i] += tmp
            end
        end
        qq[:,Base.OneTo(maxk)] ./= vec(lse)
    end
    function softmax3_coreavx1!(lse, qq, xx, tmpmax, maxk, nk)
        for k in Base.OneTo(maxk)
            @avx for i in eachindex(lse)
                tmp = exp(xx[i,k] - tmpmax[i])
                lse[i] += tmp
                qq[i,k] = tmp
            end
        end
        for k in maxk+1:nk
            @avx for i in eachindex(lse)
                tmp = exp(xx[i,k] - tmpmax[i])
                lse[i] += tmp
            end
        end
        qq[:,Base.OneTo(maxk)] ./= vec(lse)
    end
    function softmax3_core_avx1!(lse, qq, xx, tmpmax, maxk, nk)
        for k in Base.OneTo(maxk)
            @_avx for i in eachindex(lse)
                tmp = exp(xx[i,k] - tmpmax[i])
                lse[i] += tmp
                qq[i,k] = tmp
            end
        end
        for k in maxk+1:nk
            @_avx for i in eachindex(lse)
                tmp = exp(xx[i,k] - tmpmax[i])
                lse[i] += tmp
            end
        end
        qq[:,Base.OneTo(maxk)] ./= vec(lse)
    end
    function softmax3_coreavx2!(lse, qq, xx, tmpmax, maxk, nk)
        @avx for k in Base.OneTo(maxk)
            for i in eachindex(lse)
                tmp = exp(xx[i,k] - tmpmax[i])
                lse[i] += tmp
                qq[i,k] = tmp
            end
        end
        if maxk < nk
            @avx for k in maxk+1:nk
                for i in eachindex(lse)
                    tmp = exp(xx[i,k] - tmpmax[i])
                    lse[i] += tmp
                end
            end
        end
        qq[:,Base.OneTo(maxk)] ./= vec(lse)
    end
    function softmax3_core_avx2!(lse, qq, xx, tmpmax, maxk, nk)
        @_avx for k in Base.OneTo(maxk)
            for i in eachindex(lse)
                tmp = exp(xx[i,k] - tmpmax[i])
                lse[i] += tmp
                qq[i,k] = tmp
            end
        end
        if maxk < nk
            @_avx for k in maxk+1:nk
                for i in eachindex(lse)
                    tmp = exp(xx[i,k] - tmpmax[i])
                    lse[i] += tmp
                end
            end
        end
        qq[:,Base.OneTo(maxk)] ./= vec(lse)
    end
    function softmax3_coreavx3!(lse, qq, xx, tmpmax, maxk, nk)
        for k in Base.OneTo(nk)
            @avx for i in eachindex(lse)
                tmp = exp(xx[i,k] - tmpmax[i])
                lse[i] += tmp
                k <= maxk && (qq[i,k] = tmp)
            end
        end
        qq[:,Base.OneTo(maxk)] ./= vec(lse)
    end
    function softmax3_core_avx3!(lse, qq, xx, tmpmax, maxk, nk)
        for k in Base.OneTo(nk)
            @_avx for i in eachindex(lse)
                tmp = exp(xx[i,k] - tmpmax[i])
                lse[i] += tmp
                k <= maxk && (qq[i,k] = tmp)
            end
        end
        qq[:,Base.OneTo(maxk)] ./= vec(lse)
    end
    # qif = :(for i in eachindex(lse)
    #             tmp = exp(xx[i,k] - tmpmax[i])
    #             lse[i] += tmp
    #             k <= maxk && (qq[i,k] = tmp)
    #          end)
    # lsif = LoopVectorization.LoopSet(qif)
    function softmax3_coreavx4!(lse, qq, xx, tmpmax, maxk, nk)
        @avx for k in Base.OneTo(nk)
            for i in eachindex(lse)
                tmp = exp(xx[i,k] - tmpmax[i])
                lse[i] += tmp
                k <= maxk && (qq[i,k] = tmp)
            end
        end
        qq[:,Base.OneTo(maxk)] ./= vec(lse)
    end
    function softmax3_core_avx4!(lse, qq, xx, tmpmax, maxk, nk)
        @_avx for k in Base.OneTo(nk)
            for i in eachindex(lse)
                tmp = exp(xx[i,k] - tmpmax[i])
                lse[i] += tmp
                k <= maxk && (qq[i,k] = tmp)
            end
        end
        qq[:,Base.OneTo(maxk)] ./= vec(lse)
    end
    add_1_dim(x::AbstractArray) = reshape(x, size(x)..., 1)
    check_finite(x::AbstractArray) = all(isfinite.(x)) || throw(error("x not finite!"))
    function softmax3_setup!(q::AA, lse::A, tmpmax::A, x::AA, maxk=size(q, ndims(q)) ) where {T<:Real, A<:Array{T}, AA<:AbstractArray{T}}
        ndims(q) == 1+ndims(lse) || throw(DimensionMismatch())
        xsizes = size(x)
        xsizes == size(q) || throw(DimensionMismatch("size(x) = $(size(x)) but size(q) = $(size(q))"))
        nk = last(xsizes)
        for i = Base.OneTo(ndims(lse))
            size(q,i) == size(lse,i) == size(tmpmax,i) || throw(DimensionMismatch("size(x) = $(size(x)),  size(lse) = $(size(lse)), and size(tmpmax) = $(size(tmpmax))"))
        end
        0 < maxk <= nk || throw(DomainError(maxk))
        1 == LinearAlgebra.stride1(q) == LinearAlgebra.stride1(x) || throw(error("Arrays not strided"))
        isempty(x) && throw(error("x empty"))
        check_finite(x)
        maximum!(add_1_dim(tmpmax), x)
        fill!(lse, zero(T))
        xx = reshape(x, :, nk)
        qq = reshape(q, :, nk)
        lse, qq, xx, tmpmax, maxk, nk
    end
    function softmax3_base!(q::AA, lse::A, tmpmax::A, x::AA, maxk=size(q, ndims(q)) ) where {T<:Real, A<:Array{T}, AA<:AbstractArray{T}}
        lse, qq, xx, tmpmax, maxk, nk = softmax3_setup!(q, lse, tmpmax, x, maxk)
        softmax3_core!(lse, qq, xx, tmpmax, maxk, nk)
    end
    function softmax3avx1!(q::AA, lse::A, tmpmax::A, x::AA, maxk=size(q, ndims(q)) ) where {T<:Real, A<:Array{T}, AA<:AbstractArray{T}}
        lse, qq, xx, tmpmax, maxk, nk = softmax3_setup!(q, lse, tmpmax, x, maxk)
        softmax3_coreavx1!(lse, qq, xx, tmpmax, maxk, nk)
    end
    function softmax3_avx1!(q::AA, lse::A, tmpmax::A, x::AA, maxk=size(q, ndims(q)) ) where {T<:Real, A<:Array{T}, AA<:AbstractArray{T}}
        lse, qq, xx, tmpmax, maxk, nk = softmax3_setup!(q, lse, tmpmax, x, maxk)
        softmax3_core_avx1!(lse, qq, xx, tmpmax, maxk, nk)
    end
    function softmax3avx2!(q::AA, lse::A, tmpmax::A, x::AA, maxk=size(q, ndims(q)) ) where {T<:Real, A<:Array{T}, AA<:AbstractArray{T}}
        lse, qq, xx, tmpmax, maxk, nk = softmax3_setup!(q, lse, tmpmax, x, maxk)
        softmax3_coreavx2!(lse, qq, xx, tmpmax, maxk, nk)
    end
    function softmax3_avx2!(q::AA, lse::A, tmpmax::A, x::AA, maxk=size(q, ndims(q)) ) where {T<:Real, A<:Array{T}, AA<:AbstractArray{T}}
        lse, qq, xx, tmpmax, maxk, nk = softmax3_setup!(q, lse, tmpmax, x, maxk)
        softmax3_core_avx2!(lse, qq, xx, tmpmax, maxk, nk)
    end
    function softmax3avx3!(q::AA, lse::A, tmpmax::A, x::AA, maxk=size(q, ndims(q)) ) where {T<:Real, A<:Array{T}, AA<:AbstractArray{T}}
        lse, qq, xx, tmpmax, maxk, nk = softmax3_setup!(q, lse, tmpmax, x, maxk)
        softmax3_coreavx3!(lse, qq, xx, tmpmax, maxk, nk)
    end
    function softmax3_avx3!(q::AA, lse::A, tmpmax::A, x::AA, maxk=size(q, ndims(q)) ) where {T<:Real, A<:Array{T}, AA<:AbstractArray{T}}
        lse, qq, xx, tmpmax, maxk, nk = softmax3_setup!(q, lse, tmpmax, x, maxk)
        softmax3_core_avx3!(lse, qq, xx, tmpmax, maxk, nk)
    end
    function softmax3avx4!(q::AA, lse::A, tmpmax::A, x::AA, maxk=size(q, ndims(q)) ) where {T<:Real, A<:Array{T}, AA<:AbstractArray{T}}
        lse, qq, xx, tmpmax, maxk, nk = softmax3_setup!(q, lse, tmpmax, x, maxk)
        softmax3_coreavx4!(lse, qq, xx, tmpmax, maxk, nk)
    end
    function softmax3_avx4!(q::AA, lse::A, tmpmax::A, x::AA, maxk=size(q, ndims(q)) ) where {T<:Real, A<:Array{T}, AA<:AbstractArray{T}}
        lse, qq, xx, tmpmax, maxk, nk = softmax3_setup!(q, lse, tmpmax, x, maxk)
        softmax3_core_avx4!(lse, qq, xx, tmpmax, maxk, nk)
    end

    function sumprodavx(x)
        s = zero(eltype(x))
        p = one(eltype(x))
        @avx for i ∈ eachindex(x)
            s += x[i]
            p *=x[i]
        end
        s, p
    end
    function sumprod_avx(x)
        s = zero(eltype(x))
        p = one(eltype(x))
        @_avx for i ∈ eachindex(x)
            s += x[i]
            p *=x[i]
        end
        s, p
    end

    function test_bit_shift(counter)
        accu = zero(first(counter))
        @inbounds for i ∈ eachindex(counter)
            accu += counter[i] << 1
        end
        accu
    end
    function test_bit_shiftavx(counter)
        accu = zero(first(counter))
        @avx for i ∈ eachindex(counter)
            accu += counter[i] << 1
        end
        accu
    end
    function test_bit_shift_avx(counter)
        accu = zero(first(counter))
        @_avx for i ∈ eachindex(counter)
            accu += counter[i] << 1
        end
        accu
    end
    function test_for_with_different_index!(c, a, b, start_sample, num_samples)
        @inbounds for i = start_sample:num_samples + start_sample - 1
            c[i] = b[i] * a[i]
        end
    end
    function test_for_with_different_indexavx!(c, a, b, start_sample, num_samples)
        @avx for i = start_sample:num_samples + start_sample - 1
            c[i] = b[i] * a[i]
        end
    end
    function test_for_with_different_index_avx!(c, a, b, start_sample, num_samples)
        @_avx for i = start_sample:num_samples + start_sample - 1
            c[i] = b[i] * a[i]
        end
    end
    function rshift_i!(out)
        n = length(out)
        @inbounds for i in 1:n
            out[i] = out[i] << i
        end
    end
    function rshift_i_avx!(out)
        n = length(out)
        @avx for i in 1:n
            out[i] = out[i] << i
        end
    end
    function one_plus_i!(out)
        n = length(out)
        @inbounds for i in 1:n
            out[i] = 1 + i
        end
    end
    function one_plus_i_avx!(out)
        n = length(out)
        @avx for i in 1:n
            out[i] = 1 + i
        end
    end

    function addsumtoeach!(y, z)
        @inbounds @fastmath for i in axes(z, 1)
            @simd ivdep for j in axes(y, 1)
                y[j] = y[j] + z[i]
            end
        end
    end
    function addsumtoeachavx!(y, z)
        @avx for i in axes(z, 1)
            for j in axes(y, 1)
                y[j] = y[j] + z[i]
            end
        end
    end
    function crossedsumavx!(x, y, z)
        @avx for i in axes(x, 1)
            for j in axes(x, 2)
                x[i, j] = x[i, j] + z[i]
                y[j, i] = y[j, i] + z[i]
            end
        end
    end
    function crossedsum!(x, y, z)
        @inbounds @fastmath for i in axes(x, 1)
            for j in axes(x, 2)
                x[i, j] = x[i, j] + z[i]
                y[j, i] = y[j, i] + z[i]
            end
        end
    end
    function instruct_x_avx!(r::AbstractVector, loc::Int)
        @avx for lhs in 0:(length(r) >> 1) - (1 << (loc - 1))
            # mask locations before
            p = lhs + lhs & ~(1 << (loc - 1) - 1)
            q = lhs + lhs & ~(1 << (loc - 1) - 1) + 1 << (loc - 1)
            # swap rows
            tmp = r[p + 1]
            r[p + 1] = r[q + 1]
            r[q + 1] = tmp
        end
        return r
    end
    function instruct_x!(r::AbstractVector, loc::Int)
        for lhs in 0:(length(r) >> 1) - (1 << (loc - 1))
            # mask locations before
            p = lhs + lhs & ~(1 << (loc - 1) - 1)
            q = lhs + lhs & ~(1 << (loc - 1) - 1) + 1 << (loc - 1)
            # swap rows
            tmp = r[p + 1]
            r[p + 1] = r[q + 1]
            r[q + 1] = tmp
        end
        return r
    end


    function multiple_unrolls_split_depchains!(c_re::AbstractArray{T}, a_re, b_re, a_im, b_im, keep = nothing) where {T}
        for k in 1:2
            for n in 1:2
                # acc = ifelse(keep === nothing, zero(T), c_re[k, n]) # same problem
                acc = keep === nothing ? zero(T) : c_re[k, n]
                # acc = zero(T) # this works fine
                for c in 1:2
                    acc = acc + (a_re[k, n, c] * b_re[c, k] + a_im[k, n, c] * b_im[c, k])
                end
                c_re[k, n] = acc
            end
        end
        c_re
    end
    function multiple_unrolls_split_depchains_avx!(c_re::AbstractArray{T}, a_re, b_re, a_im, b_im, keep = nothing) where {T}
        @avx for k in 1:2
            for n in 1:2
                # acc = ifelse(keep === nothing, zero(T), c_re[k, n]) # same problem
                acc = keep === nothing ? zero(T) : c_re[k, n]
                # acc = zero(T) # this works fine
                for c in 1:2
                    acc = acc + (a_re[k, n, c] * b_re[c, k] + a_im[k, n, c] * b_im[c, k])
                end
                c_re[k, n] = acc
            end
        end
        c_re
    end

    function MatCalcWtDW!(m)
        l, n = size(m.Wt)
        fill!(m.Wt_D_W, 0)
        @avx for k in 1: n
            for j in 1:l
                for i in 1:l
                    m.Wt_D_W[i, j] += m.Wt[i, k] * m.Wt[j, k] * m.d[k]
                end
            end
        end
    end
    function loopinductvardivision(τ)
        M,N = size(τ)
        for t = 1:N, j = 1:M
            τ[j, t] = ((j - 1) / (M - 1))
        end
        τ
    end
    function loopinductvardivisionavx(τ)
        M,N = size(τ)
        @avx for t = 1:N, j = 1:M
            τ[j, t] = ((j - 1) / (M - 1))
        end
        τ
    end
function maxavx!(R::AbstractArray{T}, Q, keep=nothing) where T
    @avx for i in axes(Q,1)
        # acc = -999 # works fine
        acc = ifelse(isnothing(keep), typemin(T), R[i])
        for j in axes(Q,2), k in axes(Q,3)
            acc = max(acc, Q[i, j, k])
        end
        R[i] = acc
    end
    R
end
function splitintonoloop(U = randn(2,2), E1 = randn(2))
    t = 1
    a = 1.0
    _s = 0.0
    n, k = size(U)
    @avx for j = 1:k
        for i = 1:n
            u = tanh(a * U[i,j])
            v = a * (1 - t * t)
            U[i,j] = u
            _s += v
        end
        E1[j] = _s / n
    end
    U, E1
end
function splitintonoloop_reference(U = randn(2,2), E1 = randn(2))
    t = 1
    a = 1.0
    _s = 0.0
    n, k = size(U)
    for j = 1:k
        for i = 1:n
            u = tanh(a * U[i,j])
            v = a * (1 - t * t)
            U[i,j] = u
            _s += v
        end
        E1[j] = _s / n
    end
    U, E1
end
function findreducedparentfornonvecstoreavx!(U::AbstractMatrix{T}, E1::AbstractVector{T}) where T
    n,k = size(U)
    _s = zero(T)
    a = 1.0
    @avx for j = 1:k
        for i = 1:n
            t = tanh(a * U[i,j])
            U[i,j] = t
            _s += a * (1 - t^2)
        end
        E1[j] = _s / n
    end
    U,E1
end
function findreducedparentfornonvecstore!(U::AbstractMatrix{T}, E1::AbstractVector{T}) where T
    n,k = size(U)
    _s = zero(T)
    a = 1.0
    for j = 1:k
        for i = 1:n
            t = tanh(a * U[i,j])
            U[i,j] = t
            _s += a * (1 - t^2)
        end
        E1[j] = _s / n
    end
    U,E1
end




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

        @test instruct_x!(x1, 2) ≈ instruct_x_avx!(x2, 2)
        @test instruct_x!(x1, 3) ≈ instruct_x_avx!(x2, 3)
        @test instruct_x!(x1, 4) ≈ instruct_x_avx!(x2, 4)

        M, N = 47, 73;
        x = rand(T, M); A = rand(T, M, N); y = rand(T, N);
        d3 = dot3(x, A, y)
        @test dot3avx(x, A, y) ≈ d3
        @test dot3v2avx(x, A, y) ≈ d3
        @test dot3_avx(x, A, y) ≈ d3

        A2 = similar(A);
        setcolumstovectorplus100!(A, x)
        setcolumstovectorplus100avx!(A2, x)
        @test A == A2
        
        maxdeg = 20; nbasis = 1_000; dim = 15;
        r = T == Float32 ? (Int32(1):Int32(maxdeg+1)) : (1:maxdeg+1)
        basis = rand(r, (dim, nbasis));
        coeffs = rand(T, nbasis);
        P = rand(T, dim, maxdeg+1);
        # mvp(P, basis, coeffs)
        # mvpavx(P, basis, coeffs)
        mvpv = mvp(P, basis, coeffs)
        @test mvpv ≈ mvpavx(P, basis, coeffs)
        if VERSION > v"1.1"
            # Locally, this passes on version 1.1
            # However, it does not pass on Travis on 1.1.
            @test mvpv ≈ mvp_avx(P, basis, coeffs)
        end

        c = rand(T,100); x = rand(T,10^4); y1 = similar(x); y2 = similar(x);
        clenshaw!(y1,x,c)
        clenshaw_avx!(y2,x,c)
        @test y1 ≈ y2
        fill!(y2, NaN); clenshawavx!(y2,x,c)
        @test y1 ≈ y2

        C = randn(T, 199, 498);
        start_sample = 29; num_samples = 800;
        test_for_with_different_index!(B1, A, C, start_sample, num_samples)
        test_for_with_different_indexavx!(B2, A, C, start_sample, num_samples)
        r = start_sample:start_sample+num_samples - 1
        @test view(vec(B1), r) == view(vec(B2), r)
        fill!(B2, NaN); test_for_with_different_index_avx!(B2, A, C, start_sample, num_samples)
        @test view(vec(B1), r) == view(vec(B2), r)

        ni, nj, nk = (127, 113, 13)
        x = rand(T, ni, nj, nk);
        q1 = similar(x);
        q2 = similar(x);
        tmpmax = zeros(T, ni,nj);
        lse = similar(tmpmax);
        fill!(q1, 0); fill!(lse, 0);  softmax3_base!(q1, lse, tmpmax, x);

        fill!(q2, 0); fill!(lse, 0);  softmax3avx1!(q2, lse, tmpmax, x, 1);
        @test all(sum(q2; dims=3) .<= 1)
        fill!(q2, 0); fill!(lse, 0);  softmax3avx1!(q2, lse, tmpmax, x);
        @test q1 ≈ q2
        @test sum(q2; dims=3) ≈ ones(T,ni,nj)

        fill!(q2, 0); fill!(lse, 0);  softmax3_avx1!(q2, lse, tmpmax, x, 1);
        @test all(sum(q2; dims=3) .<= 1)
        fill!(q2, 0); fill!(lse, 0);  softmax3_avx1!(q2, lse, tmpmax, x);
        @test q1 ≈ q2
        @test sum(q2; dims=3) ≈ ones(T,ni,nj)

        fill!(q2, 0); fill!(lse, 0);  softmax3avx2!(q2, lse, tmpmax, x, 1);
        @test all(sum(q2; dims=3) .<= 1)
        fill!(q2, 0); fill!(lse, 0);  softmax3avx2!(q2, lse, tmpmax, x);
        @test q1 ≈ q2
        @test sum(q2; dims=3) ≈ ones(T,ni,nj)

        fill!(q2, 0); fill!(lse, 0);  softmax3_avx2!(q2, lse, tmpmax, x, 1);
        @test all(sum(q2; dims=3) .<= 1)
        fill!(q2, 0); fill!(lse, 0);  softmax3_avx2!(q2, lse, tmpmax, x);
        @test q1 ≈ q2
        @test sum(q2; dims=3) ≈ ones(T,ni,nj)

        fill!(q2, 0); fill!(lse, 0);  softmax3avx3!(q2, lse, tmpmax, x, 1);
        @test all(sum(q2; dims=3) .<= 1)
        fill!(q2, 0); fill!(lse, 0);  softmax3avx3!(q2, lse, tmpmax, x);
        @test q1 ≈ q2
        @test sum(q2; dims=3) ≈ ones(T,ni,nj)

        fill!(q2, 0); fill!(lse, 0);  softmax3_avx3!(q2, lse, tmpmax, x, 1);
        @test all(sum(q2; dims=3) .<= 1)
        fill!(q2, 0); fill!(lse, 0);  softmax3_avx3!(q2, lse, tmpmax, x);
        @test q1 ≈ q2
        @test sum(q2; dims=3) ≈ ones(T,ni,nj)

        fill!(q2, 0); fill!(lse, 0);  softmax3avx4!(q2, lse, tmpmax, x, 1);
        @test all(sum(q2; dims=3) .<= 1)
        fill!(q2, 0); fill!(lse, 0);  softmax3avx4!(q2, lse, tmpmax, x);
        @test q1 ≈ q2
        @test sum(q2; dims=3) ≈ ones(T,ni,nj)

        fill!(q2, 0); fill!(lse, 0);  softmax3_avx4!(q2, lse, tmpmax, x, 1);
        @test all(sum(q2; dims=3) .<= 1)
        fill!(q2, 0); fill!(lse, 0);  softmax3_avx4!(q2, lse, tmpmax, x);
        @test q1 ≈ q2
        @test sum(q2; dims=3) ≈ ones(T,ni,nj)

        x .+= 0.545;
        s = sum(x); p = prod(x)
        s1, p1 = sumprodavx(x)
        @test s ≈ s1
        @test p ≈ p1
        s1, p1 = sumprod_avx(x)
        @test s ≈ s1
        @test p ≈ p1
        r = T == Float32 ? (Int32(-10):Int32(107)) : (Int64(-10):Int64(107))
        s = sum(r); p = prod(r)
        s1, p1 = sumprodavx(r)
        @test s ≈ s1
        @test p ≈ p1
        s1, p1 = sumprod_avx(r)
        @test s ≈ s1
        @test p ≈ p1

        @test test_bit_shift(r) == test_bit_shiftavx(r)
        @test test_bit_shift(r) == test_bit_shift_avx(r)

        r = T(-10):T(2.3):T(1000)
        s = if VERSION >= v"1.5.0-DEV.255" || T != Float32
            sum(r)
        else
            sum(identity, r)
        end
        p = prod(r);
        s1, p1 = sumprodavx(r)
        @test s ≈ s1
        @test p ≈ p1
        s1, p1 = sumprod_avx(r)
        @test s ≈ s1
        @test p ≈ p1

        x1 = rand(47); x2 = copy(x1);
        z = rand(83);
        addsumtoeach!(x1, z)
        addsumtoeachavx!(x2, z)
        @test x1 ≈ x2
        X1 = rand(83, 47); X2 = copy(X1);
        Y1 = rand(47, 83); Y2 = copy(Y1);
        crossedsum!(X1, Y1, z)
        crossedsumavx!(X2, Y2, z)
        @test X1 ≈ X2
        @test Y1 ≈ Y2

        a_re, a_im = rand(T, 2, 2, 2), rand(T, 2, 2, 2);
        b_re, b_im = rand(T, 2, 2), rand(T, 2, 2);
        c_re_1 = ones(T, 2, 2); c_re_2 = ones(T, 2, 2);
        multiple_unrolls_split_depchains!(c_re_1, a_re, b_re, a_im, b_im, true) # [1 1; 1 1]
        multiple_unrolls_split_depchains_avx!(c_re_2, a_re, b_re, a_im, b_im, true) # [1 1; 1 1]
        @test c_re_1 ≈ c_re_2
        multiple_unrolls_split_depchains!(c_re_1, a_re, b_re, a_im, b_im) # [1 1; 1 1]
        multiple_unrolls_split_depchains_avx!(c_re_2, a_re, b_re, a_im, b_im) # [1 1; 1 1]
        @test c_re_1 ≈ c_re_2

        @test loopinductvardivision(X1) ≈ loopinductvardivisionavx(X2)
        
        mh = (
            Wt_D_W = Matrix{T}(undef, 181, 181),
            Wt = rand(T, 181, 191),
            d = rand(T, 191)
        );
        Wt_D_W = similar(mh.Wt_D_W);

        MatCalcWtDW!(mh)
        @test mh.Wt_D_W ≈ mh.Wt * Diagonal(mh.d) * mh.Wt'

        Q = rand(T, 43, 47, 51);
        R = rand(T, 43);
        @test maxavx!(R, Q) == vec(maximum(Q, dims=(2,3)))
        R .+= randn.(T); Rc = copy(R);
        @test maxavx!(R, Q, true) == max.(vec(maximum(Q, dims=(2,3))), Rc)

        U0 = randn(5,7); E0 = randn(7);
        U1, E1 = splitintonoloop_reference(copy(U0), copy(E0));
        U2, E2 = splitintonoloop(copy(U0), copy(E0));
        @test U1 ≈ U2
        @test E1 ≈ E2
        U3, E3 = findreducedparentfornonvecstoreavx!(copy(U0), copy(E0));
        findreducedparentfornonvecstore!(U0, E0);
        @test U0 ≈ U3
        @test E0 ≈ E3        
    end
    for T ∈ [Int16, Int32, Int64]
        n = 8sizeof(T) - 1
        out1 = rand(T(1):T(1_000), n);
        out2 = copy(out1);
        rshift_i!(out1)
        rshift_i_avx!(out2)
        @test out1 == out2
        one_plus_i!(out1)
        one_plus_i_avx!(out2)
        @test out1 == out2
    end

function smoothdim!(s, x, α, Rpre, irng::AbstractUnitRange, Rpost)
    ifirst, ilast = first(irng), last(irng)
    ifirst > ilast && return s
    # @inbounds @fastmath for Ipost in Rpost
    for Ipost in Rpost
        # Initialize the first value along the filtered dimension
        for Ipre in Rpre
            s[Ipre, ifirst, Ipost] = x[Ipre, ifirst, Ipost]
        end
        # Handle all other entries
        for i = ifirst+1:ilast
            for Ipre in Rpre
                s[Ipre, i, Ipost] = α*x[Ipre, i, Ipost] + (1-α)*x[Ipre, i-1, Ipost]
            end
        end
    end
    s
end
function smoothdim_avx!(s, x, α, Rpre, irng::AbstractUnitRange, Rpost)
    ifirst, ilast = first(irng), last(irng)
    ifirst > ilast && return s
    @avx for Ipost in Rpost
        for Ipre in Rpre
            s[Ipre, ifirst, Ipost] = x[Ipre, ifirst, Ipost]
            for i = ifirst+1:ilast
                s[Ipre, i, Ipost] = α*x[Ipre, i, Ipost] + (1-α)*x[Ipre, i-1, Ipost]
            end
        end
    end
    s
end
function smoothdim_ifelse_avx!(s, x, α, Rpre, irng::AbstractUnitRange, Rpost)
    ifirst, ilast = first(irng), last(irng)
    ifirst > ilast && return s
    @avx for Ipost in Rpost, i = ifirst:ilast, Ipre in Rpre
        xi = x[Ipre, i, Ipost]
        xim = i > ifirst ? x[Ipre, i-1, Ipost] : xi
        s[Ipre, i, Ipost] = α*xi + (1-α)*xim
    end
    s
end

    for T ∈ (Float32, Float64)
        @testset "Mixed CartesianIndex/Int indexing" begin
            @show T, @__LINE__
            # A demo similar to the exponential filtering demo from https://julialang.org/blog/2016/02/iteration/,
            # but with no loop-carried dependency.

            # s = dest1; 
            # ifirst, ilast = first(axes(x, d)), last(axes(x, d))
            # ls = LoopVectorization.@avx_debug for Ipost in Rpost, i = ifirst:ilast, Ipre in Rpre
            #     xi = x[Ipre, i, Ipost]
            #     xim = i > ifirst ? x[Ipre, i-1, Ipost] : xi
            #     s[Ipre, i, Ipost] = α*xi + (1-α)*xim
            # end
            # LoopVectorization.choose_order(ls);

            M = 11;
            x = rand(M,M,M,M,M);
            dest1, dest2 = similar(x), similar(x);
            α = 0.3
            for d = 1:ndims(x)
                # @show d
                Rpre  = CartesianIndices(axes(x)[1:d-1]);
                Rpost = CartesianIndices(axes(x)[d+1:end]);
                smoothdim!(dest1, x, α, Rpre, axes(x, d), Rpost);
                smoothdim_avx!(dest2, x, α, Rpre, axes(x, d), Rpost);
                @test dest1 ≈ dest2
                fill!(dest2, NaN); smoothdim_ifelse_avx!(dest2, x, α, Rpre, axes(x, d), Rpost);
                @test dest1 ≈ dest2
            end
        end
    end
end


