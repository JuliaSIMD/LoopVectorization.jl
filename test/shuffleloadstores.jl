


function mycdot(a::AbstractVector, b::AbstractVector)
    s = zero(eltype(a))
    @fastmath @inbounds @simd for i ∈ eachindex(a)
        s += a[i]' * b[i]
    end
    s
end
function mycdot_mat(a::AbstractMatrix, b::AbstractMatrix)
    re = zero(eltype(a))
    im = zero(eltype(a))
    @avx for i ∈ axes(a,2)
        re += a[1,i] * b[1,i] + a[2,i] * b[2,i]
        im += a[1,i] * b[2,i] - a[2,i] * b[1,i]
    end
    Complex(re,im)
end
function mycdot_affine(a::AbstractVector, b::AbstractVector)
    re = zero(eltype(a))
    im = zero(eltype(a))
    # with a multiplier, we go from `i = 1 -> 2i = 2` to `i = 0 -> 2i = 0
    # 2(i+1-1) = 2i + 2 - 2, so....
    @avx for i ∈ 1:length(a)>>>1
        re += a[2i-1] * b[2i-1] + a[2i] * b[2i  ]
        im += a[2i-1] * b[2i  ] - a[2i] * b[2i-1]
    end
    Complex(re,im)
end
function mycdot_stride(a::AbstractVector, b::AbstractVector)
    re = zero(eltype(a))
    im = zero(eltype(a))
    @avx for i ∈ 1:2:length(a)
        re += a[i] * b[i  ] + a[i+1] * b[i+1]
        im += a[i] * b[i+1] - a[i+1] * b[i  ]
    end
    Complex(re,im)
end

@testset "shuffles load/stores" begin
    for i ∈ 1:128
        ac = rand(Complex{Float64}, i)
        bc = rand(Complex{Float64}, i)
        acv = reinterpret(Float64, ac)
        bcv = reinterpret(Float64, bc)
        acm = reinterpret(reshape, Float64, ac)
        bcm = reinterpret(reshape, Float64, bc)
        @test mycdot(ac, bc) ≈ mycdot_mat(acm, bcm) ≈ mycdot_affine(acv, bcv) ≈ mycdot_stride(acv, bcv)
    end
end

