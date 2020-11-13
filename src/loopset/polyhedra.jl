
# struct Polyhedra{N}
#     A::NTuple{N,NTuple{2,ByteVector}}
#     B::NTuple{N,NTuple{2,ByteVector}}
#     c::NTuple{N,NTuple{2,Float64}}
#     parameters::NTuple{8,Float64}
#     paramids::NTuple{8,Int8}
# end
# struct Polyhedra
#     A::Vector{NTuple{2,ByteVector}}
#     B::Vector{NTuple{2,ByteVector}}
#     c::Vector{NTuple{2,Float64}}
#     parameters::Vector{Float64}
#     paramids::Vector{Int8}
#     loopids::Vector{Int8}
#     preallocated_subsets::Vector{Polyhedra}
# end

# """
# A' + (I ⊗ [1, -1]) ≥ c[:,1] + d[:,2]
# """
# struct Polyhedra
#     A::Matrix{Int8}
#     c::Matrix{Float64}
#     paramids::Vector{UInt8}
# end
abstract type AbstractPolyhedra end

struct RectangularPolyhedra <: AbstractPolyhedra
    c::NTuple{2,NTuple{8,Int64}}
    d::NTuple{2,NTuple{8,Int64}}
    paramids::NTuple{2,UInt64} # zero indicates no param
    # nparams::NTuple{2,Int8}
    nloops::Int8
end
function RectangularPolyhedra(
    c::NTuple{2,NTuple{N,I}},
    d::NTuple{2,NTuple{N,I}},
    paramids::NTuple{2,ByteVector{UInt64}}    
) where {N, I <: Integer}
    cₗ, cᵤ = c
    dₗ, dᵤ = d
    RectangularPolyhedra(
        (
            (Base.Cartesian.@ntuple 8 i -> i > N ? zero(Int64) : cₗ[i] % Int64),
            (Base.Cartesian.@ntuple 8 i -> i > N ? zero(Int64) : cᵤ[i] % Int64)
        ), (
            (Base.Cartesian.@ntuple 8 i -> i > N ? zero(Int64) : dₗ[i] % Int64),
            (Base.Cartesian.@ntuple 8 i -> i > N ? zero(Int64) : dᵤ[i] % Int64)
        ),
        (paramids[1].data, paramids[2].data), N % Int8
    )
end

"""
A' + (I ⊗ [1, -1]) ≥ c[:,1] + d[:,2]
"""
struct Polyhedra <: AbstractPolyhedra
    A::NTuple{2,NTuple{8,UInt64}}
    p::RectangularPolyhedra
end

function Polyhedra(A::NTuple{2,NTuple{N,ByteVector{UInt64}}}, p::RectangularPolyhedra) where {N}
    Aₗ, Aᵤ = A
    Polyhedra(
        (
            (Base.Cartesian.@ntuple 8 i -> i > N ? zero(UInt64) : Aₗ[i].data),
            (Base.Cartesian.@ntuple 8 i -> i > N ? zero(UInt64) : Aᵤ[i].data)
        ), p
    )
end

UnPack.unpack(p::Polyhedra, ::Val{S}) where {S} = UnPack.unpack(p.p, Val{S}())
function UnPack.unpack(p::Polyhedra, ::Val{:A})
    nloops = p.p.nloops
    A₁, A₂ = p.A
    A₁v = Base.Cartesian.@ntuple 8 i -> ByteVector(A₁[i], nloops)
    A₂v = Base.Cartesian.@ntuple 8 i -> ByteVector(A₂[i], nloops)
    (A₁v, A₂v)
end
function UnPack.unpack(p::RectangularPolyhedra, ::Val{:paramids})
    nloops = p.nloops
    pid₁, pid₂ = p.paramids
    ByteVector(pid₁, nloops), ByteVector(pid₂, nloops)
end

# @generated function pow_by_square(n, ::Val{P}) where {P}
#     q = Expr(:block, :(x = one(n)))
#     while !iszero(P)
#         tz = trailing_zeros(P);
#         for i ∈ 1:tz
#             push!(q.args, :(n = Base.FastMath.mul_fast(n, n)))
#         end
#         push!(q.args, :(x = Base.FastMath.mul_fast(x, n)));
#         push!(q.args, :(n = Base.FastMath.mul_fast(n, n)));
#         P >>>= (tz+1)
#     end
#     push!(q.args, :x)
#     q
# end
function falling_factorial(p, K)
    x = p
    for i ∈ 1:K-1
        x *= p - i
    end
    x
end
faulhaber(n, ::Val{0}) = n
faulhaber(n, ::Val{1}) = (n * (n + one(n))) >> 1
bin(n, p) = binomial(round(Int64, n), round(Int64, p))
bin2(n) = faulhaber(n - one(n), Val(1))
@generated function faulhaber(n, ::Val{P}) where {P}
    @assert 2 ≤ P ≤ 8
    B = ( 0.5, 0.08333333333333333, 0.0, -0.001388888888888889, 0.0, 3.3068783068783064e-5, 0.0, -8.267195767195768e-7 )
    fm = :(Base.FastMath.mul_fast)
    fa = :(Base.FastMath.add_fast)
    q = Expr(:block, :(n² = $fm(n,n)), :(norig = n))#, :(x = $fm($(B[P] * falling_factorial(P, P-1)), n)))
    xinitialized = false
    # B = (1/2, 1/6, 0.0, -1/30, 0.0, 1/42, 0.0, -1/30)
    iszero(B[P]) && push!(q.args, :(n = n²))
    for k ∈ P:-1:2
        Bₖ = B[k]
        iszero(Bₖ) && continue
        Bₖ *= falling_factorial(P, k-1)
        xadd = Expr(:call, fm, Bₖ, :n)
        if xinitialized
            xadd = Expr(:call, fa, :x, xadd)
        else
            xinitialized = true
        end
        push!(q.args, Expr(:(=), :x, xadd))
        if iszero(B[k-1])
            push!(q.args, :(n = $fm(n,n²)))
        else
            push!(q.args, :(n = $fm(n,norig)))
        end
    end
    push!(q.args, :(x = $fa(x, $fm(0.5, n))))
    push!(q.args, :(n = $fm(n, norig)))
    push!(q.args, :(Base.fptosi(Int64, $fa(x, $fm(n, $(1/(P+1)))))))
    q
end

struct BinomialFunc
    a::ByteVector{UInt64}
    cd::Int64
    coef::Int64
    b::UInt8
    active::Bool
    isvec::Bool
end
function BinomialFunc(a, cd, coef, b, active, vecid::Int)
    isvec = a[vecid] != 0
    BinomialFunc(a, cd, coef, b, active, isvec)
end
BinomialFunc() = BinomialFunc(ByteVector(), 0, 0, 0x00, false, false)
isactive(b::BinomialFunc) = b.active

struct VectorLength
    W::Int
    shifter::Int
end
VectorLength(n) = VectorLength(n, VectorizationBase.intlog2(n))
VectorLength() = VectorLength(0, 0)
Base.rem(n, vl::VectorLength) = n & (vl.W - 1)
Base.div(n::Integer, vl::VectorLength) = (n >> vl.shifter) % typeof(n)
Base.div(n::Integer, vl::VectorLength, ::RoundingMode{:NearestTiesAway}) = copysign(((abs(n) + vl.W - one(n)) >> vl.shifter) % typeof(n), n)
Base.cld(n::Integer, vl::VectorLength) = ((n + vl.W - 1) ÷ vl) % typeof(n)
Base.:(*)(a::Integer, b::VectorLength) = a * b.W
Base.:(*)(b::VectorLength, a::Integer) = a * b.W
function Base.inv(vl::VectorLength)
    x = (one(Int64) << 10) - 1
    reinterpret(Float64, (x - vl.shifter) << 52)
end
Base.:(/)(a, b::VectorLength) = a * inv(b)
# """
# Polyhedra must be sorted so that no loops ∈ `loops` depend on loops ∉ `loops`.
# Counts the lattice points in `last(loops)`.
# """
# function calculate_lattice_points(p::Polyhedra, loops::ByteVector)

#     lp
# end

# pws need to be interpreted with respect to originally paramid vector of `p`

# """
#     extreme_bound

# For loop `i` of Polyhedra `p`, where `i ∉ v`, returns the extreme bound.

# Returns
# cₗ, cᵤ, dᵤ, dₗ, pwₗ, pwᵤ
# """
# function extreme_bound(
#     p::Polyhedra, i, lower::Bool, pwₗ = ByteVector(zero(UInt64), p.nloops), pwᵤ = ByteVector(zero(UInt64), p.nloops)
# )
#     if lower
#         extreme_bound_lower(p, i, pwₗ, pwᵤ)
#     else
#         extreme_bound_upper(p, i, pwₗ, pwᵤ)
#     end
# end
function extreme_bound_lower(
    p::Polyhedra, i, pwₗ = ByteVector(zero(UInt64), p.nloops)
)
    @unpack A, c, d, paramids, nloops = p
    Aₗ, Aᵤ = A
    cₗ, cᵤ = c
    dₗ, dᵤ = d
    # pidₗ, pidᵤ = paramids
    cₗᵢ = cₗ[i];# cᵤᵢ = cᵤ[i];
    dₗᵢ = dₗ[i];# dᵤᵢ = dᵤ[i];
    Aₗᵢ = Aₗ[i]
    while !(allzero(Aₗᵢ))
        j = firstnonzeroind(Aₗᵢ)
        # cₗⱼ, cᵤⱼ, dₗⱼ, dᵤⱼ, pwₗ, pwᵤ = extreme_bound_lower(p, j, pwₗ, pwᵤ)
        Aₗᵢⱼ = Aₗᵢ[j]
        if Aₗᵢⱼ < 0
            Aₗᵢⱼ = -Aₗᵢⱼ
            cₗⱼ, dₗⱼ, pwₗ = extreme_bound_lower(p, j, pwₗ)
        else
            cₗⱼ, dₗⱼ, pwₗ = extreme_bound_upper(p, j, pwₗ)
        end
        Aₗᵢ = setindex(Aₗᵢ, zero(Int8), j)
        cₗᵢ += cₗⱼ * Aₗᵢⱼ
        dₗᵢ += dₗⱼ * Aₗᵢⱼ
        pwₗ = setindex(pwₗ, pwₗ[i] + Aₗᵢⱼ, i)
    end
    # cₗᵢ, cᵤᵢ, dₗᵢ, dᵤᵢ, pwₗ, pwᵤ
    cₗᵢ, dₗᵢ, pwₗ
end
function extreme_bound_upper(
    p::Polyhedra, i, pwᵤ = ByteVector(zero(UInt64), p.nloops)
)
    @unpack A, c, d, paramids, nloops = p
    Aₗ, Aᵤ = A
    cₗ, cᵤ = c
    dₗ, dᵤ = d
    # pidₗ, pidᵤ = paramids
    # cₗᵢ = cₗ[i];
    cᵤᵢ = cᵤ[i];
    # dₗᵢ = dₗ[i];
    dᵤᵢ = dᵤ[i];
    Aᵤᵢ = Aᵤ[i];
    while !(allzero(Aᵤᵢ))
        j = firstnonzeroind(Aᵤᵢ)
        # cₗⱼ, cᵤⱼ, dₗⱼ, dᵤⱼ, pwₗ, pwᵤ = extreme_bound_upper(p, j, lower, pwₗ, pwᵤ)
        Aᵤᵢⱼ = Aᵤᵢ[j]
        if Aᵤᵢⱼ < 0
            Aᵤᵢⱼ = -Aᵤᵢⱼ
            cᵤⱼ, dᵤⱼ, pwᵤ = extreme_bound_lower(p, j, pwᵤ)
        else
            cᵤⱼ, dᵤⱼ, pwᵤ = extreme_bound_upper(p, j, pwᵤ)
        end
        Aᵤᵢ = setindex(Aᵤᵢ, zero(Int8), j)
        cᵤᵢ += cᵤⱼ * Aᵤᵢⱼ
        dᵤᵢ += dᵤⱼ * Aᵤᵢⱼ
        pwᵤ = setindex(pwᵤ, pwᵤ[i] + Aᵤᵢⱼ, i)
    end    
    # cₗᵢ, cᵤᵢ, dₗᵢ, dᵤᵢ, pwₗ, pwᵤ
    cᵤᵢ, dᵤᵢ, pwᵤ
end

# function remove_outer_bounds(p, A₁ₗ, A₁ᵤ, cₗ₁, cᵤ₁, dₗ₁, dᵤ₁, v, pidₗ, pidᵤ, pwₗ, pwᵤ, Asum = A₁ᵤ + A₁ₗ)
function remove_outer_bounds(p, A₁ₗ, A₁ᵤ, cₗ₁, cᵤ₁, dₗ₁, dᵤ₁, v, pwₗ, pwᵤ, Asum = A₁ᵤ + A₁ₗ)
    @unpack A, c, d, paramids, nloops = p
    Aₗ, Aᵤ = A
    cₗ, cᵤ = c
    dₗ, dᵤ = d
    # pidₗ, pidᵤ = paramids

    az = true
    failure = false
    for i ∈ eachindex(Asum)
        Asᵢ = Asum[i]
        if !iszero(Asᵢ)
            if i ∉ v # eliminate
                # TODO: Must we disallow `for m ∈ 1:M, n ∈ 1+m:2m, k ∈ 1:n`
                # due to the dependence of `k` on `n`, which has a `+m` on lower and upper bounds?
                # cₗᵢ, cᵤ, dₗ, dᵤ, pwₗ, pwᵤ = extreme_bound(p, i, Asᵢ < 0)
                A₁ₗₜ = A₁ₗ[i]
                A₁ᵤₜ = A₁ᵤ[i]
                if Asᵢ < 0 # lower
                    # Asᵢ = -Asᵢ
                    A₁ₗₜ = -A₁ₗₜ
                    A₁ᵤₜ = -A₁ᵤₜ
                    ctemp, dtemp, pwₗ = extreme_bound_lower(p, i, pwₗ)
                else
                    ctemp, dtemp, pwᵤ = extreme_bound_upper(p, i, pwᵤ)
                end
                cₗ₁ += ctemp * A₁ₗₜ
                cᵤ₁ += ctemp * A₁ᵤₜ
                if !iszero(dtemp)
                    dₗ₁ += dtemp * A₁ₗₜ
                    dᵤ₁ += dtemp * A₁ᵤₜ
                    pwₗ = setindex(pwₗ, pwₗ[i] + A₁ₗₜ, i)#pidₗ[i])
                    pwᵤ = setindex(pwᵤ, pwᵤ[i] + A₁ᵤₜ, i)#pidᵤ[i])
                end
                A₁ₗ = setindex(A₁ₗ, 0x00, i)
                A₁ᵤ = setindex(A₁ᵤ, 0x00, i)
            else
                az = false
            end
        end
    end
    A₁ₗ, A₁ᵤ, cₗ₁, cᵤ₁, dₗ₁, dᵤ₁, pwₗ, pwᵤ, az, failure
end


# getloop(p::Polyhedra, v::ByteVector, vecf, veci, citers) = getloop(p::Polyhedra, v::ByteVector, vecf, veci)
"""
    getloop(p::AbstractPolyhedra, v, vl, veci, citers)

Arguments:

 - p::AbstractPolyhedra : the polyhedra representing the loop nest.
 - v::ByteVector : vector of loops from outer-most to current-loop.
 - vl::VectorLength : length of the SIMD Vvector
 - veci : Index of SIMD loop.
 - citers : iterations at the previous loop nest depth.
"""
function getloop(p::Polyhedra, v::ByteVector, vl::VectorLength, veci, citers)
    @unpack A, c, d, paramids, nloops = p
    @inbounds begin
    Aₗ, Aᵤ = A
    cₗ, cᵤ = c
    dₗ, dᵤ = d
    pidₗ, pidᵤ = paramids
    polydim = length(v)
    outid = v[polydim]
    A₁ₗ = A₁ₗoriginal = Aₗ[outid]
    A₁ᵤ = A₁ᵤoriginal = Aᵤ[outid]
    A₂ₗ = A₂ᵤ = A₃ₗ = A₃ᵤ = A₄ₗ = A₄ᵤ = ByteVector()
    Asum = A₁ᵤ + A₁ₗ
    cₗ₁ = cₗ[outid]; cᵤ₁ = cᵤ[outid]; dₗ₁ = dₗ[outid]; dᵤ₁ = dᵤ[outid];
    az = allzero(Asum)
    pwₗ = ByteVector(zero(UInt64), nloops); pwᵤ = ByteVector(zero(UInt64), nloops);
    if !az
        # maybe it is only a function of loops ∉ v
        # A₁ₗ, A₁ᵤ, cₗ₁, cᵤ₁, dₗ₁, dᵤ₁, pwₗ, pwᵤ, az, failure = remove_outer_bounds(p, A₁ₗ, A₁ᵤ, cₗ₁, cᵤ₁, dₗ₁, dᵤ₁, v, pidₗ, pidᵤ, pwₗ, pwᵤ, Asum)
        A₁ₗ, A₁ᵤ, cₗ₁, cᵤ₁, dₗ₁, dᵤ₁, pwₗ, pwᵤ, az, failure = remove_outer_bounds(p, A₁ₗ, A₁ᵤ, cₗ₁, cᵤ₁, dₗ₁, dᵤ₁, v, pwₗ, pwᵤ, Asum)
        failure && return 9223372036854775807, nullloop()
    end
    # innerdefs = 0x00
    # for i ∈ 1:polydim-1
    #     noinnerdef = 0x00 == Aₗ[v[i]] == Aᵤ[v[i]]
    #     innerdefs |= !noinnerdef
    #     innerdefs <<= 1
    # end
    # # TODO: support >1 innerdef
    # count_ones(innerdefs) > 1 && return 9223372036854775807, nullloop()
    noinnerdefs = true
    for i ∈ 1:polydim-1
        vᵢ = v[i]
        noinnerdefs = 0x00 == Aₗ[vᵢ][outid] == Aᵤ[vᵢ][outid]
        noinnerdefs || break
    end
    if az & (noinnerdefs)#(innerdefs === 0x00)
        # then this loop's iteration count is independent of the loops proceding it.
        # vecf = veci == polydim ? vecf : one(vecf)
        cdsum = 1 - (cₗ₁ + cᵤ₁ + dₗ₁ + dᵤ₁)
        # @show cdsum, cₗ₁, cᵤ₁, dₗ₁, dᵤ₁
        # citers *= round(muladd(-vecf, cdsum, vecf), RoundUp)
        citers *= veci == outid ? div(cdsum, vl, RoundNearestTiesAway) : cdsum
        loop = Loop(
            (A₁ₗ.data, A₁ᵤ.data, zero(UInt64), zero(UInt64), zero(UInt64), zero(UInt64), zero(UInt64), zero(UInt64)),
            (cₗ₁, cᵤ₁, zero(Int64), zero(Int64), zero(Int64), zero(Int64), zero(Int64), zero(Int64)),
            (dₗ₁, dᵤ₁, zero(Int64), zero(Int64), zero(Int64), zero(Int64), zero(Int64), zero(Int64)),
            (pwₗ.data, pwᵤ.data, zero(UInt64), zero(UInt64), zero(UInt64), zero(UInt64), zero(UInt64), zero(UInt64)),
            (pidₗ.data, pidᵤ.data), nloops, outid, Int8(2)
        )
        return citers, loop
    end
    Aₗ = setindex(Aₗ, A₁ₗ, outid)
    Aᵤ = setindex(Aᵤ, A₁ᵤ, outid)
    cₗ = setindex(cₗ, cₗ₁, outid)
    cᵤ = setindex(cᵤ, cᵤ₁, outid)
    dₗ = setindex(dₗ, dₗ₁, outid)
    dᵤ = setindex(dᵤ, dᵤ₁, outid)
    # if we get here, then there is a non-zero in A₁ₗ, A₁ᵤ, there is an innerdef, or both
    # Perhaps the strategy should be to pick a `!az` to start with when counting iters
    # and then proceed backwards through the deps until they're all covered?
    # we need to return a `!az` loop here anyway...
    # but it's probably simplest to do these in two separate steps.
    # 1. calculate iters
    # 2. determine loop
    # remove outers and check for `!az`
    naz = az ? zero(Int32) : outid % Int32
    Aᵤ′ = Aₗ′ = Base.Cartesian.@ntuple 8 _ -> VectorizationBase.splitint(0x0000000000000000, Int8)
    pwᵤᵥ = pwₗᵥ = Base.Cartesian.@ntuple 8 _ -> zero(UInt64)
    Aᵤ′ᵢ = Aₗ′ᵢ = Aₗ′[1]; # just to define scope here.
    # for j ∈ eachindex(Aᵢₗ)
    for j ∈ eachindex(A₁ₗ)
        Aₗ′ = setindex(Aₗ′, insertelement(Aₗ′[j], A₁ₗ[j], polydim-1), j)
        Aᵤ′ = setindex(Aᵤ′, insertelement(Aᵤ′[j], A₁ᵤ[j], polydim-1), j)
    end
    for _i ∈ 1:polydim-1
        i = (v[_i]) % Int64
        Aᵢₗ = Aₗ[i]; Aᵢᵤ = Aᵤ[i]; cᵢₗ = cₗ[i]; cᵢᵤ = cᵤ[i]; dᵢₗ = dₗ[i]; dᵢᵤ = dᵤ[i];
        # A₁ₗ, A₁ᵤ, cₗ₁, cᵤ₁, dₗ₁, dᵤ₁, pwₗ, pwᵤ, az, failure = remove_outer_bounds(p, A₁ₗ, A₁ᵤ, cₗ₁, cᵤ₁, dₗ₁, dᵤ₁, v, pidₗ, pidᵤ, pwₗ, pwᵤ, Asum)
        Aᵢₗ, Aᵢᵤ, cᵢₗ, cᵢᵤ, dᵢₗ, dᵢᵤ, pwₗᵢ, pwᵤᵢ, azᵢ, failure = remove_outer_bounds(p, Aᵢₗ, Aᵢᵤ, cᵢₗ, cᵢᵤ, dᵢₗ, dᵢᵤ, v, ByteVector(pwₗᵥ[i],nloops), ByteVector(pwᵤᵥ[i],nloops), Asum)
        failure && return 9223372036854775807, nullloop()
        # try and have naz point to a loop that's an affine combination of others
        if !azᵢ && (iszero(naz) || (!(iszero(Aᵢₗ[naz]) & iszero(Aᵢᵤ[naz]))))
            naz = i % Int32
        end
        for j ∈ eachindex(Aᵢₗ)
            Aₗ′ = setindex(Aₗ′, insertelement(Aₗ′[j], Aᵢₗ[j], i-1), j)
            Aᵤ′ = setindex(Aᵤ′, insertelement(Aᵤ′[j], Aᵢᵤ[j], i-1), j)
        end
        Aₗ = setindex(Aₗ, Aᵢₗ, i)
        Aᵤ = setindex(Aᵤ, Aᵢᵤ, i)
        cₗ = setindex(cₗ, cᵢₗ, i)
        cᵤ = setindex(cᵤ, cᵢᵤ, i)
        dₗ = setindex(dₗ, dᵢₗ, i)
        dᵤ = setindex(dᵤ, dᵢᵤ, i)
        pwₗᵥ = setindex(pwₗᵥ, pwₗᵢ.data, i)
        pwᵤᵥ = setindex(pwᵤᵥ, pwₗᵢ.data, i)
    end
    # We've removed dependencies on external loops, and found one that's a function of others. We
    # start counting iters from naz
    # determining loop
    binomials = ntuple(_ -> BinomialFunc(), Val(8))
    nbinomials = 0
    coef⁰ = 0
    coefs¹ = Base.Cartesian.@ntuple 8 i -> zero(Int64)
    coefs¹v = Base.Cartesian.@ntuple 8 i -> false
    coefs² = Base.Cartesian.@ntuple 8 i -> coefs¹
    # visited_mask = (0x0101010101010101)
    # visited_mask = VectorizationBase.splitint(0x0101010101010101 >> ((8 - polydim)*8), Bool)
    not_visited_mask = 0x0101010101010101 >> ((8 - polydim)*8)
    # coefs³ = ntuple(zero, Val(8))
    constraints = 2
    first_iter = true
    while true
        i, not_visited_mask = depending_ind(Aₗ′, Aᵤ′, not_visited_mask) # no others not yet visited depend on `i`
        Aₗᵢ = Aₗ[i] # others it depends on
        Aᵤᵢ = Aᵤ[i] # others it depends on
        Aₗᵢzero = allzero(Aₗᵢ)
        Aᵤᵢzero = allzero(Aᵤᵢ)
        cdmax = -cᵤ[i] - dᵤ[i]
        cdmin =  cₗ[i] + dₗ[i]
        cd = 1 + cdmax - cdmin
        cd = veci == i ? div(cd, vl, RoundNearestTiesAway) : cd
        Asum = Aᵤᵢ + Aₗᵢ
        allzeroAsum = allzero(Asum)
        @show Asum, cd, cdmax, cdmin
        if first_iter # initialize
            first_iter = false
            coef⁰ = cd# - (i == veci)
            coefs¹ = Base.Cartesian.@ntuple 8 j -> begin
                Asumⱼ = Asum[j] % Int64
                # iszero(Asumⱼ) ? Asumⱼ : div(Asumⱼ, vl, RoundNearestTiesAway)
                # j == veci ? Asumⱼ : (coef⁰ -= one(coef⁰); div(Asumⱼ, vl, RoundNearestTiesAway))
                j == veci ? Asumⱼ : (div(Asumⱼ, vl, RoundNearestTiesAway))
            end
            @show coefs¹
            coefs¹v = Base.Cartesian.@ntuple 8 j -> (coefs¹v[j]) | ((veci == i) & !(iszero(Asum[j])))
            continue
        end
        coefs¹ᵢ = coefs¹[i]
        coefs²ᵢ = coefs²[i]
        coefs²ᵢ = Base.Cartesian.@ntuple 8 j -> coefs²ᵢ[j] + (i == j ? 0 : coefs²[j][i])
        @show cd, coef⁰
        coef⁰_old = coef⁰
        coefs¹_old = coefs¹
        coef⁰ *= cd
        coefs¹ = Base.Cartesian.@ntuple 8 j -> coefs¹[j] * cd# + Asum[j] * coef⁰_old
        # now need to update the iᵗʰ.
        if iszero(Asum)
            coefs² = Base.Cartesian.@ntuple 8 j -> (Base.Cartesian.@ntuple 8 k -> cd * coefs²[j][k])
        else
            coefs¹ = Base.Cartesian.@ntuple 8 j -> coefs¹[j] + Asum[j] * coef⁰_old
            coefs¹v = Base.Cartesian.@ntuple 8 j -> (coefs¹v[j]) | ((veci == i) & !(iszero(Asum[j])))
            coefs² = Base.Cartesian.@ntuple 8 j -> Base.Cartesian.@ntuple 8 k -> begin
                cd * coefs²[j][k] + coefs¹_old[k] * Asum[j]
            end
        end
        nbinrange = OneTo(nbinomials)
        for b ∈ nbinrange # hockey stick
            bb = binomials[b]
            isactive(bb) || continue
            a = bb.a
            aᵢ = a[i]
            a = setindex(a, zero(Int8), i)
            allzeroa = allzero(a)
            if allzeroAsum & iszero(aᵢ)
                if allzeroa
                    isvec = bb.isvec
                    coef⁰ += bb.coef * (binomial(cdmax + 1 + bb.cd, bb.b + 1) - binomial(cdmin + bb.cd, bb.b + 1))
                    binomials = setindex(binomials, BinomialFunc(bb.a, bb.cd, bb.coef, bb.b, false, bb.isvec), b)
                else#if iszero(aᵢ)
                    binomials = setindex(binomials, BinomialFunc(bb.a, bb.cd, bb.coef * cd, bb.b, true, bb.isvec), b)
                end
            elseif iszero(aᵢ)
                # products of binomials not currently supported
                return 9223372036854775807, nullloop()
                # binomials = setindex(binomials, BinomialFunc(bb.a, bb.cd, bb.coef * cd, bb.b, true), b)
            else
                binomials = setindex(binomials, BinomialFunc(a + Aᵤᵢ, cdmax + 1 + bb.cd, bb.coef, bb.b + one(bb.b), true, bb.isvec), b)
                nbinomials += 1
                binomials = setindex(binomials, BinomialFunc(a - Aₗᵢ, cdmin + bb.cd, -bb.coef, bb.b + one(bb.b), true, bb.isvec), nbinomials)
            end
        end
        @show coef⁰ coefs¹ᵢ coefs¹ coefs¹v Aᵤᵢzero, Aₗᵢzero
        if !iszero(coefs¹ᵢ)
            @show cdmin, cdmax, (i == veci), coefs¹v[i]
            if Aᵤᵢzero
                if (i == veci) | coefs¹v[i]
                    # if true
                    if coefs¹ᵢ > 0
                        # @show cdmax
                        divvec, remvec = divrem(cdmax, vl)
                        divvec += one(divvec)
                        # divvec += coefs¹ᵢ > 0
                        # divvec += remvec > 0
                        # itersbin = bin2(divvec) * vl + remvec * divvec
                        itersbin = bin2(divvec) * vl + remvec * divvec
                    elseif i == veci
                        r = (cdmax - cdmin) % vl
                        divvec, remvec = divrem(cdmax - r, vl)
                        divvec += one(divvec)
                        itersbin = bin2(divvec) * vl + remvec * divvec
                    else
                        # divvec, remvec = divrem(cdmax, vl)
                        # divvec += one(divvec)
                        # divvec += coefs¹ᵢ > 0
                        # divvec += remvec > 0
                        # itersbin = bin2(divvec) * vl + remvec * divvec
                        # itersbin = bin2(divvec) * vl + remvec * divvec
                        divvec = div(cdmax + one(cdmax), vl, RoundNearestTiesAway)
                        itersbin = bin2(divvec) * vl# + divvec * (i == veci)
                    end
                else
                    itersbin = bin2(cdmax + 1)
                end
                coef⁰ += coefs¹ᵢ * itersbin
                @show 1, coef⁰, itersbin
            else
                nbinomials += 1
                binomials = setindex(binomials, BinomialFunc(Aᵤᵢ, cdmax + 1, coefs¹ᵢ, 0x02, true, (i == veci) | coefs¹v[i]), nbinomials)
            end
            if Aₗᵢzero
                if (i == veci) | coefs¹v[i]
                    if coefs¹ᵢ > 0
                        # FIXME: is cdmin-1 correct ?!?
                        divvec, remvec = divrem(cdmin-1, vl)
                        divvec += one(divvec)
                        # divvec += remvec > 0
                        # itersbin = bin2(divvec) * vl + remvec * divvec
                        itersbin = bin2(divvec) * vl + remvec * (divvec + one(divvec))
                    elseif i == veci#FIXME
                        # r = (1 + cdmax - cdmin) % vl
                        divvec, remvec = divrem(cdmin, vl)
                        # divvec += one(divvec)
                        itersbin = bin2(divvec) * vl + remvec * divvec
                        #     divvec, remvec = divrem(cdmin-1, vl)
                    #     divvec += one(divvec)
                    #     # divvec += remvec > 0
                    #     # itersbin = bin2(divvec) * vl + remvec * divvec
                    #     itersbin = bin2(divvec) * vl + remvec * (divvec + one(divvec))
                    else#FIXME
                        divvec, remvec = divrem(cdmin - 1 - (cdmax % vl), vl)#, RoundNearestTiesAway)
                        divvec += one(divvec)
                        itersbin = bin2(divvec) * vl + remvec * divvec
                        # itersbin = bin2(divvec) * vl + divvec * (i == veci)
                        # divvec, remvec = divrem(cdmin-1, vl)
                        # divvec += one(divvec)
                        # # divvec += remvec > 0
                        # # itersbin = bin2(divvec) * vl + remvec * divvec
                        # itersbin = bin2(divvec) * vl + remvec * (divvec + one(divvec))                        
                    end
                    # @show cdmin, itersbin
                else
                    itersbin = bin2(cdmin)
                end
                coef⁰ -= coefs¹ᵢ * itersbin
                @show 2, coef⁰, itersbin
            else
                nbinomials += 1
                binomials = setindex(binomials, BinomialFunc(-Aₗᵢ, cdmin, -coefs¹ᵢ, 0x02, true, (i == veci) | coefs¹v[i]), nbinomials)
            end
        end
        for j ∈ 1:polydim
            coefs²ᵢⱼ = coefs²ᵢ[j]
            iszero(coefs²ᵢⱼ) && continue
            if j == i
                fh2 = faulhaber(cdmax, Val(2))
                if cdmin > 0
                    fh2 -= faulhaber(cdmin - 1, Val(2))
                else
                    fh2 += faulhaber(cdmin, Val(2))
                end
                coef⁰ += coefs²ᵢⱼ * (veci == i ? div(fh2, vl, RoundNearestTiesAway) : fh2)
            elseif VectorizationBase.splitint(not_visited_mask, Bool)[j]
                coefs¹ = setindex(coefs¹, cd * coefs²ᵢⱼ, j)
            end            
        end
        (not_visited_mask === zero(UInt64)) && break
    end
    # coef⁰ should now be the number of iterations
    # this leaves us with determining the loop bounds, required for fusion checks (and used for lowering?)
    # if az
    #     return coef⁰, Loop(
    #         (A₁ₗ.data, A₁ᵤ.data, zero(UInt64), zero(UInt64), zero(UInt64), zero(UInt64), zero(UInt64), zero(UInt64)),
    #         (cₗ₁, cᵤ₁, zero(Int64), zero(Int64), zero(Int64), zero(Int64), zero(Int64), zero(Int64)),
    #         (dₗ₁, dᵤ₁, zero(Int64), zero(Int64), zero(Int64), zero(Int64), zero(Int64), zero(Int64)),
    #         (pwₗ.data, pwᵤ.data, zero(UInt64), zero(UInt64), zero(UInt64), zero(UInt64), zero(UInt64), zero(UInt64)),
    #         (pidₗ.data, pidᵤ.data), nloops, outid, Int8(2)
    #     )
    # end
    Aₒᵤₜ = Base.Cartesian.@ntuple 8 i -> (i == 1 ? A₁ₗ.data : (i == 2 ? A₁ᵤ.data : zero(UInt64)))
    cₒᵤₜ = Base.Cartesian.@ntuple 8 i -> (i == 1 ? cₗ₁ : (i == 2 ? cᵤ₁ : zero(Int64)))
    dₒᵤₜ = Base.Cartesian.@ntuple 8 i -> (i == 1 ? dₗ₁ : (i == 2 ? dᵤ₁ : zero(Int64)))
    pwₒᵤₜ = Base.Cartesian.@ntuple 8 i -> (i == 1 ? pwₗ.data : (i == 2 ?  pwᵤ.data : zero(UInt64)))
    i = 2
    for (_Anz′, _A, _c, _d, _pw) ∈ ((Aₗ′[outid], Aₗ, cₗ, dₗ, pwₗᵥ), (Aᵤ′[outid], Aᵤ, cᵤ, dᵤ, pwᵤᵥ))
        Anz = ByteVector(unsigned(VectorizationBase.fuseint(_Anz′)), nloops)
        while !(allzero(Anz))
            j = firstnonzeroind(Anz)#Aᵤ
            Anz = setindex(Anz, zero(eltype(Anz)), j)
            i += 1
            # @show i, Anz, _A
            Aₒᵤₜ = setindex(Aₒᵤₜ, _A[j].data, i)
            cₒᵤₜ = setindex(cₒᵤₜ, _c[j], i)
            dₒᵤₜ = setindex(dₒᵤₜ, _d[j], i)
            pwₒᵤₜ = setindex(pwₒᵤₜ, _pw[j], i)
        end
    end
    end
    # @show Aₒᵤₜ
    coef⁰, Loop( Aₒᵤₜ, cₒᵤₜ, dₒᵤₜ, pwₒᵤₜ, (pidₗ.data, pidᵤ.data), nloops, outid, i % Int8 )
end
function getloopiters(p::RectangularPolyhedra, v::ByteVector, vl, veci, citers)
    @unpack c, d, paramids, nloops = p
    # vecf = veci == length(v) ? vecf : one(vecf)
    c₁, c₂ = c
    c₁ᵢ = c₁[i]; c₂ᵢ = c₂[i];
    d₁, d₂ = d
    d₁ᵢ = d₁[i]; d₂ᵢ = d₂[i];
    pid₁, pid₂ = paramids
    i = last(v)
    loop = RectangularLoop(
        (c₁ᵢ, c₂ᵢ),
        (d₁ᵢ, d₂ᵢ),
        (pid₁[i], pid₂[i]),
        nloops,
        i
    )
    cdsum = 1 - (c₁ᵢ + c₂ᵢ + d₁ᵢ + d₂ᵢ)
    citers *= veci == length(v) ? div(cdsum, vl, RoundNearestTiesAway) : cdsum
    citers, loop
end




# struct Polyhedra{L <: AbstractLoop}
#     loops::Vector{L}
#     preallocated_subsets::Vector{Polyhedra{L}}
# end
# function Polyhedra{L}(N::Int) where {L <: AbstractLoop}
#     ps = Vector{Polyhedra{L}}(undef, N)
#     for n ∈ 1:N
#         ps[n] = Polyhedra(Vector{L}(undef, n), ps)
#     end
#     last(ps)
# end

# struct Polyhedra
#     A::Matrix{Int} # A * x + B * p ≥ c
#     B::Matrix{Int}
#     c::Vector{Int}
#     parameters::Vector{Float64}
#     dynamicid::Vector{Int8} # indices into global vector of params
#     affinepair::Vector{Tuple{Int8,Int8}}
#     preallocated_subsets::Vector{Polyhedra}
#     # dynamicsyms::Vector{Symbol}
#     # nvar::Int = size(A,1) - length(parameters)
# end
# nvars(p::Polyhedra) = length(p.loops)
# function prealloc_polyhedra_chain(A, params)
#     L = length(A)
#     c = Vector{Polyhedra}(undef, L)
#     for i ∈ 1:L-1
#         c[i] = Polyhedra(
#             Vector{NTuple{2,ByteVector}}(undef, i)
#             Vector{NTuple{2,ByteVector}}(undef, i),
#             Vector{NTuple{2,Float64}}(undef, i),
#             Vector{Float64}(undef, length(params)),
#             Vector{Int8}(undef, length(params)),
#             Vector{Int8}(undef, i),
#             # Vector{Tuple{Int8,Int8}}(undef, D - length(params)),
#             c
#         )
#     end
# end

# function Polyhedra(A, b, parameters, dynamicid, prealloc = prealloc_polyhedra_chain(A, parameters))
#     nv = size(A, 1) - length(parameters)
#     affinepair = fill((zero(Int8),zero(Int8)), nv)
#     for i ∈ axes(A,1)
#         nz = sum(!iszero, @view(A[i,Base.OneTo(nv)]))
#         iszero(nz) && continue
#         # currently, only simple affine pairs are allowed
#         # anz = findall(!iszero, @view(A[i,Base.OneTo(nv)]))
#         # for k ∈ anz, j ∈ anz
#         #     k == j && continue
#         #     affinepair[k]
#         # end
#         for j ∈ 1:nv
#             iszero(A[i,j]) && continue
#             for k ∈ nv
#                 j == k && continue
#                 Aᵢₖ = A[i,k]
#                 iszero(Aᵢₖ) && continue
#                 affinepair[j] = (k % Int8, i % Int8)
#                 affinepair[k] = (j % Int8, i % Int8)
#             end
#         end
#     end
#     prealloc[end] = Polyhedra(A, b, parameters, dynamicid, affinepair, prealloc)
# end

"""
# upper bound is affine func of other loop inds, A*x
# [1  0   [ i     [ c₁
#  -1 0     j ] ≥   c₂
#  0  1             c₃
#  1 -1]            c₄ ]
#  i ∈ c₁:-c₂
#  j ∈ c₃:i-c₄
#  to
# [1 -1   [ i     [ c₄
#  -1 0     j ] ≥   c₂
#  0  1             c₃
#  0 -1]            c₂ + c₄ ]
#  i ∈ max(c₄+j,c₁):-c₂
#  j ∈ c₃:-c₂-c₄
#  What if c₁ ≠ c₃ + c₄?
#  if c₁ < c₃ + c₄, inner loop doesn't run
#  if c₁ > c₃ + c₄, we need max(c₄+j,c₁)
#  TODO: handle this well
#  for now, assume c₁ == c₃ + c₄ while optimizing, but generate correct code?
#  Difficulty is that data structure would need to be able to handle this
#  Perhaps split loop?
"""
function unconditional_loop_iters!(loops::AbstractVector, loop::Loop)
    @unpack c, A, nloops, loopid = loop
    A₃, A₄ = A
    c₃, c₄ = c
    len = A₂.len
    istriangle = false
    while !allzero(A₃)
        
    end
    while !allzero(A₄)
        for (n,a) ∈ enumerate(A₂)
            iszero(a) && continue
            i = findfirst(l -> l.loopid == n, loops)::Int
            loopₙ = loops[i]
            @unpack c, A, B, p = loopₙ
            newid = loopₙ.loopid
            c₁, c₂ = c
            A₁, A₂ = A
            B₁, B₂ = B
            p₁, p₂ = p
            # in A₄, loopid is implicitly -1; make it explicit
            # in A₄, loopid is explicitly A₄[newid]; make it implicit
            A₄val = A₄[newid]
            if A₄val == (0x01 % Int8)
                (c₁ == c₃ + c₄) || return 9223372036854775807, loop
                A₁new = setindex(setindex(A₄, (0xff % Int8), loopid), zero(Int8), newid)
                A₂new = A₂
                A₃new = A₃
                A₄new = 
                c₁new = c₄
                c₂new = c₂
                c₃new = c₃
                c₄new = c₂ + c₄
            elseif A₄val == (0xff % Int8)
            else# reject
                return 9223372036854775807, loop
            end                
            loops[n] = StaticLoop( (c₁new, c₂new), (A₁new, A₂new), nloops, newid )
            A₃ = A₃new
            A₄ = A₄new
            c₃ = c₃new
            c₄ = c₄new
            break
        end
    end
    if (!istriangle) & (!isone(vecf))
        m = 0x00000000000000ff << (8*(loopid-1))
        # m₂ = 0x00000000000000ff << (8*(loopid-2))
        for n ∈ eachindex(loops)
            l = loops[n]
            # m = l.loopid > loopid > m₂ : m₁
            Au₁, Au₂ = l.A
            if (!iszero(Au₁ & m)) | (!iszero(Au₂ & m))
                istriangle = true
                break
            end
        end
    end

    if istriangle & (!isone(vecf))
        # compare ratio of triangle of vec-sized blocks to triangle of individual iters
        iters = 1.0 - c₁ - c₂
        itersdiv = round(vecf * iters, RoundUp)
        vecf⁻² = inv(vecf*vecf)
        r = vecf⁻² * (itersdiv*(itersdiv + 1.0)) / (iters*(iters + 1.0))
        iters = itersdiv * r
    else
        iters = round(muladd(-vecf, c₁ + c₂, vecf), RoundUp)
    end
    Loop( (c₁,c₂), (A₁, A₂), nloops, loopid ), iters
end
function unconditional_loop_iters!(::AbstractVector{RectangularLoop}, loop::RectangularLoop, vecf)
    c₁, c₂ = loop.c
    loop, round(muladd(-vecf, c₁ + c₂, vecf), RoundUp)
end



function poploop(p::Polyhedra, i, vecf)
    loop = p.loops[i]
    nloops = length(p.loops)
    if isone(nloops)
        return p, loop
    end
    pout = p.preallocated_subsets[nloops - 1]
    for n ∈ 1:i-1
        pout.loops[n] = p.loops[n]
    end
    for n ∈ i+1:nloops
        pout.loops[n-1] = p.loops[n]
    end
    loop, iters = unconditional_loop_iters!(pout.loops, loop, vecf)
    return pout, loop, iters
end

# """
# Checks whether the dependencies iᵣ => iₛ
# """
# function checkdependencies(p::Polyhedra, (iᵣ, iₛ)::Pair{Schedule,Schedule})
    
# end

# should make this iterable, transforming stepwise for determine_cost_looporder and scheduling...
# this function is used for code gen?
# function loopbounds(p::Polyhedra, order)
#     @unpack A, b, affinepair = p
#     np = length(p.parameters)
#     nv = nvar(p)
#     vrange = Base.OneTo(nv)
#     prange = 1+nv:size(A,1)
#     loops_lower = Vector{Union{Int,Symbol,Expr}}(undef, nv)
#     loops_upper = Vector{Union{Int,Symbol,Expr}}(undef, nv)
#     completed = ntuple(_ -> false, Val(8))
#     isaffinefunc
#     for (i,j) ∈ enumerate(order)
#         ls = us = Symbol("")
        
#         for k in axes(A,1)
#             Aₖⱼ = A[k,j]
#             if Aₖⱼ > 0 # then we have the lower bound
#                 pair, l = affinepair[j] # index of paired, row that contains both
#                 if pair > 0
#                     if completed[pair] # now we're a func of it
#                         if l == k
#                         else
#                         end
#                     elseif l == k
#                     else
#                     end
#                 elseif l > 0 && l ≠ k
                    
#                 else
#                 end
#             elseif Aₖⱼ < 0 # then we have the upper bound
#                 pair, l = affinepair[j]
#                 if pair > 0 && completed[pair] # now we're a func of it
#                 else
#                 end
#             end
#         end
#         completed = setindex(completed, true, j)
#     end
#     loops_lower, loops_upper
# end

function vertices(p::Polyhedra)
    
end

