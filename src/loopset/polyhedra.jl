
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

"""
A' + (I ⊗ [1, -1]) ≥ c[:,1] + d[:,2]
"""
struct Polyhedra <: AbstractPolyhedra
    A::NTuple{2,NTuple{8,UInt64}}
    p::RectangularPolyhedra
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
faulhaber(n, ::Val{1}) = @fastmath 0.5 * n * (n + one(n))
bin(n, x) = binomial(round(Int64, n), round(Int64, x))
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
    push!(q.args, :($fa(x, $fm(n, $(1/(P+1))))))
    q
end

struct BinomialFunc
    a::ByteVector{UInt64}
    b::Int32
    n::Int8
    coef::Int8
end
BinomialFunc() = BinomialFunc(ByteVector(zero(UInt64),zero(Int8)),zero(Int32),zero(Int8),zero(Int8))
# """
# Polyhedra must be sorted so that no loops ∈ `loops` depend on loops ∉ `loops`.
# Counts the lattice points in `last(loops)`.
# """
# function calculate_lattice_points(p::Polyhedra, loops::ByteVector)

#     lp
# end

# getloop(p::Polyhedra, v::ByteVector, vecf, veci, citers) = getloop(p::Polyhedra, v::ByteVector, vecf, veci)
function getloop(p::Polyhedra, v::ByteVector, vecf, veci, citers)
    @unpack A, c, d, paramids, nloops = p
    Aₗ, Aᵤ = A
    cₗ, cᵤ = c
    dₗ, dᵤ = d
    polydim = length(v)
    outid = v[polydim]
    A₁ₗ = A₁ₗoriginal = Aₗ[outid]
    A₁ᵤ = A₁ᵤoriginal = Aᵤ[outid]
    A₂ₗ = A₂ᵤ = A₃ₗ = A₃ᵤ = A₄ₗ = A₄ᵤ = ByteVector()
    Asum = A₁ᵤ + A₁ₗ
    cdsum = cₗ[outid] + cᵤ[outid] + dₗ[outid] + dᵤ[outid]
    az = allzero(Asum)
    if !az
        # maybe it is only a function of loops ∉ v
        az = true
        for i ∈ eachindex(Asum)
            Asᵢ = Asum[i]
            if !iszero(Asᵢ)
                if i ∉ v # eliminate
                    # TODO: Must we disallow `for m ∈ 1:M, n ∈ 1+m:2m, k ∈ 1:n`
                    # due to the dependence of `k` on `n`, which has a `+m` on lower and upper bounds?
                    if Asᵢ < 0 # i ∈ j:upper
                        ctemp, dtemp = minimum(p, i, v)
                    else#if Asᵢ > 0 # guaranteed by `!iszero(Asᵢ); # i ∈ lower:j
                        ctemp, dtemp = maximum(p, i, v)                        
                    end
                    A₁ₗₜ = A₁ₗ[i]
                    A₁ᵤₜ = A₁ᵤ[i]
                    cₗ₁ -= ctemp * A₁ₗₜ
                    cᵤ₁ -= ctemp * A₁ᵤₜ
                    dₗ₁ -= dtemp * A₁ₗₜ
                    dᵤ₁ -= dtemp * A₁ᵤₜ
                    A₁ₗ = setindex(A₁ₗ, 0x00, i)
                    A₁ᵤ = setindex(A₁ᵤ, 0x00, i)
                else
                    az = false
                end
            end
        end
    end
    noinnerdefs = true
    for i ∈ 1:polydim-1
        noinnerdefs = 0x00 == Aₗ[v[i]] == Aᵤ[v[i]]
        noinnerdefs || break
    end
    if az & noinnerdefs
        # then this loop's iteration count is independent of the loops proceding it.
        vecf = veci == polydim ? vecf : one(vecf)
        citers *= round(muladd(-vecf, cdsum, vecf), RoundUp)
        loop = Loop(
            (Aₗ.data, Aᵤ.data, zero(UInt64), zero(UInt64), zero(UInt64), zero(UInt64), zero(UInt64), zero(UInt64)),
            (cₗ[outid], cᵤ[outid], zero(Int64), zero(Int64), zero(Int64), zero(Int64), zero(Int64), zero(Int64)),
            (dₗ[outid], dᵤ[outid], zero(Int64), zero(Int64), zero(Int64), zero(Int64), zero(Int64), zero(Int64)),
            (paramids[1][outid], paramids[2][outid], zero(Int16), zero(Int16), zero(Int16), zero(Int16), zero(Int16), zero(Int16)),
            nloops, outid, one(Int8)
        )
        return citers, loop
    end
    # if we get here, then there is a non-zero in A₁ₗ, A₁ᵤ, there is an innerdef, or both
    lp = 0.0
    binomials = ntuple(_ -> BinomialFunc(), Val(8))
    coefs1 = ntuple(zero, Val(8))
    coefs2 = ntuple(zero, Val(8))
    coefs3 = ntuple(zero, Val(8))

    
    citers = 1.0
    A₁, A₂ = A
    A₁ᵢ = A₁[i]; A₂ᵢ = A₂[i];
    for i ∈ v
        
    end
    c₁, c₂ = c
    d₁, d₂ = d
    pid₁, pid₂ = paramids
    i = last(v)
    Loop(
        (A₁[i], A₂[i]),
        RectangularLoop(
            (c₁[i], c₂[i]),
            (d₁[i], d₂[i]),
            (pid₁[i], pid₂[i]),
            nloops,
            i
        )
    )
end
function getloopiters(p::RectangularPolyhedra, v::ByteVector, vecf, veci, citers)
    @unpack c, d, paramids, nloops = p
    vecf = veci == length(v) ? vecf : one(vecf)
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
    citers *= round(muladd(-vecf, c₁ᵢ + c₂ᵢ + d₁ᵢ + d₂ᵢ, vecf), RoundUp)
    loop, citers
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
                (c₁ == c₃ + c₄) || return loop, Inf
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
                return loop, Inf
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

