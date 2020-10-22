
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

struct Polyhedra{L <: AbstractLoop}
    loops::Vector{L}
    preallocated_subsets::Vector{Polyhedra{L}}
end
function Polyhedra{L}(N::Int) where {L <: AbstractLoop}
    ps = Vector{Polyhedra{L}}(undef, N)
    for n ∈ 1:N
        ps[n] = Polyhedra(Vector{L}(undef, n), ps)
    end
    last(ps)
end

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
nvars(p::Polyhedra) = length(p.loops)
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
function unconditional_loop_iters!(loops::AbstractVector, loop::StaticLoop)
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
            @unpack c, A = loopₙ
            newid = loopₙ.loopid
            c₁, c₂ = c
            A₁, A₂ = A
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
    StaticLoop( (c₁,c₂), (A₁, A₂), nloops, loopid ), iters
end
function unconditional_loop_iters!(::AbstractVector{StaticRectangularLoop}, loop::StaticRectangularLoop, vecf)
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

