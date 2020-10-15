
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

struct Polyhedra
    loops::Vector{Loop}
    preallocated_subsets::Vector{Polyhedra}
end
function Polyhedra(N::Int)
    ps = Vector{Polyhedra}(undef, N)
    for n ∈ 1:N
        ps[n] = Polyhedra(Vector{Loop}(undef, n), ps)
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

function poploop(p::Polyhedra, i)
    nloops_in_popped = length(p.A) - 1
    pout = p.preallocated_subsets[nloops_in_popped]
    
end

# """
# Checks whether the dependencies iᵣ => iₛ
# """
# function checkdependencies(p::Polyhedra, (iᵣ, iₛ)::Pair{Schedule,Schedule})
    
# end

# should make this iterable, transforming stepwise for determine_cost_looporder and scheduling...
# this function is used for code gen?
function loopbounds(p::Polyhedra, order)
    @unpack A, b, affinepair = p
    np = length(p.parametes)
    nv = nvar(p)
    vrange = Base.OneTo(nv)
    prange = 1+nv:size(A,1)
    loops_lower = Vector{Union{Int,Symbol,Expr}}(undef, nv)
    loops_upper = Vector{Union{Int,Symbol,Expr}}(undef, nv)
    completed = ntuple(_ -> false, Val(8))
    isaffinefunc
    for (i,j) ∈ enumerate(order)
        ls = us = Symbol("")
        
        for k in axes(A,1)
            Aₖⱼ = A[k,j]
            if Aₖⱼ > 0 # then we have the lower bound
                pair, l = affinepair[j] # index of paired, row that contains both
                if pair > 0
                    if completed[pair] # now we're a func of it
                        if l == k
                        else
                        end
                    elseif l == k
                    else
                    end
                elseif l > 0 && l ≠ k
                    
                else
                end
            elseif Aₖⱼ < 0 # then we have the upper bound
                pair, l = affinepair[j]
                if pair > 0 && completed[pair] # now we're a func of it
                else
                end
            end
        end
        completed = setindex(completed, true, j)
    end
    loops_lower, loops_upper
end

function vertices(p::Polyhedra)
    
end

