
# canonicalization should move `i` into indexing expression in `i+j:i+k` ?
#
abstract type AbstractLoop end
struct RectangularLoop <: AbstractLoop
    # loopids::ByteVector{UInt64} # up to 8 loop ids for loops in A
    c::NTuple{2,Int64} # constants for knowns
    d::NTuple{2,Int16} # A * x ≥ c + d # d are dynamics
    paramids::NTuple{2,Int8} # ids of the dynamics
    nloops::Int8
    loopid::Int8 # id of this loop
end
"""
Supports loop depth of up to 8.
`A` holds 8 loops; the actual loop itself is omitted, with
(1, -1) being inserted into A₁, A₂ and `loopid`.
"""
struct Loop <: AbstractLoop
    A::NTuple{8,UInt64} # A is a ByteVector in practice, see unpack(::Loop, ::Val{:A})
    c::NTuple{8,Int64}
    d::NTuple{8,Int64}
    paramweights::NTuple{8,UInt64}
    paramids::NTuple{2,UInt64}
    # nparamweights::NTuple{8,Int8}
    nloops::Int8
    loopid::Int8
    nconstraints::Int8
end
function isstatic(loop::AbstractLoop)
    pid = loop.paramids
    iszero(VectorizationBase.fuseint(Vec(pid[1],pid[2])))
end

UnPack.unpack(l::Loop, ::Val{:paramweights}) = Base.Cartesian.@ntuple 8 i -> ByteVector(l.paramweights[i], l.nparamweights[i])
UnPack.unpack(l::Loop, ::Val{:paramids}) = (ByteVector(l.paramids[1], l.nloops), ByteVector(l.paramids[2], l.nloops))


# function Loop(A::NTuple{2,ByteVector{UInt64}},)

# end

nullrectangularloop() = RectangularLoop((zero(Int64),zero(Int64)),(zero(Int16),zero(Int16)),(zero(Int8),zero(Int8)),zero(Int8),zero(Int8))
nullloop() = Loop(ntuple(_ -> zero(UInt64), Val(8)), ntuple(zero, Val(8)), ntuple(zero, Val(8)), ntuple(_ -> zero(Int16), Val(8)), zero(Int8), zero(Int8), zero(Int8))

# struct StaticLoop <: AbstractLoop
#     c::NTuple{2,Float64} # constants for knowns
#     A::NTuple{2,UInt64} # A is a ByteVector in practice, see unpack(::Loop, ::Val{:A})
#     nloops::Int8 # A * x ≥ c
#     loopid::Int8 # id of this loop
#     actuallystatic::Bool
# end
# struct RectangularLoop <: AbstractLoop
#     # loopids::ByteVector{UInt64} # up to 8 loop ids for loops in A
#     p::NTuple{2,Float64} # param value placeholders for unknowns
#     c::NTuple{2,Float64} # constants for knowns
#     B::NTuple{2,Int16} # A * x + B * p ≥ c
#     paramids::NTuple{2,Int8} # ids of the unknowns
#     nloops::Int8
#     loopid::Int8 # id of this loop
# end
# struct StaticRectangularLoop <: AbstractLoop
#     c::NTuple{2,Float64}
#     nloops::Int8
#     loopid::Int8 # id of this loop
#     actuallystatic::Bool
# end
# const RectangularLoop = Union{}

@inline function UnPack.unpack(l::AbstractLoop, ::Val{:A})
    A = l.A
    nloops = l.nloops# - one(Int8)
    Base.Cartesian.@ntuple 8 i -> ByteVector(A[i], nloops)
end
@inline function UnPack.unpack(l::RectangularLoop, ::Val{:A})
    A₁ = A₂ = zero(UInt64)
    nloops = l.l.nloops# - one(Int8)
    ByteVector(A₁, nloops), ByteVector(A₂, nloops)
end
# @inline UnPack.unpack(l::Loop, ::Val{S}) where {S} = UnPack.unpack(l.l, Val{S}())

# UnPack.unpack(l::AbstractLoop, ::Val{cₗ}) = l.c[1]
# UnPack.unpack(l::AbstractLoop, ::Val{cᵤ}) = l.c[2]

# assume_static(l::StaticLoop) = l
# function assume_static(l::Loop)
#     cₗ, cᵤ = l.c
#     Bₗ, Bᵤ = l.B
#     if Bₗ == Bᵤ == zero(Bₗ)
#         StaticLoop( (cₗ, cᵤ), l.A, l.nloops, l.loopid, true )
#     else
#         pₗ, pᵤ = l.p
#         cₗ -= Bₗ * pₗ
#         cᵤ -= Bᵤ * pᵤ
#         StaticLoop( (cₗ, cᵤ), l.A, l.nloops, l.loopid, false )
#     end
# end

function depending_ind(Aₗ′, Aᵤ′, not_visited_mask)
    i = trailing_zeros(not_visited_mask)
    # Aₗ′ᵢ = Aₗ′[(i>>>3)+1] & not_visited_mask # gives all that depend on i
    # Aᵤ′ᵢ = Aᵤ′[(i>>>3)+1] & not_visited_mask # gives all that depend on i
    # while !(allzero(Aₗ′ᵢ) & allzero(Aᵤ′ᵢ))
    A′ᵢ = (Aₗ′[(i>>>3)+1] + Aᵤ′[(i>>>3)+1]) & not_visited_mask # gives all that depend on i
    while !allzero(A′ᵢ)
        nvm = (not_visited_mask >>> (i+1))
        @assert !iszero(nvm)
        i += 1 + trailing_zeros(nvm)
        A′ᵢ = (Aₗ′[(i>>>3)+1] + Aᵤ′[(i>>>3)+1]) & not_visited_mask # gives all that depend on i
        # Aₗ′ᵢ = Aₗ′[(i>>>3)+1] & not_visited_mask
        # Aᵤ′ᵢ = Aᵤ′[(i>>>3)+1] & not_visited_mask
    end
    i = (i >>> 3) + 1
    not_visited_mask = VectorizationBase.fuseint(setindex(VectorizationBase.splitint(not_visited_mask, Bool), false, i))
    i, not_visited_mask
end

