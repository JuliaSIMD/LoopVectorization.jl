
# canonicalization should move `i` into indexing expression in `i+j:i+k` ?
#
abstract type AbstractLoop end
"""
Supports loop depth of up to 9.
`A` holds 8 loops; the actual loop itself is omitted, with
(1, -1) being inserted into A₁, A₂ and `loopid`.
"""
struct Loop <: AbstractLoop
    # loopids::ByteVector{UInt64} # up to 8 loop ids for loops in A
    p::NTuple{2,Float64} # param value placeholders for unknowns
    c::NTuple{2,Float64} # constants for knowns
    A::NTuple{2,UInt64} # A is a ByteVector in practice, see unpack(::Loop, ::Val{:A})
    B::NTuple{2,Int16} # A * x + B * p ≥ c
    paramids::NTuple{2,Int8} # ids of the unknowns
    nloops::Int8
    loopid::Int8 # id of this loop
end
struct StaticLoop <: AbstractLoop
    c::NTuple{2,Float64} # constants for knowns
    A::NTuple{2,UInt64} # A is a ByteVector in practice, see unpack(::Loop, ::Val{:A})
    nloops::Int8 # A * x ≥ c
    loopid::Int8 # id of this loop
end
# struct RectangularLoop <: AbstractLoop
#     # loopids::ByteVector{UInt64} # up to 8 loop ids for loops in A
#     p::NTuple{2,Float64} # param value placeholders for unknowns
#     c::NTuple{2,Float64} # constants for knowns
#     B::NTuple{2,Int16} # A * x + B * p ≥ c
#     paramids::NTuple{2,Int8} # ids of the unknowns
#     nloops::Int8
#     loopid::Int8 # id of this loop
# end
struct StaticRectangularLoop <: AbstractLoop
    c::NTuple{2,Float64}
    nloops::Int8
    loopid::Int8 # id of this loop
end


function UnPack.unpack(l::AbstractLoop, ::Val{:A})
    A₁, A₂ = l.A
    nloops = l.nloops
    ByteVector(A₁, nloops), ByteVector(A₂, nloops)
end
function UnPack.unpack(l::StaticRectangularLoop, ::Val{:A})
    A₁ = A₂ = zero(UInt64)
    nloops = l.nloops
    ByteVector(A₁, nloops), ByteVector(A₂, nloops)
end

# UnPack.unpack(l::AbstractLoop, ::Val{cₗ}) = l.c[1]
# UnPack.unpack(l::AbstractLoop, ::Val{cᵤ}) = l.c[2]

assume_static(l::StaticLoop) = l
function assume_static(l::Loop)
    cₗ, cᵤ = l.c
    Bₗ, Bᵤ = l.B
    pₗ, pᵤ = l.p
    cₗ -= Bₗ * pₗ
    cᵤ -= Bᵤ * pᵤ
    StaticLoop( (cₗ, cᵤ), l.A, l.nloops, l.loopid )
end
