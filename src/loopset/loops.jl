


"""
Supports loop depth of up to 9.
`A` holds 8 loops; the actual loop itself is omitted, with
(1, -1) being inserted into A₁, A₂ and `loopid`.
"""
struct Loop
    # loopids::ByteVector{UInt64} # up to 8 loop ids for loops in A
    p::NTuple{2,Float64} # param value placeholders for unknowns
    c::NTuple{2,Float64} # constants for knowns
    A::NTuple{2,UInt64} # A is a ByteVector in practice, see unpack(::Loop, ::Val{:A})
    B::NTuple{2,Int8} # A * x + B * p ≥ c
    paramids::NTuple{2,Int8} # ids of the unknowns
    nloops::Int8
    loopid::Int8 # id of this loop
end

function UnPack.unpack(l::Loop, ::Val{:A})
    A₁, A₂ = l.A
    nloops = l.nloops
    ByteVector(A₁, nloops), ByteVector(A₂, nloops)
end
