


# Maybe the polyhedra should be a vector of these?
struct Loop
    loopids::ByteVector{UInt64} # up to 8 loop ids for loops in A
    p::NTuple{2,Float64} # param values for unknowns
    c::NTuple{2,Float64} # constants for knowns
    A::NTuple{2,ByteVector{UInt64}} # A * x + B * p ≥ c
    B::NTuple{2,Int8}
    paramids::NTuple{2,Int8} # ids of the unknowns
    loopid::Int8 # id of this loop
end


