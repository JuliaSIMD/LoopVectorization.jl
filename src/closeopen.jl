
struct CloseOpen{L <: Union{Int,StaticInt}, U <: Union{Int,StaticInt}} <: AbstractUnitRange{Int}
    start::L
    upper::U
    @inline CloseOpen(s::StaticInt{L}, u::StaticInt{U}) where {L,U} = new{StaticInt{L},StaticInt{U}}(s, u)
    @inline CloseOpen(s::Integer, u::StaticInt{U}) where {U} = new{Int,StaticInt{U}}(s % Int, u)
    @inline CloseOpen(s::StaticInt{L}, u::Integer) where {L} = new{StaticInt{L},Int}(s, u % Int)
    @inline CloseOpen(s::Integer, u::Integer) = new{Int,Int}(s % Int, u % Int)
end
@inline CloseOpen(len::Integer) = CloseOpen(Zero(), len)

@inline Base.first(r::CloseOpen) = r.start
@inline Base.step(::CloseOpen) = One()
@inline Base.last(r::CloseOpen) = r.upper - One()
@inline Base.length(r::CloseOpen) = r.upper - r.start
@inline Base.length(r::CloseOpen{Zero}) = r.upper

@inline Base.iterate(r::CloseOpen) = (i = Int(first(r)); (i, i))
@inline Base.iterate(r::CloseOpen, i::Int) = (i += 1) == r.upper ? nothing : (i, i)

ArrayInterface.known_first(::Type{<:CloseOpen{StaticInt{F}}}) where {F} = F
ArrayInterface.known_step(::Type{<:CloseOpen}) = 1
ArrayInterface.known_last(::Type{<:CloseOpen{<:Any,StaticInt{L}}}) where {L} = L - 1
ArrayInterface.known_length(::Type{CloseOpen{StaticInt{F},StaticInt{L}}}) where {F,L} = L - F

@inline canonicalize_range(r::OptionallyStaticUnitRange) = r
@inline canonicalize_range(r::CloseOpen) = r
@inline canonicalize_range(r::AbstractUnitRange) = maybestaticfirst(r):maybestaticlast(r)
@inline canonicalize_range(r::CartesianIndices) = CartesianIndices(map(canonicalize_range, r.indices))


