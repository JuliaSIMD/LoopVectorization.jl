
abstract type AbstractLimitedRangeVector{T} <: AbstractVector{T} end

"""
Can hold values between -128 and 127.
Length is not a compile time constant, but limited to a maximum of 8.
"""
struct ByteVector{U<:Unsigned} <: AbstractLimitedRangeVector{Int8}
    data::U
    len::Int
end
# struct HalfByteVector{U<:Unsigned} <: AbstractLimitedRangeVector{Int8}
#     data::U
#     len::Int
# end

"""
Can hold values between `typemin(Int16)` and `typemax(Int16)`, i.e. `(-32768, 32767)`.
Length is not a compile time constant, but limited to a maximum of 8.
"""
struct WordVector <: AbstractLimitedRangeVector{Int16}
    data::UInt128
    len::Int
end
# VectorizationBase.data(v::AbstractLimitedRangeVector) = v.data
lastelem_mask(x::UInt64) = 0x00000000000000ff
lastelem_mask(x::UInt128) = 0x0000000000000000000000000000ffff
lastelem_mask(::ByteVector{UInt64}) = 0x00000000000000ff
lastelem_mask(::ByteVector{UInt128}) = 0x000000000000000000000000000000ff
lastelem_mask(x::WordVector) = 0x0000000000000000000000000000ffff
"""
Last elem (wish respect to size of `d`) of `x`.
"""
lastelem(d, x) = (lastelem_mask(d) & x) % typeof(d)
lastelem(x) = lastelem(x, x)
trailing_byte(x::UInt64) = lastelem_mask(x) & x
trailing_byte(x) = trailing_byte(x % UInt64)
# Base.length(v::AbstractLimitedRangeVector) = lastelem(data(v)) % Int
Base.length(v::AbstractLimitedRangeVector) = v.len
Base.size(v::AbstractLimitedRangeVector) = (v.len,)
# typemins(::Type{ByteVector}) = 0x8080808080808080
# typemins(::Type{WordVector}) = 0x80008000800080008000800080008000
# Base.length(v::V) where {T, V <: AbstractLimitedRangeVector{T}} = 8 - (leading_zeros(v.data ⊻ typemins(V)) ÷ (8sizeof(T)))
# Base.size(v::AbstractLimitedRangeVector) = (length(v),)
Base.getindex(v::AbstractLimitedRangeVector{T}, i) where {T} = (v.data >>> unsigned(sizeof(T)*8*(i-1))) % T

allzero(x::AbstractLimitedRangeVector) = iszero(x.data)

@inline function Base.iterate(v::AbstractLimitedRangeVector{T}, state = (v.data, v.len)) where {T}
    d, l = state
    iszero(l) ? nothing : d % T, (d >>> (8sizeof(T)), l - one(l))
end


# ByteVector() = ByteVector(typemins(ByteVector), 0)
# WordVector() = WordVector(typemins(WordVector), 0)
ByteVector() = ByteVector(zero(UInt64), 0)
ByteVector{U}() where {U} = ByteVector(zero(U), 0)
WordVector() = WordVector(zero(UInt128), 0)
function filluint(data, ::Type{T}, x::Vararg{Integer,N}) where {T,N}
    n = 0
    while true
        data |= x[N - n] % T
        n += 1
        n == N && return data
        data <<= 8*sizeof(T)
    end
end
ByteVector(x...) = ByteVector(filluint(zero(UInt64), UInt8, x...), length(x))
WordVector(x...) = WordVector(filluint(zero(UInt128), UInt16, x...), length(x))
function push(v::V, x) where {T, V <: AbstractLimitedRangeVector{T}}
    data = v.data
    shifter = 8*sizeof(T)*length(v)
    data &= ~(lastelem_mask(data) << shifter)
    V(data | ((lastelem(data, x)) << shifter), length(v) + 1)
end
function pushfirst(v::V, x) where {T, V <: AbstractLimitedRangeVector{T}}
    data = v.data << (8sizeof(T))
    V(data | lastelem(data, x), length(v) + 1)
end
function ArrayInterface.deleteat(v::V, i) where {T, V <: AbstractLimitedRangeVector{T}}
    dlead = dtrail = v.data
    leadmask = typemax(dtrail) << ((i-1)*8sizeof(T))
    trailmask = ~leadmask
    dlead >>= 8sizeof(T)
    d = (dlead & leadmask) | (dtrail & trailmask)
    V(d, length(v) - 1)
end
@inline function Base.setindex(v::V, x, i::Int) where {V <: AbstractLimitedRangeVector}
    m1 = lastelem_mask(v)
    shift = ((i - 1)*8sizeof(eltype(v)))
    m = ~(m1 << shift)
    x1 = (x % eltype(m1)) << shift
    V((v.data & m) | x1, length(v))
end

struct ShortDynamicVector{T,N} <: AbstractVector{T}
    data::NTuple{N,T}
    len::Int
end
Base.length(v::ShortDynamicVector) = v.len
Base.@propagate_inbounds Base.getindex(v::ShortDynamicVector, i) = v.data[i]
ShortDynamicVector{T,N}() where {T,N} = ShortDynamicVector(ntuple(_ -> zero(T), Val{N}()), 0)
ShortDynamicVector{T}() where {T} = ShortDynamicVector{T,8}()
ShortDynamicVector(x::Vararg{T,N}) where {T,N} = ShortDynamicVector(ntuple(n -> n > N ? zero(T) : (x[n]), Val(8)), N)
function push(v::ShortDynamicVector{T,N}, x) where {T,N}
    data = let N = length(v), d = v.data
        ntuple(Val{N}()) do n
            if n ≤ N
                d[n]
            elseif n == N + 1
                T(x)
            else
                zero(T)
            end
        end
    end
    ShortDynamicVector(data, length(v)+1)
end
function pushfirst(v::ShortDynamicVector{T,N}, x) where {T,N}
    data = let N = length(v), d = v.data
        ntuple(Val{N}()) do n
            if isone(n)
                T(x)
            elseif n ≤ N + 1
                d[n-1]
            else
                zero(T)
            end
        end
    end
    ShortDynamicVector(data, length(v)+1)
end
function ArrayInterface.deleteat(v::ShortDynamicVector{T,N}, i) where {T,N}
    data = let N = length(v), d = v.data
        ntuple(Val{N}()) do n
            if n ≥ N
                zero(T)
            elseif n < i
                d[n]
            else
                d[n-1]
            end
        end
    end
    ShortDynamicVector(data, length(v)+1)
end
function Base.setindex(v::ShortDynamicVector, x, i)
    ShortDynamicVector(Base.setindex(v, x, i), max(i, v.len))
end

_ones(::Type{UInt8}) = 0x01
_ones(::Type{UInt16}) = 0x0101
_ones(::Type{UInt32}) = 0x01010101
_ones(::Type{UInt64}) = 0x0101010101010101
_ones(::Type{UInt128}) = 0x01010101010101010101010101010101
struct DynamicCartesian{U}
    v::ByteVector{U}
end
Base.length(dc::DynamicCartesian) = prod(dc.v)
@inline function Base.iterate(dc::DynamicCartesian{U}) where {U}
    init = ByteVector{U}(_ones(U), length(dc.v))
    init, init
end
@inline function Base.iterate(dc::DynamicCartesian, state::ByteVector)
    v = dc.v
    for i ∈ eachindex(state)
        sᵢ = state[i]
        if sᵢ < v[i]
            vout = Base.setindex(state, sᵢ+1, i)
            return vout, vout
        end
        state = Base.setindex(state, one(Int8), i)
    end
    nothing
end


half(::Type{Int16}) = Int8
half(::Type{Int32}) = Int16
half(::Type{Int64}) = Int32
splitint(x::T) where {T<:Integer} = (S = half(T); ((x >>> 4sizeof(T)) % S, x % S))

