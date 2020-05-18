
struct Zero <: Number end

@inline VectorizationBase.vsub(::Zero, i) = vsub(i)
@inline VectorizationBase.vsub(i, ::Zero) = i
@inline VectorizationBase.vsub(::Zero, ::Zero) = Zero()
@inline VectorizationBase.vadd(::Zero, ::Zero) = Zero()
@inline VectorizationBase.vadd(::Zero, a) = a
@inline VectorizationBase.vadd(a, ::Zero) = a
@inline VectorizationBase.vmul(::Zero, ::Any) = Zero()
@inline VectorizationBase.vmul(::Any, ::Zero) = Zero()
for T ∈ [:Int,:SVec]
    @eval @inline VectorizationBase.vadd(::Zero, a::$T) = a
    @eval @inline VectorizationBase.vadd(a::$T, ::Zero) = a
    @eval @inline VectorizationBase.vsub(::Zero, a::$T) = vsub(a)
    @eval @inline VectorizationBase.vsub(a::$T, ::Zero) = a
    @eval @inline VectorizationBase.vmul(::Zero, ::$T) = Zero()
    @eval @inline VectorizationBase.vmul(::$T, ::Zero) = Zero()
end
@inline VectorizationBase.vadd(::Zero, a::_MM) = a
@inline VectorizationBase.vadd(a::_MM, ::Zero) = a
@inline VectorizationBase.vadd(::_MM{W,Zero}, v::VectorizationBase.AbstractSIMDVector{W,T}) where {W,T} = vadd(SIMDPirates.vrange(Val{W}(), T), v)
@inline VectorizationBase.vadd(v::VectorizationBase.AbstractSIMDVector{W,T}, ::_MM{W,Zero}) where {W,T} = vadd(SIMDPirates.vrange(Val{W}(), T), v)
@inline VectorizationBase.vadd(::_MM{W,Zero}, ::_MM{W,Zero}) where {W} = SIMDPirates.vrangemul(Val{W}(), 2, Val{0}())
@inline VectorizationBase.vmul(::_MM{W,Zero}, i) where {W} = SIMDPirates.svrangemul(Val{W}(), i, Val{0}())
@inline VectorizationBase.vmul(i, ::_MM{W,Zero}) where {W} = SIMDPirates.svrangemul(Val{W}(), i, Val{0}())
@inline VectorizationBase.vload(ptr::Ptr, ::Zero) = vload(ptr)
@inline VectorizationBase.vload(ptr::Ptr, ::_MM{W,Zero}) where {W} = vload(Val{W}(), ptr)
@inline VectorizationBase.vload(ptr::Ptr, ::_MM{W,Zero}, m::VectorizationBase.Mask) where {W} = vload(Val{W}(), ptr, m.u)
@inline VectorizationBase.vstore!(ptr::Ptr{T}, v::T, ::Zero) where {T} = vstore!(ptr, v)
@inline VectorizationBase.vnoaliasstore!(ptr::Ptr{T}, v::T, ::Zero) where {T} = vnoaliasstore!(ptr, v)
@inline VectorizationBase.vstore!(ptr::Ptr{T}, v, ::Zero) where {T} = vstore!(ptr, convert(T,v))
@inline VectorizationBase.vnoaliasstore!(ptr::Ptr{T}, v, ::Zero) where {T} = vnoaliasstore!(ptr, convert(T,v))
@inline VectorizationBase.vstore!(ptr::Ptr{T}, v::Integer, ::Zero) where {T <: Integer} = vstore!(ptr, v % T)
@inline VectorizationBase.vnoaliasstore!(ptr::Ptr{T}, v::Integer, ::Zero) where {T <: Integer} = vnoaliasstore!(ptr, v % T)
# @inline VectorizationBase.vstore!(ptr::Ptr{T}, v::T, ::Zero, m::VectorizationBase.Mask) where {T} = vstore!(ptr, v, m.u)
# @inline VectorizationBase.vnoaliasstore!(ptr::Ptr{T}, v::T, ::Zero, m::VectorizationBase.Mask) where {T} = vnoaliasstore!(ptr, v, m.u)
for V ∈ [:(NTuple{W,Core.VecElement{T}}), :(SVec{W,T})]
    @eval @inline VectorizationBase.vstore!(ptr::Ptr{T}, v::$V, ::Zero) where {W,T} = vstore!(ptr, v)
    @eval @inline VectorizationBase.vstore!(ptr::Ptr{T}, v::$V, ::_MM{W,Zero}) where {W,T} = vstore!(ptr, v)
    @eval @inline VectorizationBase.vnoaliasstore!(ptr::Ptr{T}, v::$V, ::Zero) where {W,T} = vnoaliasstore!(ptr, v)
    @eval @inline VectorizationBase.vnoaliasstore!(ptr::Ptr{T}, v::$V, ::_MM{W,Zero}) where {W,T} = vnoaliasstore!(ptr, v)
    for M ∈ [:(VectorizationBase.Mask{W}), :Unsigned]
        @eval @inline VectorizationBase.vstore!(ptr::Ptr{T}, v::$V, ::Zero, m::$M) where {W,T} = vstore!(ptr, v, m)
        @eval @inline VectorizationBase.vstore!(ptr::Ptr{T}, v::$V, ::_MM{W,Zero}, m::$M) where {W,T} = vstore!(ptr, v, m)
        @eval @inline VectorizationBase.vnoaliasstore!(ptr::Ptr{T}, v::$V, ::Zero, m::$M) where {W,T} = vnoaliasstore!(ptr, v, m)
        @eval @inline VectorizationBase.vnoaliasstore!(ptr::Ptr{T}, v::$V, ::_MM{W,Zero}, m::$M) where {W,T} = vnoaliasstore!(ptr, v, m)
    end
end

