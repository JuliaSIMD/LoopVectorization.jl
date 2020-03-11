
# Rename file to offsetarrays?

import OffsetArrays

# If ndim(::OffsetArray) == 1, we can convert to a regular strided pointer and offset.
@inline VectorizationBase.stridedpointer(a::OffsetArrays.OffsetArray{<:Any,1}) = gesp(stridedpointer(parent(a)), (-@inbounds(a.offsets[1]),))

struct OffsetStridedPointer{T, N, P <: VectorizationBase.AbstractStridedPointer{T}} <: VectorizationBase.AbstractStridedPointer{T}
    ptr::P
    offsets::NTuple{N,Int}
end
# if ndim(A::OffsetArray) ≥ 2, then eachindex(A) isa Base.OneTo, index starting at 1.
# but multiple indexing is calculated using offsets, so we need a special type to express this.
@inline function VectorizationBase.stridedpointer(A::OffsetArrays.OffsetArray)
    OffsetStridedPointer(stridedpointer(parent(A)), VectorizationBase.staticm1(A.offsets))
end
# Tuple of length == 1, use ind directly.
# @inline VectorizationBase.offset(ptr::OffsetStridedPointer, ind::Tuple{I}) where {I} = VectorizationBase.offset(ptr.ptr, ind)
# Tuple of length > 1, subtract offsets.
# @inline VectorizationBase.offset(ptr::OffsetStridedPointer{<:Any,N}, ind::Tuple) where {N} = VectorizationBase.offset(ptr.ptr, ntuple(n -> ind[n] + ptr.offsets[n], Val{N}()))
@inline VectorizationBase.offset(ptr::OffsetStridedPointer, ind::Tuple{I}) where {I} = ind
# Tuple of length > 1, subtract offsets.
@inline VectorizationBase.offset(ptr::OffsetStridedPointer{<:Any,N}, ind::Tuple) where {N} = ntuple(n -> ind[n] - ptr.offsets[n], Val{N}())
@inline Base.similar(p::OffsetStridedPointer, ptr::Ptr) = OffsetStridedPointer(similar(p.ptr, ptr), p.offsets)

# If an OffsetArray is getting indexed by a (loop-)constant value, then this particular vptr object cannot also be eachindexed, so we can safely return a stridedpointer
@inline function VectorizationBase.subsetview(ptr::OffsetStridedPointer{<:Any,N}, ::Val{I}, i) where {I,N}
    subsetview(gesp(ptr.ptr, ntuple(n -> 0 - @inbounds(ptr.offsets[n]), Val{N}())), Val{I}(), i)
end

