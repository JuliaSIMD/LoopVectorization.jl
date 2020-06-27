
# Rename file to offsetarrays?

# If ndim(::OffsetArray) == 1, we can convert to a regular strided pointer and offset.
@inline VectorizationBase.stridedpointer(a::OffsetArrays.OffsetArray{<:Any,1}) = gesp(stridedpointer(parent(a)), (-@inbounds(a.offsets[1]),))

struct OffsetStridedPointer{T, N, P <: VectorizationBase.AbstractStridedPointer{T}} <: VectorizationBase.AbstractStridedPointer{T}
    ptr::P
    offsets::NTuple{N,Int}
end
# if ndim(A::OffsetArray) ≥ 2, then eachindex(A) isa Base.OneTo, index starting at 1.
# but multiple indexing is calculated using offsets, so we need a special type to express this.
@inline function VectorizationBase.stridedpointer(A::OffsetArrays.OffsetArray)
    OffsetStridedPointer(stridedpointer(parent(A)), A.offsets)
end

@inline function VectorizationBase.stridedpointer(
    B::Adjoint{T,A}
) where {T,A<:OffsetArrays.OffsetArray{T}}
    Boff = parent(B)
    OffsetStridedPointer(
        stridedpointer(parent(Boff)'),
        Boff.offsets
    )
end
@inline function Base.transpose(A::OffsetStridedPointer)
    OffsetStridedPointer(
        transpose(A.ptr), A.offsets
    )
end
# Tuple of length == 1, use ind directly.
# @inline VectorizationBase.offset(ptr::OffsetStridedPointer, ind::Tuple{I}) where {I} = VectorizationBase.offset(ptr.ptr, ind)
# Tuple of length > 1, subtract offsets.
# @inline VectorizationBase.offset(ptr::OffsetStridedPointer{<:Any,N}, ind::Tuple) where {N} = VectorizationBase.offset(ptr.ptr, ntuple(n -> ind[n] + ptr.offsets[n], Val{N}()))
@inline VectorizationBase.offset(ptr::OffsetStridedPointer, ind::Tuple{I}) where {I} = VectorizationBase.offset(ptr.ptr, ind)
# Tuple of length > 1, subtract offsets.
@inline VectorizationBase.offset(ptr::OffsetStridedPointer{<:Any,N}, ind::Tuple) where {N} = VectorizationBase.offset(ptr.ptr, ntuple(n -> vsub(ind[n], ptr.offsets[n]), Val{N}()))
@inline Base.similar(p::OffsetStridedPointer, ptr::Ptr) = OffsetStridedPointer(similar(p.ptr, ptr), p.offsets)
@inline Base.pointer(p::OffsetStridedPointer) = pointer(p.ptr)
@inline VectorizationBase.gesp(p::OffsetStridedPointer, i) = similar(p.ptr, gep(p, i))
# @inline VectorizationBase.gesp(p::OffsetStridedPointer, i) = similar(p, gep(p.ptr, i))
# If an OffsetArray is getting indexed by a (loop-)constant value, then this particular vptr object cannot also be eachindexed, so we can safely return a stridedpointer
@inline function VectorizationBase.subsetview(ptr::OffsetStridedPointer{<:Any,N}, ::Val{I}, i) where {I,N}
    subsetview(gesp(ptr.ptr, ntuple(n -> 0 - @inbounds(ptr.offsets[n]), Val{N}())), Val{I}(), i)
end

@inline VectorizationBase.offset(ptr::OffsetStridedPointer{<:Any,<:Any,<:VectorizationBase.AbstractBitPointer}, ind::Tuple{I}) where {I} = VectorizationBase.offset(ptr.ptr, (vsub(ind[1], ptr.offsets[1]),))
@inline VectorizationBase.gesp(ptr::VectorizationBase.AbstractBitPointer, i) = OffsetStridedPointer(ptr, vsub.(-1, unwrap.(i)))
