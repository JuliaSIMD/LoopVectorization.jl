
function vfilter!(f::F, x::Vector{T}, y::AbstractArray{T}) where {F,T <: NativeTypes}
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    N = length(y)
    Nrep = N >>> Wshift
    Nrem = N & (W - 1)
    j = 0
    st = VectorizationBase.static_sizeof(T)
    zero_index = MM(W, Static(0), st)
    GC.@preserve x y begin
        ptr_x = llvmptr(x)
        ptr_y = llvmptr(y)
        for _ ∈ 1:Nrep
            vy = VectorizationBase.__vload(ptr_y, zero_index, False(), register_size())
            mask = f(vy)
            VectorizationBase.compressstore!(gep(ptr_x, VectorizationBase.lazymul(st, j)), vy, mask)
            ptr_y = gep(ptr_y, register_size())
            j = vadd_fast(j, count_ones(mask))
        end
        rem_mask = VectorizationBase.mask(T, Nrem)
        vy = VectorizationBase.__vload(ptr_y, zero_index, rem_mask, False(), register_size())
        mask = rem_mask & f(vy)
        VectorizationBase.compressstore!(gep(ptr_x, VectorizationBase.lazymul(st, j)), vy, mask)
        j = vadd_fast(j, count_ones(mask))
        Base._deleteend!(x, N-j) # resize!(x, j)
    end
    x
end
vfilter!(f::F, x::Vector{T}) where {F, T<:NativeTypes} = vfilter!(f, x, x)
vfilter(f::F, y::AbstractArray{T}) where {F, T<:NativeTypes} = vfilter!(f, Vector{T}(undef, length(y)), y)
vfilter(f::F, y) where {F} = filter(f, y)
vfilter!(f::F, y) where {F} = filter!(f, y)

"""
    vfilter(f, a::AbstractArray)

SIMD-vectorized `filter`, returning an array containing the elements of `a` for which `f` return `true`.

This function requires AVX512 to be faster than `Base.filter`, as it adds compressstore instructions.
"""
vfilter

"""
    vfilter!(f, a::AbstractArray)

SIMD-vectorized `filter!`, removing the element of `a` for which `f` is false.
"""
vfilter!
