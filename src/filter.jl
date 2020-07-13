
if (Base.libllvm_version ≥ v"7" && VectorizationBase.AVX512F) || Base.libllvm_version ≥ v"9"
    function vfilter!(f::F, x::Vector{T}, y::AbstractArray{T}) where {F,T <: NativeTypes}
        W, Wshift = VectorizationBase.pick_vector_width_shift(T)
        N = length(y)
        Nrep = N >>> Wshift
        Nrem = N & (W - 1)
        j = 0
        GC.@preserve x y begin
            ptr_x = pointer(x)
            ptr_y = pointer(y)
            for _ ∈ 1:Nrep
                vy = vload(Vec{W,T}, ptr_y)
                mask = f(SVec(vy))
                SIMDPirates.compressstore!(gep(ptr_x, j), vy, mask)
                ptr_y = gepbyte(ptr_y, VectorizationBase.REGISTER_SIZE)
                j = vadd(j, count_ones(mask))
            end
            rem_mask = VectorizationBase.mask(T, Nrem)
            vy = vload(Vec{W,T}, ptr_y, rem_mask)
            mask = rem_mask & f(SVec(vy))
            SIMDPirates.compressstore!(gep(ptr_x, j), vy, mask)
            j = vadd(j, count_ones(mask))
            Base._deleteend!(x, N-j) # resize!(x, j)
        end
        x
    end
    vfilter!(f::F, x::Vector{T}) where {F, T<:NativeTypes} = vfilter!(f, x, x)
    vfilter(f::F, y::AbstractArray{T}) where {F, T<:NativeTypes} = vfilter!(f, Vector{T}(undef, length(y)), y)
end
vfilter(f::F, y) where {F} = filter(f, y)
vfilter!(f::F, y) where {F} = filter!(f, y)

"""
    vfilter(f, a::AbstractArray)

SIMD-vectorized `filter`, returning an array containing the elements of `a` for which `f` return `true`.
"""
vfilter

"""
    vfilter!(f, a::AbstractArray)

SIMD-vectorized `filter!`, removing the element of `a` for which `f` is false.
"""
vfilter!
