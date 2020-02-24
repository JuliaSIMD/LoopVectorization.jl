
function vfilter!(f::F, x::Vector{T}, y::AbstractArray{T}) where {F,T <: SUPPORTED_TYPES}
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    N = length(y)
    Nrep = N >>> Wshift
    Nrem = N & (W - 1)
    i = 0
    j = 0
    GC.@preserve x y begin
        ptr_x = pointer(x)
        ptr_y = pointer(y)
        for _ ∈ 1:Nrep
            vy = load(Vec{W,T}, ptr_y, i)
            mask = f(SVec(vy))
            SIMDPirates.compressstore!(gep(ptr_x, j), vy, mask)
            i += W
            j += count_ones(mask)
        end
        rem_mask = VectorizationBase.mask(T, Nrem)
        vy = load(Vec{W,T}, gep(ptr_y, i), rem_mask)
        mask = rem_mask & f(SVec(vy))
        SIMDPirates.compressstore!(gep(ptr_x, j), vy, mask)
        j += count_ones(mask)
        Base._deleteend!(x, N-j) # resize!(x, j)
    end
    x
end
vfilter(f, y::AbstractArray{T}) where {T<:SUPPORTED_TYPES} = vfilter!(f, Vector{T}(undef, length(y)), y)

