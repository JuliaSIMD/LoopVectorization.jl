
const DenseNativeArray = DenseArray{<:NativeTypes}

"""
`vstorent!` (non-temporal store) requires data to be aligned.
`alignstores!` will align `y` in preparation for the non-temporal maps.
"""
function alignstores!(
    f::F, y::DenseArray{T},
    args::Vararg{DenseNativeArray,A}
) where {F, T <: Base.HWReal, A}
    N = length(y)
    ptry = VectorizationBase.zstridedpointer(y)
    ptrargs = VectorizationBase.zstridedpointer.(args)
    V = VectorizationBase.pick_vector_width_val(T)
    W = unwrap(V)
    zero_index = MM{W}(Static(0))
    uintptry = reinterpret(UInt, pointer(ptry))
    @assert iszero(uintptry & (sizeof(T) - 1)) "The destination vector (`dest`) must be aligned to `sizeof(eltype(dest)) == $(sizeof(T))` bytes."
    alignment = uintptry & (VectorizationBase.REGISTER_SIZE - 1)
    if alignment > 0
        i = reinterpret(Int, W - (alignment >>> VectorizationBase.intlog2(sizeof(T))))
        m = mask(T, i)
        if N < i
            m &= mask(T, N & (W - 1))
        end
        vnoaliasstore!(ptry, f(vload.(ptrargs, ((zero_index,),), m)...), (zero_index,), m)
        gesp(ptry, (i,)), gesp.(ptrargs, ((i,),)), N - i
    else
        ptry, ptrargs, N
    end
end

function vmap_singlethread!(
    f::F, y::DenseArray{T},
    ::Val{NonTemporal},
    args::Vararg{DenseNativeArray,A}
) where {F,T <: Base.HWReal, A, NonTemporal}
    if NonTemporal # if stores into `y` aren't aligned, we'll get a crash
        ptry, ptrargs, N = alignstores!(f, y, args...)
    else
        N = length(y)
        ptry = VectorizationBase.zstridedpointer(y)
        ptrargs = VectorizationBase.zstridedpointer.(args)
    end
    i = 0
    V = VectorizationBase.pick_vector_width_val(T)
    W = unwrap(V)
    st = VectorizationBase.static_sizeof(T)
    zero_index = MM{W}(Static(0), st)
    while i < N - ((W << 2) - 1)

        # vstore!(stridedpointer(B), VectorizationBase.VecUnroll((v1,v2,v3)), VectorizationBase.Unroll{AU,1,3,AV,W64,zero(UInt)}((i, j, k)))
        # vload(stridedpointer(B), VectorizationBase.Unroll{1,1,4,1,W,0x0000000000000000}((i,)))
        
        index = VectorizationBase.Unroll{1,1,4,1,W,0x0000000000000000}((i,))
        v = f(vload.(ptrargs, index)...)
        if NonTemporal
            vstorent!(ptry, v, index)
        else
            vnoaliasstore!(ptry, v, index)
        end
        i = vadd_fast(i, 4W)
    end
    while i < N - (W - 1) # stops at 16 when
        vᵣ = f(vload.(ptrargs, ((MM{W}(i),),))...)
        if NonTemporal
            vstorent!(ptry, vᵣ, (MM{W}(i),))
        else
            vnoaliasstore!(ptry, vᵣ, (MM{W}(i),))
        end
        i = vadd_fast(i, W)
    end
    if i < N
        m = mask(T, N & (W - 1))
        vnoaliasstore!(ptry, f(vload.(ptrargs, ((MM{W}(i),),), m)...), (MM{W}(i,),), m)
    end
    y
end

function vmap_multithreaded!(
    f::F,
    y::DenseArray{T},
    ::Val{true},
    args::Vararg{DenseNativeArray,A}
) where {F,T,A}
    ptry, ptrargs, N = alignstores!(f, y, args...)
    N > 0 || return y
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    V = VectorizationBase.pick_vector_width_val(T)
    Wsh = Wshift + 2
    Niter = N >>> Wsh
    Base.Threads.@threads :static for j ∈ 0:Niter-1
        index = VectorizationBase.Unroll{1,1,4,1,W,0x0000000000000000}((j << Wsh,))
        vstorent!(ptry, f(vload.(ptrargs, index)...), index)
    end
    ii = Niter << Wsh
    while ii < N - (W - 1) # stops at 16 when
        vstorent!(ptry, f(vload.(ptrargs, ((MM{W}(ii),),))...), (MM{W}(ii),))
        ii = vadd_fast(ii, W)
    end
    if ii < N
        m = mask(T, N & (W - 1))
        vnoaliasstore!(ptry, f(vload.(ptrargs, ((MM{W}(ii),),), m)...), (MM{W}(ii),), m)
    end
    y
end
function vmap_multithreaded!(
    f::F,
    y::DenseArray{T},
    ::Val{false},
    args::Vararg{DenseNativeArray,A}
) where {F,T,A}
    N = length(y)
    ptry = VectorizationBase.zstridedpointer(y)
    ptrargs = VectorizationBase.zstridedpointer.(args)
    N > 0 || return y
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    V = VectorizationBase.pick_vector_width_val(T)
    Wsh = Wshift + 2
    Niter = N >>> Wsh
    Base.Threads.@threads :static for j ∈ 0:Niter-1
        index = VectorizationBase.Unroll{1,1,4,1,W,0x0000000000000000}((j << Wsh,))
        vnoaliasstore!(ptry, f(vload.(ptrargs, index)...), index)
    end
    ii = Niter << Wsh
    while ii < N - (W - 1) # stops at 16 when
        vnoaliasstore!(ptry, f(vload.(ptrargs, ((MM{W}(ii),),))...), (MM{W}(ii),))
        ii = vadd_fast(ii, W)
    end
    if ii < N
        m = mask(T, N & (W - 1))
        vnoaliasstore!(ptry, f(vload.(ptrargs, ((MM{W}(ii),),), m)...), (MM{W}(ii),), m)
    end
    y
end


"""
    vmap!(f, destination, a::AbstractArray)
    vmap!(f, destination, a::AbstractArray, b::AbstractArray, ...)

Vectorized-`map!`, applying `f` to each element of `a` (or paired elements of `a`, `b`, ...)
and storing the result in `destination`.
"""
function vmap!(
    f::F, y::DenseArray{T}, args::Vararg{DenseNativeArray,A}
) where {F,T<:Base.HWReal,A}
    vmap_singlethread!(f, y, Val{false}(), args...)
end


"""
    vmapt!(::Function, dest, args...)

Like `vmap!` (see `vmap!`), but uses `Threads.@threads` for parallel execution.
"""
function vmapt!(
    f::F, y::DenseArray{T}, args::Vararg{DenseNativeArray,A}
) where {F,T<:Base.HWReal,A}
    vmap_multithreaded!(f, y, Val{false}(), args...)
end


"""
    vmapnt!(::Function, dest, args...)


This is a vectorized map implementation using nontemporal store operations. This means that the write operations to the destination will not go to the CPU's cache.
If you will not immediately be reading from these values, this can improve performance because the writes won't pollute your cache. This can especially be the case if your arguments are very long.

```julia
julia> using LoopVectorization, BenchmarkTools

julia> x = rand(10^8); y = rand(10^8); z = similar(x);

julia> f(x,y) = exp(-0.5abs2(x - y))
f (generic function with 1 method)

julia> @benchmark map!(f, \$z, \$x, \$y)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     439.613 ms (0.00% GC)
  median time:      440.729 ms (0.00% GC)
  mean time:        440.695 ms (0.00% GC)
  maximum time:     441.665 ms (0.00% GC)
  --------------
  samples:          12
  evals/sample:     1

julia> @benchmark vmap!(f, \$z, \$x, \$y)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     178.147 ms (0.00% GC)
  median time:      178.381 ms (0.00% GC)
  mean time:        178.430 ms (0.00% GC)
  maximum time:     179.054 ms (0.00% GC)
  --------------
  samples:          29
  evals/sample:     1

julia> @benchmark vmapnt!(f, \$z, \$x, \$y)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     144.183 ms (0.00% GC)
  median time:      144.338 ms (0.00% GC)
  mean time:        144.349 ms (0.00% GC)
  maximum time:     144.641 ms (0.00% GC)
  --------------
  samples:          35
  evals/sample:     1
```
"""
function vmapnt!(
    f::F, y::DenseArray{T}, args::Vararg{DenseNativeArray,A}
) where {F,T<:Base.HWReal,A}
    vmap_singlethread!(f, y, Val{true}(), args...)
end

"""
    vmapntt!(::Function, dest, args...)

Like `vmapnt!` (see `vmapnt!`), but uses `Threads.@threads` for parallel execution.
"""
function vmapntt!(
    f::F, y::DenseArray{T}, args::Vararg{DenseNativeArray,A}
) where {F,T<:Base.HWReal,A}
    vmap_multithreaded!(f, y, Val{true}(), args...)
end

# generic fallbacks
@inline vmap!(f, args...) = map!(f, args...)
@inline vmapt!(f, args...) = map!(f, args...)
@inline vmapnt!(f, args...) = map!(f, args...)
@inline vmapntt!(f, args...) = map!(f, args...)

function vmap_call(f::F, vm!::V, args::Vararg{Any,N}) where {V,F,N}
    T = Base._return_type(f, Base.Broadcast.eltypes(args))
    dest = similar(first(args), T)
    vm!(f, dest, args...)
end

"""
    vmap(f, a::AbstractArray)
    vmap(f, a::AbstractArray, b::AbstractArray, ...)

SIMD-vectorized `map`, applying `f` to each element of `a` (or paired elements of `a`, `b`, ...)
and returning a new array.
"""
vmap(f::F, args::Vararg{Any,N}) where {F,N} = vmap_call(f, vmap!, args...)

"""
    vmapt(f, a::AbstractArray)
    vmapt(f, a::AbstractArray, b::AbstractArray, ...)

A threaded variant of [`vmap`](@ref).
"""
vmapt(f::F, args::Vararg{Any,N}) where {F,N} = vmap_call(f, vmapt!, args...)

"""
    vmapnt(f, a::AbstractArray)
    vmapnt(f, a::AbstractArray, b::AbstractArray, ...)

A "non-temporal" variant of [`vmap`](@ref). This can improve performance in cases where
`destination` will not be needed soon.
"""
vmapnt(f::F, args::Vararg{Any,N}) where {F,N} = vmap_call(f, vmapnt!, args...)

"""
    vmapntt(f, a::AbstractArray)
    vmapntt(f, a::AbstractArray, b::AbstractArray, ...)

A threaded variant of [`vmapnt`](@ref).
"""
vmapntt(f::F, args::Vararg{Any,N}) where {F,N} = vmap_call(f, vmapntt!, args...)


# @inline vmap!(f, y, x...) = @avx y .= f.(x...)
# @inline vmap(f, x...) = @avx f.(x...)
