
"""
`vstorent!` (non-temporal store) requires data to be aligned.
`alignstores!` will align `y` in preparation for the non-temporal maps.
"""
function alignstores!(f::F, y::AbstractVector{T}, args::Vararg{<:Any,A}) where {F,T,A}
    N = length(y)
    ptry = pointer(y)
    ptrargs = pointer.(args)
    W = VectorizationBase.pick_vector_width(T)
    V = VectorizationBase.pick_vector_width_val(T)
    @assert iszero(reinterpret(UInt, ptry) & (sizeof(T) - 1)) "The destination vector (`dest`) must be aligned at least to `sizeof(eltype(dest))`."
    alignment = reinterpret(UInt, ptry) & (VectorizationBase.REGISTER_SIZE - 1)
    if alignment > 0
        i = reinterpret(Int, W - (alignment >>> VectorizationBase.intlog2(sizeof(T))))
        m = mask(T, i)
        if N < i
            m &= mask(T, N & (W - 1))
        end
        vnoaliasstore!(ptry, extract_data(f(vload.(V, ptrargs, m)...)), m)
        gep(ptry, i), gep.(ptrargs, i), N - i
    else
        ptry, ptrargs, N
    end
end

function vmap_singlethread!(f::F, y::AbstractVector{T}, ::Val{NonTemporal}, args::Vararg{<:Any,A}) where {F,T,A,NonTemporal}
    if NonTemporal
        ptry, ptrargs, N = alignstores!(f, y, args...)
    else
        N = length(y)
        ptry = pointer(y)
        ptrargs = pointer.(args)
    end
    i = 0
    W = VectorizationBase.pick_vector_width(T)
    V = VectorizationBase.pick_vector_width_val(T)
    while i < N - ((W << 2) - 1)
        v₁ = extract_data(f(vload.(V, gep.(ptrargs,      i     ))...))
        v₂ = extract_data(f(vload.(V, gep.(ptrargs, vadd(i,  W)))...))
        v₃ = extract_data(f(vload.(V, gep.(ptrargs, vadd(i, 2W)))...))
        v₄ = extract_data(f(vload.(V, gep.(ptrargs, vadd(i, 3W)))...))
        if NonTemporal
            vstorent!(gep(ptry,      i     ), v₁)
            vstorent!(gep(ptry, vadd(i,  W)), v₂)
            vstorent!(gep(ptry, vadd(i, 2W)), v₃)
            vstorent!(gep(ptry, vadd(i, 3W)), v₄)
        else
            vnoaliasstore!(gep(ptry,      i     ), v₁)
            vnoaliasstore!(gep(ptry, vadd(i,  W)), v₂)
            vnoaliasstore!(gep(ptry, vadd(i, 2W)), v₃)
            vnoaliasstore!(gep(ptry, vadd(i, 3W)), v₄)
        end
        i = vadd(i, 4W)
    end
    while i < N - (W - 1) # stops at 16 when
        vᵢ = extract_data(f(vload.(V, gep.(ptrargs, i))...))
        if NonTemporal
            vstorent!(gep(ptry, i), vᵢ)
        else
            vnoaliasstore!(gep(ptry, i), vᵢ)
        end
        i = vadd(i, W)
    end
    if i < N
        m = mask(T, N & (W - 1))
        vnoaliasstore!(gep(ptry, i), extract_data(f(vload.(V, gep.(ptrargs, i), m)...)), m)
    end
    y
end

function vmap_multithreaded!(f::F, y::AbstractVector{T}, ::Val{NonTemporal}, args::Vararg{<:Any,A}) where {F,T,A,NonTemporal}
    if NonTemporal
        ptry, ptrargs, N = alignstores!(f, y, args...)
    else
        N = length(y)
        ptry = pointer(y)
        ptrargs = pointer.(args)
    end
    N > 0 || return y
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    V = VectorizationBase.pick_vector_width_val(T)
    Wsh = Wshift + 2
    Niter = N >>> Wsh
    Base.Threads.@threads for j ∈ 0:Niter-1
        i = j << Wsh
        v₁ = extract_data(f(vload.(V, gep.(ptrargs,      i     ))...))
        v₂ = extract_data(f(vload.(V, gep.(ptrargs, vadd(i,  W)))...))
        v₃ = extract_data(f(vload.(V, gep.(ptrargs, vadd(i, 2W)))...))
        v₄ = extract_data(f(vload.(V, gep.(ptrargs, vadd(i, 3W)))...))
        if NonTemporal
            vstorent!(gep(ptry,      i     ), v₁)
            vstorent!(gep(ptry, vadd(i,  W)), v₂)
            vstorent!(gep(ptry, vadd(i, 2W)), v₃)
            vstorent!(gep(ptry, vadd(i, 3W)), v₄)
        else
            vnoaliasstore!(gep(ptry,      i     ), v₁)
            vnoaliasstore!(gep(ptry, vadd(i,  W)), v₂)
            vnoaliasstore!(gep(ptry, vadd(i, 2W)), v₃)
            vnoaliasstore!(gep(ptry, vadd(i, 3W)), v₄)
        end
    end
    ii = Niter << Wsh
    while ii < N - (W - 1) # stops at 16 when
        vᵢ = extract_data(f(vload.(V, gep.(ptrargs, ii))...))
        if NonTemporal
            vstorent!(gep(ptry, ii), vᵢ)
        else
            vnoaliasstore!(gep(ptry, ii), vᵢ)
        end
        ii = vadd(ii, W)
    end
    if ii < N
        m = mask(T, N & (W - 1))
        vnoaliasstore!(gep(ptry, ii), extract_data(f(vload.(V, gep.(ptrargs, ii), m)...)), m)
    end
    y
end


"""
    vmap!(f, destination, a::AbstractArray)
    vmap!(f, destination, a::AbstractArray, b::AbstractArray, ...)

Vectorized-`map!`, applying `f` to each element of `a` (or paired elements of `a`, `b`, ...)
and storing the result in `destination`.
"""
function vmap!(f::F, y::AbstractVector{T}, args::Vararg{<:Any,A}) where {F,T,A}
    vmap_singlethread!(f, y, Val{false}(), args...)
end


"""
    vmapt!(::Function, dest, args...)

Like `vmap!` (see `vmap!`), but uses `Threads.@threads` for parallel execution.
"""
function vmapt!(f::F, y::AbstractVector{T}, args::Vararg{<:Any,A}) where {F,T,A}
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
function vmapnt!(f::F, y::AbstractVector{T}, args::Vararg{<:Any,A}) where {F,T,A}
    vmap_singlethread!(f, y, Val{true}(), args...)
end

"""
    vmapntt!(::Function, dest, args...)

Like `vmapnt!` (see `vmapnt!`), but uses `Threads.@threads` for parallel execution.
"""
function vmapntt!(f::F, y::AbstractVector{T}, args::Vararg{<:Any,A}) where {F,T,A}
    vmap_multithreaded!(f, y, Val{true}(), args...)
end

function vmap_call(f::F, vm!::V, args::Vararg{<:Any,N}) where {V,F,N}
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
vmap(f::F, args::Vararg{<:Any,N}) where {F,N} = vmap_call(f, vmap!, args...)

"""
    vmapt(f, a::AbstractArray)
    vmapt(f, a::AbstractArray, b::AbstractArray, ...)

A threaded variant of [`vmap`](@ref).
"""
vmapt(f::F, args::Vararg{<:Any,N}) where {F,N} = vmap_call(f, vmapt!, args...)

"""
    vmapnt(f, a::AbstractArray)
    vmapnt(f, a::AbstractArray, b::AbstractArray, ...)

A "non-temporal" variant of [`vmap`](@ref). This can improve performance in cases where
`destination` will not be needed soon.
"""
vmapnt(f::F, args::Vararg{<:Any,N}) where {F,N} = vmap_call(f, vmapnt!, args...)

"""
    vmapntt(f, a::AbstractArray)
    vmapntt(f, a::AbstractArray, b::AbstractArray, ...)

A threaded variant of [`vmapnt`](@ref).
"""
vmapntt(f::F, args::Vararg{<:Any,N}) where {F,N} = vmap_call(f, vmapntt!, args...)


# @inline vmap!(f, y, x...) = @avx y .= f.(x...)
# @inline vmap(f, x...) = @avx f.(x...)
