

function vmap_quote(N, ::Type{T}) where {T}
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    val = Expr(:call, Expr(:curly, :Val, W))
    q = Expr(:block, Expr(:(=), :M, Expr(:call, :length, :dest)), Expr(:(=), :vdest, Expr(:call, :pointer, :dest)), Expr(:(=), :m, 0))
    fcall = Expr(:call, :f)
    loopbody = Expr(:block, Expr(:call, :vstore!, :vdest, fcall, :m), Expr(:(+=), :m, W))
    fcallmask = Expr(:call, :f)
    bodymask = Expr(:block, Expr(:(=), :__mask__, Expr(:call, :mask, val, Expr(:call, :&, :M, W-1))), Expr(:call, :vstore!, :vdest, fcallmask, :m, :__mask__))
    for n ∈ 1:N
        arg_n = Symbol(:varg_,n)
        push!(q.args, Expr(:(=), arg_n, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__,Symbol(@__FILE__)), Expr(:call, :pointer, Expr(:ref, :args, n)))))
        push!(fcall.args, Expr(:call, :vload, val, arg_n, :m))
        push!(fcallmask.args, Expr(:call, :vload, val, arg_n, :m, :__mask__))
    end
    loop = Expr(:for, Expr(:(=), :_, Expr(:call, :(:), 0, Expr(:call, :-, Expr(:call, :(>>>), :M, Wshift), 1))), loopbody)
    push!(q.args, loop)
    ifmask = Expr(:if, Expr(:call, :(!=), :m, :M), bodymask)
    push!(q.args, ifmask)
    push!(q.args, :dest)
    q
end
@generated function vmap!(f::F, dest::AbstractArray{T}, args::Vararg{<:AbstractArray,N}) where {F,T,N}
    vmap_quote(N, T)
end

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
        vstore!(ptry, extract_data(f(vload.(V, ptrargs, m)...)), m)
        gep(ptry, i), gep.(ptrargs, i), N - i
    else
        ptry, ptrargs, N
    end
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
    ptry, ptrargs, N = alignstores!(f, y, args...)
    i = 0
    W = VectorizationBase.pick_vector_width(T)
    V = VectorizationBase.pick_vector_width_val(T)
    while i < N - ((W << 2) - 1)
        vstorent!(ptry, extract_data(f(vload.(V, ptrargs, i)...)), i); i += W
        vstorent!(ptry, extract_data(f(vload.(V, ptrargs, i)...)), i); i += W
        vstorent!(ptry, extract_data(f(vload.(V, ptrargs, i)...)), i); i += W
        vstorent!(ptry, extract_data(f(vload.(V, ptrargs, i)...)), i); i += W
    end
    while i < N - (W - 1) # stops at 16 when 
        vstorent!(ptry, extract_data(f(vload.(V, ptrargs, i)...)), i); i += W
    end
    if i < N
        m = mask(T, N & (W - 1))
        vstore!(ptry, extract_data(f(vload.(V, ptrargs, i, m)...)), i, m)
    end
    y
end

"""
    vmapntt!(::Function, dest, args...)

Like `vmapnt!` (see `vmapnt!`), but but uses `Threads.@threads` for parallel execution.
"""
function vmapntt!(f::F, y::AbstractVector{T}, args::Vararg{<:Any,A}) where {F,T,A}
    ptry, ptrargs, N = alignstores!(f, y, args...)
    N > 0 || return y
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    V = VectorizationBase.pick_vector_width_val(T)
    Wsh = Wshift + 2
    Niter = N >>> Wsh
    Base.Threads.@threads for j ∈ 0:Niter-1
        i = j << Wsh
        vstorent!(ptry, extract_data(f(vload.(V, ptrargs, i)...)), i); i += W
        vstorent!(ptry, extract_data(f(vload.(V, ptrargs, i)...)), i); i += W
        vstorent!(ptry, extract_data(f(vload.(V, ptrargs, i)...)), i); i += W
        vstorent!(ptry, extract_data(f(vload.(V, ptrargs, i)...)), i)
    end
    ii = Niter << Wsh
    while ii < N - (W - 1) # stops at 16 when 
        vstorent!(ptry, extract_data(f(vload.(V, ptrargs, ii)...)), ii); ii += W
    end
    if ii < N
        m = mask(T, N & (W - 1))
        vstore!(ptry, extract_data(f(vload.(V, ptrargs, ii, m)...)), ii, m)
    end
    y
end

function vmap_call(f::F, vm!::V, args::Vararg{<:Any,N}) where {V,F,N}
    T = Base._return_type(f, Base.Broadcast.eltypes(args))
    dest = similar(first(args), T)
    vm!(f, dest, args...)
end
vmap(f::F, args::Vararg{<:Any,N}) where {F,N} = vmap_call(f, vmap!, args...)
vmapnt(f::F, args::Vararg{<:Any,N}) where {F,N} = vmap_call(f, vmapnt!, args...)
vmapntt(f::F, args::Vararg{<:Any,N}) where {F,N} = vmap_call(f, vmapntt!, args...)


# @inline vmap!(f, y, x...) = @avx y .= f.(x...)
# @inline vmap(f, x...) = @avx f.(x...)


