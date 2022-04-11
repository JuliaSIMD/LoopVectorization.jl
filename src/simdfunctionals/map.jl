"""
`vstorent!` (non-temporal store) requires data to be aligned.
`alignstores!` will align `y` in preparation for the non-temporal maps.
"""
function setup_vmap!(
  f::F,
  y::AbstractArray{T},
  ::Val{true},
  args::Vararg{AbstractArray,A},
) where {F,T<:Base.HWReal,A}
  N = length(y)
  ptry = VectorizationBase.zstridedpointer(y)
  ptrargs = map(VectorizationBase.zstridedpointer, args)
  V = pick_vector_width(T)
  W = unwrap(V)
  zero_index = (MM{W}(Static(0)),)
  uintptry = reinterpret(UInt, pointer(ptry))
  @assert iszero(uintptry & (sizeof(T) - 1)) "The destination vector (`dest`) must be aligned to `sizeof(eltype(dest)) == $(sizeof(T))` bytes."
  alignment = uintptry & (register_size() - 1)
  if alignment > 0
    i = reinterpret(Int, W - (alignment >>> VectorizationBase.intlog2(sizeof(T))))
    m = mask(T, i)
    if N < i
      m &= mask(T, N & (W - 1))
    end
    _vstore!(
      ptry,
      f(map1(vload, ptrargs, zero_index, m)...),
      zero_index,
      m,
      False(),
      True(),
      False(),
      register_size(),
    )
    gesp(ptry, (i,)), map1(gesp, ptrargs, (i,)), N - i
  else
    ptry, ptrargs, N
  end
end
function map1_quote(K::Int, args::Int)
  t = Expr(:tuple)
  gf = GlobalRef(Core, :getfield)
  for k ∈ 1:K
    c = Expr(:call, :f, Expr(:call, gf, :x_1, k, false))
    for a ∈ 2:args
      push!(c.args, Symbol(:x_, a))
    end
    push!(t.args, c)
  end
  Expr(:block, Expr(:meta, :inline), t)
end
@generated map1(f::F, x_1::Tuple{Vararg{Any,K}}, x_2) where {F,K} = map1_quote(K, 2)
@generated map1(f::F, x_1::Tuple{Vararg{Any,K}}, x_2, x_3) where {F,K} = map1_quote(K, 3)

@inline function setup_vmap!(f, y, ::Val{false}, args::Vararg{AbstractArray,A}) where {A}
  N = length(y)
  ptry = VectorizationBase.zstridedpointer(y)
  ptrargs = map(VectorizationBase.zstridedpointer, args)
  ptry, ptrargs, N
end

function vmap_singlethread!(
  f::F,
  y::AbstractArray{T},
  ::Val{NonTemporal},
  args::Vararg{AbstractArray,A},
) where {F,T<:NativeTypes,A,NonTemporal}
  ptry, ptrargs, N = setup_vmap!(f, y, Val{NonTemporal}(), args...)
  _vmap_singlethread!(f, ptry, Zero(), N, Val{NonTemporal}(), ptrargs)
  nothing
end
function _vmap_singlethread!(
  f::F,
  ptry::AbstractStridedPointer{T},
  start,
  N,
  ::Val{NonTemporal},
  ptrargs::Tuple{Vararg{Any,A}},
) where {F,T,NonTemporal,A}
  i = convert(Int, start)
  V = VectorizationBase.pick_vector_width(
    promote_type(T, reduce(promote_type, map(eltype, ptrargs))),
  )
  W = unwrap(V)
  UNROLL = 4
  LOG2UNROLL = 2
  while i < vsub_nsw(N, ((W << LOG2UNROLL) - 1))
    index = VectorizationBase.Unroll{1,W,UNROLL,1,W,zero(UInt)}((i,))
    v = f(VectorizationBase.fmap(vload, ptrargs, index)...)
    if NonTemporal
      _vstore!(ptry, v, index, True(), True(), True(), register_size())
    else
      _vstore!(ptry, v, index, False(), True(), False(), register_size())
    end
    i = vadd_nw(i, StaticInt{UNROLL}() * W)
  end
  # if Base.libllvm_version ≥ v"11" # this seems to be slower
  #     Nm1 = vsub_nw(N, 1)
  #     while i < N # stops at 16 when
  #         m = mask(V, i, Nm1)
  #         vnoaliasstore!(ptry, f(vload.(ptrargs, ((MM{W}(i),),), m)...), (MM{W}(i,),), m)
  #         i = vadd_nw(i, W)
  #     end
  # else
  while i < vsub_nsw(N, (W - 1)) # stops at 16 when
    vᵣ = f(map1(vload, ptrargs, (MM{W}(i),))...)
    if NonTemporal
      _vstore!(ptry, vᵣ, (MM{W}(i),), True(), True(), True(), register_size())
    else
      _vstore!(ptry, vᵣ, (MM{W}(i),), False(), True(), False(), register_size())
    end
    i = vadd_nw(i, W)
  end
  if i < N
    m = mask(StaticInt(W), N & (W - 1))
    vfinal = f(map1(vload, ptrargs, (MM{W}(i),), m)...)
    if NonTemporal
      _vstore!(ptry, vfinal, (MM{W}(i),), m, True(), True(), False(), register_size())
    else
      _vstore!(ptry, vfinal, (MM{W}(i),), m, False(), True(), False(), register_size())
    end
  end
  # end
  nothing
end

abstract type AbstractVmapClosure{NonTemporal,F,D,N,A<:Tuple{Vararg{Any,N}}} <: Function end
struct VmapClosure{NonTemporal,F,D,N,A} <: AbstractVmapClosure{NonTemporal,F,D,N,A}
  f::F
  function VmapClosure{NonTemporal}(
    f::F,
    ::D,
    ::A,
  ) where {NonTemporal,F,D,N,A<:Tuple{Vararg{Any,N}}}
    new{NonTemporal,F,D,N,A}(f)
  end
end
# struct VmapKnownClosure{NonTemporal,F,D,N,A} <: AbstractVmapClosure{NonTemporal,F,D,N,A} end

# @generated function (::VmapKnownClosure{NonTemporal,F,D,N,A})(p::Ptr{UInt})  where {NonTemporal,F,D,N,A}
#     :(_vmap_thread_call!($(F.instance), p, $D, $A, Val{$NonTemporal}()))
# end
function (m::VmapClosure{NonTemporal,F,D,N,A})(p::Ptr{UInt}) where {NonTemporal,F,D,N,A}
  (offset, dest) = ThreadingUtilities.load(p, D, 2 * sizeof(UInt))
  (offset, args) = ThreadingUtilities.load(p, A, offset)

  (offset, start) = ThreadingUtilities.load(p, Int, offset)
  (offset, stop) = ThreadingUtilities.load(p, Int, offset)

  _vmap_singlethread!(m.f, dest, start, stop, Val{NonTemporal}(), args)
  ThreadingUtilities._atomic_store!(p, ThreadingUtilities.SPIN)
  NonTemporal && Threads.atomic_fence()
  nothing
end

@inline function _get_fptr(cfunc::Base.CFunction)
  Base.unsafe_convert(Ptr{Cvoid}, cfunc)
end
# @generated function _get_fptr(cfunc::F) where {F<:VmapKnownClosure}
#     precompile(F(), (Ptr{UInt},))
#     quote
#         $(Expr(:meta,:inline))
#         @cfunction($(F()), Cvoid, (Ptr{UInt},))
#     end
# end

@inline function setup_thread_vmap!(p, cfunc, ptry, ptrargs, start, stop)
  fptr = _get_fptr(cfunc)
  offset = ThreadingUtilities.store!(p, fptr, sizeof(UInt))
  offset = ThreadingUtilities.store!(p, ptry, offset)
  offset = ThreadingUtilities.store!(p, ptrargs, offset)
  offset = ThreadingUtilities.store!(p, start, offset)
  offset = ThreadingUtilities.store!(p, stop, offset)
  nothing
end
@inline function launch_thread_vmap!(tid, cfunc, ptry, ptrargs, start, stop)
  ThreadingUtilities.launch(
    tid,
    cfunc,
    ptry,
    ptrargs,
    start,
    stop,
  ) do p, cfunc, ptry, ptrargs, start, stop
    setup_thread_vmap!(p, cfunc, ptry, ptrargs, start, stop)
  end
end

@inline function vmap_closure(
  f::F,
  ptry::D,
  ptrargs::A,
  ::Val{NonTemporal},
) where {F,D<:StridedPointer,N,A<:Tuple{Vararg{Any,N}},NonTemporal}
  vmc = VmapClosure{NonTemporal}(f, ptry, ptrargs)
  @cfunction($vmc, Cvoid, (Ptr{UInt},))
end

function vmap_multithread!(
  f::F,
  y::AbstractArray{T},
  ::Val{NonTemporal},
  args::Vararg{AbstractArray,A},
) where {F,T,A,NonTemporal}
  W, Wshift = VectorizationBase.pick_vector_width_shift(T)
  ptry, ptrargs, N = setup_vmap!(f, y, Val{NonTemporal}(), args...)
  # nt = min(Threads.nthreads(), VectorizationBase.SYS_CPU_THREADS, N >> (Wshift + 3))
  nt = min(Threads.nthreads(), num_cores(), N >> (Wshift + 5))

  # if !((nt > 1) && iszero(ccall(:jl_in_threaded_region, Cint, ())))
  if nt < 2
    _vmap_singlethread!(f, ptry, Zero(), N, Val{NonTemporal}(), ptrargs)
    return
  end

  cfunc = vmap_closure(f, ptry, ptrargs, Val{NonTemporal}())
  Nveciter = (N + (W - 1)) >> Wshift
  Nd, Nr = divrem(Nveciter, nt)
  NdW = Nd << Wshift
  NdWr = NdW + W
  GC.@preserve cfunc begin
    start = 0
    for tid ∈ 1:nt-1
      stop = start + ifelse(tid ≤ Nr, NdWr, NdW)
      launch_thread_vmap!(tid, cfunc, ptry, ptrargs, start, stop)
      start = stop
    end
    _vmap_singlethread!(f, ptry, start, N, Val{NonTemporal}(), ptrargs)
    for tid ∈ 1:nt-1
      ThreadingUtilities.wait(tid)
    end
  end
  nothing
end
@generated function gc_preserve_vmap!(
  f::F,
  y::AbstractArray,
  ::Val{NonTemporal},
  ::Val{Threaded},
  args::Vararg{AbstractArray,A},
) where {F,A,NonTemporal,Threaded}
  m = Threaded ? :vmap_multithread! : :vmap_singlethread!
  call = Expr(:call, m, :f, :y, Expr(:call, Expr(:curly, :Val, NonTemporal)))
  q = Expr(:block, Expr(:meta, :inline))
  gcpres = Expr(:gc_preserve, call)
  for a ∈ 1:A
    arg = Symbol(:arg_, a)
    parg = Symbol(:parg_, a)
    push!(q.args, Expr(:(=), arg, :(@inbounds args[$a])))#Expr(:ref, :args, a)))
    push!(q.args, Expr(:(=), parg, Expr(:call, :preserve_buffer, arg)))
    push!(call.args, arg)
    push!(gcpres.args, parg)
  end
  push!(q.args, gcpres, :y)
  q
end


@inline _all_dense(t::Tuple{ArrayInterface.True}) = true
@inline _all_dense(t::Tuple{ArrayInterface.True,ArrayInterface.True,Vararg}) =
  _all_dense(Base.tail(t))
@inline _all_dense(t::Tuple{ArrayInterface.True,ArrayInterface.False,Vararg}) = false
@inline _all_dense(t::Tuple{ArrayInterface.False,Vararg}) = false
@inline all_dense() = true
@inline all_dense(A::Array) = true
@inline all_dense(A::AbstractArray) = _all_dense(ArrayInterface.dense_dims(A))
@inline all_dense(
  A::AbstractArray,
  B::AbstractArray,
  C::Vararg{AbstractArray,K},
) where {K} = all_dense(A) && all_dense(B, C...)

"""
    vmap!(f, destination, a::AbstractArray)
    vmap!(f, destination, a::AbstractArray, b::AbstractArray, ...)
Vectorized-`map!`, applying `f` to batches of elements of `a` (or paired batches of `a`, `b`, ...)
and storing the result in `destination`.

The function `f` must accept `VectorizationBase.AbstractSIMD` inputs. Ideally, all this requires
is making sure that `f` is defined to be agnostic with respect to input types, but if the function `f`
contains branches or loops, more work will probably be needed. For example, a function
```julia
f(x) = x > 0 ? log(x) : inv(x)
```
can be rewritten into
```julia
using IfElse
f(x) = IfElse.ifelse(x > 0, log(x), inv(x))
```
"""
function vmap!(
  f::F,
  y::AbstractArray,
  arg1::AbstractArray,
  arg2::AbstractArray,
  args::Vararg{AbstractArray,A},
) where {F,A}
  if check_args(y, arg1, arg2, args...) && all_dense(y, arg1, arg2, args...)
    gc_preserve_vmap!(f, y, Val{false}(), Val{false}(), arg1, arg2, args...)
  else
    map!(f, y, arg1, arg2, args...)
  end
end
function vmap!(f::F, y::AbstractArray, arg::AbstractArray) where {F}
  if check_args(y, arg) && all_dense(y, arg)
    gc_preserve_vmap!(f, y, Val{false}(), Val{false}(), arg)
  else
    map!(f, y, arg)
  end
end


"""
    vmapt!(::Function, dest, args...)
A threaded variant of [`vmap!`](@ref).
"""
function vmapt!(f::F, y::AbstractArray, args::Vararg{AbstractArray,A}) where {F,A}
  if check_args(y, args...) && all_dense(y, args...)
    gc_preserve_vmap!(f, y, Val{false}(), Val{true}(), args...)
  else
    map!(f, y, args...)
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
function vmapnt!(f::F, y::AbstractArray, args::Vararg{AbstractArray,A}) where {F,A}
  if check_args(y, args...) && all_dense(y, args...)
    gc_preserve_vmap!(f, y, Val{true}(), Val{false}(), args...)
  else
    map!(f, y, args...)
  end
end

"""
    vmapntt!(::Function, dest, args...)
A threaded variant of [`vmapnt!`](@ref).
"""
function vmapntt!(f::F, y::AbstractArray, args::Vararg{AbstractArray,A}) where {F,A}
  if check_args(y, args...) && all_dense(y, args...)
    gc_preserve_vmap!(f, y, Val{true}(), Val{true}(), args...)
  else
    map!(f, y, args...)
  end
end

# generic fallbacks
@inline vmap!(f, args...) = map!(f, args...)
@inline vmapt!(f, args...) = map!(f, args...)
@inline vmapnt!(f, args...) = map!(f, args...)
@inline vmapntt!(f, args...) = map!(f, args...)

# similar_bit(x, ::Type{T}) where {T} = similar(x, T)
# similar_bit(x, ::Type{Bool}) = BitArray(undef, size(x))

function vmap_call(f::F, vm!::V, args::Vararg{Any,N}) where {V,F,N}
  T = Base._return_type(f, Base.Broadcast.eltypes(args))
  dest = similar(first(args), T)
  # dest = similar_bit(first(args), T)
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
