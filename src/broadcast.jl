@generated function append_true(::Val{D}, ::Val{N}) where {D,N}
  length(D) == N && return D
  t = Expr(:tuple)
  for d in D
    push!(t.args, d)
  end
  for n = length(D)+1:N
    push!(t.args, true)
  end
  t
end
struct LowDimArray{D,T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
  data::A
  function LowDimArray{D}(data::A) where {D,T,N,A<:AbstractArray{T,N}}
    new{append_true(Val{D}(), Val{N}()),T,N,A}(data)
  end
  function LowDimArray{D,T,N,A}(data::A) where {D,T,N,A<:AbstractArray{T,N}}
    new{append_true(Val{D}(), Val{N}()),T,N,A}(data)
  end
end
function LowDimArray{D0}(
  data::LowDimArray{D1,T,N,A}
) where {D0,T,N,D1,A<:AbstractArray{T,N}}
  LowDimArray{map(|, D0, D1),T,N,A}(parent(data))
end
Base.@propagate_inbounds Base.getindex(
  A::LowDimArray,
  i::Vararg{Union{StaticInt,Integer,CartesianIndex},K}
) where {K} = getindex(A.data, i...)
@inline Base.size(A::LowDimArray) = Base.size(A.data)
@inline Base.size(A::LowDimArray, i) = Base.size(A.data, i)

@inline _pick_lowdim_known(::Tuple{}, x) = x
@inline function _pick_lowdim_known(b::Tuple{Bool,Vararg}, x)
  f = first(b) ? first(x) : 1
  l = _pick_lowdim_known(Base.tail(b), Base.tail(x))
  (f, l...)
end
@inline function ArrayInterface.known_size(
  ::Type{LowDimArray{D,T,N,A}}
) where {D,T,N,A}
  _pick_lowdim_known(D, ArrayInterface.known_size(A))
end
@inline ArrayInterface.parent_type(
  ::Type{LowDimArray{D,T,N,A}}
) where {T,D,N,A} = A
@inline Base.strides(A::LowDimArray) = map(Int, static_strides(A))
@inline ArrayInterface.device(::LowDimArray) = ArrayInterface.CPUPointer()
@generated function ArrayInterface.static_size(
  A::LowDimArray{D,T,N}
) where {D,T,N}
  t = Expr(:tuple)
  for n ∈ 1:N
    if n > length(D) || D[n]
      push!(t.args, Expr(:call, getfield, :s, n))
    else
      push!(t.args, Expr(:call, Expr(:curly, lv(:StaticInt), 1)))
    end
  end
  Expr(:block, Expr(:meta, :inline), :(s = ArrayInterface.size(parent(A))), t)
end
Base.parent(A::LowDimArray) = getfield(A, :data)
Base.unsafe_convert(::Type{Ptr{T}}, A::LowDimArray{D,T}) where {D,T} =
  pointer(parent(A))
ArrayInterface.contiguous_axis(A::LowDimArray) =
  ArrayInterface.contiguous_axis(parent(A))
ArrayInterface.contiguous_batch_size(A::LowDimArray) =
  ArrayInterface.contiguous_batch_size(parent(A))
ArrayInterface.stride_rank(A::LowDimArray) =
  ArrayInterface.stride_rank(parent(A))
ArrayInterface.offsets(A::LowDimArray) = ArrayInterface.offsets(parent(A))

@generated function _lowdimfilter(
  ::Val{D},
  tup::Tuple{Vararg{Any,N}}
) where {D,N}
  t = Expr(:tuple)
  for n ∈ 1:N
    if n > length(D) || D[n]
      push!(t.args, Expr(:call, getfield, :tup, n))
    end
  end
  Expr(:block, Expr(:meta, :inline), t)
end

struct ForBroadcast{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
  data::A
end
@inline Base.parent(fb::ForBroadcast) = getfield(fb, :data)
@inline ArrayInterface.parent_type(::Type{ForBroadcast{T,N,A}}) where {T,N,A} =
  A
Base.@propagate_inbounds Base.getindex(
  A::ForBroadcast,
  i::Vararg{Any,K}
) where {K} = parent(A)[i...]
const LowDimArrayForBroadcast{D,T,N,A} = ForBroadcast{T,N,LowDimArray{D,T,N,A}}
@inline function VectorizationBase.contiguous_axis(
  fb::LowDimArrayForBroadcast{D,T,N,A}
) where {D,T,N,A}
  _contiguous_axis(
    Val{D}(),
    VectorizationBase.contiguous_axis(parent(parent(fb)))
  )
end
@inline forbroadcast(A::AbstractArray) = ForBroadcast(A)
@inline forbroadcast(A::AbstractRange) = A
@inline forbroadcast(A) = A
# @inline forbroadcast(A::Adjoint) = forbroadcast(parent(A))
# @inline forbroadcast(A::Transpose) = forbroadcast(parent(A))
@inline function ArrayInterface.static_strides(
  A::Union{LowDimArray,ForBroadcast}
)
  B = parent(A)
  _strides(
    static_size(A),
    static_strides(B),
    VectorizationBase.val_stride_rank(B),
    VectorizationBase.val_dense_dims(B)
  )
end

# @inline function VectorizationBase.contiguous_batch_size(fb::LowDimArrayForBroadcast{D,T,N,A}) where {D,T,N,A}
#     _contiguous_axis(Val{D}(), VectorizationBase.contiguous_batch_size(parent(parent(fb))))
# end
@generated function _contiguous_axis(::Val{D}, ::StaticInt{C}) where {D,C}
  Dlen = length(D)
  (C > 0) || return Expr(:block, Expr(:meta, :inline), staticexpr(C))
  if C ≤ Dlen
    D[C] || return Expr(:block, Expr(:meta, :inline), staticexpr(-1))
  end
  Cnew = 0
  for n ∈ 1:C
    Cnew += ((n > Dlen)) || D[n]
  end
  Expr(:block, Expr(:meta, :inline), staticexpr(Cnew))
end
function ArrayInterface.contiguous_axis(
  ::Type{LowDimArrayForBroadcast{D,T,N,A}}
) where {D,T,N,A}
  ArrayInterface.contiguous_axis(A)
end
@inline function ArrayInterface.stride_rank(
  ::Type{LowDimArrayForBroadcast{D,T,N,A}}
) where {D,T,N,A}
  _lowdimfilter(Val(D), ArrayInterface.stride_rank(A))
end
@inline function ArrayInterface.dense_dims(
  ::Type{LowDimArrayForBroadcast{D,T,N,A}}
) where {D,T,N,A}
  _lowdimfilter(Val(D), ArrayInterface.dense_dims(A))
end
@inline function ArrayInterface.static_strides(
  fb::LowDimArrayForBroadcast{D}
) where {D}
  _lowdimfilter(Val(D), static_strides(parent(fb)))
end
@inline function ArrayInterface.offsets(
  fb::LowDimArrayForBroadcast{D}
) where {D}
  _lowdimfilter(Val(D), ArrayInterface.offsets(parent(parent(fb))))
end
@inline function ArrayInterface.StrideIndex(
  a::A
) where {A<:LowDimArrayForBroadcast}
  _stride_index(
    ArrayInterface.stride_rank(A),
    ArrayInterface.contiguous_axis(A),
    a
  )
end
@inline function _stride_index(
  r::Tuple{Vararg{StaticInt,N}},
  ::StaticInt{C},
  A
) where {N,C}
  StrideIndex{N,ArrayInterface.known(r),C}(A)
end

for f ∈ [ # groupedstridedpointer support
  :(ArrayInterface.contiguous_axis),
  :(ArrayInterface.contiguous_batch_size),
  :(ArrayInterface.device),
  :(ArrayInterface.stride_rank)
]
  @eval @inline $f(::Type{ForBroadcast{T,N,A}}) where {T,N,A} = $f(A)
end
for f ∈ [ # groupedstridedpointer support
  :(LayoutPointers.memory_reference),
  :(ArrayInterface.contiguous_axis),
  :(ArrayInterface.contiguous_batch_size),
  :(ArrayInterface.device),
  :(ArrayInterface.stride_rank),
  :(VectorizationBase.val_dense_dims),
  :(ArrayInterface.offsets),
  :(Base.size)#, :(ArrayInterface.strides)
]
  @eval @inline $f(fb::ForBroadcast) = $f(getfield(fb, :data))
end

function is_column_major(x)
  for (i, j) ∈ enumerate(x)
    i == j || return false
  end
  true
end
is_row_major(x) = is_column_major(reverse(x))
# @inline _bytestrides(s,paren) = VectorizationBase.bytestrides(paren)
function _strides_expr(
  @nospecialize(s),
  @nospecialize(x),
  R::Vector{Int},
  D::Vector{Bool}
)
  N = length(R)
  q = Expr(:block, Expr(:meta, :inline))
  strd_tup = Expr(:tuple)
  ifel = GlobalRef(Core, :ifelse)
  Nrange = 1:1:N # type stability w/ respect to reverse
  use_stride_acc = true
  stride_acc::Int = 1
  if is_column_major(R)
    # elseif is_row_major(R)
    #   Nrange = reverse(Nrange)
  else # not worth my time optimizing this case at the moment...
    # will write something generic stride-rank agnostic eventually
    use_stride_acc = false
    stride_acc = 0
  end
  sₙ_value::Int = 0
  for n ∈ Nrange
    xₙ_type = x[n]
    xₙ_static = xₙ_type <: StaticInt
    xₙ_value::Int = xₙ_static ? (xₙ_type.parameters[1])::Int : 0
    s_type = s[n]
    sₙ_static = s_type <: StaticInt
    if sₙ_static
      sₙ_value = s_type.parameters[1]
      if s_type === One
        push!(strd_tup.args, Expr(:call, lv(:Zero)))
      elseif stride_acc ≠ 0
        push!(strd_tup.args, staticexpr(stride_acc))
      else
        push!(strd_tup.args, :($getfield(x, $n)))
      end
    else
      if xₙ_static
        push!(strd_tup.args, staticexpr(xₙ_value))
      elseif stride_acc ≠ 0
        push!(strd_tup.args, staticexpr(stride_acc))
      else
        push!(
          strd_tup.args,
          :($ifel(isone($getfield(s, $n)), zero($xₙ_type), $getfield(x, $n)))
        )
      end
    end
    if (n ≠ last(Nrange)) && use_stride_acc
      nnext = n + step(Nrange)
      if D[nnext]
        if xₙ_static & sₙ_static
          stride_acc = xₙ_value * sₙ_value
        elseif sₙ_static
          if stride_acc ≠ 0
            stride_acc *= sₙ_value
          end
        else
          stride_acc = 0
        end
      else
        stride_acc = 0
      end
    end
  end
  push!(q.args, strd_tup)
  q
end
@generated function _strides(
  s::Tuple{Vararg{Union{StaticInt,Integer},N}},
  x::Tuple{Vararg{Union{StaticInt,Integer},N}},
  ::Val{R},
  ::Val{D}
) where {N,R,D}
  Rv = Vector{Int}(undef, N)
  Dv = Vector{Bool}(undef, N)
  for n = 1:N
    Rv[n] = R[n]
    Dv[n] = D[n]
  end
  _strides_expr(s.parameters, x.parameters, Rv, Dv)
end

struct Product{A,B}
  a::A
  b::B
end
@inline function Base.size(p::Product)
  M = @inbounds size(p.a)[1]
  (M, Base.tail(size(p.b))...)
end
@inline function Base.size(p::Product, i::Integer)
  i == 1 && return @inbounds size(p.a)[1]
  @inbounds size(p.b)[i]
end
@inline Base.length(p::Product) = prod(size(p))
@inline Base.broadcastable(p::Product) = p
@inline numdims(A) = ndims(A) # fallback
@inline numdims(::Type{Product{A,B}}) where {A,B} = numdims(B)
@inline Base.ndims(::Type{Product{A,B}}) where {A,B} = numdims(B)
# This numdims nonsense is a hack to avoid type piracy in defining:
@inline numdims(
  ::Type{B}
) where {
  N,
  S<:Base.Broadcast.AbstractArrayStyle{N},
  B<:Base.Broadcast.Broadcasted{S}
} = N

Base.Broadcast._broadcast_getindex_eltype(
  ::Product{A,B}
) where {T,A<:AbstractVecOrMat{T},B<:AbstractVecOrMat{T}} = T
function Base.Broadcast._broadcast_getindex_eltype(p::Product)
  promote_type(
    Base.Broadcast._broadcast_getindex_eltype(p.a),
    Base.Broadcast._broadcast_getindex_eltype(p.b)
  )
end

_is_one(x) = x !== 1
@inline _dontbc(x) = map(_is_one, ArrayInterface.known_size(x))
@inline _dontbc(x::Product) = _dontbc(x.a), _dontbc(x.b)
@inline _dontbc(bc::Base.Broadcast.Broadcasted) = map(_dontbc, bc.args)

"""
    A *ˡ B

A lazy product of `A` and `B`. While functionally identical to `A * B`, this may avoid the
need for intermediate storage for any computations in `A` or `B`.  Example:

    @turbo @. a + B *ˡ (c + d')

which is equivalent to

     a .+ B * (c .+ d')

It should only be used inside an `@turbo` block, and to materialize the result it cannot be
the final operation.
"""
@inline *ˡ(a::A, b::B) where {A,B} = Product{A,B}(a, b)
@inline Base.Broadcast.broadcasted(::typeof(*ˡ), a::A, b::B) where {A,B} =
  Product{A,B}(a, b)
# TODO: Need to make this handle A or B being (1 or 2)-D broadcast objects.
function add_broadcast!(
  ls::LoopSet,
  mC::Symbol,
  bcname::Symbol,
  loopsyms::Vector{Symbol},
  @nospecialize(prod::Type{<:Product}),
  dontbc,
  elementbytes::Int
)
  A, B = prod.parameters
  Krange = gensym!(ls, "K")
  Klen = gensym!(ls, "K")
  mA = gensym!(ls, "Aₘₖ")
  mB = gensym!(ls, "Bₖₙ")
  pushprepreamble!(ls, Expr(:(=), mA, Expr(:(.), bcname, QuoteNode(:a))))
  pushprepreamble!(ls, Expr(:(=), mB, Expr(:(.), bcname, QuoteNode(:b))))
  pushprepreamble!(
    ls,
    Expr(:(=), Klen, Expr(:call, getfield, Expr(:call, :static_size, mB), 1))
  )
  pushpreamble!(ls, Expr(:(=), Krange, Expr(:call, :(:), staticexpr(1), Klen)))
  k = gensym!(ls, "k")
  add_loop!(ls, Loop(k, 1, Klen, 1, Krange, Klen), k)
  m = loopsyms[1]
  if numdims(B) == 1
    bloopsyms = Symbol[k]
    cloopsyms = Symbol[m]
    reductdeps = Symbol[m, k]
    kvec = bloopsyms
  elseif numdims(B) == 2
    n = loopsyms[2]
    bloopsyms = Symbol[k, n]
    cloopsyms = Symbol[m, n]
    reductdeps = Symbol[m, k, n]
    kvec = Symbol[k]
  else
    throw("B must be a vector or matrix.")
  end
  # load A
  # loadA = add_load!(ls, gensym!(ls, :A), productref(A, mA, m, k), elementbytes)
  loadA = add_broadcast!(
    ls,
    gensym!(ls, "A"),
    mA,
    Symbol[m, k],
    A,
    dontbc[1],
    elementbytes
  )
  # load B
  loadB = add_broadcast!(
    ls,
    gensym!(ls, "B"),
    mB,
    bloopsyms,
    B,
    dontbc[2],
    elementbytes
  )
  # set Cₘₙ = 0
  # setC = add_constant!(ls, zero(promote_type(recursive_eltype(A), recursive_eltype(B))), cloopsyms, mC, elementbytes)
  # targetC will be used for reduce_to_add
  mCt = gensym!(ls, mC)
  targetC = add_constant!(
    ls,
    gensym!(ls, "zero"),
    cloopsyms,
    mCt,
    elementbytes,
    :numericconstant
  )
  push!(ls.preamble_zeros, (identifier(targetC), IntOrFloat))
  setC = add_constant!(
    ls,
    gensym!(ls, "zero"),
    cloopsyms,
    mC,
    elementbytes,
    :numericconstant
  )
  push!(ls.preamble_zeros, (identifier(setC), IntOrFloat))
  setC.reduced_children = kvec
  # compute Cₘₙ += Aₘₖ * Bₖₙ
  instrsym = Base.libllvm_version < v"11.0.0" ? :vfmadd231 : :vfmadd
  reductop = Operation(
    ls,
    mC,
    elementbytes,
    instrsym,
    compute,
    reductdeps,
    kvec,
    Operation[loadA, loadB, setC]
  )
  reductop = pushop!(ls, reductop, mC)
  reductfinal = Operation(
    ls,
    mCt,
    elementbytes,
    :reduce_to_add,
    compute,
    cloopsyms,
    kvec,
    Operation[reductop, targetC]
  )
  pushop!(ls, reductfinal, mCt)
end

function extract_all_1_array!(
  ls::LoopSet,
  bcname::Symbol,
  N::Int,
  elementbytes::Int
)
  refextract = gensym!(ls, bcname)
  ref = Expr(:ref, bcname)
  for _ ∈ 1:N
    push!(ref.args, :begin)
  end
  pushprepreamble!(ls, Expr(:(=), refextract, :(@inbounds $ref)))
  return add_constant!(ls, refextract, elementbytes) # or replace elementbytes with sizeof(T) ? u
end
function doaddref!(ls::LoopSet, op::Operation)
  push!(ls.syms_aliasing_refs, name(op))
  push!(ls.refs_aliasing_syms, op.ref)
  op
end

function add_broadcast!(
  ls::LoopSet,
  destname::Symbol,
  bcname::Symbol,
  loopsyms::Vector{Symbol},
  @nospecialize(_::Type{<:AbstractArray{T,N}}),
  @nospecialize(dontbc::NTuple{N,Bool}),
  elementbytes::Int
) where {T,N}
  any(dontbc) || return extract_all_1_array!(ls, bcname, N, elementbytes)
  bcname2 = gensym!(ls, bcname)
  ref = if all(dontbc)
    pushprepreamble!(ls, Expr(:(=), bcname2, Expr(:call, forbroadcast, bcname)))
    ArrayReference(bcname2, loopsyms[1:N])
  else
    fulldims = Symbol[]
    subset = Expr(:tuple)
    for n = 1:N
      push!(subset.args, dontbc[n])
      dontbc[n] && push!(fulldims, loopsyms[n])
    end
    lda = Expr(:call, Expr(:curly, LowDimArray, subset), bcname)
    pushprepreamble!(ls, Expr(:(=), bcname2, Expr(:call, forbroadcast, lda)))
    ArrayReference(bcname2, fulldims)
  end

  loadop = add_simple_load!(ls, destname, ref, elementbytes, true)::Operation
  doaddref!(ls, loadop)
end
function add_broadcast!(
  ls::LoopSet,
  ::Symbol,
  bcname::Symbol,
  loopsyms::Vector{Symbol},
  @nospecialize(_::Type{T}),
  @nospecialize(__),
  elementbytes::Int
) where {T<:Number}
  add_constant!(ls, bcname, elementbytes) # or replace elementbytes with sizeof(T) ? u
end
function add_broadcast!(
  ls::LoopSet,
  ::Symbol,
  bcname::Symbol,
  loopsyms::Vector{Symbol},
  @nospecialize(_::Type{Base.RefValue{T}}),
  @nospecialize(__),
  elementbytes::Int
) where {T}
  refextract = gensym!(ls, bcname)
  pushprepreamble!(ls, Expr(:(=), refextract, Expr(:ref, bcname)))
  add_constant!(ls, refextract, elementbytes) # or replace elementbytes with sizeof(T) ? u
end
const BroadcastedArray{S<:Broadcast.AbstractArrayStyle,F,A} =
  Broadcasted{S,Nothing,F,A}
function add_broadcast!(
  ls::LoopSet,
  destname::Symbol,
  bcname::Symbol,
  loopsyms::Vector{Symbol},
  @nospecialize(B::Type{<:BroadcastedArray}),
  @nospecialize(dontbc),
  elementbytes::Int
)
  S, _, F, A = B.parameters
  instr = get(FUNCTIONSYMBOLS, F) do
    f = gensym!(ls, "func")
    pushprepreamble!(ls, Expr(:(=), f, Expr(:(.), bcname, QuoteNode(:f))))
    Instruction(bcname, f)
  end
  args = A.parameters
  bcargs = gensym!(ls, "bcargs")
  pushprepreamble!(ls, Expr(:(=), bcargs, Expr(:(.), bcname, QuoteNode(:args))))
  # this is the var name in the loop
  parents = Operation[]
  deps = Symbol[]
  # reduceddeps = Symbol[]
  for (i, arg) ∈ enumerate(args)
    argname = gensym!(ls, "arg")
    pushprepreamble!(ls, Expr(:(=), argname, Expr(:call, getfield, bcargs, i)))
    # dynamic dispatch
    parent = add_broadcast!(
      ls,
      gensym!(ls, "temp"),
      argname,
      loopsyms,
      arg,
      dontbc[i],
      elementbytes
    )::Operation
    push!(parents, parent)
    mergesetdiffv!(deps, loopdependencies(parent), reduceddependencies(parent))
  end
  op = Operation(
    length(operations(ls)),
    destname,
    elementbytes,
    instr,
    compute,
    deps,
    NODEPENDENCY,
    parents
  )
  pushop!(ls, op, destname)
end

function add_broadcast_loops!(
  ls::LoopSet,
  loopsyms::Vector{Symbol},
  destsym::Symbol
)
  axes_tuple = Expr(:tuple)
  pushpreamble!(ls, Expr(:(=), axes_tuple, Expr(:call, :static_axes, destsym)))
  for itersym ∈ loopsyms
    Nrange = gensym!(ls, "N")
    Nlower = gensym!(ls, "N")
    Nupper = gensym!(ls, "N")
    Nlen = gensym!(ls, "N")
    add_loop!(ls, Loop(itersym, Nlower, Nupper, 1, Nrange, Nlen), itersym)
    push!(axes_tuple.args, Nrange)
    pushpreamble!(
      ls,
      Expr(:(=), Nlower, Expr(:call, lv(:maybestaticfirst), Nrange))
    )
    pushpreamble!(
      ls,
      Expr(:(=), Nupper, Expr(:call, lv(:maybestaticlast), Nrange))
    )
    pushpreamble!(
      ls,
      Expr(
        :(=),
        Nlen,
        Expr(:call, GlobalRef(ArrayInterface, :static_length), Nrange)
      )
    )
  end
end

function vmaterialize_fun(
  sizeofT::Int,
  N,
  @nospecialize(_::Type{BC}),
  Mod,
  UNROLL,
  dontbc,
  transpose::Bool
) where {BC}
  # 2 + 1
  # we have an N dimensional loop.
  # need to construct the LoopSet
  ls = LoopSet(Mod)
  inline, u₁, u₂, v, isbroadcast, _, rs, rc, cls, threads, warncheckarg, safe =
    UNROLL
  set_hw!(ls, rs, rc, cls)
  ls.isbroadcast = isbroadcast # maybe set `false` in a DiffEq-like `@..` macro
  loopsyms = [gensym!(ls, "n") for _ ∈ 1:N]
  transpose &&
    pushprepreamble!(ls, Expr(:(=), :dest, Expr(:call, :parent, :dest′)))
  ret = transpose ? :dest′ : :dest
  add_broadcast_loops!(ls, loopsyms, ret)
  elementbytes = sizeofT
  add_broadcast!(ls, :destination, :bc, loopsyms, BC, dontbc, elementbytes)
  transpose && reverse!(loopsyms)
  storeop = add_simple_store!(
    ls,
    :destination,
    ArrayReference(:dest, loopsyms),
    elementbytes
  )
  doaddref!(ls, storeop)
  resize!(ls.loop_order, num_loops(ls)) # num_loops may be greater than N, eg Product
  # return ls
  sc = setup_call(
    ls,
    :(Base.Broadcast.materialize!($ret, bc)),
    LineNumberNode(0),
    inline,
    false,
    u₁,
    u₂,
    v,
    threads % Int,
    warncheckarg,
    safe
  )
  Expr(:block, Expr(:meta, :inline), sc, ret)
end

# size of dest determines loops
# function vmaterialize!(
@generated function vmaterialize!(
  dest::AbstractArray{T,N},
  bc::BC,
  ::Val{Mod},
  ::Val{UNROLL},
  ::Val{dontbc}
) where {T<:NativeTypes,N,BC<:Union{Broadcasted,Product},Mod,UNROLL,dontbc}
  vmaterialize_fun(sizeof(T), N, BC, Mod, UNROLL, dontbc, false)
end
@generated function vmaterialize!(
  dest′::Union{Adjoint{T,A},Transpose{T,A}},
  bc::BC,
  ::Val{Mod},
  ::Val{UNROLL},
  ::Val{dontbc}
) where {
  T<:NativeTypes,
  N,
  A<:AbstractArray{T,N},
  BC<:Union{Broadcasted,Product},
  Mod,
  UNROLL,
  dontbc
}
  vmaterialize_fun(sizeof(T), N, BC, Mod, UNROLL, dontbc, true)
end
# these are marked `@inline` so the `@turbo` itself can choose whether or not to inline.
@generated function vmaterialize!(
  dest::AbstractArray{T,N},
  bc::Broadcasted{
    Base.Broadcast.DefaultArrayStyle{0},
    Nothing,
    typeof(identity),
    Tuple{T2}
  },
  ::Val{Mod},
  ::Val{UNROLL},
  ::Val{dontbc}
) where {T<:NativeTypes,N,T2<:Number,Mod,UNROLL,dontbc}
  inline, u₁, u₂, v, isbroadcast, W, rs, rc, cls, threads, warncheckarg, safe =
    UNROLL
  quote
    $(Expr(:meta, :inline))
    arg = T(first(bc.args))
    @turbo inline = $inline unroll = ($u₁, $u₂) thread = $threads vectorize = $v for i ∈
                                                                                     eachindex(
      dest
    )
      dest[i] = arg
    end
    dest
  end
end
@generated function vmaterialize!(
  dest′::Union{Adjoint{T,A},Transpose{T,A}},
  bc::Broadcasted{
    Base.Broadcast.DefaultArrayStyle{0},
    Nothing,
    typeof(identity),
    Tuple{T2}
  },
  ::Val{Mod},
  ::Val{UNROLL},
  ::Val{dontbc}
) where {T<:NativeTypes,N,A<:AbstractArray{T,N},T2<:Number,Mod,UNROLL,dontbc}
  inline, u₁, u₂, v, isbroadcast, W, rs, rc, cls, threads, warncheckarg, safe =
    UNROLL
  quote
    $(Expr(:meta, :inline))
    arg = T(first(bc.args))
    dest = parent(dest′)
    @turbo inline = $inline unroll = ($u₁, $u₂) thread = $threads vectorize = $v for i ∈
                                                                                     eachindex(
      dest
    )
      dest[i] = arg
    end
    dest′
  end
end
@inline function vmaterialize!(
  dest,
  bc,
  ::Val{Mod},
  ::Val{UNROLL}
) where {Mod,UNROLL}
  vmaterialize!(dest, bc, Val{Mod}(), Val{UNROLL}(), Val(_dontbc(bc)))
end

@inline function vmaterialize(
  bc::Broadcasted{Base.Broadcast.DefaultArrayStyle{0}},
  ::Val{Mod},
  ::Val{UNROLL}
) where {Mod,UNROLL}
  Base.materialize(bc)
end
@inline function vmaterialize(
  bc::Broadcasted,
  ::Val{Mod},
  ::Val{UNROLL}
) where {Mod,UNROLL}
  ElType = Base.Broadcast.combine_eltypes(bc.f, bc.args)
  dest = similar(bc, ElType)
  vmaterialize!(dest, bc, Val{Mod}(), Val{UNROLL}(), Val(_dontbc(bc)))
end

# vmaterialize!(dest, bc, ::Val, ::Val, ::StaticInt, ::StaticInt, ::StaticInt) =
# Base.Broadcast.materialize!(dest, bc)
