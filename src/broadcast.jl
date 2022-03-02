struct LowDimArray{D,T,N,A<:DenseArray{T,N}} <: DenseArray{T,N}
    data::A
end
function LowDimArray{D}(data::A) where {D,T,N,A <: AbstractArray{T,N}}
    LowDimArray{D,T,N,A}(data)
end
Base.@propagate_inbounds Base.getindex(A::LowDimArray, i::Vararg{Union{Integer,CartesianIndex},K}) where {K} = getindex(A.data, i...)
@inline Base.size(A::LowDimArray) = Base.size(A.data)
@inline Base.size(A::LowDimArray, i) = Base.size(A.data, i)

@inline ArrayInterface.parent_type(::Type{LowDimArray{D,T,N,A}}) where {T,D,N,A} = A
@inline Base.strides(A::LowDimArray) = map(Int, strides(A))
@inline ArrayInterface.device(::LowDimArray) = ArrayInterface.CPUPointer()
@generated function ArrayInterface.size(A::LowDimArray{D,T,N}) where {D,T,N}
  t = Expr(:tuple)
  gf = GlobalRef(Core,:getfield)
  for n ∈ 1:N
    if n > length(D) || D[n]
      push!(t.args, Expr(:call, gf, :s, n, false))
    else
      push!(t.args, Expr(:call, Expr(:curly, lv(:StaticInt), 1)))
    end
  end
  Expr(:block, Expr(:meta,:inline), :(s = ArrayInterface.size(parent(A))), t)
end
Base.parent(A::LowDimArray) = getfield(A, :data)
Base.unsafe_convert(::Type{Ptr{T}}, A::LowDimArray{D,T}) where {D,T} = pointer(parent(A))
ArrayInterface.contiguous_axis(A::LowDimArray) = ArrayInterface.contiguous_axis(parent(A))
ArrayInterface.contiguous_batch_size(A::LowDimArray) = ArrayInterface.contiguous_batch_size(parent(A))
ArrayInterface.stride_rank(A::LowDimArray) = ArrayInterface.stride_rank(parent(A))
ArrayInterface.offsets(A::LowDimArray) = ArrayInterface.offsets(parent(A))

@generated function _lowdimfilter(::Val{D}, tup::Tuple{Vararg{Any,N}}) where {D,N}
  t = Expr(:tuple)
  gf = GlobalRef(Core,:getfield)
  for n ∈ 1:N
    if n > length(D) || D[n]
      push!(t.args, Expr(:call, gf, :tup, n, false))
    end
  end
  Expr(:block, Expr(:meta,:inline), t)
end

struct ForBroadcast{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
  data::A
end
@inline Base.parent(fb::ForBroadcast) = getfield(fb, :data)
@inline ArrayInterface.parent_type(::Type{ForBroadcast{T,N,A}}) where {T,N,A} = A
Base.@propagate_inbounds Base.getindex(A::ForBroadcast, i::Vararg{Any,K}) where {K} = parent(A)[i...]
const LowDimArrayForBroadcast{D,T,N,A} = ForBroadcast{T,N,LowDimArray{D,T,N,A}}
@inline function VectorizationBase.contiguous_axis(fb::LowDimArrayForBroadcast{D,T,N,A}) where {D,T,N,A}
  _contiguous_axis(Val{D}(), VectorizationBase.contiguous_axis(parent(parent(fb))))
end
@inline forbroadcast(A::AbstractArray) = ForBroadcast(A)
@inline forbroadcast(A::AbstractRange) = A
@inline forbroadcast(A) = A
@inline forbroadcast(A::Adjoint) = forbroadcast(parent(A))
@inline forbroadcast(A::Transpose) = forbroadcast(parent(A))
@inline function ArrayInterface.strides(A::Union{LowDimArray,ForBroadcast})
  B = parent(A)
  _strides(size(A), strides(B), VectorizationBase.val_stride_rank(B), VectorizationBase.val_dense_dims(B))
end


# @inline function VectorizationBase.contiguous_batch_size(fb::LowDimArrayForBroadcast{D,T,N,A}) where {D,T,N,A}
#     _contiguous_axis(Val{D}(), VectorizationBase.contiguous_batch_size(parent(parent(fb))))
# end
@generated function _contiguous_axis(::Val{D}, ::StaticInt{C}) where {D,C}
  Dlen = length(D)
  (C > 0) || return Expr(:block,Expr(:meta,:inline), staticexpr(C))
  if C ≤ Dlen
    D[C] || return Expr(:block,Expr(:meta,:inline), staticexpr(-1))
  end
  Cnew = 0
  for n ∈ 1:C
    Cnew += ((n > Dlen)) || D[n]
  end
  Expr(:block,Expr(:meta,:inline), staticexpr(Cnew))
end
@inline function ArrayInterface.stride_rank(::Type{LowDimArrayForBroadcast{D,T,N,A}}) where {D,T,N,A}
  _lowdimfilter(Val(D), ArrayInterface.stride_rank(A))
end
@inline function ArrayInterface.dense_dims(::Type{LowDimArrayForBroadcast{D,T,N,A}}) where {D,T,N,A}
  _lowdimfilter(Val(D), ArrayInterface.dense_dims(A))
end
@inline function ArrayInterface.strides(fb::LowDimArrayForBroadcast{D}) where {D}
  _lowdimfilter(Val(D), strides(parent(fb)))
end
@inline function ArrayInterface.offsets(fb::LowDimArrayForBroadcast{D}) where {D}
  _lowdimfilter(Val(D), ArrayInterface.offsets(parent(parent(fb))))
end
@inline function ArrayInterface.StrideIndex(a::A) where {A<:LowDimArrayForBroadcast}
  _stride_index(ArrayInterface.stride_rank(A), ArrayInterface.contiguous_axis(A), a)
end
@inline function _stride_index(r::Tuple{Vararg{StaticInt,N}}, ::StaticInt{C}, A) where {N,C}
  StrideIndex{N,ArrayInterface.known(r),C}(A)
end

for f ∈ [ # groupedstridedpointer support
    :(ArrayInterface.contiguous_axis),
    :(ArrayInterface.contiguous_batch_size),
    :(ArrayInterface.device),
    :(ArrayInterface.stride_rank),
]
    @eval @inline $f(::Type{ForBroadcast{T,N,A}}) where {T,N,A} = $f(A)
end
for f ∈ [ # groupedstridedpointer support
    :(VectorizationBase.memory_reference),
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
function _strides_expr(@nospecialize(s), @nospecialize(x), R::Vector{Int}, D::Vector{Bool})
  N = length(R)
  q = Expr(:block, Expr(:meta,:inline))
  strd_tup = Expr(:tuple)
  gf = GlobalRef(Core, :getfield)
  ifel = GlobalRef(Core, :ifelse)
  Nrange = 1:1:N # type stability w/ respect to reverse
  use_stride_acc = true
  stride_acc::Int = 1
  if is_column_major(R)
  elseif is_row_major(R)
    Nrange = reverse(Nrange)
  else # not worth my time optimizing this case at the moment...
    # will write something generic stride-rank agnostic eventually
    use_stride_acc = false
    stride_acc = 0
  end
  sₙ_value::Int = 0
  for n ∈ Nrange
    xₙ_type = x[n]
    # xₙ_type = typeof(x).parameters[n]
    xₙ_static = xₙ_type <: Static
    xₙ_value::Int = xₙ_static ? (xₙ_type.parameters[1])::Int : 0
    s_type = s[n]
    # s_type = typeof(s).parameters[n]
    sₙ_static = s_type <: Static
    if sₙ_static
      sₙ_value = s_type.parameters[1]
      if s_type === One
        push!(strd_tup.args, Expr(:call, lv(:Zero)))
      elseif stride_acc ≠ 0
        push!(strd_tup.args, staticexpr(stride_acc))
      else
        push!(strd_tup.args, :($gf(x, $n, false)))
      end
    else
      if xₙ_static
        push!(strd_tup.args, staticexpr(xₙ_value))
      elseif stride_acc ≠ 0
        push!(strd_tup.args, staticexpr(stride_acc))
      else
        push!(strd_tup.args, :($ifel(isone($gf(s, $n, false)), zero($xₙ_type), $gf(x, $n, false))))
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
  s::Tuple{Vararg{Integer,N}}, x::Tuple{Vararg{Integer,N}}, ::Val{R}, ::Val{D}
) where {N,R,D}
  Rv = Vector{Int}(undef, N)
  Dv = Vector{Bool}(undef, N)
  for n in 1:N
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
@inline numdims(::Type{B}) where {N, S <: Base.Broadcast.AbstractArrayStyle{N}, B <: Base.Broadcast.Broadcasted{S}} = N

Base.Broadcast._broadcast_getindex_eltype(::Product{A,B}) where {T, A <: AbstractVecOrMat{T}, B <: AbstractVecOrMat{T}} = T
function Base.Broadcast._broadcast_getindex_eltype(p::Product)
    promote_type(
        Base.Broadcast._broadcast_getindex_eltype(p.a),
        Base.Broadcast._broadcast_getindex_eltype(p.b)
    )
end

# recursive_eltype(::Type{A}) where {T, A <: AbstractArray{T}} = T
# recursive_eltype(::Type{NTuple{N,T}}) where {N,T<:Union{Float32,Float64}} = T
# recursive_eltype(::Type{Float32}) = Float32
# recursive_eltype(::Type{Float64}) = Float64
# recursive_eltype(::Type{Tuple{T}}) where {T} = T
# recursive_eltype(::Type{Tuple{T1,T2}}) where {T1,T2} = promote_type(recursive_eltype(T1), recursive_eltype(T2))
# recursive_eltype(::Type{Tuple{T1,T2,T3}}) where {T1,T2,T3} = promote_type(recursive_eltype(T1), recursive_eltype(T2), recursive_eltype(T3))
# recursive_eltype(::Type{Tuple{T1,T2,T3,T4}}) where {T1,T2,T3,T4} = promote_type(recursive_eltype(T1), recursive_eltype(T2), recursive_eltype(T3), recursive_eltype(T4))
# recursive_eltype(::Type{Tuple{T1,T2,T3,T4,T5}}) where {T1,T2,T3,T4,T5} = promote_type(recursive_eltype(T1), recursive_eltype(T2), recursive_eltype(T3), recursive_eltype(T4), recursive_eltype(T5))

# function recursive_eltype(::Type{Broadcasted{S,A,F,ARGS}}) where {S,A,F,ARGS}
#     recursive_eltype(ARGS)
# end

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
@inline Base.Broadcast.broadcasted(::typeof(*ˡ), a::A, b::B) where {A, B} = Product{A,B}(a, b)
# TODO: Need to make this handle A or B being (1 or 2)-D broadcast objects.
function add_broadcast!(
    ls::LoopSet, mC::Symbol, bcname::Symbol, loopsyms::Vector{Symbol},
    @nospecialize(prod::Type{<:Product}), elementbytes::Int
)
    A, B = prod.parameters
    Krange = gensym!(ls, "K")
    Klen = gensym!(ls, "K")
    mA = gensym!(ls, "Aₘₖ")
    mB = gensym!(ls, "Bₖₙ")
    gf = GlobalRef(Core,:getfield)
    pushprepreamble!(ls, Expr(:(=), mA, Expr(:call, :forbroadcast, Expr(:(.), bcname, QuoteNode(:a)))))
    pushprepreamble!(ls, Expr(:(=), mB, Expr(:call, :forbroadcast, Expr(:(.), bcname, QuoteNode(:b)))))
    pushprepreamble!(ls, Expr(:(=), Klen, Expr(:call, gf, Expr(:call, :size, mB), 1, false)))
    pushpreamble!(ls, Expr(:(=), Krange, Expr(:call, :(:), staticexpr(1), Klen)))
    k = gensym!(ls, "k")
    add_loop!(ls, Loop(k, 1, Klen, 1, Krange, Klen), k)
    m = loopsyms[1];
    if numdims(B) == 1
        bloopsyms = Symbol[k]
        cloopsyms = Symbol[m]
        reductdeps = Symbol[m, k]
        kvec = bloopsyms
    elseif numdims(B) == 2
        n = loopsyms[2];
        bloopsyms = Symbol[k,n]
        cloopsyms = Symbol[m,n]
        reductdeps = Symbol[m, k, n]
        kvec = Symbol[k]
    else
        throw("B must be a vector or matrix.")
    end
    # load A
    # loadA = add_load!(ls, gensym!(ls, :A), productref(A, mA, m, k), elementbytes)
    loadA = add_broadcast!(ls, gensym!(ls, "A"), mA, Symbol[m,k], A, elementbytes)
    # load B
    loadB = add_broadcast!(ls, gensym!(ls, "B"), mB, bloopsyms, B, elementbytes)
    # set Cₘₙ = 0
    # setC = add_constant!(ls, zero(promote_type(recursive_eltype(A), recursive_eltype(B))), cloopsyms, mC, elementbytes)
    # targetC will be used for reduce_to_add
    mCt = gensym!(ls, mC)
    targetC = add_constant!(ls, gensym!(ls, "zero"), cloopsyms, mCt, elementbytes, :numericconstant)
    push!(ls.preamble_zeros, (identifier(targetC), IntOrFloat))
    setC = add_constant!(ls, gensym!(ls, "zero"), cloopsyms, mC, elementbytes, :numericconstant)
    push!(ls.preamble_zeros, (identifier(setC), IntOrFloat))
    setC.reduced_children = kvec
    # compute Cₘₙ += Aₘₖ * Bₖₙ
    instrsym = Base.libllvm_version < v"11.0.0" ? :vfmadd231 : :vfmadd
    reductop = Operation(
        ls, mC, elementbytes, instrsym, compute, reductdeps, kvec, Operation[loadA, loadB, setC]
    )
    reductop = pushop!(ls, reductop, mC)
    reductfinal = Operation(
        ls, mCt, elementbytes, :reduce_to_add, compute, cloopsyms, kvec, Operation[reductop, targetC]
    )
    pushop!(ls, reductfinal, mCt)
end

function extract_all_1_array!(ls::LoopSet, bcname::Symbol, N::Int, elementbytes::Int)
    refextract = gensym!(ls, bcname)
    ref = Expr(:ref, bcname);
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
    ls::LoopSet, destname::Symbol, bcname::Symbol, loopsyms::Vector{Symbol},
    @nospecialize(_::Type{LowDimArray{D,T,N,A}}), elementbytes::Int
) where {D,T,N,A}
    Dlen = length(D)
    if Dlen == N && !any(D) # array is a scalar, as it is broadcasted on all dimensions
        return extract_all_1_array!(ls, bcname, N, elementbytes)
    end
    fulldims = Symbol[loopsyms[n] for n ∈ 1:N if ((Dlen < n) || D[n]::Bool)]
    ref = ArrayReference(bcname, fulldims)
    loadop = add_simple_load!(ls, destname, ref, elementbytes, true )::Operation
    doaddref!(ls, loadop)
end
function add_broadcast_adjoint_array!(
    ls::LoopSet, destname::Symbol, bcname::Symbol, loopsyms::Vector{Symbol}, ::Type{A}, elementbytes::Int
) where {T,N,A<:AbstractArray{T,N}}
    # parent = gensym!(ls, "parent")
    # pushprepreamble!(ls, Expr(:(=), parent, Expr(:call, :parent, bcname)))
    # isone(length(loopsyms)) && return extract_all_1_array!(ls, bcname, N, elementbytes)
    ref = ArrayReference(bcname, Symbol[loopsyms[N + 1 - n] for n ∈ 1:N])
    loadop = add_simple_load!( ls, destname, ref, elementbytes, true )::Operation
    doaddref!(ls, loadop)
end
function add_broadcast_adjoint_array!(
    ls::LoopSet, destname::Symbol, bcname::Symbol, loopsyms::Vector{Symbol}, ::Type{<:AbstractVector}, elementbytes::Int
)
    # isone(length(loopsyms)) && return extract_all_1_array!(ls, bcname, N, elementbytes)
    # parent = gensym!(ls, "parent")
    # pushprepreamble!(ls, Expr(:(=), parent, Expr(:call, :parent, bcname)))

    ref = ArrayReference(bcname, Symbol[loopsyms[2]])
    loadop = add_simple_load!( ls, destname, ref, elementbytes, true )::Operation
    doaddref!(ls, loadop)
end
function add_broadcast!(
    ls::LoopSet, destname::Symbol, bcname::Symbol, loopsyms::Vector{Symbol},
    ::Type{Adjoint{T,A}}, elementbytes::Int
) where {T, A <: AbstractArray{T}}
    add_broadcast_adjoint_array!( ls, destname, bcname, loopsyms, A, elementbytes )
end
function add_broadcast!(
    ls::LoopSet, destname::Symbol, bcname::Symbol, loopsyms::Vector{Symbol},
    ::Type{Transpose{T,A}}, elementbytes::Int
) where {T, A <: AbstractArray{T}}
    add_broadcast_adjoint_array!( ls, destname, bcname, loopsyms, A, elementbytes )
end
function add_broadcast!(
    ls::LoopSet, destname::Symbol, bcname::Symbol, loopsyms::Vector{Symbol},
    ::Type{<:AbstractArray{T,N}}, elementbytes::Int
) where {T,N}
    loadop = add_simple_load!(ls, destname, ArrayReference(bcname, @view(loopsyms[1:N])), elementbytes, true)
    doaddref!(ls, loadop)
end
function add_broadcast!(
    ls::LoopSet, ::Symbol, bcname::Symbol, loopsyms::Vector{Symbol}, ::Type{T}, elementbytes::Int
) where {T<:Number}
    add_constant!(ls, bcname, elementbytes) # or replace elementbytes with sizeof(T) ? u
end
function add_broadcast!(
    ls::LoopSet, ::Symbol, bcname::Symbol, loopsyms::Vector{Symbol}, ::Type{Base.RefValue{T}}, elementbytes::Int
) where {T}
    refextract = gensym!(ls, bcname)
    pushprepreamble!(ls, Expr(:(=), refextract, Expr(:ref, bcname)))
    add_constant!(ls, refextract, elementbytes) # or replace elementbytes with sizeof(T) ? u
end
function add_broadcast!(
    ls::LoopSet, destname::Symbol, bcname::Symbol, loopsyms::Vector{Symbol},
    @nospecialize(_::Type{SubArray{T,N,A,S,B}}), elementbytes::Int
) where {T,N,N2,A<:AbstractArray{T,N2},B,N3,S <: Tuple{Int,Vararg{Any,N3}}}
    inds = Vector{Symbol}(undef, N+1)
    inds[1] = DISCONTIGUOUS
    inds[2:end] .= @view(loopsyms[1:N])
    loadop = add_simple_load!(ls, destname, ArrayReference(bcname, inds), elementbytes, true)
    doaddref!(ls, loadop)
end
const BroadcastedArray{S<:Broadcast.AbstractArrayStyle,F,A} = Broadcasted{S,Nothing,F,A}
function add_broadcast!(
    ls::LoopSet, destname::Symbol, bcname::Symbol, loopsyms::Vector{Symbol},
    @nospecialize(B::Type{<:BroadcastedArray}),
    elementbytes::Int
)
    S,_,F,A = B.parameters
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
    gf = GlobalRef(Core, :getfield)
    for (i,arg) ∈ enumerate(args)
        argname = gensym!(ls, "arg")
        pushprepreamble!(ls, Expr(:(=), argname, Expr(:call, :forbroadcast, Expr(:call, gf, bcargs, i, false))))
        # dynamic dispatch
        parent = add_broadcast!(ls, gensym!(ls, "temp"), argname, loopsyms, arg, elementbytes)::Operation
        push!(parents, parent)
        mergesetdiffv!(deps, loopdependencies(parent), reduceddependencies(parent))
    end
    op = Operation(
        length(operations(ls)), destname, elementbytes, instr, compute, deps, NODEPENDENCY, parents
    )
    pushop!(ls, op, destname)
end

function add_broadcast_loops!(ls::LoopSet, loopsyms::Vector{Symbol}, destsym::Symbol)
    axes_tuple = Expr(:tuple)
    pushpreamble!(ls, Expr(:(=), axes_tuple, Expr(:call, :axes, destsym)))
    for itersym ∈ loopsyms
        Nrange = gensym!(ls, "N")
        Nlower = gensym!(ls, "N")
        Nupper = gensym!(ls, "N")
        Nlen = gensym!(ls, "N")
        add_loop!(ls, Loop(itersym, Nlower, Nupper, 1, Nrange, Nlen), itersym)
        push!(axes_tuple.args, Nrange)
        pushpreamble!(ls, Expr(:(=), Nlower, Expr(:call, lv(:maybestaticfirst), Nrange)))
        pushpreamble!(ls, Expr(:(=), Nupper, Expr(:call, lv(:maybestaticlast), Nrange)))
        pushpreamble!(ls, Expr(:(=), Nlen, Expr(:call, GlobalRef(ArrayInterface,:static_length), Nrange)))
    end
end
# size of dest determines loops
# function vmaterialize!(
@generated function vmaterialize!(
    dest::AbstractArray{T,N}, bc::BC, ::Val{Mod}, ::Val{UNROLL}
) where {T <: NativeTypes, N, BC <: Union{Broadcasted,Product}, Mod, UNROLL}
  2+1
  # we have an N dimensional loop.
  # need to construct the LoopSet
  # @show typeof(dest)
  ls = LoopSet(Mod)
  inline, u₁, u₂, v, isbroadcast, _, rs, rc, cls, l1, l2, l3, threads, warncheckarg = UNROLL
  set_hw!(ls, rs, rc, cls, l1, l2, l3)
  ls.isbroadcast = isbroadcast # maybe set `false` in a DiffEq-like `@..` macro
  loopsyms = [gensym!(ls, "n") for _ ∈ 1:N]
  add_broadcast_loops!(ls, loopsyms, :dest)
  elementbytes = sizeof(T)
  add_broadcast!(ls, :destination, :bc, loopsyms, BC, elementbytes)
  storeop = add_simple_store!(ls, :destination, ArrayReference(:dest, loopsyms), elementbytes)
  doaddref!(ls, storeop)
  resize!(ls.loop_order, num_loops(ls)) # num_loops may be greater than N, eg Product
  # return ls
  sc = setup_call(ls, :(Base.Broadcast.materialize!(dest, bc)), LineNumberNode(0), inline, false, u₁, u₂, v, threads%Int, warncheckarg)
  # for n in loopsyms; push!(sc.args, :(@show $n)); end
  Expr(:block, Expr(:meta,:inline), sc, :dest)
end
@generated function vmaterialize!(
    dest′::Union{Adjoint{T,A},Transpose{T,A}}, bc::BC, ::Val{Mod}, ::Val{UNROLL}
) where {T <: NativeTypes, N, A <: AbstractArray{T,N}, BC <: Union{Broadcasted,Product}, Mod, UNROLL}
  # we have an N dimensional loop.
  # need to construct the LoopSet
  ls = LoopSet(Mod)
  inline, u₁, u₂, v, isbroadcast, _, rs, rc, cls, l1, l2, l3, threads, warncheckarg = UNROLL
  set_hw!(ls, rs, rc, cls, l1, l2, l3)
  ls.isbroadcast = isbroadcast # maybe set `false` in a DiffEq-like `@..` macro
  loopsyms = [gensym!(ls, "n") for _ ∈ 1:N]
  pushprepreamble!(ls, Expr(:(=), :dest, Expr(:call, :parent, :dest′)))
  add_broadcast_loops!(ls, loopsyms, :dest′)
  elementbytes = sizeof(T)
  add_broadcast!(ls, :destination, :bc, loopsyms, BC, elementbytes)
  storeop = add_simple_store!(ls, :destination, ArrayReference(:dest, reverse(loopsyms)), elementbytes)
  doaddref!(ls, storeop)
  resize!(ls.loop_order, num_loops(ls)) # num_loops may be greater than N, eg Product
  Expr(:block, Expr(:meta,:inline), setup_call(ls, :(Base.Broadcast.materialize!(dest′, bc)), LineNumberNode(0), inline, false, u₁, u₂, v, threads%Int, warncheckarg), :dest′)
end
# these are marked `@inline` so the `@turbo` itself can choose whether or not to inline.
@generated function vmaterialize!(
  dest::AbstractArray{T,N}, bc::Broadcasted{Base.Broadcast.DefaultArrayStyle{0},Nothing,typeof(identity),Tuple{T2}}, ::Val{Mod}, ::Val{UNROLL}
  ) where {T <: NativeTypes, N, T2 <: Number, Mod, UNROLL}
  inline, u₁, u₂, v, isbroadcast, W, rs, rc, cls, l1, l2, l3, threads = UNROLL
  quote
    $(Expr(:meta,:inline))
    arg = T(first(bc.args))
    @turbo inline=$inline unroll=($u₁,$u₂) thread=$threads vectorize=$v for i ∈ eachindex(dest)
      dest[i] = arg
    end
    dest
  end
end
@generated function vmaterialize!(
  dest′::Union{Adjoint{T,A},Transpose{T,A}}, bc::Broadcasted{Base.Broadcast.DefaultArrayStyle{0},Nothing,typeof(identity),Tuple{T2}}, ::Val{Mod}, ::Val{UNROLL}
  ) where {T <: NativeTypes, N, A <: AbstractArray{T,N}, T2 <: Number, Mod, UNROLL}
  inline, u₁, u₂, v, isbroadcast, W, rs, rc, cls, l1, l2, l3, threads = UNROLL
  quote
    $(Expr(:meta,:inline))
    arg = T(first(bc.args))
    dest = parent(dest′)
    @turbo inline=$inline unroll=($u₁,$u₂) thread=$threads vectorize=$v for i ∈ eachindex(dest)
      dest[i] = arg
    end
    dest′
  end
end

@inline function vmaterialize(bc::Broadcasted{Base.Broadcast.DefaultArrayStyle{0}}, ::Val{Mod}, ::Val{UNROLL}) where {Mod,UNROLL}
  Base.materialize(bc)
end
@inline function vmaterialize(
    bc::Broadcasted, ::Val{Mod}, ::Val{UNROLL}
) where {Mod,UNROLL}
  ElType = Base.Broadcast.combine_eltypes(bc.f, bc.args)
  dest = similar(bc, ElType)
  vmaterialize!(dest, bc, Val{Mod}(), Val{UNROLL}())
end

vmaterialize!(dest, bc, ::Val, ::Val, ::StaticInt, ::StaticInt, ::StaticInt) = Base.Broadcast.materialize!(dest, bc)

