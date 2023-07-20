import VectorizationBase: vsum

@inline vreduce(::typeof(+), v::VectorizationBase.AbstractSIMDVector) = vsum(v)
@inline vreduce(::typeof(*), v::VectorizationBase.AbstractSIMDVector) = vprod(v)
@inline vreduce(::typeof(max), v::VectorizationBase.AbstractSIMDVector) =
  vmaximum(v)
@inline vreduce(::typeof(min), v::VectorizationBase.AbstractSIMDVector) =
  vminimum(v)
@inline vreduce(op, v::VectorizationBase.AbstractSIMDVector) =
  vec_vreduce(op, v)
@inline vec_reduce(op, v::VectorizationBase.AbstractSIMDVector) =
  vec_reduce(op, Vec(v))
vec_vreduce(op, v::Vec{1}) = VectorizationBase.extractelement(v, 0)
@inline function vec_vreduce(op, v::Vec{W}) where {W}
  a = op(
    VectorizationBase.extractelement(v, 0),
    VectorizationBase.extractelement(v, 1)
  )
  for i ∈ 2:W-1
    a = op(a, VectorizationBase.extractelement(v, i))
  end
  a
end

function mapreduce_simple(
  f::F,
  op::OP,
  args::Vararg{AbstractArray,A}
) where {F,OP,A}
  ptrargs = ntuple(a -> pointer(args[a]), Val(A))
  N = length(first(args))
  iszero(N) && throw("Length of vector is 0!")
  st = ntuple(a -> VectorizationBase.static_sizeof(eltype(args[a])), Val(A))
  a_0 = f(VectorizationBase.__vload.(ptrargs, False(), register_size())...)
  i = 1
  while i < N
    a_0 = op(
      a_0,
      f(
        VectorizationBase.__vload.(
          ptrargs,
          VectorizationBase.lazymul.(st, i),
          False(),
          register_size()
        )...
      )
    )
    i += 1
  end
  a_0
end

"""
    vmapreduce(f, op, A::DenseArray...)

Vectorized version of `mapreduce`. Applies `f` to each element of the arrays `A`, and reduces the result with `op`.
"""
@inline function vmapreduce(
  f::F,
  op::OP,
  arg1::AbstractArray{T},
  args::Vararg{AbstractArray{T},A}
) where {F,OP,T<:NativeTypes,A}
  if !(check_args(arg1, args...) && all_dense(arg1, args...))
    return mapreduce(f, op, arg1, args...)
  end
  N = length(arg1)
  iszero(A) || @assert all(length.(args) .== N)
  W = VectorizationBase.pick_vector_width(T)
  if N < W
    mapreduce_simple(f, op, arg1, args...)
  else
    _vmapreduce(f, op, W, N, T, arg1, args...)
  end
end
@inline function _vmapreduce(
  f::F,
  op::OP,
  ::StaticInt{W},
  N,
  ::Type{T},
  args::Vararg{AbstractArray{<:NativeTypes},A}
) where {F,OP,A,W,T}
  ptrargs = VectorizationBase.zero_offsets.(stridedpointer.(args))
  if N ≥ 4W
    index = VectorizationBase.Unroll{1,W,4,1,W,zero(UInt)}((Zero(),))
    i = 4W
    au = f(vload.(ptrargs, index)...)
    while i < N - ((W << 2) - 1)
      index = VectorizationBase.Unroll{1,W,4,1,W,zero(UInt)}((i,))
      i += 4W
      au = op(au, f(vload.(ptrargs, index)...))
    end
    a_0 = VectorizationBase.reduce_to_onevec(op, au)
  else
    a_0 = f(vload.(ptrargs, ((MM{W}(Zero()),),))...)
    i = W
  end
  while i < N - (W - 1)
    a_0 = op(a_0, f(vload.(ptrargs, ((MM{W}(i),),))...))
    i += W
  end
  if i < N
    m = mask(T, N & (W - 1))
    a_0 = ifelse(m, op(a_0, f(vload.(ptrargs, ((MM{W}(i),),))...)), a_0)
  end
  vreduce(op, a_0)
end
@inline vmapreduce(f, op, args...) = mapreduce(f, op, args...)

"""
    vsum(A::DenseArray)
    vsum(f, A::DenseArray)

Vectorized version of `sum`. Providing a function as the first argument
will apply the function to each element of `A` before summing.
"""
@inline vsum(f::F, A::AbstractArray{T}) where {F,T<:NativeTypes} =
  vmapreduce(f, +, A)
@inline vsum(A::AbstractArray{T}) where {T<:NativeTypes} = vsum(identity, A)

length_one_axis(::Base.OneTo) = Base.OneTo(1)
length_one_axis(::Any) = 1:1

"""
    vreduce(op, A::DenseArray; [dims::Int])

Vectorized version of `reduce`. Reduces the array `A` using the operator `op`.
At most one dimension may be supplied as kwarg.
"""
@inline vreduce(op, arg) = vmapreduce(identity, op, arg)

for (op, init) in zip((:+, :max, :min), (:zero, :typemin, :typemax))
  @eval @inline function vreduce(::typeof($op), arg; dims = nothing)
    if !(check_args(arg) && all_dense(arg))
      return reduce($op, arg; dims = dims)
    end
    dims === nothing && return _vreduce($op, arg)
    isone(ndims(arg)) && return [_vreduce($op, arg)]
    @assert length(dims) == 1
    axes_arg = axes(arg)
    axes_out = Base.setindex(axes_arg, length_one_axis(axes_arg[dims]), dims)
    out = similar(arg, axes_out)
    # fill!(out, $init(first(arg)))
    # TODO: generated function with Base.Cartesian.@nif to set to ndim(arg)
    Base.Cartesian.@nif 5 d -> (d <= ndims(arg) && dims == d) d -> begin
      Rpre = CartesianIndices(ntuple(i -> axes_arg[i], d - 1))
      Rpost = CartesianIndices(ntuple(i -> axes_arg[i+d], ndims(arg) - d))
      _vreduce_dims!(out, $op, Rpre, static_axes(arg, dims), Rpost, arg)
    end d -> begin
      Rpre = CartesianIndices(axes_arg[1:dims-1])
      Rpost = CartesianIndices(axes_arg[dims+1:end])
      _vreduce_dims!(out, $op, Rpre, static_axes(arg, dims), Rpost, arg)
    end
  end

  @eval @inline function _vreduce_dims!(
    out,
    ::typeof($op),
    Rpre,
    is,
    Rpost,
    arg
  )
    s = $init(first(arg))
    @turbo for Ipost in Rpost, Ipre in Rpre
      accum = s
      for i in is
        accum = $op(accum, arg[Ipre, i, Ipost])
      end
      out[Ipre, 1, Ipost] = accum
    end
    return out
  end

  @eval @inline function _vreduce(::typeof($op), arg)
    s = $init(first(arg))
    @turbo for i in eachindex(arg)
      s = $op(s, arg[i])
    end
    return s
  end
end
