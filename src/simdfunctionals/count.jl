_vcount(f) = 0
function _vcount(f::F, args::Vararg{DenseArray,M}) where {F,M}
  x = first(args)
  y = Base.tail(args)
  foreach(a -> @assert(size(a) == size(x)), y)
  N = length(x)
  ptrargs = map(VectorizationBase.zstridedpointer, args)
  i = 0
  V = VectorizationBase.pick_vector_width(
    reduce(promote_type, map(eltype, ptrargs))
  )
  W = unwrap(V)
  UNROLL = 4
  LOG2UNROLL = 2
  _counts = if VERSION >= v"1.7"
    VecUnroll(ntuple(Returns(0), Val(UNROLL)))
  else
    VecUnroll(ntuple(_ -> (0), Val(UNROLL)))
  end
  counts::typeof(_counts) = _counts
  while i < vsub_nsw(N, ((W << LOG2UNROLL) - 1))
    index = VectorizationBase.Unroll{1,W,UNROLL,1,W,zero(UInt)}((i,))
    counts += count_ones(f(VectorizationBase.fmap(vload, ptrargs, index)...))
    i = vadd_nw(i, StaticInt{UNROLL}() * W)
  end
  count = reduce_tup(+, data(counts))
  while i < vsub_nsw(N, (W - 1)) # stops at 16 when
    count += count_ones(f(map1(vload, ptrargs, (MM{W}(i),))...))
    i = vadd_nw(i, W)
  end
  if i < N
    m = mask(StaticInt(W), N & (W - 1))
    vfinal = f(map1(vload, ptrargs, (MM{W}(i),), m)...)
    count += count_ones(vfinal & m)
  end
  count
end

@generated function vcount(f::F, args::Vararg{DenseArray,M}) where {F,M}
  call = Expr(:call, :_vcount, :f)
  gc_preserve_call_quote(call, M::Int)
end
vcount(::typeof(identity), x::AbstractArray{Bool}) =
  vcount(VectorizationBase.tomask, x)
vcount(x::AbstractArray{Bool}) = vcount(VectorizationBase.tomask, x)
