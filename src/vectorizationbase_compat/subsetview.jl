
function copy_sp_data_range!(newR, newstrd, newoffsets, ind, R, rg)
  for i ∈ rg
    push!(newR.args, R[i])
    push!(newstrd.args, :($getfield(strd, $i, false)))
    push!(newoffsets.args, :($getfield(offs, $i, false)))
    push!(ind.args, :($getfield(offs, $i, false)))
  end
  nothing
end
@generated function subsetview(
  ptr::AbstractStridedPointer{T,N,C,B,R,X,O},
  ::StaticInt{I},
  i::Union{Integer,StaticInt}
) where {T,N,C,B,R,X,O,I}
  I > N && return :ptr
  @assert B ≤ 0 "Batched dims not currently supported."
  newC = C == I ? -1 : (C < I ? C : C - 1)
  newR = Expr(:tuple)
  newstrd = Expr(:tuple)
  newoffsets = Expr(:tuple)
  ind = Expr(:tuple)
  copy_sp_data_range!(newR, newstrd, newoffsets, ind, R, 1:I-1)
  push!(ind.args, :i)
  copy_sp_data_range!(newR, newstrd, newoffsets, ind, R, I+1:N)
  gptr = Expr(:call, :gep, :ptr, ind)
  quote
    $(Expr(:meta, :inline))
    strd = strides(ptr)
    offs = offsets(ptr)
    si = StaticArrayInterface.StrideIndex{$(N - 1),$newR,$newC}($newstrd, $newoffsets)
    stridedpointer($gptr, si, StaticInt{$B}())
  end
end
@inline _subsetview(
  ptr::AbstractStridedPointer,
  ::StaticInt{I},
  J::Tuple{}
) where {I} = ptr
@inline _subsetview(
  ptr::AbstractStridedPointer,
  ::StaticInt{I},
  J::Tuple{J1}
) where {I,J1} = subsetview(ptr, StaticInt{I}(), first(J))
@inline _subsetview(
  ptr::AbstractStridedPointer,
  ::StaticInt{I},
  J::Tuple{J1,J2,Vararg}
) where {I,J1,J2} = _subsetview(
  subsetview(ptr, StaticInt{I}(), first(J)),
  StaticInt{I}(),
  Base.tail(J)
)
@inline subsetview(
  ptr::AbstractStridedPointer,
  ::StaticInt{I},
  J::CartesianIndex
) where {I} = _subsetview(ptr, StaticInt{I}(), Tuple(J))

@inline _gesp(
  sp::VectorizationBase.FastRange,
  ::StaticInt{1},
  i,
  ::StaticInt{1}
) = gesp(sp, (i,))
@generated function _gesp(
  sp::AbstractStridedPointer{T,N},
  ::StaticInt{I},
  i::Union{Integer,StaticInt},
  ::StaticInt{D}
) where {I,N,T,D}
  t = Expr(:tuple)
  for j ∈ 1:I-1
    # push!(t.args, staticexpr(0))
    push!(t.args, VectorizationBase.NullStep())
  end
  push!(t.args, :i)
  if D > 1
    for j ∈ I+1:N
      # push!(t.args, staticexpr(0))
      push!(t.args, VectorizationBase.NullStep())
    end
  end
  quote
    $(Expr(:meta, :inline))
    gesp(sp, $t)
  end
end
