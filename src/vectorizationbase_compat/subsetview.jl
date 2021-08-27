
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
    ptr::AbstractStridedPointer{T,N,C,B,R,X,O}, ::StaticInt{I}, i::Integer
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
    $(Expr(:meta,:inline))
    strd = strides(ptr)
    offs = offsets(ptr)
    si = ArrayInterface.StrideIndex{$(N-1),$newR,$newC}($newstrd, $newoffsets)
    stridedpointer($gptr, si, StaticInt{$B}())
  end
end
@inline _gesp(sp::VectorizationBase.FastRange, ::StaticInt{1}, i) = gesp(sp, (i,))
@generated function _gesp(sp::AbstractStridedPointer{T,N}, ::StaticInt{I}, i::Integer) where {I,N,T}
  t = Expr(:tuple)
  for j ∈ 1:I-1
    push!(t.args, staticexpr(0))
  end
  push!(t.args, :i)
  if I > 1
    for j ∈ I+1:N
      push!(t.args, staticexpr(0))
    end
  end
  quote
    $(Expr(:meta,:inline))
    gesp(sp, $t)
  end
end


