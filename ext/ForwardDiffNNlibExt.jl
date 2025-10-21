module ForwardDiffNNlibExt
import ForwardDiff
using LoopVectorization, VectorizationBase, SLEEFPirates, ForwardDiff, NNlib

@generated function NNlib.relu(
  x::ForwardDiff.Dual{T,<:LoopVectorization.AbstractSIMD,N}
) where {T,N}
  quote
    $(Expr(:meta, :inline))
    v = x.value
    z = zero(v)
    cmp = v < z
    r = ifelse(cmp, z, v)
    p = x.partials
    ForwardDiff.Dual{T}(
      r,
      ForwardDiff.Partials(Base.Cartesian.@ntuple $N n -> ifelse(cmp, z, p[n]))
    )
  end
end

@generated function NNlib.leakyrelu(
  x::ForwardDiff.Dual{T,<:LoopVectorization.AbstractSIMD,N},
  a = 0.01
) where {T,N}
  quote
    $(Expr(:meta, :inline))
    v = x.value
    z = zero(v)

    α = convert(typeof(v), a)
    cmp = v < z
    r = ifelse(cmp, α * v, v)
    p = x.partials
    ForwardDiff.Dual{T}(
      r,
      ForwardDiff.Partials(Base.Cartesian.@ntuple $N n -> ifelse(cmp, α * p[n], p[n]))
    )
  end
end

end
