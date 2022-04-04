import .ForwardDiff
using SIMDDualNumbers

@generated function init_dual(v::Tuple{Vararg{AbstractSIMD,A}}) where {A}
  res = Expr(:tuple)
  q = Expr(:block, Expr(:meta, :inline))
  for a ∈ 1:A
    v_a = Symbol(:v_, a)
    push!(q.args, Expr(:(=), v_a, Expr(:ref, :v, a)))
    partials = Expr(:tuple)
    for i ∈ 1:A
      push!(partials.args, Expr(:call, i == a ? :one : :zero, v_a))
    end
    push!(res.args, :(ForwardDiff.Dual($v_a, ForwardDiff.Partials($partials))))
  end
  push!(q.args, res)
  q
end
@generated function dual_store!(
  ∂p::Tuple{Vararg{AbstractStridedPointer,A}},
  p::AbstractStridedPointer,
  ∂v,
  im::Vararg{Any,N},
) where {A,N}
  quote
    $(Expr(:meta, :inline))
    v = ∂v.value
    ∂ = ∂v.partials
    Base.Cartesian.@nextract $N im im
    Base.Cartesian.@ncall $N VectorizationBase.vnoaliasstore! p v im  # store
    Base.Cartesian.@nexprs $A a -> begin # for each of `A` partials
      ∂p_a = ∂p[a]
      ∂_a = ∂[a]
      Base.Cartesian.@ncall $N VectorizationBase.vnoaliasstore! ∂p_a ∂_a im # store
    end
    nothing
  end
end
