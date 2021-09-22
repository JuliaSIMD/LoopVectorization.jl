import .ForwardDiff

@generated function SLEEFPirates.tanh_fast(x::ForwardDiff.Dual{T,S,N}) where {T,S,N}
  quote
    $(Expr(:meta,:inline))
    t = tanh_fast(x.value)
    ∂t = vfnmadd_fast(t, t, one(S))
    p = x.partials
    ForwardDiff.Dual{T}(t, ForwardDiff.Partials(Base.Cartesian.@ntuple $N n -> mul_fast(∂t, p[n])))
  end
end
@generated function SLEEFPirates.sigmoid_fast(x::ForwardDiff.Dual{T,S,N}) where {T,S,N}
  quote
    $(Expr(:meta,:inline))
    s = sigmoid_fast(x.value)
    ∂s = vfnmadd_fast(s,s,s)
    p = x.partials
    ForwardDiff.Dual{T}(s, ForwardDiff.Partials(Base.Cartesian.@ntuple $N n -> mul_fast(∂s, p[n])))
  end
end
@generated function VectorizationBase.relu(x::ForwardDiff.Dual{T,S,N}) where {T,S,N}
  quote
    $(Expr(:meta,:inline))
    v = x.value
    z = zero(v)
    cmp = v < z
    r = ifelse(cmp, z, v)
    p = x.partials
    ForwardDiff.Dual{T}(r, ForwardDiff.Partials(Base.Cartesian.@ntuple $N n -> ifelse(cmp, z, p[n])))
  end
end
@generated function init_dual(v::Tuple{Vararg{AbstractSIMD,A}}) where {A}
  res = Expr(:tuple)
  q = Expr(:block, Expr(:meta,:inline))
  for a ∈ 1:A
    v_a = Symbol(:v_,a)
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
@generated function dual_store!(∂p::Tuple{Vararg{AbstractStridedPointer,A}}, p::AbstractStridedPointer, ∂v, im::Vararg{Any,N}) where {A,N}
  quote
    $(Expr(:meta,:inline))
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


