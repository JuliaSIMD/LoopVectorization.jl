module ForwardDiffExt
import ForwardDiff, ChainRulesCore
using SIMDDualNumbers, LoopVectorization
using LoopVectorization:
  AbstractSIMD,
  AbstractStridedPointer,
  relu,
  vmap,
  VectorizationBase,
  vmapt,
  vmapnt,
  vmapntt,
  MM,
  StaticInt,
  vadd_nw,
  vsub_nsw,
  vload,
  mask,
  vfnmadd_fast,
  mul_fast
using VectorizationBase: zero_offsets

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
  im::Vararg{Any,N}
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

if isdefined(ChainRulesCore, :ZeroTangent)
  const ChainRulesZero = ChainRulesCore.ZeroTangent
else
  const ChainRulesZero = ChainRulesCore.Zero
end

function ChainRulesCore.rrule(::typeof(tanh_fast), x)
  t = tanh_fast(x)
  ∂ = let t = t
    y -> (ChainRulesZero(), mul_fast(vfnmadd_fast(t, t, one(t)), y))
  end
  t, ∂
end
function ChainRulesCore.rrule(::typeof(sigmoid_fast), x)
  s = sigmoid_fast(x)
  ∂ = let s = s
    y -> (ChainRulesZero(), mul_fast(vfnmadd_fast(s, s, s), y))
  end
  s, ∂
end
function ChainRulesCore.rrule(::typeof(relu), v)
  z = zero(v)
  cmp = v < z
  r = ifelse(cmp, z, v)
  ∂ = let cmp = cmp
    y -> (ChainRulesZero(), ifelse(cmp, zero(y), y))
  end
  r, ∂
end

function ∂vmap_singlethread!(
  f::F,
  ∂y::Tuple{Vararg{DenseArray{T},A}},
  y::DenseArray{T},
  args::Vararg{DenseArray{<:Base.HWReal},A}
) where {F,T<:Base.HWReal,A}
  N = length(y)
  ptry = zero_offsets(stridedpointer(y))
  ptrargs = map(zero_offsets, map(stridedpointer, args))
  ptr∂y = map(zero_offsets, map(stridedpointer, ∂y))
  i = 0
  V = VectorizationBase.pick_vector_width(T)
  W = Int(V)
  while i < vsub_nsw(N, ((W << 2) - 1))
    index = VectorizationBase.Unroll{1,W,4,1,W,zero(UInt)}((i,))
    v = f(init_dual(map(Base.Fix2(vload, index), ptrargs))...)
    dual_store!(ptr∂y, ptry, v, index)
    i = vadd_nw(i, 4W)
  end
  while i < vsub_nsw(N, (W - 1))
    loader = Base.Fix2(vload, (MM{W}(i),))
    vᵣ = f(init_dual(map(loader, ptrargs))...)
    dual_store!(ptr∂y, ptry, vᵣ, (MM{W}(i),))
    i = vadd_nw(i, W)
  end
  if i < N
    m = mask(T, N & (W - 1))
    mloader = let i = i, m = m
      p -> vload(p, (MM{W}(i),), m)
    end
    dual_store!(
      ptr∂y,
      ptry,
      f(init_dual(map(mloader, ptrargs))...),
      (MM{W}(i),),
      m
    )
  end
  nothing
end

struct SIMDMapBack{K,T<:Tuple{Vararg{Any,K}}}
  jacs::T
end
@generated function (b::SIMDMapBack{K,T})(Δ::A) where {K,T,A}
  preloop = Expr(:block, :(jacs = b.jacs))
  loop_body = Expr(:block, :(Δᵢ = Δ[i]))
  ret = Expr(:tuple, ChainRulesZero(), ChainRulesZero())
  for k ∈ 1:K
    jₖ = Symbol(:j_, k)
    push!(preloop.args, :($jₖ = jacs[$k]))
    push!(loop_body.args, :($jₖ[i] *= Δᵢ))
    push!(ret.args, jₖ)
  end
  quote
    $preloop
    @turbo for i ∈ eachindex(Δ)
      $loop_body
    end
    $ret
  end
end

function ChainRulesCore.rrule(
  ::typeof(vmap),
  f::F,
  args::Vararg{Any,K}
) where {F,K}
  out = similar(first(args))
  jacs = map(similar, args)
  ∂vmap_singlethread!(f, jacs, out, args...)
  out, SIMDMapBack(jacs)
end
for f in (:vmapt, :vmapnt, :vmapntt)
  @eval function ChainRulesCore.rrule(
    ::typeof($f),
    f::F,
    args::Vararg{Any,K}
  ) where {F,K}
    ChainRulesCore.rrule(typeof($vmap), f, args...)
  end
end
end
