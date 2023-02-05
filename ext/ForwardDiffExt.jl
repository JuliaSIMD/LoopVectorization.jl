module ForwardDiffExt
import ForwardDiff, ChainRulesCore
using LoopVectorization, VectorizationBase, SLEEFPirates, ForwardDiff

import IfElse: ifelse
using VectorizationBase: AbstractSIMD, AbstractMask, zero_offsets

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

@generated function Base.abs(
  x::ForwardDiff.Dual{TAG,S,N}
) where {TAG,S<:AbstractSIMD,N}
  quote
    $(Expr(:meta, :inline))
    val = x.value
    p = x.partials
    cmp = val < zero($S)
    absx = $ifelse(cmp, -val, val)
    Base.Cartesian.@nexprs $N n -> p_n = p[n]
    ForwardDiff.Dual{$TAG}(
      absx,
      ForwardDiff.Partials(
        Base.Cartesian.@ntuple $N n -> $ifelse(cmp, -p_n, p_n)
      )
    )
  end
end
@inline function Base.max(
  x::ForwardDiff.Dual{TAG,<:AbstractSIMD,N},
  y::ForwardDiff.Dual{TAG,<:AbstractSIMD,N}
) where {TAG,N}
  vx = ForwardDiff.value(x)
  vy = ForwardDiff.value(y)
  xgy = vx > vy
  z = ifelse(xgy, vx, vy)
  p = VectorizationBase.fmap(
    ifelse,
    xgy,
    ForwardDiff.partials(x).values,
    ForwardDiff.partials(y).values
  )
  ForwardDiff.Dual{TAG}(z, ForwardDiff.Partials(p))
end

@inline Base.max(
  x::T,
  y::Real
) where {N,T<:ForwardDiff.Dual{<:Any,<:AbstractSIMD,N}} = max(x, T(y))
@inline Base.max(
  y::Real,
  x::T
) where {N,T<:ForwardDiff.Dual{<:Any,<:AbstractSIMD,N}} = max(x, T(y))
@inline Base.max(
  x::T,
  y::Int
) where {N,T<:ForwardDiff.Dual{<:Any,<:AbstractSIMD,N}} = max(x, T(y))
@inline Base.max(
  y::Int,
  x::T
) where {N,T<:ForwardDiff.Dual{<:Any,<:AbstractSIMD,N}} = max(x, T(y))

@inline function Base.min(
  x::ForwardDiff.Dual{TAG,<:AbstractSIMD,N},
  y::ForwardDiff.Dual{TAG,<:AbstractSIMD,N}
) where {TAG,N}
  vx = ForwardDiff.value(x)
  vy = ForwardDiff.value(y)
  xgy = vx < vy
  z = ifelse(xgy, vx, vy)
  p = VectorizationBase.fmap(
    ifelse,
    xgy,
    ForwardDiff.partials(x).values,
    ForwardDiff.partials(y).values
  )
  ForwardDiff.Dual{TAG}(z, ForwardDiff.Partials(p))
end
@inline Base.min(
  x::T,
  y::Real
) where {N,T<:ForwardDiff.Dual{<:Any,<:AbstractSIMD,N}} = min(x, T(y))
@inline Base.min(
  y::Real,
  x::T
) where {N,T<:ForwardDiff.Dual{<:Any,<:AbstractSIMD,N}} = min(x, T(y))
@inline Base.min(
  x::T,
  y::Int
) where {N,T<:ForwardDiff.Dual{<:Any,<:AbstractSIMD,N}} = min(x, T(y))
@inline Base.min(
  y::Int,
  x::T
) where {N,T<:ForwardDiff.Dual{<:Any,<:AbstractSIMD,N}} = min(x, T(y))

@generated function SLEEFPirates.tanh_fast(
  x::ForwardDiff.Dual{T,S,N}
) where {T,S,N}
  quote
    $(Expr(:meta, :inline))
    t = tanh_fast(x.value)
    ∂t = $(VectorizationBase.vfnmadd_fast)(t, t, one(S))
    p = x.partials
    ForwardDiff.Dual{T}(
      t,
      ForwardDiff.Partials(
        Base.Cartesian.@ntuple $N n -> $(Base.FastMath.mul_fast)(∂t, p[n])
      )
    )
  end
end
@generated function SLEEFPirates.sigmoid_fast(
  x::ForwardDiff.Dual{T,S,N}
) where {T,S,N}
  quote
    $(Expr(:meta, :inline))
    s = sigmoid_fast(x.value)
    ∂s = $(VectorizationBase.vfnmadd_fast)(s, s, s)
    p = x.partials
    ForwardDiff.Dual{T}(
      s,
      ForwardDiff.Partials(
        Base.Cartesian.@ntuple $N n -> $(Base.FastMath.mul_fast)(∂s, p[n])
      )
    )
  end
end
@generated function VectorizationBase.relu(
  x::ForwardDiff.Dual{T,S,N}
) where {T,S,N}
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

@generated function ifelse(
  m::AbstractMask,
  x::ForwardDiff.Dual{TAG,V,P},
  y::ForwardDiff.Dual{TAG,V,P}
) where {TAG,V,P}
  quote
    $(Expr(:meta, :inline))
    z = $ifelse(m, ForwardDiff.value(x), ForwardDiff.value(y))
    px = ForwardDiff.partials(x)
    py = ForwardDiff.partials(y)
    p = Base.Cartesian.@ntuple $P p -> $ifelse(m, px[p], py[p])
    ForwardDiff.Dual{$TAG}(z, ForwardDiff.Partials(p))
  end
end
@generated function ifelse(
  m::AbstractMask,
  x::Number,
  y::ForwardDiff.Dual{TAG,V,P}
) where {TAG,V,P}
  quote
    $(Expr(:meta, :inline))
    z = $ifelse(m, x, ForwardDiff.value(y))
    py = ForwardDiff.partials(y)
    p = Base.Cartesian.@ntuple $P p -> $ifelse(m, zero($V), py[p])
    ForwardDiff.Dual{$TAG}(z, ForwardDiff.Partials(p))
  end
end
@generated function ifelse(
  m::AbstractMask,
  x::ForwardDiff.Dual{TAG,V,P},
  y::Number
) where {TAG,V,P}
  quote
    $(Expr(:meta, :inline))
    z = $ifelse(m, ForwardDiff.value(x), y)
    px = ForwardDiff.partials(x)
    p = Base.Cartesian.@ntuple $P p -> $ifelse(m, px[p], zero($V))
    ForwardDiff.Dual{$TAG}(z, ForwardDiff.Partials(p))
  end
end
@inline function SLEEFPirates.softplus(x::ForwardDiff.Dual{TAG}) where {TAG}
  val = ForwardDiff.value(x)
  expx = exp(val)
  vx = log1p(expx)
  px = Base.FastMath.inv_fast(one(val) + Base.FastMath.inv_fast(expx))
  ForwardDiff.Dual{TAG}(vx, Base.FastMath.mul_fast(ForwardDiff.partials(x), px))
end

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
