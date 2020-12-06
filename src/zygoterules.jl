
import ZygoteRules # ChainRules doesn't yet allow calling AD inside a rule, like _pullback

ZygoteRules.@adjoint function vmap(f, args::AbstractArray...)
    ∇vmap(__context__, f, args...)
end

function ∇vmap(cx, f, args...)
    ys_and_backs = vmap((xs...) -> ZygoteRules._pullback(cx, f, xs...), args...)
    if isempty(ys_and_backs)
        println("got empty case!")
        return ys_and_backs, _ -> nothing
    else
        ys, backs = _zygote_unzip(ys_and_backs)
        function back(Δ)
            # Δf_and_args = _zygote_unzip(vmap((b, δ) -> b(δ), backs, Δ)) # MethodError: no method matching map!(::LoopVectorization.var"#601#606", ::Array{Tuple{Nothing,Float64},1}, ::Array{Zygote.var"#1848#back#217"{Zygote.var"#215#216"{Float64}},1}, ::Tuple{Float64,Float64,Float64})
            Δf_and_args = _zygote_unzip(map((b, δ) -> b(δ), backs, Δ))
            Δf = reduce(_zygote_accum, Δf_and_args[1])
            (Δf, Δf_and_args[2:end]...)
        end
        back(::Nothing) = map(_ -> nothing, ys_and_backs)
        return ys, back
    end
end

# unzip -- from src/lib/array.jl

struct _zygote_StaticGetter{i} end

(::_zygote_StaticGetter{i})(v) where {i} = v[i]

function _zygote_unzip(tuples)
    N = length(first(tuples))
    _zygote__unzip(tuples, Val(N))
end

@generated function _zygote__unzip(tuples, ::Val{N}) where {N}
    Expr(:tuple, [:(map($(_zygote_StaticGetter{i}()), tuples)) for i in 1:N]...)
end

# accum -- from src/lib/lib.jl

_zygote_accum() = nothing
_zygote_accum(x) = x

_zygote_accum(x, y) =
  x === nothing ? y :
  y === nothing ? x :
  x + y

_zygote_accum(x, y, zs...) = _zygote_accum(_zygote_accum(x, y), zs...)

_zygote_accum(x::Tuple, y::Tuple) = _zygote_accum.(x, y)
_zygote_accum(x::AbstractArray, y::AbstractArray) = _zygote_accum.(x, y)

@generated function _zygote_accum(x::NamedTuple, y::NamedTuple)
  grad(x) = x in fieldnames(y) ? :(y.$x) : :nothing
  Expr(:tuple, [:($f=_zygote_accum(x.$f, $(grad(f)))) for f in fieldnames(x)]...)
end

function _zygote_accum(x::Base.RefValue, y::Base.RefValue)
  @assert x === y
  return x
end
