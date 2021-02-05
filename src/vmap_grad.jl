
using VectorizationBase: AbstractSIMD
using .ForwardDiff
using .ChainRulesCore

@generated function SLEEFPirates.tanh_fast(x::ForwardDiff.Dual{T,S,N}) where {T,S,N}
    quote
        $(Expr(:meta,:inline))
        t = tanh_fast(x.value)
        ∂t = vfnmadd_fast(t, t, one(S))
        p = x.partials
        ForwardDiff.Dual(t, ForwardDiff.Partials(Base.Cartesian.@ntuple $N n -> mul_fast(∂t, p[n])))
    end
end
function ChainRulesCore.rrule(::typeof(tanh_fast), x)
    t = tanh_fast(x)
    ∂ = let t = t
        y -> (ChainRulesCore.Zero(), mul_fast(vfnmadd_fast(t, t, one(t)), y))
    end
    t, ∂
end
@generated function SLEEFPirates.sigmoid_fast(x::ForwardDiff.Dual{T,S,N}) where {T,S,N}
    quote
        $(Expr(:meta,:inline))
        s = sigmoid_fast(x.value)
        ∂s = vfnmadd_fast(s,s,s)
        p = x.partials
        ForwardDiff.Dual(s, ForwardDiff.Partials(Base.Cartesian.@ntuple $N n -> mul_fast(∂s, p[n])))
    end
end
function ChainRulesCore.rrule(::typeof(sigmoid_fast), x)
    s = sigmoid_fast(x)
    ∂ = let s = s
        y -> (ChainRulesCore.Zero(), mul_fast(vfnmadd_fast(s, s, s), y))
    end
    s, ∂
end
@generated function VectorizationBase.relu(x::ForwardDiff.Dual{T,S,N}) where {T,S,N}
    quote
        $(Expr(:meta,:inline))
        v = x.value
        z = zero(v)
        cmp = v < z
        r = ifelse(cmp, z, v)
        p = x.partials
        ForwardDiff.Dual(r, ForwardDiff.Partials(Base.Cartesian.@ntuple $N n -> ifelse(cmp, z, p[n])))
    end
end
function ChainRulesCore.rrule(::typeof(relu), v)
    z = zero(v)
    cmp = v < z
    r = ifelse(cmp, z, v)
    ∂ = let cmp = cmp
        y -> (ChainRulesCore.Zero(), ifelse(cmp, zero(y), y))
    end
    r, ∂
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

function ∂vmap_singlethread!(
    f::F, ∂y::Tuple{Vararg{DenseArray{T},A}}, y::DenseArray{T},
    args::Vararg{<:DenseArray{<:Base.HWReal},A}
) where {F,T <: Base.HWReal, A}
    N = length(y)
    ptry = VectorizationBase.zero_offsets(stridedpointer(y))
    ptrargs = VectorizationBase.zero_offsets.(stridedpointer.(args))
    ptr∂y = VectorizationBase.zero_offsets.(stridedpointer.(∂y))
    
    i = 0
    V = VectorizationBase.pick_vector_width(T)
    W = Int(V)
    st = VectorizationBase.static_sizeof(T)
    zero_index = MM{W}(StaticInt(0), st)
    while i < N - ((W << 2) - 1)
        index = VectorizationBase.Unroll{1,W,4,1,W,0x0000000000000000}((i,))
        v = f(init_dual(vload.(ptrargs, index))...)
        dual_store!(ptr∂y, ptry, v, index)
        i = vadd_fast(i, 4W)
    end
    while i < N - (W - 1) 
        vᵣ = f(init_dual(vload.(ptrargs, ((MM{W}(i),),)))...)
        dual_store!(ptr∂y, ptry, vᵣ, (MM{W}(i),))
        i = vadd_fast(i, W)
    end
    if i < N
        m = mask(T, N & (W - 1))
        dual_store!(ptr∂y, ptry, f(init_dual(vload.(ptrargs, ((MM{W}(i),),), m))...), (MM{W}(i,),), m)
    end
    nothing
end


struct SIMDMapBack{K,T<:Tuple{Vararg{Any,K}}}
    jacs::T
end
@generated function (b::SIMDMapBack{K,T})(Δ::A) where {K,T,A}
    preloop = Expr(:block, :(jacs = b.jacs))
    loop_body = Expr(:block, :(Δᵢ = Δ[i]))
    ret = Expr(:tuple, ChainRulesCore.Zero(), ChainRulesCore.Zero())
    for k ∈1:K
        jₖ = Symbol(:j_, k)
        push!(preloop.args, :($jₖ = jacs[$k]))
        push!(loop_body.args, :($jₖ[i] *= Δᵢ))
        push!(ret.args, jₖ)
    end
    quote
        $preloop
        @avx for i ∈ eachindex(Δ)
            $loop_body
        end
        $ret
    end
end

function ChainRulesCore.rrule(::typeof(vmap), f::F, args::Vararg{Any,K}) where {F,K}
    out = similar(first(args))
    jacs = map(similar, args)
    ∂vmap_singlethread!(f, jacs, out, args...)
    out, SIMDMapBack(jacs)
end


