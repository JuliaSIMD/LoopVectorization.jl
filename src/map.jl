

function vmap_quote(N, ::Type{T}) where {T}
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    val = Expr(:call, Expr(:curly, :Val, W))
    q = Expr(:block, Expr(:(=), :M, Expr(:call, :length, :dest)), Expr(:(=), :vdest, Expr(:call, :vectorizable, :dest)), Expr(:(=), :m, 0))
    fcall = Expr(:call, :f)
    loopbody = Expr(:block, Expr(:call, :vstore!, :vdest, fcall, :m), Expr(:(+=), :m, W))
    fcallmask = Expr(:call, :f)
    bodymask = Expr(:block, Expr(:(=), :__mask__, Expr(:call, :mask, val, Expr(:call, :&, :M, W-1))), Expr(:call, :vstore!, :vdest, fcallmask, :m, :__mask__))
    for n ∈ 1:N
        arg_n = Symbol(:varg_,n)
        push!(q.args, Expr(:(=), arg_n, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__,@__FILE__), Expr(:call, :vectorizable, Expr(:ref, :args, n)))))
        push!(fcall.args, Expr(:call, :vload, val, arg_n, :m))
        push!(fcallmask.args, Expr(:call, :vload, val, arg_n, :m, :__mask__))
    end
    loop = Expr(:for, Expr(:(=), :_, Expr(:call, :(:), 0, Expr(:call, :-, Expr(:call, :(>>>), :M, Wshift), 1))), loopbody)
    push!(q.args, loop)
    ifmask = Expr(:if, Expr(:call, :(!=), :m, :M), bodymask)
    push!(q.args, ifmask)
    push!(q.args, :dest)
    q
end
@generated function vmap!(f::F, dest::AbstractArray{T}, args::Vararg{<:AbstractArray,N}) where {F,T,N}
    vmap_quote(N, T)
end
function vmap(f::F, args...) where {F}
    T = Base._return_type(f, Base.Broadcast.eltypes(args))
    dest = similar(first(args), T)
    vmap!(f, dest, args...)
end


