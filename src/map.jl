

function vmap_quote(N, ::Type{T}) where {T}
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    val = Expr(:call, Expr(:curly, :Val, W))
    q = Expr(:block, Expr(:(=), :M, Expr(:call, :length, :dest)), Expr(:(=), :vdest, Expr(:call, :pointer, :dest)), Expr(:(=), :m, 0))
    fcall = Expr(:call, :f)
    loopbody = Expr(:block, Expr(:call, :vstore!, :vdest, fcall, :m), Expr(:(+=), :m, W))
    fcallmask = Expr(:call, :f)
    bodymask = Expr(:block, Expr(:(=), :__mask__, Expr(:call, :mask, val, Expr(:call, :&, :M, W-1))), Expr(:call, :vstore!, :vdest, fcallmask, :m, :__mask__))
    for n ∈ 1:N
        arg_n = Symbol(:varg_,n)
        push!(q.args, Expr(:(=), arg_n, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__,Symbol(@__FILE__)), Expr(:call, :pointer, Expr(:ref, :args, n)))))
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

function vmapnt!(f::F, y::AbstractVector{T}, args::Vararg{<:Any,A}) where {F,T,A}
    ptry = pointer(y)
    @assert reinterpret(UInt, ptry) & (VectorizationBase.REGISTER_SIZE - 1) == 0
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    ptrargs = pointer.(args)
    V = VectorizationBase.pick_vector_width_val(T)
    N = length(y)
    i = 0
    while i < N - ((W << 2) - 1)
        vstorent!(ptry, extract_data(f(vload.(V, ptrargs, i)...)), i); i += W
        vstorent!(ptry, extract_data(f(vload.(V, ptrargs, i)...)), i); i += W
        vstorent!(ptry, extract_data(f(vload.(V, ptrargs, i)...)), i); i += W
        vstorent!(ptry, extract_data(f(vload.(V, ptrargs, i)...)), i); i += W
    end
    while i < N - (W - 1) # stops at 16 when 
        vstorent!(ptry, extract_data(f(vload.(V, ptrargs, i)...)), i); i += W
    end
    if i < N
        m = mask(T, N & (W - 1))
        vstore!(ptry, extract_data(f(vload.(V, ptrargs, i, m)...)), i, m)
    end
    y
end
function vmapntt!(f::F, y::AbstractVector{T}, args::Vararg{<:Any,A}) where {F,T,A}
    ptry = pointer(y)
    @assert reinterpret(UInt, ptry) & (VectorizationBase.REGISTER_SIZE - 1) == 0
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    ptrargs = pointer.(args)
    V = VectorizationBase.pick_vector_width_val(T)
    N = length(y)
    Wsh = Wshift + 2
    Niter = N >>> Wsh
    Base.Threads.@threads for j ∈ 0:Niter-1
        i = j << Wsh
        vstorent!(ptry, extract_data(f(vload.(V, ptrargs, i)...)), i); i += W
        vstorent!(ptry, extract_data(f(vload.(V, ptrargs, i)...)), i); i += W
        vstorent!(ptry, extract_data(f(vload.(V, ptrargs, i)...)), i); i += W
        vstorent!(ptry, extract_data(f(vload.(V, ptrargs, i)...)), i)
    end
    ii = Niter << Wsh
    while ii < N - (W - 1) # stops at 16 when 
        vstorent!(ptry, extract_data(f(vload.(V, ptrargs, ii)...)), ii); ii += W
    end
    if ii < N
        m = mask(T, N & (W - 1))
        vstore!(ptry, extract_data(f(vload.(V, ptrargs, ii, m)...)), ii, m)
    end
    y
end

function vmap_call(f::F, vm!::V, args::Vararg{<:Any,N}) where {V,F,N}
    T = Base._return_type(f, Base.Broadcast.eltypes(args))
    dest = similar(first(args), T)
    vm!(f, dest, args...)
end
vmap(f::F, args::Vararg{<:Any,N}) where {F,N} = vmap_call(f, vmap!, args...)
vmapnt(f::F, args::Vararg{<:Any,N}) where {F,N} = vmap_call(f, vmapnt!, args...)
vmapntt(f::F, args::Vararg{<:Any,N}) where {F,N} = vmap_call(f, vmapntt!, args...)


# @inline vmap!(f, y, x...) = @avx y .= f.(x...)
# @inline vmap(f, x...) = @avx f.(x...)


