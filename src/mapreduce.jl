
@inline vreduce(::typeof(+), v::VectorizationBase.AbstractSIMDVector) = vsum(v)
@inline vreduce(::typeof(*), v::VectorizationBase.AbstractSIMDVector) = vprod(v)
@inline vreduce(::typeof(max), v::VectorizationBase.AbstractSIMDVector) = vmaximum(v)
@inline vreduce(::typeof(min), v::VectorizationBase.AbstractSIMDVector) = vminimum(v)
@inline vreduce(op, v::VectorizationBase.AbstractSIMDVector) = _vreduce(op, v)
@inline _vreduce(op, v::VectorizationBase.AbstractSIMDVector) = _reduce(op, Vec(v))
@inline function _vreduce(op, v::Vec)
    isone(length(v)) && return v[1]
    a = op(v[1], v[2])
    for i ∈ 3:length(v)
        a = op(a, v[i])
    end
    a
end

function mapreduce_simple(f::F, op::OP, args::Vararg{DenseArray{<:NativeTypes},A}) where {F,OP,A}
    ptrargs = ntuple(a -> pointer(args[a]), Val(A))
    N = length(first(args))
    iszero(N) && throw("Length of vector is 0!")
    a_0 = f(vload.(ptrargs)...); i = 1
    while i < N
        a_0 = op(a_0, f(vload.(gep.(ptrargs, i))...)); i += 1
    end
    a_0
end


"""
    vmapreduce(f, op, A::DenseArray...)

Vectorized version of `mapreduce`. Applies `f` to each element of the arrays `A`, and reduces the result with `op`.
"""
function vmapreduce(f::F, op::OP, arg1::DenseArray{T}, args::Vararg{DenseArray{T},A}) where {F,OP,T<:NativeTypes,A}
    N = length(arg1)
    iszero(A) || @assert all(length.(args) .== N)
    W = VectorizationBase.pick_vector_width(T)
    V = VectorizationBase.pick_vector_width_val(T)
    if N < W
        mapreduce_simple(f, op, arg1, args...)
    else
        _vmapreduce(f, op, V, N, T, arg1, args...)
    end
end
function _vmapreduce(f::F, op::OP, ::StaticInt{W}, N, ::Type{T}, args::Vararg{DenseArray{<:NativeTypes},A}) where {F,OP,A,W,T}
    ptrargs = pointer.(args)
    a_0 = f(vload.(Val{W}(), ptrargs)...); i = W
    if N ≥ 4W
        a_1 = f(vload.(Val{W}(), gep.(ptrargs, i))...); i += W
        a_2 = f(vload.(Val{W}(), gep.(ptrargs, i))...); i += W
        a_3 = f(vload.(Val{W}(), gep.(ptrargs, i))...); i += W
        while i < N - ((W << 2) - 1)
            a_0 = op(a_0, f(vload.(Val{W}(), gep.(ptrargs, i))...)); i += W
            a_1 = op(a_1, f(vload.(Val{W}(), gep.(ptrargs, i))...)); i += W
            a_2 = op(a_2, f(vload.(Val{W}(), gep.(ptrargs, i))...)); i += W
            a_3 = op(a_3, f(vload.(Val{W}(), gep.(ptrargs, i))...)); i += W
        end
        a_0 = op(a_0, a_1)
        a_2 = op(a_2, a_3)
        a_0 = op(a_0, a_2)
    end
    while i < N - (W - 1)
        a_0 = op(a_0, f(vload.(Val{W}(), gep.(ptrargs, i))...)); i += W
    end
    if i < N
        m = mask(T, N & (W - 1))
        a_0 = ifelse(m, op(a_0, f(vload.(Val{W}(), gep.(ptrargs, i))...)), a_0)
    end
    vreduce(op, a_0)
end

@inline vmapreduce(f, op, args...) = mapreduce(f, op, args...)

length_one_axis(::Base.OneTo) = Base.OneTo(1)
length_one_axis(::Any) = 1:1

"""
    vreduce(op, destination, A::DenseArray...)

Vectorized version of `reduce`. Reduces the array `A` using the operator `op`.
"""
@inline vreduce(op, arg) = vmapreduce(identity, op, arg)

for (op, init) in zip((:+, :max, :min), (:zero, :typemin, :typemax))
    @eval function vreduce(::typeof($op), arg; dims = nothing)
        isnothing(dims) && return _vreduce($op, arg)
        isone(ndims(arg)) && return [_vreduce($op, arg)]
        @assert length(dims) == 1
        axes_arg = axes(arg)
        axes_out = Base.setindex(axes_arg, length_one_axis(axes_arg[dims]), dims)
        out = similar(arg, axes_out)
        # fill!(out, $init(first(arg)))
        # TODO: generated function with Base.Cartesian.@nif to set to ndim(arg)
        Base.Cartesian.@nif 5 d -> (d <= ndims(arg) && dims == d) d -> begin
            Rpre = CartesianIndices(ntuple(i -> axes_arg[i], d-1))
            Rpost = CartesianIndices(ntuple(i -> axes_arg[i+d], ndims(arg) - d))
            _vreduce_dims!(out, $op, Rpre, 1:size(arg, dims), Rpost, arg)
        end d -> begin
            Rpre = CartesianIndices(axes_arg[1:dims-1])
            Rpost = CartesianIndices(axes_arg[dims+1:end])
            _vreduce_dims!(out, $op, Rpre, 1:size(arg, dims), Rpost, arg)
        end
    end

    @eval function _vreduce_dims!(out, ::typeof($op), Rpre, is, Rpost, arg)
        s = $init(first(arg))
        @avx for Ipost in Rpost, Ipre in Rpre
            accum = s
            for i in is
                accum = $op(accum, arg[Ipre, i, Ipost])
            end
            out[Ipre, 1, Ipost] = accum
        end
        return out
    end

    @eval function _vreduce(::typeof($op), arg)
        s = $init(first(arg))
        @avx for i in eachindex(arg)
            s = $op(s, arg[i])
        end
        return s
    end
end
