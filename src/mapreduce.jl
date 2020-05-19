
@inline vreduce(::typeof(+), v::VectorizationBase.AbstractSIMDVector) = vsum(v)
@inline vreduce(::typeof(*), v::VectorizationBase.AbstractSIMDVector) = vprod(v)
@inline vreduce(::typeof(max), v::VectorizationBase.AbstractSIMDVector) = vmaximum(v)
@inline vreduce(::typeof(min), v::VectorizationBase.AbstractSIMDVector) = vminimum(v)
@inline vreduce(op, v::VectorizationBase.AbstractSIMDVector) = _vreduce(op, v)
@inline _vreduce(op, v::VectorizationBase.AbstractSIMDVector) = _reduce(op, SVec(v))
@inline function _vreduce(op, v::SVec)
    isone(length(v)) && return v[1]
    a = op(v[1], v[2])
    for i ∈ 3:length(v)
        a = op(a, v[i])
    end
    a
end

function mapreduce_simple(f::F, op::OP, args::Vararg{DenseArray{T},A}) where {F,OP,T<:NativeTypes,A}
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
function vmapreduce(f::F, op::OP, args::Vararg{DenseArray{T},A}) where {F,OP,T<:NativeTypes,A}
    N = length(first(args))
    A > 1 && @assert all(isequal(length.(args)...))
    W = VectorizationBase.pick_vector_width(T)
    V = VectorizationBase.pick_vector_width_val(T)
    N < W && return mapreduce_simple(f, op, args...)
    ptrargs = pointer.(args)
    
    a_0 = f(vload.(V, ptrargs)...); i = W
    if N ≥ 4W
        a_1 = f(vload.(V, gep.(ptrargs, i))...); i += W
        a_2 = f(vload.(V, gep.(ptrargs, i))...); i += W
        a_3 = f(vload.(V, gep.(ptrargs, i))...); i += W
        while i < N - ((W << 2) - 1)
            a_0 = op(a_0, f(vload.(V, gep.(ptrargs, i))...)); i += W
            a_1 = op(a_1, f(vload.(V, gep.(ptrargs, i))...)); i += W
            a_2 = op(a_2, f(vload.(V, gep.(ptrargs, i))...)); i += W
            a_3 = op(a_3, f(vload.(V, gep.(ptrargs, i))...)); i += W
        end
        a_0 = op(a_0, a_1)
        a_2 = op(a_2, a_3)
        a_0 = op(a_0, a_2)
    end
    while i < N - (W - 1)
        a_0 = op(a_0, f(vload.(V, gep.(ptrargs, i))...)); i += W
    end
    if i < N
        m = mask(T, N & (W - 1))
        a_0 = vifelse(m, op(a_0, f(vload.(V, gep.(ptrargs, i))...)), a_0)
    end
    vreduce(op, a_0)
end

@inline vmapreduce(f, op, args...) = mapreduce(f, op, args...)


"""
    vreduce(op, destination, A::DenseArray...)

Vectorized version of `reduce`. Reduces the array `A` using the operator `op`.
"""
@inline vreduce(op, arg) = vmapreduce(identity, op, arg)

