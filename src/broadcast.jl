struct Product{A,B}
    a::A
    b::B
end
@inline function Base.size(p::Product)
    M = @inbounds size(p.a)[1]
    (M, Base.tail(size(p.b))...)
end
@inline function Base.size(p::Product, i::Integer)
    i == 1 && return @inbounds size(p.a)[1]
    @inbounds size(p.b)[i]
end
@inline Base.length(p::Product) = prod(size(p))
@inline Base.broadcastable(p::Product) = p
@inline numdims(A) = ndims(A) # fallback
@inline numdims(::Type{Product{A,B}}) where {A,B} = numdims(B)
@inline Base.ndims(::Type{Product{A,B}}) where {A,B} = numdims(B)
# This numdims nonsense is a hack to avoid type piracy in defining:
@inline numdims(::Type{B}) where {N, S <: Base.Broadcast.AbstractArrayStyle{N}, B <: Base.Broadcast.Broadcasted{S}} = N

Base.Broadcast._broadcast_getindex_eltype(::Product{A,B}) where {T, A <: AbstractVecOrMat{T}, B <: AbstractVecOrMat{T}} = T
function Base.Broadcast._broadcast_getindex_eltype(p::Product)
    promote_type(
        Base.Broadcast._broadcast_getindex_eltype(p.a),
        Base.Broadcast._broadcast_getindex_eltype(p.b)
    )
end

# recursive_eltype(::Type{A}) where {T, A <: AbstractArray{T}} = T
# recursive_eltype(::Type{NTuple{N,T}}) where {N,T<:Union{Float32,Float64}} = T
# recursive_eltype(::Type{Float32}) = Float32
# recursive_eltype(::Type{Float64}) = Float64
# recursive_eltype(::Type{Tuple{T}}) where {T} = T
# recursive_eltype(::Type{Tuple{T1,T2}}) where {T1,T2} = promote_type(recursive_eltype(T1), recursive_eltype(T2))
# recursive_eltype(::Type{Tuple{T1,T2,T3}}) where {T1,T2,T3} = promote_type(recursive_eltype(T1), recursive_eltype(T2), recursive_eltype(T3))
# recursive_eltype(::Type{Tuple{T1,T2,T3,T4}}) where {T1,T2,T3,T4} = promote_type(recursive_eltype(T1), recursive_eltype(T2), recursive_eltype(T3), recursive_eltype(T4))
# recursive_eltype(::Type{Tuple{T1,T2,T3,T4,T5}}) where {T1,T2,T3,T4,T5} = promote_type(recursive_eltype(T1), recursive_eltype(T2), recursive_eltype(T3), recursive_eltype(T4), recursive_eltype(T5))

# function recursive_eltype(::Type{Broadcasted{S,A,F,ARGS}}) where {S,A,F,ARGS}
#     recursive_eltype(ARGS)
# end

"""
    A *ˡ B

A lazy product of `A` and `B`. While functionally identical to `A * B`, this may avoid the
need for intermediate storage for any computations in `A` or `B`.  Example:

    @avx @. a + B *ˡ (c + d')

which is equivalent to

     a .+ B * (c .+ d')

It should only be used inside an `@avx` block, and to materialize the result it cannot be
the final operation.
"""
@inline *ˡ(a::A, b::B) where {A,B} = Product{A,B}(a, b)
@inline Base.Broadcast.broadcasted(::typeof(*ˡ), a::A, b::B) where {A, B} = Product{A,B}(a, b)
# TODO: Need to make this handle A or B being (1 or 2)-D broadcast objects.
function add_broadcast!(
    ls::LoopSet, mC::Symbol, bcname::Symbol, loopsyms::Vector{Symbol},
    @nospecialize(prod::Type{<:Product}), elementbytes::Int
)
    A, B = prod.parameters
    K = gensym(:K)
    mA = gensym(:Aₘₖ)
    mB = gensym(:Bₖₙ)
    pushprepreamble!(ls, Expr(:(=), mA, Expr(:(.), bcname, QuoteNode(:a))))
    pushprepreamble!(ls, Expr(:(=), mB, Expr(:(.), bcname, QuoteNode(:b))))
    pushprepreamble!(ls, Expr(:(=), K, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__,Symbol(@__FILE__)), Expr(:ref, Expr(:call, :size, mB), 1))))
    k = gensym(:k)
    add_loop!(ls, Loop(k, 1, K), k)
    m = loopsyms[1];
    if numdims(B) == 1
        bloopsyms = Symbol[k]
        cloopsyms = Symbol[m]
        reductdeps = Symbol[m, k]
        kvec = bloopsyms
    elseif numdims(B) == 2
        n = loopsyms[2];
        bloopsyms = Symbol[k,n]
        cloopsyms = Symbol[m,n]
        reductdeps = Symbol[m, k, n]
        kvec = Symbol[k]
    else
        throw("B must be a vector or matrix.")
    end
    # load A
    # loadA = add_load!(ls, gensym(:A), productref(A, mA, m, k), elementbytes)
    loadA = add_broadcast!(ls, gensym(:A), mA, Symbol[m,k], A, elementbytes)
    # load B
    loadB = add_broadcast!(ls, gensym(:B), mB, bloopsyms, B, elementbytes)
    # set Cₘₙ = 0
    # setC = add_constant!(ls, zero(promote_type(recursive_eltype(A), recursive_eltype(B))), cloopsyms, mC, elementbytes)
    # targetC will be used for reduce_to_add
    mCt = gensym(mC)
    targetC = add_constant!(ls, gensym(:zero), cloopsyms, mCt, elementbytes, :numericconstant)
    push!(ls.preamble_zeros, (identifier(targetC), IntOrFloat))
    setC = add_constant!(ls, gensym(:zero), cloopsyms, mC, elementbytes, :numericconstant)
    push!(ls.preamble_zeros, (identifier(setC), IntOrFloat))
    setC.reduced_children = kvec
    # compute Cₘₙ += Aₘₖ * Bₖₙ
    reductop = Operation(
        ls, mC, elementbytes, :vfmadd231, compute, reductdeps, kvec, Operation[loadA, loadB, setC]
    )
    reductop = pushop!(ls, reductop, mC)
    reductfinal = Operation(
        ls, mCt, elementbytes, :reduce_to_add, compute, cloopsyms, kvec, Operation[reductop, targetC]
    )
    pushop!(ls, reductfinal, mCt)
end

struct LowDimArray{D,T,N,A<:DenseArray{T,N}} <: DenseArray{T,N}
    data::A
end
@inline Base.pointer(A::LowDimArray) = pointer(A.data)
Base.@propagate_inbounds Base.getindex(A::LowDimArray, i...) = getindex(A.data, i...)
@inline Base.size(A::LowDimArray) = Base.size(A.data)
@inline Base.size(A::LowDimArray, i) = Base.size(A.data, i)
@generated function VectorizationBase.stridedpointer(A::LowDimArray{D,T,N}) where {D,T,N}
    smul = Expr(:(.), Expr(:(.), :LoopVectorization, QuoteNode(:VectorizationBase)), QuoteNode(:staticmul))
    multup = Expr(:tuple)
    for n ∈ D[1]+1:N
        if length(D) < n
            push!(multup.args, Expr(:call, :ifelse, :(isone(size(A,$n))), 0, Expr(:ref, :strideA, n)))
        elseif D[n]
            push!(multup.args, Expr(:ref, :strideA, n))
        end
    end
    s = Expr(:call, smul, T, multup)
    f = D[1] ? :PackedStridedPointer : :SparseStridedPointer
    Expr(:block, Expr(:meta,:inline), Expr(:(=), :strideA, Expr(:call, :strides, Expr(:(.), :A, QuoteNode(:data)))),
         Expr(:call, Expr(:(.), :VectorizationBase, QuoteNode(f)), Expr(:call, :pointer, :A), s))
end
function LowDimArray{D}(data::A) where {D,T,N,A <: AbstractArray{T,N}}
    LowDimArray{D,T,N,A}(data)
end
function extract_all_1_array!(ls::LoopSet, bcname::Symbol, N::Int, elementbytes::Int)
    refextract = gensym(bcname)
    ref = Expr(:ref, bcname); append!(ref.args, [1 for n ∈ 1:N])
    pushprepreamble!(ls, Expr(:(=), refextract, ref))
    return add_constant!(ls, refextract, elementbytes) # or replace elementbytes with sizeof(T) ? u
end
function add_broadcast!(
    ls::LoopSet, destname::Symbol, bcname::Symbol, loopsyms::Vector{Symbol},
    @nospecialize(LDA::Type{<:LowDimArray}), elementbytes::Int
)
    D,T,N::Int,_ = LDA.parameters
    Dlen = length(D)
    if Dlen == N && !any(D)
        return extract_all_1_array!(ls, bcname, N, elementbytes)
    end
    fulldims = Symbol[loopsyms[n] for n ∈ 1:N if ((Dlen < n) || D[n]::Bool)]
    ref = ArrayReference(bcname, fulldims)
    add_simple_load!(ls, destname, ref, elementbytes, true, false )::Operation
end
function add_broadcast_adjoint_array!(
    ls::LoopSet, destname::Symbol, bcname::Symbol, loopsyms::Vector{Symbol}, ::Type{A}, elementbytes::Int
) where {T,N,A<:AbstractArray{T,N}}
    parent = gensym(:parent)
    pushprepreamble!(ls, Expr(:(=), parent, Expr(:call, :parent, bcname)))
    # isone(length(loopsyms)) && return extract_all_1_array!(ls, bcname, N, elementbytes)
    ref = ArrayReference(parent, Symbol[loopsyms[N + 1 - n] for n ∈ 1:N])
    add_simple_load!( ls, destname, ref, elementbytes, true, true )::Operation
end
function add_broadcast_adjoint_array!(
    ls::LoopSet, destname::Symbol, bcname::Symbol, loopsyms::Vector{Symbol}, ::Type{<:AbstractVector}, elementbytes::Int
)
    # isone(length(loopsyms)) && return extract_all_1_array!(ls, bcname, N, elementbytes)
    ref = ArrayReference(bcname, Symbol[loopsyms[2]])
    add_simple_load!( ls, destname, ref, elementbytes, true, true )
end
function add_broadcast!(
    ls::LoopSet, destname::Symbol, bcname::Symbol, loopsyms::Vector{Symbol},
    ::Type{Adjoint{T,A}}, elementbytes::Int
) where {T, A <: AbstractArray{T}}
    add_broadcast_adjoint_array!( ls, destname, bcname, loopsyms, A, elementbytes )
end
function add_broadcast!(
    ls::LoopSet, destname::Symbol, bcname::Symbol, loopsyms::Vector{Symbol},
    ::Type{Transpose{T,A}}, elementbytes::Int
) where {T, A <: AbstractArray{T}}
    add_broadcast_adjoint_array!( ls, destname, bcname, loopsyms, A, elementbytes )
end
function add_broadcast!(
    ls::LoopSet, destname::Symbol, bcname::Symbol, loopsyms::Vector{Symbol},
    ::Type{<:AbstractArray{T,N}}, elementbytes::Int
) where {T,N}
    add_simple_load!(ls, destname, ArrayReference(bcname, @view(loopsyms[1:N])), elementbytes, true, true)
end
function add_broadcast!(
    ls::LoopSet, ::Symbol, bcname::Symbol, loopsyms::Vector{Symbol}, ::Type{T}, elementbytes::Int
) where {T<:Number}
    add_constant!(ls, bcname, elementbytes) # or replace elementbytes with sizeof(T) ? u
end
function add_broadcast!(
    ls::LoopSet, ::Symbol, bcname::Symbol, loopsyms::Vector{Symbol}, ::Type{Base.RefValue{T}}, elementbytes::Int
) where {T}
    refextract = gensym(bcname)
    pushprepreamble!(ls, Expr(:(=), refextract, Expr(:ref, bcname)))
    add_constant!(ls, refextract, elementbytes) # or replace elementbytes with sizeof(T) ? u
end
function add_broadcast!(
    ls::LoopSet, destname::Symbol, bcname::Symbol, loopsyms::Vector{Symbol},
    @nospecialize(_::Type{SubArray{T,N,A,S,B}}), elementbytes::Int
) where {T,N,N2,A<:AbstractArray{T,N2},B,N3,S <: Tuple{Int,Vararg{Any,N3}}}
    inds = Vector{Symbol}(undef, N+1)
    inds[1] = Symbol("##DISCONTIGUOUSSUBARRAY##")
    inds[2:end] .= @view(loopsyms[1:N])
    add_simple_load!(ls, destname, ArrayReference(bcname, inds), elementbytes, true, true)
end
const BroadcastedArray{S<:Broadcast.AbstractArrayStyle,F,A} = Broadcasted{S,Nothing,F,A}
function add_broadcast!(
    ls::LoopSet, destname::Symbol, bcname::Symbol, loopsyms::Vector{Symbol},
    @nospecialize(B::Type{<:BroadcastedArray}),
    elementbytes::Int
)
    S,_,F,A = B.parameters
    instr = get(FUNCTIONSYMBOLS, F) do
        f = gensym(:func)
        pushprepreamble!(ls, Expr(:(=), f, Expr(:(.), bcname, QuoteNode(:f))))
        Instruction(bcname, f)
    end
    args = A.parameters
    Nargs = length(args)
    bcargs = Expr(:(.), bcname, QuoteNode(:args))
    # this is the var name in the loop
    parents = Operation[]
    deps = Symbol[]
    # reduceddeps = Symbol[]
    for (i,arg) ∈ enumerate(args)
        argname = gensym(:arg)
        pushprepreamble!(ls, Expr(:(=), argname, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__,Symbol(@__FILE__)), Expr(:ref, bcargs, i))))
        # dynamic dispatch
        parent = add_broadcast!(ls, gensym(:temp), argname, loopsyms, arg, elementbytes)::Operation
        push!(parents, parent)
        mergesetdiffv!(deps, loopdependencies(parent), reduceddependencies(parent))
    end
    op = Operation(
        length(operations(ls)), destname, elementbytes, instr, compute, deps, NOPARENTS, parents
    )
    pushop!(ls, op, destname)
end

# size of dest determines loops
# function vmaterialize!(
@generated function vmaterialize!(
    dest::StridedArray{T,N}, bc::BC, ::Val{Mod}
) where {T <: NativeTypes, N, BC <: Union{Broadcasted,Product}, Mod}
    # we have an N dimensional loop.
    # need to construct the LoopSet
    # @show typeof(dest)
    loopsyms = [gensym(:n) for n ∈ 1:N]
    ls = LoopSet(Mod)
    ls.isbroadcast[] = true
    sizes = Expr(:tuple)
    for (n,itersym) ∈ enumerate(loopsyms)
        Nsym = gensym(:N)
        add_loop!(ls, Loop(itersym, 1, Nsym), itersym)
        push!(sizes.args, Nsym)
    end
    pushpreamble!(ls, Expr(:(=), sizes, Expr(:call, :size, :dest)))
    elementbytes = sizeof(T)
    add_broadcast!(ls, :dest, :bc, loopsyms, BC, elementbytes)
    add_simple_store!(ls, :dest, ArrayReference(:dest, loopsyms), elementbytes)
    resize!(ls.loop_order, num_loops(ls)) # num_loops may be greater than N, eg Product
    # return ls
    q = lower(ls)
    push!(q.args, :dest)
    # @show q
    # q
    q = Expr(:block, ls.prepreamble, Expr(:if, check_args_call(ls), q, :(Base.Broadcast.materialize!(dest, bc))))
    isone(N) && pushfirst!(q.args, Expr(:meta,:inline))
    q
     # ls
end
@generated function vmaterialize!(
    dest′::Union{Adjoint{T,A},Transpose{T,A}}, bc::BC, ::Val{Mod}
) where {T <: NativeTypes, N, A <: StridedArray{T,N}, BC <: Union{Broadcasted,Product}, Mod}
    # we have an N dimensional loop.
    # need to construct the LoopSet
    loopsyms = [gensym(:n) for n ∈ 1:N]
    ls = LoopSet(Mod)
    ls.isbroadcast[] = true
    pushprepreamble!(ls, Expr(:(=), :dest, Expr(:call, :parent, :dest′)))
    sizes = Expr(:tuple)
    for (n,itersym) ∈ enumerate(loopsyms)
        Nsym = gensym(:N)
        add_loop!(ls, Loop(itersym, 1, Nsym), itersym)
        push!(sizes.args, Nsym)
    end
    pushpreamble!(ls, Expr(:(=), sizes, Expr(:call, :size, :dest′)))
    elementbytes = sizeof(T)
    add_broadcast!(ls, :dest, :bc, loopsyms, BC, elementbytes)
    add_simple_store!(ls, :dest, ArrayReference(:dest, reverse(loopsyms)), elementbytes)
    resize!(ls.loop_order, num_loops(ls)) # num_loops may be greater than N, eg Product
    q = lower(ls)
    push!(q.args, :dest′)
    q = Expr(:block, ls.prepreamble, Expr(:if, check_args_call(ls), q, :(Base.Broadcast.materialize!(dest′, bc))))
    isone(N) && pushfirst!(q.args, Expr(:meta,:inline))
    q
    # ls
end
function vmaterialize!(
    dest::StridedArray{T,N}, bc::Broadcasted{Base.Broadcast.DefaultArrayStyle{0},Nothing,typeof(identity),Tuple{T2}}, ::Val{Mod}
) where {T <: NativeTypes, N, T2 <: Number, Mod}
    arg = T(first(bc.args))
    @avx for i ∈ eachindex(dest)
        dest[i] = arg
    end
    dest
end
function vmaterialize!(
    dest′::Union{Adjoint{T,A},Transpose{T,A}}, bc::Broadcasted{Base.Broadcast.DefaultArrayStyle{0},Nothing,typeof(identity),Tuple{T2}}, ::Val{Mod}
) where {T <: NativeTypes, N, A <: StridedArray{T,N}, T2 <: Number, Mod}
    arg = T(first(bc.args))
    dest = parent(dest′)
    @avx for i ∈ eachindex(dest)
        dest[i] = arg
    end
    dest′
end

@inline function vmaterialize(bc::Broadcasted, ::Val{Mod}) where {Mod}
    ElType = Base.Broadcast.combine_eltypes(bc.f, bc.args)
    vmaterialize!(similar(bc, ElType), bc, Val{Mod}())
end

vmaterialize!(dest, bc, ::Val{mod}) where {mod} = Base.Broadcast.materialize!(dest, bc)

