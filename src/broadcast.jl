
@inline stridedpointer_for_broadcast(A) = stridedpointer_for_broadcast(ArrayInterface.size(A), stridedpointer(A))
@inline stridedpointer_for_broadcast(s, ptr) = ptr
# function stridedpointer_for_broadcast(s, ptr::VectorizationBase.AbstractStridedPointer)
#      # FIXME: this is unsafe for AbstractStridedPointers
#     throw("Broadcasting not currently supported for arrays where typeof(stridedpointer(A)) === $(typeof(ptr))")
# end
function stridedpointer_for_broadcast_quote(typ, N, S, X)
    q = Expr(:block, Expr(:meta,:inline), :(strd = ptr.strd))
    strd_tup = Expr(:tuple)
    for n ∈ 1:N
        s_type = S[n]
        if s_type <: Static
            if s_type === Static{1}
                push!(strd_tup.args, Expr(:call, lv(:Zero)))
            else
                push!(strd_tup.args, :(strd[$n]))
            end
        else
            Xₙ_type = X[n]
            if Xₙ_type <: Static # FIXME; what to do here? Dynamic dispatch? 
                push!(strd_tup.args, :(strd[$n]))
            else
                push!(strd_tup.args, :(Base.ifelse(isone(s[$n]), zero($Xₙ_type), strd[$n])))
            end
        end
    end
    push!(q.args, :(@inbounds $typ(ptr.p, $strd_tup, ptr.offsets)))
    q
end
@generated function stridedpointer_for_broadcast(s::Tuple{Vararg{Any,N}}, ptr::StridedPointer{T,N,C,B,R,X,O}) where {T,N,C,B,R,X,O}
    typ = Expr(:curly, :StridedPointer, T, N, C, B, R)
    stridedpointer_for_broadcast_quote(typ, N, s.parameters, X.parameters)
end
@generated function stridedpointer_for_broadcast(s::Tuple{Vararg{Any,N}}, ptr::VectorizationBase.StridedBitPointer{N,C,B,R,X,O}) where {N,C,B,R,X,O}
    typ = Expr(:curly, lv(:StridedBitPointer), N, C, B, R)
    stridedpointer_for_broadcast_quote(typ, N, s.parameters, X.parameters)
end

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
    Krange = gensym!(ls, "K")
    Klen = gensym!(ls, "K")
    mA = gensym!(ls, "Aₘₖ")
    mB = gensym!(ls, "Bₖₙ")
    pushprepreamble!(ls, Expr(:(=), mA, Expr(:(.), bcname, QuoteNode(:a))))
    pushprepreamble!(ls, Expr(:(=), mB, Expr(:(.), bcname, QuoteNode(:b))))
    pushprepreamble!(ls, Expr(:(=), Klen, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__,Symbol(@__FILE__)), Expr(:ref, Expr(:call, :size, mB), 1))))
    pushpreamble!(ls, Expr(:(=), Krange, Expr(:call, :(:), staticexpr(1), Klen)))
    k = gensym!(ls, "k")
    add_loop!(ls, Loop(k, 1, Klen, Krange, Klen), k)
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
    # loadA = add_load!(ls, gensym!(ls, :A), productref(A, mA, m, k), elementbytes)
    loadA = add_broadcast!(ls, gensym!(ls, "A"), mA, Symbol[m,k], A, elementbytes)
    # load B
    loadB = add_broadcast!(ls, gensym!(ls, "B"), mB, bloopsyms, B, elementbytes)
    # set Cₘₙ = 0
    # setC = add_constant!(ls, zero(promote_type(recursive_eltype(A), recursive_eltype(B))), cloopsyms, mC, elementbytes)
    # targetC will be used for reduce_to_add
    mCt = gensym!(ls, mC)
    targetC = add_constant!(ls, gensym!(ls, "zero"), cloopsyms, mCt, elementbytes, :numericconstant)
    push!(ls.preamble_zeros, (identifier(targetC), IntOrFloat))
    setC = add_constant!(ls, gensym!(ls, "zero"), cloopsyms, mC, elementbytes, :numericconstant)
    push!(ls.preamble_zeros, (identifier(setC), IntOrFloat))
    setC.reduced_children = kvec
    # compute Cₘₙ += Aₘₖ * Bₖₙ
    instrsym = Base.libllvm_version < v"11.0.0" ? :vfmadd231 : :vfmadd
    reductop = Operation(
        ls, mC, elementbytes, instrsym, compute, reductdeps, kvec, Operation[loadA, loadB, setC]
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
Base.@propagate_inbounds Base.getindex(A::LowDimArray, i...) = getindex(A.data, i...)
@inline Base.size(A::LowDimArray) = Base.size(A.data)
@inline Base.size(A::LowDimArray, i) = Base.size(A.data, i)
@inline Base.strides(A::LowDimArray) = strides(A.data)
@inline ArrayInterface.parent_type(::Type{LowDimArray{D,T,N,A}}) where {T,D,N,A} = A
@inline ArrayInterface.strides(A::LowDimArray) = ArrayInterface.strides(A.data)
@generated function ArrayInterface.size(A::LowDimArray{D,T,N}) where {D,T,N}
    t = Expr(:tuple)
    for n ∈ 1:N
        if n > length(D) || D[n]
            push!(t.args, Expr(:ref, :s, n))
        else
            push!(t.args, Expr(:call, Expr(:curly, lv(:Static), 1)))
        end
    end
    Expr(:block, Expr(:meta,:inline), :(s = ArrayInterface.size(parent(A))), t)
end
Base.parent(A::LowDimArray) = A.data
Base.unsafe_convert(::Type{Ptr{T}}, A::LowDimArray{D,T}) where {D,T} = pointer(A.data)
ArrayInterface.contiguous_axis(A::LowDimArray) = ArrayInterface.contiguous_axis(A.data)
ArrayInterface.contiguous_batch_size(A::LowDimArray) = ArrayInterface.contiguous_batch_size(A.data)
ArrayInterface.stride_rank(A::LowDimArray) = ArrayInterface.stride_rank(A.data)
ArrayInterface.offsets(A::LowDimArray) = ArrayInterface.offsets(A.data)

@inline function stridedpointer_for_broadcast(A::LowDimArray{D}) where {D}
    _stridedpointer(stridedpointer_for_broadcast(parent(A)), Val{D}())
end

@generated function _stridedpointer(p::StridedPointer{T,N,C,B,R}, ::Val{D}) where {T,N,C,B,R,D}
    lenD = length(D)
    strd = Expr(:tuple)
    offsets = Expr(:tuple)
    Rtup = Expr(:tuple)
    Cnew = -1
    Bnew = -1
    Nnew = 0
    for n ∈ 1:N
        ((n ≤ lenD) && (!D[n])) && continue
        if n == C
            Cnew = n
        end
        if n == B
            Bnew = n
        end
        push!(Rtup.args, R[n])
        push!(offsets.args, Expr(:ref, :offs, n))
        push!(strd.args, Expr(:ref, :strd, n))
        Nnew += 1
    end
    typ = Expr(:curly, :StridedPointer, T, Nnew, Cnew, Bnew, Rtup)
    ptr = Expr(:call, typ, :(pointer(p)), strd, offsets)
    Expr(:block, Expr(:meta,:inline), :(strd = p.strd), :(offs = p.offsets), ptr)
end
# @generated function VectorizationBase.stridedpointer(A::LowDimArray{D,T,N}) where {D,T,N}
#     smul = Expr(:(.), Expr(:(.), :LoopVectorization, QuoteNode(:VectorizationBase)), QuoteNode(:staticmul))
#     multup = Expr(:tuple)
#     for n ∈ D[1]+1:N
#         if length(D) < n
#             push!(multup.args, Expr(:call, :ifelse, :(isone(size(A,$n))), 0, Expr(:ref, :strideA, n)))
#         elseif D[n]
#             push!(multup.args, Expr(:ref, :strideA, n))
#         end
#     end
#     s = Expr(:call, smul, T, multup)
#     f = D[1] ? :PackedStridedPointer : :SparseStridedPointer
#     Expr(:block, Expr(:meta,:inline), Expr(:(=), :strideA, Expr(:call, :strides, Expr(:(.), :A, QuoteNode(:data)))),
#          Expr(:call, Expr(:(.), :VectorizationBase, QuoteNode(f)), Expr(:call, :pointer, :A), s))
# end
function LowDimArray{D}(data::A) where {D,T,N,A <: AbstractArray{T,N}}
    LowDimArray{D,T,N,A}(data)
end
function extract_all_1_array!(ls::LoopSet, bcname::Symbol, N::Int, elementbytes::Int)
    refextract = gensym!(ls, bcname)
    ref = Expr(:ref, bcname); foreach(_ -> push!(ref.args, :begin), 1:N)
    pushprepreamble!(ls, Expr(:(=), refextract, ref))
    return add_constant!(ls, refextract, elementbytes) # or replace elementbytes with sizeof(T) ? u
end
function add_broadcast!(
    ls::LoopSet, destname::Symbol, bcname::Symbol, loopsyms::Vector{Symbol},
    @nospecialize(LDA::Type{LowDimArray{D,T,N,A}}), elementbytes::Int
) where {D,T,N,A}
    # D,T,N::Int,_ = LDA.parameters
    Dlen = length(D)
    if Dlen == N && !any(D) # array is a scalar, as it is broadcasted on all dimensions
        return extract_all_1_array!(ls, bcname, N, elementbytes)
    end
    fulldims = Symbol[loopsyms[n] for n ∈ 1:N if ((Dlen < n) || D[n]::Bool)]
    ref = ArrayReference(bcname, fulldims)
    add_simple_load!(ls, destname, ref, elementbytes, true, true )::Operation
end
function add_broadcast_adjoint_array!(
    ls::LoopSet, destname::Symbol, bcname::Symbol, loopsyms::Vector{Symbol}, ::Type{A}, elementbytes::Int
) where {T,N,A<:AbstractArray{T,N}}
    parent = gensym!(ls, "parent")
    pushprepreamble!(ls, Expr(:(=), parent, Expr(:call, :parent, bcname)))
    # isone(length(loopsyms)) && return extract_all_1_array!(ls, bcname, N, elementbytes)
    ref = ArrayReference(parent, Symbol[loopsyms[N + 1 - n] for n ∈ 1:N])
    add_simple_load!( ls, destname, ref, elementbytes, true, true )::Operation
end
function add_broadcast_adjoint_array!(
    ls::LoopSet, destname::Symbol, bcname::Symbol, loopsyms::Vector{Symbol}, ::Type{<:AbstractVector}, elementbytes::Int
)
    # isone(length(loopsyms)) && return extract_all_1_array!(ls, bcname, N, elementbytes)
    parent = gensym!(ls, "parent")
    pushprepreamble!(ls, Expr(:(=), parent, Expr(:call, :parent, bcname)))

    ref = ArrayReference(parent, Symbol[loopsyms[2]])
    add_simple_load!( ls, destname, ref, elementbytes, true, true )::Operation
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
    refextract = gensym!(ls, bcname)
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
        f = gensym!(ls, "func")
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
        argname = gensym!(ls, "arg")
        pushprepreamble!(ls, Expr(:(=), argname, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__,Symbol(@__FILE__)), Expr(:ref, bcargs, i))))
        # dynamic dispatch
        parent = add_broadcast!(ls, gensym!(ls, "temp"), argname, loopsyms, arg, elementbytes)::Operation
        push!(parents, parent)
        mergesetdiffv!(deps, loopdependencies(parent), reduceddependencies(parent))
    end
    op = Operation(
        length(operations(ls)), destname, elementbytes, instr, compute, deps, NOPARENTS, parents
    )
    pushop!(ls, op, destname)
end

function add_broadcast_loops!(ls::LoopSet, loopsyms::Vector{Symbol}, destsym::Symbol)
    axes_tuple = Expr(:tuple)
    pushpreamble!(ls, Expr(:(=), axes_tuple, Expr(:call, :axes, destsym)))
    for (n,itersym) ∈ enumerate(loopsyms)
        Nrange = gensym!(ls, "N")
        Nlower = gensym!(ls, "N")
        Nupper = gensym!(ls, "N")
        Nlen = gensym!(ls, "N")
        add_loop!(ls, Loop(itersym, Nlower, Nupper, Nrange, Nlen), itersym)
        push!(axes_tuple.args, Nrange)
        pushpreamble!(ls, Expr(:(=), Nlower, Expr(:call, lv(:maybestaticfirst), Nrange)))
        pushpreamble!(ls, Expr(:(=), Nupper, Expr(:call, lv(:maybestaticlast), Nrange)))
        pushpreamble!(ls, Expr(:(=), Nlen, Expr(:call, lv(:maybestaticlength), Nrange)))
    end
end
# size of dest determines loops
# function vmaterialize!(
@generated function vmaterialize!(
    dest::AbstractArray{T,N}, bc::BC,
    ::Val{Mod}, ::StaticInt{RS}, ::StaticInt{RC}, ::StaticInt{CLS}
) where {T <: NativeTypes, N, BC <: Union{Broadcasted,Product}, Mod, RS, RC, CLS}
    # 2+1
    # we have an N dimensional loop.
    # need to construct the LoopSet
    # @show typeof(dest)
    ls = LoopSet(Mod)
    set_hw!(ls, RS, RC, CLS)
    loopsyms = [gensym!(ls, "n") for n ∈ 1:N]
    ls.isbroadcast[] = true
    add_broadcast_loops!(ls, loopsyms, :dest)
    elementbytes = sizeof(T)
    add_broadcast!(ls, :dest, :bc, loopsyms, BC, elementbytes)
    add_simple_store!(ls, :dest, ArrayReference(:dest, loopsyms), elementbytes)
    resize!(ls.loop_order, num_loops(ls)) # num_loops may be greater than N, eg Product
    # return ls
    q = lower(ls, 0)
    push!(q.args, :dest)
    # @show q
    # q
    q = Expr(
        :block,
        ls.prepreamble,
        # Expr(:if, check_args_call(ls), Expr(:block, :(println("Primary code path!")), q), Expr(:block, :(println("Back up code path!")), :(Base.Broadcast.materialize!(dest, bc))))
        Expr(:if, check_args_call(ls), q, :(Base.Broadcast.materialize!(dest, bc)))
    )
    # isone(N) && pushfirst!(q.args, Expr(:meta,:inline))
    q
     # ls
end
@generated function vmaterialize!(
    dest′::Union{Adjoint{T,A},Transpose{T,A}}, bc::BC,
    ::Val{Mod}, ::StaticInt{RS}, ::StaticInt{RC}, ::StaticInt{CLS}
) where {T <: NativeTypes, N, A <: AbstractArray{T,N}, BC <: Union{Broadcasted,Product}, Mod, RS, RC, CLS}
    # we have an N dimensional loop.
    # need to construct the LoopSet
    ls = LoopSet(Mod)
    set_hw!(ls, RS, RC, CLS)
    loopsyms = [gensym!(ls, "n") for n ∈ 1:N]
    ls.isbroadcast[] = true
    pushprepreamble!(ls, Expr(:(=), :dest, Expr(:call, :parent, :dest′)))
    add_broadcast_loops!(ls, loopsyms, :dest′)
    elementbytes = sizeof(T)
    add_broadcast!(ls, :dest, :bc, loopsyms, BC, elementbytes)
    add_simple_store!(ls, :dest, ArrayReference(:dest, reverse(loopsyms)), elementbytes)
    resize!(ls.loop_order, num_loops(ls)) # num_loops may be greater than N, eg Product
    q = lower(ls, 0)
    push!(q.args, :dest′)
    q = Expr(
        :block,
        ls.prepreamble,
        Expr(:if, check_args_call(ls), q, :(Base.Broadcast.materialize!(dest′, bc)))
    )
    # isone(N) && pushfirst!(q.args, Expr(:meta,:inline))
    q
    # ls
end
# these are marked `@inline` so the `@avx` itself can choose whether or not to inline.
@inline function vmaterialize!(
    dest::AbstractArray{T,N}, bc::Broadcasted{Base.Broadcast.DefaultArrayStyle{0},Nothing,typeof(identity),Tuple{T2}},
    ::Val{Mod}, RS::Static, RC::Static, CLS::Static
) where {T <: NativeTypes, N, T2 <: Number, Mod}
    arg = T(first(bc.args))
    @avx for i ∈ eachindex(dest)
        dest[i] = arg
    end
    dest
end
@inline function vmaterialize!(
    dest′::Union{Adjoint{T,A},Transpose{T,A}}, bc::Broadcasted{Base.Broadcast.DefaultArrayStyle{0},Nothing,typeof(identity),Tuple{T2}},
    ::Val{Mod}, RS::Static, RC::Static, CLS::Static
) where {T <: NativeTypes, N, A <: AbstractArray{T,N}, T2 <: Number, Mod}
    arg = T(first(bc.args))
    dest = parent(dest′)
    @avx for i ∈ eachindex(dest)
        dest[i] = arg
    end
    dest′
end

@inline function vmaterialize(bc::Broadcasted, ::Val{Mod}, RS::Static, RC::Static, CLS::Static) where {Mod}
    ElType = Base.Broadcast.combine_eltypes(bc.f, bc.args)
    vmaterialize!(similar(bc, ElType), bc, Val{Mod}(), RS, RC, CLS)
end

vmaterialize!(dest, bc, ::Val{mod}, ::StaticInt, ::StaticInt, ::StaticInt) where {mod} = Base.Broadcast.materialize!(dest, bc)

