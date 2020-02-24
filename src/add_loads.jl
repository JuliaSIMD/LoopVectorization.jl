function add_load!(ls::LoopSet, op::Operation, actualarray::Bool = true, broadcast::Bool = false)
    @assert isload(op)
    ref = op.ref
    id = findfirst(r -> r == ref, ls.refs_aliasing_syms)
    # try to CSE
    if id === nothing
        push!(ls.syms_aliasing_refs, name(op))
        push!(ls.refs_aliasing_syms, ref)
    else
        opp = ls.opdict[ls.syms_aliasing_refs[id]] # throw an error if not found.
        return isstore(opp) ? getop(ls, first(parents(opp))) : opp
    end    
    add_vptr!(ls, op.ref.ref.array, vptr(op.ref), actualarray, broadcast)
    pushop!(ls, op, name(op))
end

function add_load!(
    ls::LoopSet, var::Symbol, array::Symbol, rawindices, elementbytes::Int
)
    mpref = array_reference_meta!(ls, array, rawindices, elementbytes)
    add_load!(ls, var, mpref, elementbytes)
end
function add_load!(
    ls::LoopSet, var::Symbol, mpref::ArrayReferenceMetaPosition, elementbytes::Int
)
    length(mpref.loopdependencies) == 0 && return add_constant!(ls, var, mpref, elementbytes)
    ref = mpref.mref
    op = Operation( ls, var, elementbytes, :getindex, memload, mpref )
    add_load!(ls, op, true, false)
end

# for use with broadcasting
function add_simple_load!(
    ls::LoopSet, var::Symbol, ref::ArrayReference, elementbytes::Int, actualarray::Bool = true, broadcast::Bool = false
)
    loopdeps = Symbol[s for s ∈ ref.indices]
    mref = ArrayReferenceMeta(
        ref, fill(true, length(loopdeps))
    )
    op = Operation(
        length(operations(ls)), var, elementbytes,
        :getindex, memload, loopdeps,
        NODEPENDENCY, NOPARENTS, mref
    )
    add_vptr!(ls, op.ref.ref.array, vptr(op.ref), actualarray, broadcast)
    pushop!(ls, op, var)
end
function add_load_ref!(ls::LoopSet, var::Symbol, ex::Expr, elementbytes::Int)
    array, rawindices = ref_from_ref(ex)
    add_load!(ls, var, array, rawindices, elementbytes)
end
function add_load_getindex!(ls::LoopSet, var::Symbol, ex::Expr, elementbytes::Int)
    array, rawindices = ref_from_getindex(ex)
    add_load!(ls, var, array, rawindices, elementbytes)
end

function add_loopvalue!(ls::LoopSet, arg::Symbol, elementbytes::Int)
    # check for CSE opportunity
    loopsym = Symbol("##LOOPSYMBOL##", arg)
    ar = ArrayReference(loopsym, [arg])
    for op ∈ operations(ls)
        if isload(op) && op.ref.ref == ar
            return op
        end
    end
    pushpreamble!(ls, Expr(:(=), loopsym, LoopValue()))
    loopsymop = add_simple_load!(ls, gensym(loopsym), ar, elementbytes, false)
    push!(ls.syms_aliasing_refs, name(loopsymop))
    push!(ls.refs_aliasing_syms, loopsymop.ref)
    loopsymop
end


struct LoopValue end
@inline VectorizationBase.stridedpointer(::LoopValue) = LoopValue()
@inline VectorizationBase.vload(::LoopValue, i::Tuple{_MM{W}}) where {W} = _MM{W}(@inbounds(i[1].i) + 1)
@inline VectorizationBase.vload(::LoopValue, i::Tuple{_MM{W}}, ::Unsigned) where {W} = _MM{W}(@inbounds(i[1].i) + 1)
@inline VectorizationBase.vload(::LoopValue, i::Integer) = i + one(i)
@inline VectorizationBase.vload(::LoopValue, i::Tuple{I}) where {I<:Integer} = @inbounds(i[1]) + one(I)
@inline Base.eltype(::LoopValue) = Int8

