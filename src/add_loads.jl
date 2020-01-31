
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
    # try to CSE
    id = findfirst(r -> r == ref, ls.refs_aliasing_syms)
    if id === nothing
        push!(ls.syms_aliasing_refs, var)
        push!(ls.refs_aliasing_syms, ref)
    else
        opp = getop(ls, ls.syms_aliasing_refs[id], elementbytes)
        return isstore(opp) ? getop(ls, first(parents(opp))) : opp
    end
    # else, don't
    op = Operation( ls, var, elementbytes, :getindex, memload, mpref )
    add_vptr!(ls, op)
    pushop!(ls, op, var)
end

# for use with broadcasting
function add_simple_load!(
    ls::LoopSet, var::Symbol, ref::ArrayReference, elementbytes::Int, actualarray::Bool = true
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
    add_vptr!(ls, op.ref.ref.array, vptr(op.ref), actualarray)
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


struct LoopValue end
@inline VectorizationBase.stridedpointer(::LoopValue) = LoopValue()
@inline SIMDPirates.vload(::LoopValue, i::Tuple{_MM{W}}) where {W} = SVec(SIMDPirates.vrangeincr(Val{W}(), @inbounds(i[1].i), Val{1}()))
@inline SIMDPirates.vload(::LoopValue, i::Tuple{_MM{W}}, ::Unsigned) where {W} = SVec(SIMDPirates.vrangeincr(Val{W}(), @inbounds(i[1].i), Val{1}()))
@inline VectorizationBase.load(::LoopValue, i::Integer) = i + one(i)
@inline VectorizationBase.load(::LoopValue, i::Tuple{I}) where {I<:Integer} = @inbounds(i[1]) + one(I)
@inline Base.eltype(::LoopValue) = Int8

