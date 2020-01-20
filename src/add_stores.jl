function add_unique_store!(ls::LoopSet, op::Operation)
    add_vptr!(ls, op)
    pushop!(ls, op, name(op.ref))
end
function cse_store!(ls::LoopSet, op::Operation)
    id = identifier(op)
    ls.operations[id] = op
    ls.opdict[op.variable] = op
    op
end
function add_store!(ls::LoopSet, op::Operation)
    nops = length(ls.operations)
    id = op.identifier
    id == nops ? add_unique_store!(ls, op) : cse_store!(ls, op)
end
function add_store!(
    ls::LoopSet, var::Symbol, mpref::ArrayReferenceMetaPosition, elementbytes::Int = 8
)
    parents = mpref.parents
    ldref = mpref.loopdependencies
    reduceddeps = mpref.reduceddeps
    parent = getop(ls, var, ldref, elementbytes)
    # pushfirst!(parents, parent)
    pvar = parent.variable
    nops = length(ls.operations)
    id = nops
    if pvar ∉ ls.syms_aliasing_refs
        push!(ls.syms_aliasing_refs, pvar)
        push!(ls.refs_aliasing_syms, mpref.mref)
        # add_unique_store!(ls, mref, parents, ldref, reduceddeps, elementbytes)
    else
        # try to cse store
        # different from cse load, because the other op here must be a store
        ref = mpref.mref.ref
        for opp ∈ operations(ls)
            isstore(opp) || continue
            if ref == opp.ref.ref# && return cse_store!(ls, identifier(opp), mref, parents, ldref, reduceddeps, elementbytes)
                id = opp.identifier
            end
        end
        # add_unique_store!(ls, mref, parents, ldref, reduceddeps, elementbytes)        
    end
    pushparent!(parents, ldref, reduceddeps, parent)
    op = Operation( id, name(mpref), elementbytes, :setindex!, memstore, mpref )#loopdependencies, reduceddeps, parents, mpref.mref )
    add_store!(ls, op)
end
function add_store!(
    ls::LoopSet, var::Symbol, array::Symbol, rawindices, elementbytes::Int = 8
)
    mpref = array_reference_meta!(ls, array, rawindices, elementbytes)
    add_store!(ls, var, mpref, elementbytes)
end
function add_simple_store!(ls::LoopSet, var::Symbol, ref::ArrayReference, elementbytes::Int = 8)
    mref = ArrayReferenceMeta(
        ref, fill(true, length(getindices(ref)))
    )
    parents = [getop(ls, var, elementbytes)]
    ldref = convert(Vector{Symbol}, getindices(ref))
    op = Operation( ls, name(mref), elementbytes, :setindex!, memstore, ldref, NODEPENDENCY, parents, mref )
    add_unique_store!(ls, op)
end
function add_store_ref!(ls::LoopSet, var::Symbol, ex::Expr, elementbytes::Int = 8)
    array, raw_indices = ref_from_ref(ex)
    add_store!(ls, var, array, raw_indices, elementbytes)
end
function add_store_setindex!(ls::LoopSet, ex::Expr, elementbytes::Int = 8)
    array, raw_indices = ref_from_setindex(ex)
    add_store!(ls, (ex.args[2])::Symbol, array, rawindices, elementbytes)
end

# For now, it is illegal to load from a conditional store.
# if you want that sort of behavior, do a conditional reassignment, and store that result unconditionally.
function add_conditional_store!(ls::LoopSet, LHS, condop::Operation, storeop::Operation, elementbytes::Int)
    array, raw_indices = ref_from_ref(ex)
    ref = ArrayReference(array, raw_indices)
    mref = ArrayReferenceMeta(
        ref, fill(true, length(getindices(ref)))
    )
    parents = [storeop, condop]
    ldref = convert(Vector{Symbol}, getindices(ref))
    op = Operation( ls, name(mref), elementbytes, :condtionalstore!, memstore, ldref, NODEPENDENCY, parents, mref )
    add_unique_store!(ls, op)    
end

