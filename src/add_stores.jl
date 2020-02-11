function add_unique_store!(ls::LoopSet, op::Operation)
    add_vptr!(ls, op)
    pushop!(ls, op, name(op.ref))
end
function cse_store!(ls::LoopSet, op::Operation)
    id = identifier(op)
    ls.operations[id] = op
    ls.opdict[op.variable] = op
end
function add_store!(ls::LoopSet, op::Operation, add_pvar::Bool = name(first(parents(op))) ∉ ls.syms_aliasing_refs)
    @assert isstore(op)
    if add_pvar
        push!(ls.syms_aliasing_refs, name(first(parents(op))))
        push!(ls.refs_aliasing_syms, op.ref)
    end
    id = op.identifier
    id == length(operations(ls)) ? add_unique_store!(ls, op) : cse_store!(ls, op)
end
function add_copystore!(
    ls::LoopSet, parent::Operation, mpref::ArrayReferenceMetaPosition, elementbytes::Int
)
    op = add_compute!(ls, gensym(), :identity, [parent], elementbytes)
    # pushfirst!(mpref.parents, parent)
    add_store!(ls, name(op), mpref, elementbytes, op)
end


function add_store!(
    ls::LoopSet, var::Symbol, mpref::ArrayReferenceMetaPosition, elementbytes::Int, parent = getop(ls, var, mpref.loopdependencies, elementbytes)
)
    isload(parent) && return add_copystore!(ls, parent, mpref, elementbytes)
    vparents = mpref.parents
    ldref = mpref.loopdependencies
    reduceddeps = mpref.reduceddeps
    pvar = name(parent)
    id = length(ls.operations)
    # try to cse store, by replacing the previous one
    ref = mpref.mref.ref
    add_pvar = true
    for opp ∈ operations(ls)
        isstore(opp) || continue
        if ref == opp.ref.ref
            id = opp.identifier
            break
        end
        add_pvar &= (name(first(parents(opp))) != pvar)
    end
    pushfirst!(vparents, parent)
    update_deps!(ldref, reduceddeps, parent)
    op = Operation( id, name(mpref), elementbytes, :setindex!, memstore, mpref )
    add_store!(ls, op, add_pvar)
end

function add_store!(
    ls::LoopSet, var::Symbol, array::Symbol, rawindices, elementbytes::Int
)
    mpref = array_reference_meta!(ls, array, rawindices, elementbytes)
    add_store!(ls, var, mpref, elementbytes)
end
function add_simple_store!(ls::LoopSet, parent::Operation, ref::ArrayReference, elementbytes::Int)
    mref = ArrayReferenceMeta(
        ref, fill(true, length(getindices(ref)))
    )
    op = Operation( ls, name(mref), elementbytes, :setindex!, memstore, getindices(ref), NODEPENDENCY, [parent], mref )
    add_unique_store!(ls, op)
end
function add_simple_store!(ls::LoopSet, var::Symbol, ref::ArrayReference, elementbytes::Int)
    add_simple_store!(ls, getop(ls, var, elementbytes), ref, elementbytes)
end
function add_store_ref!(ls::LoopSet, var::Symbol, ex::Expr, elementbytes::Int)
    array, raw_indices = ref_from_ref(ex)
    add_store!(ls, var, array, raw_indices, elementbytes)
end
function add_store_ref!(ls::LoopSet, var, ex::Expr, elementbytes::Int)
    # array, raw_indices = ref_from_ref(ex)
    # mpref = array_reference_meta!(ls, array, raw_indices, elementbytes)
    # c = add_constant!(ls, var, loopdependencies(mpref), gensym(:storeconst), elementbytes)
    # add_store!(ls, name(c), mpref, elementbytes)
    c = add_constant!(ls, var, elementbytes)
    add_store_ref!(ls, name(c), ex, elementbytes)
end
function add_store_setindex!(ls::LoopSet, ex::Expr, elementbytes::Int)
    array, raw_indices = ref_from_setindex(ex)
    add_store!(ls, (ex.args[3])::Symbol, array, raw_indices, elementbytes)
end

# For now, it is illegal to load from a conditional store.
# if you want that sort of behavior, do a conditional reassignment, and store that result unconditionally.
function add_conditional_store!(ls::LoopSet, LHS, condop::Operation, storeop::Operation, elementbytes::Int)
    array, rawindices = ref_from_ref(LHS)
    mpref = array_reference_meta!(ls, array, rawindices, elementbytes)
    mref = mpref.mref
    ldref = mpref.loopdependencies

    pvar = name(storeop)
    id = length(ls.operations)
    @assert pvar ∉ ls.syms_aliasing_refs
    # if pvar ∉ ls.syms_aliasing_refs
    # FIXME properly handle CSE of conditional stores.
    push!(ls.syms_aliasing_refs, pvar)
    push!(ls.refs_aliasing_syms, mref)
    storeparents = [storeop, condop]
    # else
    #     # for now, we don't try to cse the store
    #     # later, as an optimization, we could:
    #     # 1. cse the store
    #     # 2. use the mask to combine the vector we're trying to store here with the vector that would have been stored in the now cse-ed 1.
    #     # 3. use a regular (non-masked) store on that vector.
    #     ref = mpref.mref.ref
    #     for opp ∈ operations(ls)
    #         isstore(opp) || continue
    #         if ref == opp.ref.ref# && return cse_store!(ls, identifier(opp), mref, parents, ldref, reduceddeps, elementbytes)
    #             id = opp.identifier
    #             break
    #         end
    #     end
    #     if id != length(ls.operations) # then there was a previous store
    #         prevstore = getop(ls, id + 1)
    #         # @show prevstore prevstore.node_type, loopdependencies(prevstore)
    #         # @show operations(ls)
    #         storeop = add_compute!(ls, gensym(:combinedstoreop), Instruction(:vifelse), [condop, storeop, first(parents(prevstore))], elementbytes)
    #         storeparents = [storeop]
    #         storeinstr = if prevstore.instruction.instr === :conditionalstore!
    #             push!(storeparents, add_compute!(ls, gensym(:combinedmask), Instruction(:|), [condop, last(parents(prevstore))], elementbytes))
    #             :conditionalstore!
    #         else
    #             :setindex!
    #         end
    #         op = Operation( id, name(mref), elementbytes, storeinstr, memstore, ldref, NODEPENDENCY, storeparents, mref )
    #         return cse_store!(ls, op)
    #     end
    # end
    op = Operation( id, name(mref), elementbytes, :conditionalstore!, memstore, ldref, NODEPENDENCY, storeparents, mref )
    add_unique_store!(ls, op)
end

