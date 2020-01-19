function add_vptr!(ls::LoopSet, op::Operation)
    ref = op.ref
    indexed = name(ref)
    id = identifier(op)
    if includesarray(ls, indexed) < 0
        push!(ls.includedarrays, (indexed, id))
        pushpreamble!(ls, Expr(:(=), vptr(op), Expr(:call, lv(:stridedpointer), indexed)))
    end
    nothing
end
# function intersection(depsplus, ls)
    # deps = Symbol[]
    # for dep ∈ depsplus
        # dep ∈ ls && push!(deps, dep)
    # end
    # deps
# end

function array_reference_meta!(ls::LoopSet, array::Symbol, rawindices, elementbytes::Int = 8)
    indices = Vector{Union{Symbol,Int}}(undef, length(rawindices))
    loopedindex = fill(false, length(indices))
    parents = Operation[]
    loopdependencies = Symbol[]
    reduceddeps = Symbol[]
    loopset = keys(ls.loops)
    for i ∈ eachindex(indices)
        ind = rawindices[i]
        if ind isa Integer
            indices[i] = ind - 1
        elseif ind isa Symbol
            indices[i] = ind
            if ind ∈ loopset
                loopedindex[i] = true
                push!(loopdependencies, ind)
            end
        elseif ind isa Expr
            parent = add_operation!(ls, gensym(:indexpr), ind, elementbytes)
            pushparent!(parents, loopdependencies, reduceddeps, parent)
            # var = get(ls.opdict, ind, nothing)
            indices[i] = name(parent)#mangledvar(parent)
        else
            throw("Unrecognized loop index: $ind.")
        end
    end
    length(parents) == 0 || pushfirst!(indices, Symbol("##DISCONTIGUOUSSUBARRAY##"))
    mref = ArrayReferenceMeta(ArrayReference( array, indices ), loopedindex)
    ArrayReferenceMetaPosition(mref, parents, loopdependencies, reduceddeps)
end
function tryrefconvert(ls::LoopSet, ex::Expr, elementbytes::Int = 8)::Tuple{Bool,ArrayReferenceMetaPosition}
    ya, yinds = if ex.head === :ref
        ref_from_ref(ex)
    elseif ex.head === :call
        f = first(ex.args)
        if f === :getindex
            ref_from_getindex(ex)
        elseif f === :setindex!
            ref_from_setindex(ex)
        else
            return false, NOTAREFERENCEMP
        end
    else
        return false, NOTAREFERENCEMP
    end
    true, array_reference_meta!(ls, ya, yinds, elementbytes)
end

