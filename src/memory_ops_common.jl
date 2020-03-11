function ref_from_expr(ex, offset1::Int, offset2::Int)
    (ex.args[1 + offset1])::Symbol, @view(ex.args[2 + offset2:end])
end
ref_from_ref(ex::Expr) = ref_from_expr(ex, 0, 0)
ref_from_getindex(ex::Expr) = ref_from_expr(ex, 1, 1)
ref_from_setindex(ex::Expr) = ref_from_expr(ex, 1, 2)
function ref_from_expr(ex::Expr)
    if ex.head === :ref
        ref_from_ref(ex)
    else#if ex.head === :call
        f = first(ex.args)::Symbol
        f === :getindex ? ref_from_getindex(ex) : ref_from_setindex(ex)
    end
end

add_vptr!(ls::LoopSet, op::Operation) = add_vptr!(ls, op.ref)
add_vptr!(ls::LoopSet, mref::ArrayReferenceMeta) = add_vptr!(ls, mref.ref.array, vptr(mref))
function add_vptr!(ls::LoopSet, array::Symbol, vptrarray::Symbol = vptr(array), actualarray::Bool = true, broadcast::Bool = false)
    if !includesarray(ls, array)
        push!(ls.includedarrays, array)
        actualarray && push!(ls.includedactualarrays, array)
        if broadcast
            pushpreamble!(ls, Expr(:(=), vptrarray, Expr(:call, lv(:stridedpointer_for_broadcast), array)))
        else
            pushpreamble!(ls, Expr(:(=), vptrarray, Expr(:call, lv(:stridedpointer), array)))
        end
    end
    nothing
end
function subset_vptr!(ls::LoopSet, vptr::Symbol, indnum::Int, ind::Union{Symbol,Int})
    subsetvptr = Symbol(vptr, "_subset_$(indnum)_with_$(ind)##")
    inde = ind isa Symbol ? Expr(:call, :-, ind, 1) : ind - 1
    pushpreamble!(ls, Expr(:(=), subsetvptr, Expr(:call, lv(:subsetview), vptr, Expr(:call, Expr(:curly, :Val, indnum)), inde)))
    subsetvptr
end
const DISCONTIGUOUS = Symbol("##DISCONTIGUOUSSUBARRAY##")
function array_reference_meta!(ls::LoopSet, array::Symbol, rawindices, elementbytes::Int)
    vptrarray = vptr(array)
    add_vptr!(ls, array, vptrarray) # now, subset
    
    indices = Symbol[]
    loopedindex = Bool[]
    parents = Operation[]
    loopdependencies = Symbol[]
    reduceddeps = Symbol[]
    loopset = ls.loopsymbols
    ninds = 1
    for ind ∈ rawindices        
        if ind isa Integer # subset
            vptrarray = subset_vptr!(ls, vptrarray, ninds, ind)
            length(indices) == 0 && push!(indices, DISCONTIGUOUS)
        elseif ind isa Expr
            #FIXME: position (in loopnest) wont be length(ls.loopsymbols) in general
            parent = add_operation!(ls, gensym(:indexpr), ind, elementbytes, length(ls.loopsymbols))
            pushparent!(parents, loopdependencies, reduceddeps, parent)
            # var = get(ls.opdict, ind, nothing)
            push!(indices, name(parent)); ninds += 1
            push!(loopedindex, false)
        elseif ind isa Symbol
            if ind ∈ loopset
                push!(indices, ind); ninds += 1
                push!(loopedindex, true)
                push!(loopdependencies, ind)
            else
                indop = get(ls.opdict, ind, nothing)
                if indop !== nothing  && !isconstant(indop)
                    pushparent!(parents, loopdependencies, reduceddeps, parent)
                    # var = get(ls.opdict, ind, nothing)
                    push!(indices, name(parent)); ninds += 1
                    push!(loopedindex, false)
                else
                    vptrarray = subset_vptr!(ls, vptrarray, ninds, ind)
                    length(indices) == 0 && push!(indices, DISCONTIGUOUS)
                end
            end
        else
            throw("Unrecognized loop index: $ind.")
        end
    end
    # (length(parents) != 0 && first(indices) !== Symbol("##DISCONTIGUOUSSUBARRAY##")) && pushfirst!(indices, Symbol("##DISCONTIGUOUSSUBARRAY##"))
    mref = ArrayReferenceMeta(ArrayReference( array, indices ), loopedindex, vptrarray)
    ArrayReferenceMetaPosition(mref, parents, loopdependencies, reduceddeps)
end
function tryrefconvert(ls::LoopSet, ex::Expr, elementbytes::Int)::Tuple{Bool,ArrayReferenceMetaPosition}
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

