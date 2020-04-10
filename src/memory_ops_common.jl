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
using VectorizationBase: noaliasstridedpointer
function add_vptr!(ls::LoopSet, array::Symbol, vptrarray::Symbol = vptr(array), actualarray::Bool = true, broadcast::Bool = false)
    if !includesarray(ls, array)
        push!(ls.includedarrays, array)
        actualarray && push!(ls.includedactualarrays, array)
        if broadcast
            pushpreamble!(ls, Expr(:(=), vptrarray, Expr(:call, lv(:stridedpointer_for_broadcast), array)))
        else
            pushpreamble!(ls, Expr(:(=), vptrarray, Expr(:call, lv(:stridedpointer), array)))
            # pushpreamble!(ls, Expr(:(=), vptrarray, Expr(:call, lv(:noaliasstridedpointer), array)))
        end
    end
    nothing
end

@inline valsum() = Val{0}()
@inline valsum(::Val{M}) where {M} = Val{M}()
@generated valsum(::Val{M}, ::Val{N}) where {M,N} = Val{M+N}()
@inline valsum(::Val{M}, ::Val{N}, ::Val{K}, args...) where {M,N,K} = valsum(valsum(Val{M}(), Val{N}()), Val{K}(), args...)
@inline valdims(::Any) = Val{1}()
@inline valdims(::CartesianIndices{N}) where {N} = Val{N}()

function append_loop_valdims!(valcall::Expr, loop::Loop)
    if isstaticloop(loop)
        push!(valcall.args, :(Val{1}()))
    else
        push!(valcall.args, Expr(:call, lv(:valdims), loop_boundary(loop)))
    end
    nothing
end
function subset_vptr!(ls::LoopSet, vptr::Symbol, indnum::Int, ind, previndices, loopindex)
    subsetvptr = Symbol(vptr, "_subset_$(indnum)_with_$(ind)##")
    valcall = Expr(:call, Expr(:curly, :Val, 1))
    if indnum > 1
        valcall = Expr(:call, lv(:valsum), valcall)
        for i ∈ 1:indnum-1
            if loopindex[i]
                append_loop_valdims!(valcall, getloop(ls, previndices[i]))
            else
                for loopdep ∈ loopdependencies(ls.opdict[previndices[i]])
                    append_loop_valdims!(valcall, getloop(ls, loopdep))
                end
            end
        end
    end
    # @show valcall
    indm1 = ind isa Integer ? ind - 1 : Expr(:call, :-, ind, 1)
    pushpreamble!(ls, Expr(:(=), subsetvptr, Expr(:call, lv(:subsetview), vptr, valcall, indm1)))
    subsetvptr
end

function addoffset!(ls, indices, offsets, loopedindex, loopdependencies, ind, offset)
    if typemin(Int8) ≤ offset ≤ typemax(Int8)
        push!(indices, ind);
        push!(offsets, offset % Int8)
        push!(loopedindex, true)
        push!(loopdependencies, ind)
        true
    else
        false
    end
end

function checkforoffset!(
    ls::LoopSet, indices::Vector{Symbol}, offsets::Vector{Int8}, loopedindex::Vector{Bool}, loopdependencies::Vector{Symbol}, ind::Expr
)
    ind.head === :call || return false
    f = first(ind.args)
    (((f === :+) || (f === :-)) && (length(ind.args) == 3)) || return false
    factor = f === :+ ? 1 : -1
    arg1 = ind.args[2]
    arg2 = ind.args[3]
    if arg1 isa Integer
        if arg2 isa Symbol && arg2 ∈ ls.loopsymbols
            addoffset!(ls, indices, offsets, loopedindex, loopdependencies, arg2, arg1 * factor)
        else
            false
        end
    elseif arg2 isa Integer
        if arg1 isa Symbol && arg1 ∈ ls.loopsymbols
            addoffset!(ls, indices, offsets, loopedindex, loopdependencies, arg1, arg2 * factor)
        else
            false
        end
    else
        false
    end        
end

const DISCONTIGUOUS = Symbol("##DISCONTIGUOUSSUBARRAY##")
function array_reference_meta!(ls::LoopSet, array::Symbol, rawindices, elementbytes::Int, var::Union{Nothing,Symbol} = nothing)
    vptrarray = vptr(array)
    add_vptr!(ls, array, vptrarray) # now, subset
    indices = Symbol[]
    offsets = Int8[]
    loopedindex = Bool[]
    parents = Operation[]
    loopdependencies = Symbol[]
    reduceddeps = Symbol[]
    loopset = ls.loopsymbols
    ninds = 1
    for ind ∈ rawindices        
        if ind isa Integer # subset
            vptrarray = subset_vptr!(ls, vptrarray, ninds, ind, indices, loopedindex)
            length(indices) == 0 && push!(indices, DISCONTIGUOUS)
        elseif ind isa Expr
            #FIXME: position (in loopnest) wont be length(ls.loopsymbols) in general
            if !checkforoffset!(ls, indices, offsets, loopedindex, loopdependencies, ind)
                parent = add_operation!(ls, gensym(:indexpr), ind, elementbytes, length(ls.loopsymbols))
                pushparent!(parents, loopdependencies, reduceddeps, parent)
                push!(indices, name(parent)); 
                push!(offsets, zero(Int8))
                push!(loopedindex, false)
            end
            ninds += 1
        elseif ind isa Symbol
            if ind ∈ loopset
                push!(indices, ind); ninds += 1
                push!(offsets, zero(Int8))
                push!(loopedindex, true)
                push!(loopdependencies, ind)
            else
                indop = get(ls.opdict, ind, nothing)
                if indop !== nothing  && !isconstant(indop)
                    pushparent!(parents, loopdependencies, reduceddeps, indop)
                    push!(indices, name(indop)); ninds += 1
                    push!(offsets, zero(Int8))
                    push!(loopedindex, false)
                else
                    vptrarray = subset_vptr!(ls, vptrarray, ninds, ind, indices, loopedindex)
                    length(indices) == 0 && push!(indices, DISCONTIGUOUS)
                end
            end
        else
            throw("Unrecognized loop index: $ind.")
        end
    end
    # (length(parents) != 0 && first(indices) !== Symbol("##DISCONTIGUOUSSUBARRAY##")) && pushfirst!(indices, Symbol("##DISCONTIGUOUSSUBARRAY##"))
    mref = ArrayReferenceMeta(ArrayReference( array, indices, offsets ), loopedindex, vptrarray)
    ArrayReferenceMetaPosition(mref, parents, loopdependencies, reduceddeps, isnothing(var) ? Symbol("") : var )
end
function tryrefconvert(ls::LoopSet, ex::Expr, elementbytes::Int, var::Union{Nothing,Symbol} = nothing)::Tuple{Bool,ArrayReferenceMetaPosition}
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
    true, array_reference_meta!(ls, ya, yinds, elementbytes, var)
end

