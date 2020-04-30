
# struct ParentsBeforeChildrenIterator
#     ops::Vector{Operation}
#     visited::Vector{Bool}
# end
# function iterate(pbci::ParentsBeforeChildrenIterator)
#     for (i,op) ∈ enumerate(pbci.ops)
#         if iszero(length(parents(op)))
#             pbci.visited[i] = true
#             return op, pbci
#         end
#     end
#     nothing
# end
# function iterate()

# end

function dependent_outer_reducts(ls::LoopSet, op)
    for i ∈ ls.outer_reductions
        search_tree(parents(operations(ls)[i]), name(op)) && return true
    end
    false
end

function isnopidentity(ls::LoopSet, op::Operation, u₁loop::Symbol, u₂loop::Symbol, vectorized::Symbol, suffix)
    parents_op = parents(op)
    if iscompute(op) && instruction(op).instr === :identity && name(first(parents_op)) === name(op) && isone(length(parents_op))
        mvar, u₁unrolledsym, u₂unrolledsym = variable_name_and_unrolled(op, u₁loop, u₂loop, suffix)
        parents_u₁syms, parents_u₂syms = parent_unroll_status(op, u₁loop, u₂loop, suffix)
        if (u₁unrolledsym == first(parents_u₁syms)) && ((!isnothing(suffix)) == parents_u₂syms[1])
            #TODO: identifer(first(parents_op)) ∉ ls.outer_reductions is going to miss a lot of cases
            #Should probably replace that with `DVec` (demoting Vec) types, that demote to scalar.
            if (vectorized ∈ loopdependencies(first(parents_op)) && vectorized ∉ loopdependencies(op)) && !dependent_outer_reducts(ls, op)
                op.instruction = reduction_to_scalar(instruction(first(parents_op)))
                op.mangledvariable = gensym(op.mangledvariable)
                false
            else
                true
            end
        else
            false
        end
    else
        false
    end
end

function set_upstream_family!(adal::Vector{T}, op::Operation, val::T, ld::Vector{Symbol}, id::Int) where {T}
    adal[identifier(op)] == val && return # must already have been set
    if ld != loopdependencies(op) || id == identifier(op)
        (adal[identifier(op)] = val)
    end
    for opp ∈ parents(op)
        identifier(opp) == identifier(op) && continue
        set_upstream_family!(adal, opp, val, ld, id)
    end
end

function addoptoorder!(
    ls::LoopSet, included_vars::Vector{Bool}, place_after_loop::Vector{Bool}, op::Operation,
    loopsym::Symbol, _n::Int, u₁loop::Symbol, u₂loop::Symbol, vectorized::Symbol, loopistiled::Bool
)
    lo = ls.loop_order
    id = identifier(op)
    included_vars[id] && return nothing
    loopsym ∈ loopdependencies(op) || return nothing
    for opp ∈ parents(op) # ensure parents are added first
        addoptoorder!(ls, included_vars, place_after_loop, opp, loopsym, _n, u₁loop, u₂loop, vectorized, loopistiled)
    end
    included_vars[id] && return nothing
    included_vars[id] = true
    isunrolled = (u₁loop ∈ loopdependencies(op)) + 1
    istiled = u₂loop ∈ loopdependencies(op)
    # optype = Int(op.node_type) + 1
    after_loop = place_after_loop[id] + 1
    if !isloopvalue(op)
        if istiled
            isnopidentity(ls, op, u₁loop, u₂loop, vectorized, 0) || push!(lo[isunrolled,2,after_loop,_n], op)
        else
            isnopidentity(ls, op, u₁loop, u₂loop, vectorized, nothing) || push!(lo[isunrolled,1,after_loop,_n], op)
        end
    end
    # @show op, after_loop
    # isloopvalue(op) || push!(lo[isunrolled,istiled,after_loop,_n], op)
    # all(opp -> iszero(length(reduceddependencies(opp))), parents(op)) &&
    set_upstream_family!(place_after_loop, op, false, loopdependencies(op), identifier(op)) # parents that have already been included are not moved, so no need to check included_vars to filter
    nothing
end

function fillorder!(ls::LoopSet, order::Vector{Symbol}, u₁loop::Symbol, u₂loop::Symbol, loopistiled::Bool, vectorized::Symbol)
    lo = ls.loop_order
    resize!(lo, length(ls.loopsymbols))
    ro = lo.loopnames # reverse order; will have same order as lo
    nloops = length(order)
    ops = operations(ls)
    nops = length(ops)
    included_vars = fill!(resize!(ls.included_vars, nops), false)
    place_after_loop = fill!(resize!(ls.place_after_loop, nops), true)
    # to go inside out, we just have to include all those not-yet included depending on the current sym
    empty!(lo)
    for _n ∈ 1:nloops
        n = 1 + nloops - _n
        ro[_n] = loopsym = order[n]
        #loopsym = order[n]
        for op ∈ ops
            addoptoorder!( ls, included_vars, place_after_loop, op, loopsym, _n, u₁loop, u₂loop, vectorized, loopistiled )
        end
    end
end

