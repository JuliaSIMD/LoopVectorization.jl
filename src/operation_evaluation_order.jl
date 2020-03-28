
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

function set_upstream_family!(adal::Vector{T}, op::Operation, val::T, ld::Vector{Symbol}, id::Int) where {T}
    adal[identifier(op)] == val && return # must already have been set
    # @show op
    if ld != loopdependencies(op) || id == identifier(op)
        (adal[identifier(op)] = val)
    end
    for opp ∈ parents(op)
        identifier(opp) == identifier(op) && continue
        set_upstream_family!(adal, opp, val, ld, id)
    end
end

function addoptoorder!(
    lo::LoopOrder, included_vars::Vector{Bool}, place_after_loop::Vector{Bool}, op::Operation, loopsym::Symbol, _n::Int, unrolled::Symbol, tiled::Symbol, loopistiled::Bool
)
    id = identifier(op)
    included_vars[id] && return nothing
    loopsym ∈ loopdependencies(op) || return nothing
    for opp ∈ parents(op) # ensure parents are added first
        addoptoorder!(lo, included_vars, place_after_loop, opp, loopsym, _n, unrolled, tiled, loopistiled)
    end
    included_vars[id] && return nothing
    included_vars[id] = true
    isunrolled = (unrolled ∈ loopdependencies(op)) + 1
    istiled = (loopistiled ? (tiled ∈ loopdependencies(op)) : false) + 1
    # optype = Int(op.node_type) + 1
    after_loop = place_after_loop[id] + 1
    # @show place_after_loop[id], op
    isloopvalue(op) || push!(lo[isunrolled,istiled,after_loop,_n], op)
    # all(opp -> iszero(length(reduceddependencies(opp))), parents(op)) &&
    set_upstream_family!(place_after_loop, op, false, loopdependencies(op), identifier(op)) # parents that have already been included are not moved, so no need to check included_vars to filter
    nothing
end

function fillorder!(ls::LoopSet, order::Vector{Symbol}, unrolled::Symbol, tiled::Symbol, loopistiled::Bool)
    lo = ls.loop_order
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
            addoptoorder!( lo, included_vars, place_after_loop, op, loopsym, _n, unrolled, tiled, loopistiled )
        end
    end
end

