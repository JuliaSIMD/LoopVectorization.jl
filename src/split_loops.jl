

function add_operation!(ls_new::LoopSet, included::Vector{Int}, ls::LoopSet, op::Operation)
    newid = included[identifier(op)]
    iszero(newid) || return operations(ls_new)[newid]
    vparents = Operation[]
    for opp ∈ parents(op)
        push!(vparents, add_operation!(ls_new, included, ls, opp))
    end
    opnew = Operation(
        length(operations(ls_new)), name(op), op.elementbytes, instruction(op), op.node_type,
        loopdependencies(op), reduceddependencies(op), vparents, ref(op), reducedchildren(op)
    )
    included[identifier(op)] = identifier(opnew)
    opnew
end

function append_if_included!(vnew, vold, included)
    for (i, v) ∈ vold
        id = included[i]
        iszero(id) && continue
        push!(vnew, (id, v))
    end
end

function split_loopset(ls::LoopSet, ids)
    ls_new = LoopSet(:LoopVectorization)
    included = zeros(Int, length(operations(ls)))
    for i ∈ ids
        add_operation!(ls_new, included, ls, operations(ls)[i])
    end
    for op ∈ operations(ls_new)
        for l ∈ loopdependencies(op)
            if l ∉ ls_new.loopsymbols
                add_loop!(ls_new, getloop(ls, l))
            end
        end
        length(ls_new.loopsymbols) == length(ls.loopsymbols) && break
    end
    append_if_included!(ls_new.preamble_symsym, ls.preamble_symsym, included)
    append_if_included!(ls_new.preamble_symint, ls.preamble_symint, included)
    append_if_included!(ls_new.preamble_symfloat, ls.preamble_symfloat, included)
    append_if_included!(ls_new.preamble_zeros, ls.preamble_zeros, included)
    append_if_included!(ls_new.preamble_ones, ls.preamble_ones, included)
    ls_new
end


function lower_and_split_loops(ls::LoopSet)
    ops = operations(ls)
    split_candidates = Int[]
    for op ∈ ops
        isstore(op) && push!(split_candidates, identifier(op))
    end
    for i ∈ ls.outer_reductions
        push!(split_candidates, i)
    end
    length(split_candidates) > 1 || return lower(ls)
    order_fused, unrolled_fused, tiled_fused, vectorized_fused, U_fused, T_fused, cost_fused = choose_order(ls)
    remaining_ops = Vector{Int}(undef, length(split_candidates) - 1); split_1 = Int[0];
    for (ind,i) ∈ enumerate(split_candidates)
        split_1[1] = i
        ls_1 = split_loopset(ls, split_1)
        order_1, unrolled_1, tiled_1, vectorized_1, U_1, T_1, cost_1 = choose_order(ls_1)
        reaminig_ops[1:ind-1] .= @view(split_candidates[1:ind-1]); reaminig_ops[ind:end] .= @view(split_candidates[ind+1:end])
        ls_2 = split_loopset(ls, remaining_ops)
        order_2, unrolled_2, tiled_2, vectorized_2, U_2, T_2, cost_2 = choose_order(ls_2)
        if cost_1 + cost_2 < cost_fused
            ls_2_lowered = if length(remaining_ops) > 1
                lower_and_split_loops(ls_2)
            else
                lower(ls_2, unrolled_2, tiled_2, vectorized_2, U_2, T_2)
            end
            Expr(
                :block,
                ls.preamble,
                lower(ls_1, unrolled_1, tiled_1, vectorized_1, U_1, T_1),
                ls_2_lowered
            )
        end
    end
    lower(ls, order_fused, unrolled_fused, tiled_fused, vectorized_fused, U_fused, T_fused)
end




