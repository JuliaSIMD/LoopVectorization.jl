

function add_operation!(ls_new::LoopSet, included::Vector{Int}, ls::LoopSet, op::Operation, ids::Vector{Int}, issecond::Bool)
  newid = included[identifier(op)]
  iszero(newid) || return operations(ls_new)[newid]
  vparents = Operation[]
  for opp ∈ parents(op)
    # TODO: get it so that
    # a[i] = f(a[i]) will split into one loop computing and storing f(a[i]), and the other loading from that storage if it needs it.
    if issecond && (iscompute(opp) & (!isstore(op)))
      found = false
      for opc ∈ children(opp)
        if isstore(opc) && identifier(opc) ∉ ids
          # @show opp opc op
          # replace opp with a load from opc
          parentsopc = parents(opc)
          parentsnew = length(parentsopc) > 1 ? Operation[] : NOPARENTS
          opnew = Operation(
            length(operations(ls_new)), name(opp), opc.elementbytes, instruction(:getindex), memload,
            loopdependencies(opc), reduceddependencies(opc), parentsnew, opc.ref, reducedchildren(opc)
          )
          addsetv!(ls_new.includedactualarrays, vptr(opc.ref))
          push!(operations(ls_new), opnew)
          push!(vparents, opnew)
          for i ∈ 2:length(parentsopc)
            push!(parentsnew, add_operation!(ls_new, included, ls, parentsopc[i], ids, issecond))
          end
          included[identifier(opp)] = identifier(opnew)
          found = true
          break
        end
      end
      found && continue
    end
    push!(vparents, add_operation!(ls_new, included, ls, opp, ids, issecond))
  end
  opnew = Operation(
    length(operations(ls_new)), name(op), op.elementbytes, instruction(op), op.node_type,
    loopdependencies(op), reduceddependencies(op), vparents, op.ref, reducedchildren(op)
  )
  accesses_memory(op) && addsetv!(ls_new.includedactualarrays, vptr(op.ref))
  push!(operations(ls_new), opnew)
  included[identifier(op)] = identifier(opnew)
  opnew
end

function append_if_included!(vnew, vold, included)
  for (i, v) ∈ vold
    id = included[i]
    iszero(id) || push!(vnew, (id, v))
  end
end

function split_loopset(ls::LoopSet, ids::Vector{Int}, issecond::Bool)
  ls_new = LoopSet(:LoopVectorization)
  included = zeros(Int, length(operations(ls)))
  for i ∈ ids
    add_operation!(ls_new, included, ls, operations(ls)[i], ids, issecond)
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
  append_if_included!(ls_new.preamble_funcofeltypes, ls.preamble_funcofeltypes, included)
  for i ∈ ls.outer_reductions
    id = included[i]
    iszero(id) || push!(ls_new.outer_reductions, id)
  end
  # TODO: allow them to differ. E.g., non-AVX2 x86 cpus don't have efficient integer calculations
  # Therefore, it would be profitable to split for this reason.
  # However, currently the default assumption in vector width will be wrong, so we should calculate
  # it correctly (like ls.vector_width); wrong (too high) value will encourage splitting when
  # it shouldn't.
  # Current behavior is incorrect when VECWIDTH chosen does actually differ between
  # split loops and the loops are statically sized, because code gen will then assume it is correct...
  l1, l2, l3 = cache_sze(ls)
  set_hw!(ls_new, reg_size(ls), reg_count(ls), cache_lnsze(ls), l1, l2, l3)
  ls_new.vector_width = ls.vector_width
  fill_offset_memop_collection!(ls)
  # println("ls_new operations:")
  # display(ls_new.operations)
  # println()
  ls_new
end

function returned_ops(ls::LoopSet)
    ops = operations(ls)
    retops = Int[]
    for op ∈ ops
        isstore(op) && push!(retops, identifier(op))
    end
    for i ∈ ls.outer_reductions
        push!(retops, i)
    end
    retops
end

function lower_and_split_loops(ls::LoopSet, inline::Int)
  split_candidates = returned_ops(ls)
  length(split_candidates) > 1 || return lower(ls, inline)
  order_fused, unrolled_fused, tiled_fused, vectorized_fused, U_fused, T_fused, cost_fused, shouldinline_fused = choose_order_cost(ls)
  remaining_ops = Vector{Int}(undef, length(split_candidates) - 1); split_1 = Int[0];
  # for (ind,i) ∈ enumerate(split_candidates)
  looplenpen = 0.05
  ls_looplen = looplengthprod(ls)*looplenpen
  for (ind,i) ∈ enumerate(split_candidates)
    split_1[1] = i
    ls_1 = split_loopset(ls, split_1, false)
    order_1, unrolled_1, tiled_1, vectorized_1, U_1, T_1, cost_1, shouldinline_1 = choose_order_cost(ls_1)
    remaining_ops[1:ind-1] .= @view(split_candidates[1:ind-1]); remaining_ops[ind:end] .= @view(split_candidates[ind+1:end])
    ls_2 = split_loopset(ls, remaining_ops, true)
    order_2, unrolled_2, tiled_2, vectorized_2, U_2, T_2, cost_2, shouldinline_2 = choose_order_cost(ls_2)
    # U_1 = T_1 = U_2 = T_2 = 2
    # return ls_1, ls_2
    if cost_1 + cost_2 + looplenpen*(looplengthprod(ls_1) + looplengthprod(ls_2)) ≤ muladd(0.9, cost_fused, ls_looplen)
      ls_2_lowered = if length(remaining_ops) > 1
        inline = iszero(inline) ? (shouldinline_1 % Int) : inline
        lower_and_split_loops(ls_2, inline)
      else
        doinline = inlinedecision(inline, shouldinline_1 | shouldinline_2)
        lower(ls_2, order_2, unrolled_2, tiled_2, vectorized_2, U_2, T_2, doinline)
      end
      return Expr(
        :block,
        ls.preamble,
        lower(ls_1, order_1, unrolled_1, tiled_1, vectorized_1, U_1, T_1, false),
        ls_2_lowered,
        nothing
      )
    end
    length(split_candidates) == 2 && break
  end
  doinline = inlinedecision(inline, shouldinline_fused)
  lower(ls, order_fused, unrolled_fused, tiled_fused, vectorized_fused, U_fused, T_fused, doinline)
end




