
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

# function dependent_outer_reducts(ls::LoopSet, op)
#     for i ∈ ls.outer_reductions
#         search_tree(parents(operations(ls)[i]), name(op)) && return true
#     end
#     false
# end

function isnopidentity(
  ls::LoopSet,
  op::Operation,
  u₁loop::Symbol,
  u₂loop::Symbol,
  vectorized::Symbol,
  u₂max::Int
)
  parents_op = parents(op)
  if iscompute(op) && instruction(op).instr === :identity
    # loopistiled = u₂max ≠ -1
    # parents_u₁syms, parents_u₂syms = parent_unroll_status(op, u₁loop, u₂loop, u₂max)
    # if (u₁unrolledsym == first(parents_u₁syms)) && (isu₂unrolled(op) == parents_u₂syms[1])
    oppstate = Base.iterate(parents_op)
    oppstate === nothing && return false
    opp, state = oppstate
    Base.iterate(parents_op, state) === nothing || return false
    name(opp) === name(op) || return false
    # @show op opp isu₁unrolled(op), isu₁unrolled(opp), isu₂unrolled(op), isu₂unrolled(opp)
    (isu₁unrolled(op) == isu₁unrolled(opp)) &
    (isu₂unrolled(op) == isu₂unrolled(opp))
  else
    false
  end
end

function set_upstream_family!(
  adal::Vector{T},
  op::Operation,
  val::T,
  ld::Vector{Symbol},
  id::Int
) where {T}
  adal[identifier(op)] == val && return # must already have been set
  if ld != loopdependencies(op) || id == identifier(op)
    (adal[identifier(op)] = val)
  end
  for opp ∈ parents(op)
    identifier(opp) == identifier(op) && continue
    set_upstream_family!(adal, opp, val, ld, id)
  end
end
function search_for_reductinit!(
  op::Operation,
  opswap::Operation,
  var::Symbol,
  loopdeps::Vector{Symbol}
)
  for (i, opp) ∈ enumerate(parents(op))
    if (name(opp) === var) &&
       (length(reduceddependencies(opp)) == 0) &&
       (length(loopdependencies(opp)) == length(loopdeps)) &&
       (length(children(opp)) == 1)
      if all(in(loopdeps), loopdependencies(opp))
        parents(op)[i] = opswap
        return opp
      end
    end
    opcheck = search_for_reductinit!(opp, opswap, var, loopdeps)
    opcheck === opp || return opcheck
  end
  return op
end
function addoptoorder!(
  ls::LoopSet,
  included_vars::Vector{Bool},
  place_after_loop::Vector{Bool},
  op::Operation,
  loopsym::Symbol,
  _n::Int,
  u₁loop::Symbol,
  u₂loop::Symbol,
  vectorized::Symbol,
  u₂max::Int
)
  lo = ls.loop_order
  id = identifier(op)
  included_vars[id] || return nothing
  loopsym ∈ loopdependencies(op) || return nothing
  for opp ∈ parents(op) # ensure parents are added first
    addoptoorder!(
      ls,
      included_vars,
      place_after_loop,
      opp,
      loopsym,
      _n,
      u₁loop,
      u₂loop,
      vectorized,
      u₂max
    )
  end
  included_vars[id] || return nothing
  included_vars[id] = false
  isunrolled = (isu₁unrolled(op)) + 1
  istiled = isu₂unrolled(op) + 1
  # optype = Int(op.node_type) + 1
  after_loop = place_after_loop[id] + 1
  if !isloopvalue(op)
    isnopidentity(ls, op, u₁loop, u₂loop, vectorized, u₂max) ||
      push!(lo[isunrolled, istiled, after_loop, _n], op)
    # if istiled
    #     isnopidentity(ls, op, u₁loop, u₂loop, vectorized, u₂max, u₂max) || push!(lo[isunrolled,2,after_loop,_n], op)
    # else
    #     isnopidentity(ls, op, u₁loop, u₂loop, vectorized, u₂max, nothing) || push!(lo[isunrolled,1,after_loop,_n], op)
    # end
  end
  # @show op, after_loop
  # isloopvalue(op) || push!(lo[isunrolled,istiled,after_loop,_n], op)
  # all(opp -> iszero(length(reduceddependencies(opp))), parents(op)) &&
  set_upstream_family!(
    place_after_loop,
    op,
    false,
    loopdependencies(op),
    identifier(op)
  ) # parents that have already been included are not moved, so no need to check included_vars to filter
  nothing
end
function replace_reduct_init!(
  ls::LoopSet,
  op::Operation,
  opsub::Operation,
  opcheck::Operation
)
  deleteat!(parents(op), 2)
  op.variable = opcheck.variable
  opsub.variable = opcheck.variable
  op.mangledvariable = opcheck.mangledvariable
  opsub.mangledvariable = opcheck.mangledvariable
  op.instruction = instruction(:identity)
  fill_children!(ls)
end
function nounrollreduction(
  op::Operation,
  u₁loop::Symbol,
  u₂loop::Symbol,
  vectorized::Symbol
)
  reduceddeps = reduceddependencies(op)
  (vectorized ∉ reduceddeps) && (u₁loop ∉ reduceddeps) && (u₂loop ∉ reduceddeps)
end
function load_short_static_reduction_first!(
  ls::LoopSet,
  u₁loop::Symbol,
  u₂loop::Symbol,
  vectorized::Symbol
)
  for op ∈ operations(ls)
    iscompute(op) || continue
    length(reduceddependencies(op)) == 0 && continue
    parents_op = parents(op)
    length(parents_op) == 2 || continue
    found = false
    parent₁deps = loopdependencies(parents_op[1])
    parent₂deps = loopdependencies(parents_op[2])
    for reduced_dep ∈ reduceddependencies(op)
      if (reduced_dep ∈ parent₁deps) || (reduced_dep ∈ parent₂deps)
        found = true
        break
      end
    end
    found || continue
    if (instruction(op).instr === :reduced_add)
      vecloop = getloop(ls, vectorized)
      if isstaticloop(vecloop) &&
         (length(vecloop) ≤ 16) &&
         nounrollreduction(op, u₁loop, u₂loop, vectorized)
        opsub = parents(op)[2]
        length(children(opsub)) == 1 || continue
        opsearch = parents(op)[1]
        opcheck = search_for_reductinit!(
          opsearch,
          opsub,
          name(opsearch),
          loopdependencies(op)
        )
        opcheck === opsearch || replace_reduct_init!(ls, op, opsub, opcheck)
      end
    elseif (instruction(op).instr === :add_fast) &&
           (instruction(first(parents(op))).instr === :identity)
      vecloop = getloop(ls, vectorized)
      if isstaticloop(vecloop) &&
         (length(vecloop) ≤ 16) &&
         nounrollreduction(op, u₁loop, u₂loop, vectorized)
        opsub = parents(op)[2]
        (
          (length(reduceddependencies(opsub)) == 0) &
          (length(children(opsub)) == 1)
        ) || continue
        opsearch = parents(op)[1]
        opcheck = search_for_reductinit!(
          opsearch,
          opsub,
          name(opsearch),
          loopdependencies(op)
        )
        opcheck === opsearch || replace_reduct_init!(ls, op, opsub, opcheck)
      end
    end
  end
end

function fillorder!(
  ls::LoopSet,
  order::Vector{Symbol},
  u₁loop::Symbol,
  u₂loop::Symbol,
  u₂max::Int,
  vectorized::Symbol
)
  load_short_static_reduction_first!(ls, u₁loop, u₂loop, vectorized)
  lo = ls.loop_order
  resize!(lo, length(ls.loopsymbols))
  ro = lo.loopnames # reverse order; will have same order as lo
  nloops = length(order)
  ops = operations(ls)
  nops = length(ops)
  included_vars = getroots!(resize!(ls.included_vars, nops), ls)
  #for i ∈ eachindex(included_vars)
  #    included_vars[i] = !included_vars[i]
  #end
  #included_vars = fill!(resize!(ls.included_vars, nops), false)
  place_after_loop = fill!(resize!(ls.place_after_loop, nops), true)
  # to go inside out, we just have to include all those not-yet included depending on the current sym
  empty!(lo)
  for _n ∈ 1:nloops
    n = 1 + nloops - _n
    ro[_n] = loopsym = order[n]
    #loopsym = order[n]
    for op ∈ ops
      addoptoorder!(
        ls,
        included_vars,
        place_after_loop,
        op,
        loopsym,
        _n,
        u₁loop,
        u₂loop,
        vectorized,
        u₂max
      )
    end
  end
end
