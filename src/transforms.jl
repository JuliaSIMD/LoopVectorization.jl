# file for misc loopset transforms

function hoist_constant_memory_accesses!(ls::LoopSet)
  hoist_stores = false
  for op ∈ operations(ls)
    if isload(op)
      length(getindicesonly(op)) == 0 && hoist_constant_vload!(ls, op)
    elseif isstore(op) && iszero(length(getindicesonly(op)))
      hoist_stores = true
    end
  end
  hoist_stores && return hoist_constant_memory_accesses_nocheck!(ls)
  ls.preamble
end

function hoist_constant_memory_accesses_nocheck!(ls::LoopSet)
  post = Expr(:block)
  for op ∈ operations(ls)
    if isstore(op) && length(getindicesonly(op)) == 0
      hoist_constant_store!(post, ls, op)
    end
  end
  post
end
function hoist_constant_vload!(ls::LoopSet, op::Operation)
  op.instr = LOOPCONSTANT
  op.node_type = constant
  add_constant_vload!(ls, op, ArrayReferenceMetaPosition(op.ref, parents(op), loopdependencies(op), reduceddependencies(op), name(op)), elementbytes)
end

function return_empty_reductinit(op::Operation, var::Symbol)
  for opp ∈ parents(op)
    if (name(opp) === var) && (length(reduceddependencies(opp)) == 0) && (length(loopdependencies(opp)) == 0) && (length(children(opp)) == 1)
      return opp
    end
    opcheck = return_empty_reductinit(opp, var)
    opcheck === opp || return opcheck
  end
  return op
end



function constant_symbol!(ls::LoopSet, op::Operation)
  # hack
  # relowers, but should make it work
  # TODO: DRY with `lower_licm_constants!` from `src/codegen/lower_constants.jl`
  skip_constant(instruction(op)) || return instruction(op).instr
  idcheck = identifier(op)
  symname = constantopname(op)
  for (id, sym) ∈ ls.preamble_symsym
    (idcheck ≢ nothing) && ((idcheck == id) && continue)
    pushpreamble!(ls, Expr(:(=), symname, sym))
    return symname
    # setconstantop!(ls, op,  sym)
    # setconstantop!(ls, op, Expr(:call, lv(:maybeconvert), ls.T, sym))
  end
  for (id,(intval,intsz,signed)) ∈ ls.preamble_symint
    (idcheck ≢ nothing) && ((idcheck == id) && continue)
    if intsz == 1
      pushpreamble!(ls, Expr(:(=), symname, intval % Bool))
    else
      pushpreamble!(ls, Expr(:(=), symname, sizeequivalent_symint_expr(intval, signed)))
    end
    return symname
  end
  for (id,floatval) ∈ ls.preamble_symfloat
    (idcheck ≢ nothing) && ((idcheck == id) && continue)
    pushpreamble!(ls, Expr(:(=), symname, Expr(:call, lv(:sizeequivalentfloat), ELTYPESYMBOL, floatval)))
    return symname
  end
  for (id,typ) ∈ ls.preamble_zeros
    (idcheck ≢ nothing) && ((idcheck == id) && continue)
    instruction(op) === LOOPCONSTANT || continue
    if typ == IntOrFloat
      pushpreamble!(ls, Expr(:(=), symname, Expr(:call, :zero, ELTYPESYMBOL)))
    elseif typ == HardInt
      pushpreamble!(ls, Expr(:(=), symname, Expr(:call, lv(:zerointeger), ELTYPESYMBOL)))
    else#if typ == HardFloat
      pushpreamble!(ls, Expr(:(=), symname, Expr(:call, lv(:zerofloat), ELTYPESYMBOL)))
    end
    return symname
  end
  for (id,f) ∈ ls.preamble_funcofeltypes
    (idcheck ≢ nothing) && ((idcheck == id) && continue)
    pushpreamble!(ls, Expr(:(=), symname, Expr(:call, reduction_zero(f), ELTYPESYMBOL)))
    return symname
  end
  throw("Constant operation symbol not found.")
end

function hoist_constant_store!(q::Expr, ls::LoopSet, op::Operation)
  op.instruction = DROPPEDCONSTANT
  op.node_type = constant

  opr = only(parents(op))
  while opr.instruction.instr === :identity
    opr.instruction = DROPPEDCONSTANT
    opr.node_type = constant
    opr = only(parents(opr))
  end
  push!(ls.outer_reductions, identifier(opr))
  
  initop = return_empty_reductinit(opr, name(opr))
  # @show last(ls.preamble.args)
  init = constant_symbol!(ls, initop)
  # @show last(ls.preamble.args)
  pushpreamble!(ls, Expr(:(=), outer_reduct_init_typename(opr), Expr(:call, lv(:typeof), init)))
  qpre = Expr(:block)
  push!(q.args, Expr(:call, lv(:unsafe_store!), Expr(:call, lv(:pointer), op.ref.ptr), outer_reduction_to_scalar_reduceq!(qpre, opr, init)))
  length(qpre.args) == 0 || pushpreamble!(ls, qpre) # creating `Expr` and pushing because `outer_reduction_to_scalar_reduceq!` uses `pushfirst!(q.args`, and we don't want it at the start of the preamble
  return nothing
end

