function maybeaddref!(ls::LoopSet, op::Operation)
  ref = op.ref
  id = findfirst(==(ref), ls.refs_aliasing_syms)
  # try to CSE
  if id === nothing
    push!(ls.syms_aliasing_refs, name(op))
    push!(ls.refs_aliasing_syms, ref)
    0
  else
    id
  end
end

function add_load!(ls::LoopSet, op::Operation, actualarray::Bool = true)
  @assert isload(op)
  if (id = maybeaddref!(ls, op)) > 0 # try to CSE
    opp = ls.opdict[ls.syms_aliasing_refs[id]]
    op_inds = getindicesonly(op)
    li = op.ref.loopedindex
    allmatch = true
    parents_op = parents(op)
    parents_opp = parents(opp)
    for i ∈ eachindex(op_inds)
      li[i] && continue
      ind = op_inds[i]
      if (id_op = parentind(ind, op)) > 0
        if (id_opp = parentind(ind, opp)) > 0
          if !matches(parents_op[id_op], parents_opp[id_opp])
            allmatch = false
            break
          end
        end
      end
    end
    if allmatch
      opp = isstore(opp) ? first(parents(opp)) : opp
      ls.opdict[name(op)] = opp
      return opp
    end
  end
  add_vptr!(ls, op.ref.ref.array, vptr(op), actualarray)
  pushop!(ls, op, name(op))
end

function add_load!(ls::LoopSet, var::Symbol, array::Symbol, rawindices, elementbytes::Int)
  mpref = array_reference_meta!(ls, array, rawindices, elementbytes, var)
  add_load!(ls, mpref, elementbytes)
end
function load_is_constant(mpref::ArrayReferenceMetaPosition)
  li = mpref.mref.loopedindex
  inds = getindicesonly(mpref)
  for i ∈ eachindex(li)
    li[i] && return false
    if (id = parentind(inds[i], mpref)) > 0
      isinitializedconst(parents(mpref)[id]) || return false
    end
  end
  true
end
function add_load!(ls::LoopSet, mpref::ArrayReferenceMetaPosition, elementbytes::Int)
  if length(mpref.loopdependencies) == 0 || load_is_constant(mpref)
    return add_constant!(ls, mpref, elementbytes)
  end
  op = Operation(ls, varname(mpref), elementbytes, :getindex, memload, mpref)
  add_load!(ls, op, true)
end

# for use with broadcasting
function add_simple_load!(
  ls::LoopSet,
  var::Symbol,
  ref::ArrayReference,
  elementbytes::Int,
  actualarray::Bool = true,
)
  loopdeps = copy(getindicesonly(ref))
  mref = ArrayReferenceMeta(ref, fill(true, length(loopdeps)))
  add_simple_load!(ls, var, mref, loopdeps, elementbytes, actualarray)
end
function add_simple_load!(
  ls::LoopSet,
  var::Symbol,
  mref::ArrayReferenceMeta,
  loopdeps::Vector{Symbol},
  elementbytes::Int,
  actualarray::Bool = true,
)
  op = Operation(
    length(operations(ls)),
    var,
    elementbytes,
    :getindex,
    memload,
    loopdeps,
    NODEPENDENCY,
    NOPARENTS,
    mref,
  )
  add_vptr!(ls, op.ref.ref.array, vptr(op), actualarray)
  pushop!(ls, op, var)
end
function add_load_ref!(ls::LoopSet, var::Symbol, ex::Expr, elementbytes::Int)
  array, rawindices = ref_from_ref!(ls, ex)
  add_load!(ls, var, array, rawindices, elementbytes)
end
function add_load_getindex!(ls::LoopSet, var::Symbol, ex::Expr, elementbytes::Int)
  array, rawindices = ref_from_getindex!(ls, ex)
  add_load!(ls, var, array, rawindices, elementbytes)
end

function add_loopvalue!(ls::LoopSet, arg::Symbol, elementbytes::Int)
  # check for CSE opportunity
  instr = Instruction(arg, arg)
  for op ∈ operations(ls)#check to CSE
    (op.variable === arg && instr == instruction(op)) && return op
  end
  op = Operation(length(operations(ls)), arg, elementbytes, instr, loopvalue, [arg])
  pushop!(ls, op, arg)
end
