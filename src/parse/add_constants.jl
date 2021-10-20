const CONSTANT_SYMBOLS = (:nothing, :Float64, :Float32, :Int8, :UInt8, :Int16, :UInt16, :Int32, :UInt32, :Int64, :UInt64)
function add_constant!(ls::LoopSet, var::Symbol, elementbytes::Int)
  var ∈ ls.loopsymbols && return add_loopvalue!(ls, var, elementbytes)
  globalconst = Base.sym_in(var, CONSTANT_SYMBOLS)
  instr = globalconst ? Instruction(GLOBALCONSTANT, var) : LOOPCONSTANT
  op = Operation(length(operations(ls)), var, elementbytes, instr, constant, NODEPENDENCY, Symbol[], NOPARENTS)
  rop = pushop!(ls, op, var)
  (!globalconst && (rop === op)) && pushpreamble!(ls, op, var)
  rop
end
# function add_constant!(ls::LoopSet, var, elementbytes::Int = 8)
#     sym = gensym(:loopconstant)
#     pushpreamble!(ls, Expr(:(=), sym, var))
#     add_constant!(ls, sym, elementbytes)
# end
function add_constant!(ls::LoopSet, var::Number, elementbytes::Int = 8, varname = gensym!(ls, "loopconstnumber"))
  op = Operation(length(operations(ls)), varname, elementbytes, LOOPCONSTANT, constant, NODEPENDENCY, Symbol[], NOPARENTS)
  ops = operations(ls)
  typ = var isa Integer ? HardInt : HardFloat
  if iszero(var)
    for (id,typ_) ∈ ls.preamble_zeros
      (instruction(ops[id]) == LOOPCONSTANT && typ == typ_) && return ops[id]
    end
    push!(ls.preamble_zeros, (identifier(op),typ))
  elseif var isa Integer
    idescript = integer_description(var)
    for (id,descript) ∈ ls.preamble_symint
      if (instruction(ops[id]) == LOOPCONSTANT) && (idescript == descript)
        return ops[id]
      end
    end
    push!(ls.preamble_symint, (identifier(op), idescript))
  else#if var isa FloatX
    for (id,fvar) ∈ ls.preamble_symfloat
      (instruction(ops[id]) == LOOPCONSTANT && fvar == var) && return ops[id]
    end
    push!(ls.preamble_symfloat, (identifier(op), var))
  end
  rop = pushop!(ls, op)
  rop === op || return rop
  pushpreamble!(ls, Expr(:(=), name(op), var))
  rop
end
function ensure_constant_lowered!(ls::LoopSet, op::Operation)
  if iscompute(op)
    call = callexpr(instruction(op))
    for opp ∈ parents(op)
      ensure_constant_lowered!(ls, opp)
      push!(call.args, name(opp))
    end
    pushpreamble!(ls, Expr(:(=), name(op), call))
  elseif isconstant(op) & !isconstantop(op)
    opid = identifier(op)
    for (id, sym) ∈ ls.preamble_symsym
      if id == opid
        pushpreamble!(ls, Expr(:(=), name(op), sym))
        return
      end
    end
    for (id,(intval,intsz,signed)) ∈ ls.preamble_symint
      if id == opid
        if intsz == 1
          pushpreamble!(ls, Expr(:(=), name(op), intval % Bool))
        elseif signed
          pushpreamble!(ls, Expr(:(=), name(op), intval))
        else
          pushpreamble!(ls, Expr(:(=), name(op), intval % UInt))
        end
        return
      end
    end
    for (id,floatval) ∈ ls.preamble_symfloat
      if id == opid
        pushpreamble!(ls, Expr(:(=), name(op), floatval))
        return
      end
      
    end
    for (id,typ) ∈ ls.preamble_zeros
      if id == opid
        pushpreamble!(ls, Expr(:(=), name(op), staticexpr(0)))
        return
      end
    end
    for (id,f) ∈ ls.preamble_funcofeltypes
      if id == opid
        pushpreamble!(ls, Expr(:(=), name(op), Expr(:call, reduction_zero(f), Float64)))
        return
      end
    end
  end
end
function ensure_constant_lowered!(ls::LoopSet, mpref::ArrayReferenceMetaPosition, ind::Symbol)
  length(loopdependencies(mpref)) == 0 && return
  for (id,opp) ∈ enumerate(parents(mpref))
    if name(opp) === ind
      ensure_constant_lowered!(ls, opp)
    end
  end
  return nothing
end
function add_constant_vload!(ls::LoopSet, op::Operation, mpref::ArrayReferenceMetaPosition, elementbytes::Int)
  temp = gensym!(ls, "intermediateconstref")
  use_getindex = vptr(name(mpref)) === mpref.mref.ptr
  vloadcall = use_getindex ? Expr(:call, Base.getindex, name(mpref)) : Expr(:call, lv(:_vload), mpref.mref.ptr)
  nindices = length(getindices(op))
  # getoffsets(op) .+= 1
  if nindices > 0
    dummyloop = first(ls.loops)
    for ind ∈ getindicesonly(op)
      ensure_constant_lowered!(ls, mpref, ind)
    end
    if use_getindex
      append!(vloadcall.args, mem_offset(op, UnrollArgs(dummyloop, dummyloop, dummyloop, 0, 0, 0), fill(false,nindices), true, ls, false).args)
    else
      push!(vloadcall.args, mem_offset(op, UnrollArgs(dummyloop, dummyloop, dummyloop, 0, 0, 0), fill(false,nindices), true, ls, false))
    end
  end
  if use_getindex
    pushpreamble!(ls, Expr(:(=), temp, :(@inbounds $vloadcall)))
  else
    push!(vloadcall.args, Expr(:call, lv(:False)), staticexpr(reg_size(ls)))
    pushpreamble!(ls, Expr(:(=), temp, vloadcall))
  end
  pushpreamble!(ls, Expr(:(=), name(op), temp))
  pushpreamble!(ls, op, temp)
  return temp
end
function add_constant!(ls::LoopSet, mpref::ArrayReferenceMetaPosition, elementbytes::Int)
  op = Operation(length(operations(ls)), varname(mpref), elementbytes, LOOPCONSTANT, constant, NODEPENDENCY, Symbol[], NOPARENTS, mpref.mref)
  add_vptr!(ls, op)
  temp = add_constant_vload!(ls, op, mpref, elementbytes)
  pushop!(ls, op, temp)
end
# This version has loop dependencies. var gets assigned to sym when lowering.
# value is what will get assigned within the loop.
# assignedsym will be assigned to value within the preamble
function add_constant!(
    ls::LoopSet, value::Symbol, deps::Vector{Symbol}, assignedsym::Symbol, elementbytes::Int, f::Symbol = Symbol("")
)
  value ∈ ls.loopsymbols && return add_loopvalue!(ls, value, elementbytes)
  retop = get(ls.opdict, value, nothing)
  if retop === nothing
    op = Operation(length(operations(ls)), assignedsym, elementbytes, Instruction(f, value), constant, deps, NODEPENDENCY, NOPARENTS)
  else
    op = Operation(length(operations(ls)), assignedsym, elementbytes, :identity, compute, deps, reduceddependencies(retop), [retop])
  end
  pushop!(ls, op, assignedsym)
end
# function add_constant!(
#     ls::LoopSet, value, deps::Vector{Symbol}, assignedsym::Symbol, elementbytes::Int, f::Symbol = Symbol("")
# )
#     intermediary = gensym(:intermediate) # hack, passing meta info here
#     pushpreamble!(ls, Expr(:(=), intermediary, value))
#     add_constant!(ls, intermediary, deps, assignedsym, f, elementbytes)
# end
function add_constant!(
    ls::LoopSet, value::Number, deps::Vector{Symbol}, assignedsym::Symbol, elementbytes::Int
)
  op = add_constant!(ls, gensym!(ls, string(value)), deps, assignedsym, elementbytes, :numericconstant)
  pushpreamble!(ls, op, value)
  op
end
