function addsetv!(s::AbstractVector{T}, v::T) where {T}
  for sᵢ ∈ s
    sᵢ === v && return nothing
  end
  push!(s, v)
  nothing
end
function mergesetv!(s1::AbstractVector{T}, s2::AbstractVector{T}) where {T}
  for s ∈ s2
    addsetv!(s1, s)
  end
  nothing
end
function mergesetdiffv!(
  s1::AbstractVector{T},
  s2::AbstractVector{T},
  s3::AbstractVector{T}
) where {T}
  for s ∈ s2
    s ∉ s3 && addsetv!(s1, s)
  end
  nothing
end
# Everything in arg2 (s1) that isn't in arg3 (s2) is added to arg1 (s3)
function setdiffv!(
  s3::AbstractVector{T},
  s1::AbstractVector{T},
  s2::AbstractVector{T}
) where {T}
  for s ∈ s1
    (s ∈ s2) || (s ∉ s3 && push!(s3, s))
  end
end
function setdiffv!(
  s4::AbstractVector{T},
  s3::AbstractVector{T},
  s1::AbstractVector{T},
  s2::AbstractVector{T}
) where {T}
  for s ∈ s1
    (s ∈ s2) ? (s ∉ s4 && push!(s4, s)) : (s ∉ s3 && push!(s3, s))
  end
end
function update_deps!(
  deps::Vector{Symbol},
  reduceddeps::Vector{Symbol},
  parent::Operation
)
  mergesetv!(deps, loopdependencies(parent))#, reduceddependencies(parent))
  if !(isload(parent) || isconstant(parent)) #&& !isreductcombineinstr(parent)
    mergesetv!(reduceddeps, reduceddependencies(parent))
  end
  nothing
end

function pushparent!(
  parents::Vector{Operation},
  deps::Vector{Symbol},
  reduceddeps::Vector{Symbol},
  parent::Operation
)
  @assert parents !== NOPARENTS
  push!(parents, parent)
  update_deps!(deps, reduceddeps, parent)
end
# function pushparent!(mpref::ArrayReferenceMetaPosition, parent::Operation)
#     pushparent!(mpref.parents, mpref.loopdependencies, mpref.reduceddeps, parent)
# end
function add_parent!(
  vparents::Vector{Operation},
  deps::Vector{Symbol},
  reduceddeps::Vector{Symbol},
  ls::LoopSet,
  var,
  elementbytes::Int,
  position::Int
)
  parent = if var isa Symbol
    # if var === :kern_1_1
    #     @show operations(ls) ls.preamble_symsym
    # end
    opp = getop(ls, var, elementbytes)
    # if var === :kern_1_1
    #     @show operations(ls) ls.preamble_symsym
    # end
    # @show var opp first(operations(ls)) opp === first(operations(ls))
    if iscompute(opp) &&
       instruction(opp).instr === :identity &&
       length(loopdependencies(opp)) < position &&
       isone(length(parents(opp))) &&
       name(opp) === name(first(parents(opp)))
      first(parents(opp))
    else
      opp
    end
  elseif var isa Expr #CSE candidate
    add_operation!(ls, gensym!(ls, "temp"), var, elementbytes, position)
  else # assumed constant
    add_constant!(ls, var, elementbytes)
    # add_constant!(ls, var, deps, gensym(:loopredefconst), elementbytes)
  end
  pushparent!(vparents, deps, reduceddeps, parent)
end
# function add_reduction!(
#     vparents::Vector{Operation}, deps::Vector{Symbol}, reduceddeps::Vector{Symbol}, ls::LoopSet, var::Symbol, elementbytes::Int
# )
#     get!(ls.opdict, var) do
#         add_constant!(ls, var, elementbytes)
#     end
# end
function search_tree(opv::Vector{Operation}, var::Symbol) # relies on cycles being forbidden
  for opp ∈ opv
    name(opp) === var && return true
    search_tree(parents(opp), var) && return true
  end
  false
end

search_tree_for_ref(::LoopSet, ::Vector{Operation}, ::Nothing, var::Symbol) =
  var, false
function search_tree_for_ref(
  ls::LoopSet,
  opv::Vector{Operation},
  mpref::ArrayReferenceMetaPosition,
  var::Symbol
) # relies on cycles being forbidden
  for opp ∈ opv
    if opp.ref == mpref.mref
      if varname(mpref) === var
        id = findfirst(==(mpref.mref), ls.refs_aliasing_syms)
        mpref.varname = var = id === nothing ? var : ls.syms_aliasing_refs[id]
        return var, true
      end
    end
    var, found = search_tree_for_ref(ls, parents(opp), mpref, var)
    found && return (var, found)
  end
  var, false
end
function search_tree(opv::Vector{Operation}, var::Operation) # relies on cycles being forbidden
  for opp ∈ opv
    opp === var && return true
    search_tree(parents(opp), var) && return true
  end
  false
end
function update_reduction_status!(
  parentvec::Vector{Operation},
  deps::Vector{Symbol},
  parent::Symbol
)
  for opp ∈ parentvec
    if name(opp) === parent
      mergesetv!(reducedchildren(opp), deps)
      break
    elseif search_tree(parents(opp), parent)
      mergesetv!(reducedchildren(opp), deps)
      update_reduction_status!(parents(opp), deps, parent)
      break
    end
  end
end
# function add_compute!(ls::LoopSet, op::Operation)
# @assert iscompute(op)
# pushop!(ls, child, name(op))
# end
# function isreductzero(op::Operation, ls::LoopSet, reduct_zero::Symbol)
#     isconstant(op) || return false
#     reduct_zero === op.instruction.mod && return true
#     if reduct_zero === :zero
#         iszero(ls, op) && return true
#     elseif reduct_zero === :one
#         isone(ls, op) && return true
#     end
#     false
# end
function add_reduced_deps!(op::Operation, reduceddeps::Vector{Symbol})
  # op.dependencies = copy(loopdependencies(op))
  # mergesetv!(loopdependencies(op), reduceddeps)
  reduceddepsop = reduceddependencies(op)
  if reduceddepsop === NODEPENDENCY
    op.reduced_deps = copy(reduceddeps)
  else
    mergesetv!(reduceddepsop, reduceddeps)
  end
  # reduceddepsop = reducedchildren(op)
  # if reduceddepsop === NODEPENDENCY
  #     op.reduced_children = copy(reduceddeps)
  # else
  #     mergesetv!(reduceddepsop, reduceddeps)
  # end
  nothing
end

function substitute_op_in_parents!(
  vparents::Vector{Operation},
  replacer::Operation,
  replacee::Operation,
  reduceddeps::Vector{Symbol},
  reductsym::Symbol
)
  found = false
  for i ∈ eachindex(vparents)
    opp = vparents[i]
    if opp === replacee
      vparents[i] = replacer
      found = true
    else
      fopp = substitute_op_in_parents!(
        parents(opp),
        replacer,
        replacee,
        reduceddeps,
        reductsym
      )
      if fopp
        add_reduced_deps!(opp, reduceddeps)
        # FIXME: https://github.com/JuliaSIMD/LoopVectorization.jl/issues/259
        # 
        opp.variable = reductsym
        opp.mangledvariable = Symbol("##", reductsym, :_)
      end
      found |= fopp
    end
  end
  found
end

function add_reduction_update_parent!(
  vparents::Vector{Operation},
  deps::Vector{Symbol},
  reduceddeps::Vector{Symbol},
  ls::LoopSet,
  parent::Operation,
  instr::Instruction,
  reduction_ind::Int,
  elementbytes::Int
)
  var = name(parent)
  # isouterreduction = iszero(length(loopdependencies(parent))) && (parent.instruction === LOOPCONSTANT)
  isouterreduction = parent.instruction === LOOPCONSTANT
  # @show instr, vparents, parent, reduction_ind
  # if parent is not an outer reduction...
  # if !isouterreduction && !isreductzero(parent, ls, reduct_zero)
  # @show isouterreduction, parent, length(loopdependencies(parent))
  add_reduct_instruct =
    (!isouterreduction && !isconstant(parent)) && (instr.instr ≢ :ifelse)
  if add_reduct_instruct
    # if instr.instr === :ifelse
    # @assert length(vparents) == 2
    # instrclass = reduction_instruction_class(instruction(vparents[2])) # key allows for faster lookups
    # else
    instrclass = reduction_instruction_class(instr) # key allows for faster lookups
    # end
    reduct_zero = reduction_zero(instrclass)
    # We add
    reductcombine = reduction_scalar_combine(instrclass)
    # reductcombine = :identity
    reductsym = gensym!(ls, "reduction")
    reductzero_sym = gensym!(ls, "reduction##zero")
    # reductsym = gensym(:reduction)
    reductinit = add_constant!(
      ls,
      reductzero_sym,
      loopdependencies(parent),
      reductsym,
      elementbytes,
      :numericconstant
    )
    if reduct_zero === :zero
      push!(ls.preamble_zeros, (identifier(reductinit), IntOrFloat))
    else
      push!(ls.preamble_funcofeltypes, (identifier(reductinit), instrclass))
    end
  else
    reductinit = parent
    reductsym = var
    reductcombine = :identity#Symbol("")
  end
  combineddeps = copy(deps)
  mergesetv!(combineddeps, reduceddeps)
  # directdependency && pushparent!(vparents, deps, reduceddeps, reductinit)#parent) # deps and reduced deps will not be disjoint
  if reduction_ind > 0 # if is directdependency
    insert!(vparents, reduction_ind, reductinit)
    if instr.instr ∈ (:-, :vsub!, :vsub, :/, :vfdiv!, :vfidiv!)
      update_deps!(deps, reduceddeps, reductinit)#parent) # deps and reduced deps will not be disjoint
    end
  elseif !isouterreduction && reductinit !== parent
    substitute_op_in_parents!(
      vparents,
      reductinit,
      parent,
      reduceddeps,
      reductsym
    )
  end
  update_reduction_status!(vparents, reduceddeps, name(reductinit))
  # this is the op added by add_compute
  op = Operation(
    length(operations(ls)),
    reductsym,
    elementbytes,
    instr,
    compute,
    deps,
    reduceddeps,
    vparents
  )
  isouterreduction && push!(ls.outer_reductions, identifier(op))
  opout = pushop!(ls, op, var) # note this overwrites the entry in the operations dict, but not the vector
  # isouterreduction || iszero(length(reduceddeps)) && return opout
  # return opout
  isouterreduction && return opout
  # create child op, which is the reduction combination
  childrdeps = Symbol[]
  childparents = Operation[op]#, parent ]
  if add_reduct_instruct
    @assert childparents !== NOPARENTS
    push!(childparents, parent)
  end
  childdeps = loopdependencies(reductinit)
  setdiffv!(childrdeps, loopdependencies(op), childdeps)
  child = Operation(
    length(operations(ls)),
    name(parent),
    elementbytes,
    reductcombine,
    compute,
    childdeps,
    childrdeps,
    childparents
  )
  # child = Operation(
  #     length(operations(ls)), name(parent), elementbytes, Instruction(reductcombine,:identity), compute, childdeps, childrdeps, childparents
  # )
  pushop!(ls, child, name(parent))
  opout
end

function substitute!(ex::Expr, d::Dict{Symbol,Symbol})
  for (i, arg) ∈ enumerate(ex.args)
    if arg isa Symbol
      ex.args[i] = get(d, arg, arg)
    elseif arg isa Expr
      substitute!(arg, d)
    end
  end
end
function argsymbol(
  ls::LoopSet,
  arg,
  mpref,
  elementbytes::Int,
  position::Int
)::Symbol
  argsym = gensym!(ls, "anonarg")
  if mpref === nothing
    add_operation!(ls, argsym, arg, elementbytes, position)
  else
    add_operation!(ls, argsym, arg, mpref, elementbytes, position)
  end
  return argsym
end
function add_anon_func!(
  ls::LoopSet,
  LHS::Symbol,
  f::Expr,
  ex::Expr,
  position::Int,
  mpref::Union{Nothing,ArrayReferenceMetaPosition},
  elementbytes::Int
)::Operation
  d = Dict{Symbol,Symbol}()
  anonargs = f.args[1]
  if anonargs isa Symbol
    @assert length(ex.args) == 2
    arg = ex.args[2]
    if !(arg isa Symbol)
      argsym = argsymbol(ls, arg, mpref, elementbytes, position)
    else
      argsym = arg::Symbol
    end
    d[anonargs] = argsym
  else
    @assert Meta.isexpr(anonargs, :tuple)
    callargs = @view(ex.args[2:end])
    for i ∈ eachindex(anonargs.args, callargs)
      arg = callargs[i]
      if !(arg isa Symbol)
        argsym = argsymbol(ls, arg, mpref, elementbytes, position)
      else
        argsym = arg::Symbol
      end
      d[anonargs.args[i]] = argsym
    end
  end
  anonbody = f.args[2]
  substitute!(anonbody, d)
  for i ∈ 1:length(anonbody.args)-1
    exᵢ = anonbody.args[i]
    exᵢ isa Expr && push!(ls, exᵢ, elementbytes, position, mpref)
  end
  lastline = last(anonbody.args)
  retop = if lastline isa Symbol
    add_compute!(
      ls,
      LHS,
      instruction(:identity),
      Operation[getop(ls, lastline)],
      elementbytes
    )
  elseif Meta.isexpr(lastline, :call)
    add_compute!(ls, LHS, lastline, elementbytes, position, mpref)
  elseif Meta.isexpr(lastline, :(=))
    add_compute!(ls, LHS, lastline.args[2], elementbytes, position, mpref)
  else
    throw(LoopError("Last line of anon func not understood: $lastline"))
  end
  return retop
end
# TODO: DRY, this is similar to `find_samename_constparent` in `condense_loopset.jl`
function find_inner_reduct_parent(op::Operation, opname::Symbol)
  for opp ∈ parents(op)
    (((isconstant(opp)) && (name(opp) === opname))) && return opp
    opptemp = find_samename_constparent(opp, opname)
    opptemp === opp || return opptemp
  end
  op
end

function maybe_fix_reduced_deps!(
  ::LoopSet,
  deps::Vector{Symbol},
  reduceddeps::Vector{Symbol},
  parent::Operation,
  mpref::ArrayReferenceMetaPosition,
  ::Int
)
  loopdeps_parent = loopdependencies(parent)
  reduceddeps_parent = reduceddependencies(parent)
  loopdeps_mpref = loopdependencies(mpref)
  loopdeps_new = Symbol[]
  pparent_id = findfirst(Base.Fix2(===, name(parent)) ∘ name, parents(parent))
  pparent_id === nothing && return deps, reduceddeps
  pparent = parents(parent)[pparent_id]
  @assert length(loopdependencies(pparent)) ==
          length(loopdeps_parent) + length(reduceddeps_parent)
  reduceddeps_pparent = reduceddependencies(pparent)
  for ld ∈ loopdeps_parent
    if ld ∈ loopdeps_mpref
      push!(loopdeps_new, ld)
    else
      push!(reduceddeps_parent, ld)
      push!(reduceddeps_pparent, ld)
    end
  end
  parent.dependencies = loopdeps_new
  reduct_init = find_inner_reduct_parent(pparent, name(pparent))
  reduct_init.dependencies = loopdeps_new
  reduct_init.reduced_children = reduceddeps_pparent
  # ld = loopdependencies(mpref)
  return loopdeps_new, copy(reduceddeps_parent)
end
function add_compute!(
  ls::LoopSet,
  var::Symbol,
  ex::Expr,
  elementbytes::Int,
  position::Int,
  mpref::Union{Nothing,ArrayReferenceMetaPosition} = nothing
)::Operation
  @assert ex.head === :call
  fexpr = first(ex.args)
  Meta.isexpr(fexpr, :(->)) &&
    return add_anon_func!(ls, var, fexpr, ex, position, mpref, elementbytes)
  # instr = instruction(first(ex.args))::Symbol
  instr = instruction!(ls, first(ex.args))::Instruction

  args = @view(ex.args[2:end])
  if (instr.instr === :pow_fast || instr.instr === :(^)) && length(args) == 2
    arg1 = args[1]
    arg2 = args[2]
    if arg1 isa Number && convert(Float64, arg1) === -1.0
      return add_compute!(
        ls,
        var,
        :(2iseven($arg2) - 1),
        elementbytes,
        position,
        mpref
      )
    end
    if arg2 isa Number
      return add_pow!(ls, var, args[1], arg2, elementbytes, position)
    elseif arg2 isa Val
      arg2num = Int(static(ex.args[3]))::Int
      return add_pow!(ls, var, args[1], arg2num, elementbytes, position)
    end
  elseif instr.instr === :oftype && length(args) == 2
    return get_arg!(ls, args[2], elementbytes, position)
  end
  vparents = Operation[]
  deps = Symbol[]
  reduceddeps = Symbol[]
  reduction_ind = 0
  # @show ex first(operations(ls)) === getop(ls, :kern_1_1, elementbytes) first(operations(ls)) getop(ls, :kern_1_1, elementbytes)
  for (ind, arg) ∈ enumerate(args)
    if var === arg
      reduction_ind = ind
      # add_reduction!(vparents, deps, reduceddeps, ls, arg, elementbytes)
      getop(ls, var, elementbytes)
    elseif arg isa Expr
      isref, argref = tryrefconvert(ls, arg, elementbytes, varname(mpref))
      if isref
        if mpref == argref
          if varname(mpref) === var
            id = findfirst(==(mpref.mref), ls.refs_aliasing_syms)
            mpref.varname =
              var = id === nothing ? var : ls.syms_aliasing_refs[id]
            reduction_ind = ind
            mergesetv!(
              deps,
              loopdependencies(add_load!(ls, argref, elementbytes))
            )
          else
            pushparent!(
              vparents,
              deps,
              reduceddeps,
              add_load!(ls, argref, elementbytes)
            )
          end
        else
          argref.varname = gensym!(ls, "tempload")
          pushparent!(
            vparents,
            deps,
            reduceddeps,
            add_load!(ls, argref, elementbytes)
          )
        end
      else
        add_parent!(
          vparents,
          deps,
          reduceddeps,
          ls,
          arg,
          elementbytes,
          position
        )
      end
    elseif arg ∈ ls.loopsymbols
      loopsymop = add_loopvalue!(ls, arg, elementbytes)
      pushparent!(vparents, deps, reduceddeps, loopsymop)
    else
      add_parent!(vparents, deps, reduceddeps, ls, arg, elementbytes, position)
    end
  end
  reduction = reduction_ind > 0
  loopnestview = view(ls.loopsymbols, 1:position)
  if iszero(length(deps)) && reduction
    append!(deps, loopnestview)
    append!(reduceddeps, loopnestview)
  else
    newloopdeps = Symbol[]
    newreduceddeps = Symbol[]
    setdiffv!(newloopdeps, newreduceddeps, deps, loopnestview)
    mergesetv!(newreduceddeps, reduceddeps)
    deps = newloopdeps
    reduceddeps = newreduceddeps
  end
  # fix for #347
  if (mpref ≢ nothing) &&
     (
       (reduction_ind ≠ 0) &
       (mpref.varname === var) &
       (position > length(loopdependencies(mpref)))
     ) &&
     isone(length(vparents)) &&
     (position == length(loopdependencies(only(vparents))))
    deps, reduceddeps = maybe_fix_reduced_deps!(
      ls,
      deps,
      reduceddeps,
      only(vparents),
      mpref,
      position
    )
  end
  # @show reduction, search_tree(vparents, var) ex var vparents mpref get(ls.opdict, var, nothing) search_tree_for_ref(ls, vparents, mpref, var) # relies on cycles being forbidden
  if reduction || search_tree(vparents, var)
    return add_reduction!(
      ls,
      var,
      reduceddeps,
      deps,
      vparents,
      reduction_ind,
      elementbytes,
      instr
    )
  else
    if mpref ≢ nothing && (
      (length(loopdependencies(mpref)) < position) |
      (length(reduceddependencies(mpref)) > 0)
    )
      var, found = search_tree_for_ref(ls, vparents, mpref, var)
      found && return add_reduction!(
        ls,
        var,
        reduceddeps,
        deps,
        vparents,
        reduction_ind,
        elementbytes,
        instr
      )
    end
    op = Operation(
      length(operations(ls)),
      var,
      elementbytes,
      instr,
      compute,
      deps,
      reduceddeps,
      vparents
    )
    return pushop!(ls, op, var)
  end
  # maybe_const_compute!(ls, op, elementbytes, position)
end
function add_reduction!(
  ls::LoopSet,
  var::Symbol,
  reduceddeps,
  deps,
  vparents,
  reduction_ind,
  elementbytes,
  instr
)
  parent = ls.opdict[var]
  setdiffv!(reduceddeps, deps, loopdependencies(parent))
  # parent = getop(ls, var, elementbytes)
  # if length(reduceddeps) == 0
  if all(!in(deps), reduceddeps)
    if reduction_ind != 0
      insert!(vparents, reduction_ind, parent)
      mergesetv!(deps, loopdependencies(parent))
    end
    op = Operation(
      length(operations(ls)),
      var,
      elementbytes,
      instr,
      compute,
      deps,
      reduceddeps,
      vparents
    )
    pushop!(ls, op, var)
  else
    add_reduction_update_parent!(
      vparents,
      deps,
      reduceddeps,
      ls,
      parent,
      instr,
      reduction_ind,
      elementbytes
    )
  end
end

function add_compute!(
  ls::LoopSet,
  LHS::Symbol,
  instr,
  vparents::Vector{Operation},
  elementbytes::Int
)
  deps = Symbol[]
  reduceddeps = Symbol[]
  for parent ∈ vparents
    update_deps!(deps, reduceddeps, parent)
  end
  op = Operation(
    length(operations(ls)),
    LHS,
    elementbytes,
    instr,
    compute,
    deps,
    reduceddeps,
    vparents
  )
  pushop!(ls, op, LHS)
end
# checks for reductions
function add_compute_ifelse!(
  ls::LoopSet,
  LHS::Symbol,
  cond::Operation,
  iftrue::Operation,
  iffalse::Operation,
  elementbytes::Int
)
  deps = Symbol[]
  reduceddeps = Symbol[]
  update_deps!(deps, reduceddeps, cond)
  update_deps!(deps, reduceddeps, iftrue)
  update_deps!(deps, reduceddeps, iffalse)
  if name(iftrue) === LHS
    if name(iffalse) === LHS # a = ifelse(condition, a, a) # -- why??? Let's just eliminate it.
      return iftrue
    end
    vparents = Operation[cond, iffalse]
    setdiffv!(reduceddeps, deps, loopdependencies(iftrue))
    if any(in(deps), reduceddeps)
      return add_reduction_update_parent!(
        vparents,
        deps,
        reduceddeps,
        ls,
        iftrue,
        Instruction(:LoopVectorization, :ifelse),
        2,
        elementbytes
      )
    end
  elseif name(iffalse) === LHS
    vparents = Operation[cond, iftrue]
    setdiffv!(reduceddeps, deps, loopdependencies(iffalse))
    if any(in(deps), reduceddeps)
      return add_reduction_update_parent!(
        vparents,
        deps,
        reduceddeps,
        ls,
        iffalse,
        Instruction(:LoopVectorization, :ifelse),
        3,
        elementbytes
      )
    end
  end
  vparents = Operation[cond, iftrue, iffalse]
  op = Operation(
    length(operations(ls)),
    LHS,
    elementbytes,
    :ifelse,
    compute,
    deps,
    reduceddeps,
    vparents
  )
  pushop!(ls, op, LHS)
end
function get_arg!(
  ls::LoopSet,
  @nospecialize(x),
  elementbytes::Int,
  position::Int
)::Operation
  if x isa Expr
    add_operation!(
      ls,
      Symbol("###xpow###$(length(operations(ls)))###"),
      x,
      elementbytes,
      position
    )::Operation
  elseif x isa Symbol
    if x ∈ ls.loopsymbols
      add_loopvalue!(ls, x, elementbytes)
    else
      xo = get(ls.opdict, x, nothing)
      xo === nothing && return add_constant!(ls, x, elementbytes)::Operation
      return xo
    end
  elseif x isa Number
    return add_constant!(ls, x^p, elementbytes, var)::Operation
  else
    throw("objects of type $x not supported as arg")
  end
end

# adds x ^ (p::Real)
function add_pow!(
  ls::LoopSet,
  var::Symbol,
  @nospecialize(x),
  p::Real,
  elementbytes::Int,
  position::Int
)
  xop::Operation = if x isa Expr
    add_operation!(
      ls,
      Symbol("###xpow###$(length(operations(ls)))###"),
      x,
      elementbytes,
      position
    )
  elseif x isa Symbol
    if x ∈ ls.loopsymbols
      add_loopvalue!(ls, x, elementbytes)
    else
      xo = get(ls.opdict, x, nothing)
      if xo === nothing
        if round(Int, p) ≠ p
          pushpreamble!(ls, Expr(:(=), var, Expr(:call, :(^), x, p)))
          return add_constant!(ls, var, elementbytes)
        else
          xo = add_constant!(ls, x, elementbytes)::Operation
        end
      end
      xo
    end
  elseif x isa Number
    return add_constant!(ls, x^p, elementbytes, var)::Operation
  end
  if p isa Integer
    pnum = Int(p)::Int
    pden = 1
  else
    prational = rationalize(p)
    @unpack num, den = prational
    pnum = convert(Int, num)::Int
    pden = convert(Int, den)::Int
  end
  if pden == 1
    nothing
  elseif pden == 2
    if pnum == 1
      return add_compute!(ls, var, :sqrt, [xop], elementbytes)
    else
      xop = add_compute!(ls, gensym!(ls, "root"), :sqrt, [xop], elementbytes)
    end
  elseif pden == 3
    if pnum == 1
      return add_compute!(ls, var, :cbrt, [xop], elementbytes)
    else
      xop = add_compute!(ls, gensym!(ls, "cbroot"), :cbrt, [xop], elementbytes)
    end
  elseif pden == 4
    xop = add_compute!(ls, gensym!(ls, "root"), :sqrt, [xop], elementbytes)
    if pnum == 1
      return add_compute!(ls, var, :sqrt, [xop], elementbytes)
    else
      xop = add_compute!(ls, gensym!(ls, "root"), :sqrt, [xop], elementbytes)
    end
  else
    pop = add_constant!(ls, p, elementbytes)
    return add_compute!(ls, var, :^, [xop, pop], elementbytes)
  end
  pint = pnum
  if pint == -1
    return add_compute!(ls, var, :inv, [xop], elementbytes)
  elseif pint < 0
    xop = add_compute!(ls, gensym!(ls, "inverse"), :inv, [xop], elementbytes)
    pint = -pint
  end
  if pint == 0
    op = Operation(
      length(operations(ls)),
      var,
      elementbytes,
      LOOPCONSTANT,
      constant,
      NODEPENDENCY,
      Symbol[],
      NOPARENTS
    )
    push!(
      ls.preamble_funcofeltypes,
      (identifier(op), MULTIPLICATIVE_IN_REDUCTIONS)
    )
    return pushop!(ls, op)
  elseif pint == 1#requires `pden ≠ 1`.
    return add_compute!(ls, var, :identity, [xop], elementbytes)
  elseif pint == 2
    return add_compute!(ls, var, :abs2_fast, [xop], elementbytes)
  end
  # Implementation from https://github.com/JuliaLang/julia/blob/a965580ba7fd0e8314001521df254e30d686afbf/base/intfuncs.jl#L216
  t = trailing_zeros(pint) + 1
  pint >>= t
  while (t -= 1) > 0
    varname = (iszero(pint) && isone(t)) ? var : gensym!(ls, "pbs")
    xop = add_compute!(ls, varname, :abs2_fast, [xop], elementbytes)
  end
  yop = xop
  while pint > 0
    t = trailing_zeros(pint) + 1
    pint >>= t
    while (t -= 1) >= 0
      xop =
        add_compute!(ls, gensym!(ls, "pbs"), :abs2_fast, [xop], elementbytes)
    end
    yop = add_compute!(
      ls,
      iszero(pint) ? var : gensym!(ls, "pbs"),
      :mul_fast,
      [xop, yop],
      elementbytes
    )
  end
  yop
end
