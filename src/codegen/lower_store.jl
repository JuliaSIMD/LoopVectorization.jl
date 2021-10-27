function opisreduced(op::Operation)
  for rdep ∈ reduceddependencies(op)
    rdep ∉ loopdependencies(op) && return true
  end
  false
end
function storeinstr_preprend(op::Operation, vloopsym::Symbol)
  # defaultstoreop = :vstore!
  # defaultstoreop = :vnoaliasstore!
  isvectorized(op) && return Symbol("")
  vloopsym ∉ reduceddependencies(op) && return Symbol("")
  # vectorized is not a loopdep, but is a reduced dep
  opp::Operation = first(parents(op))
  # while vectorized ∉ loopdependencies(opp)
  while ((!isvectorized(opp)) || opisreduced(opp))
    oppold = opp
    for oppp ∈ parents(opp)
      if vloopsym ∈ reduceddependencies(oppp)
        @assert opp !== oppp "More than one parent is a reduction over the vectorized variable."
        opp = oppp
      end
    end
    @assert opp !== oppold "Failed to find any parents "
  end
  instr = instruction(opp).instr
  instr === :ifelse && return Symbol("") #throw(LoopError("ifelse not yet supported for inner reductions"))
  return reduction_to_scalar(instr)
  # if instr ≢ :ifelse
  #   Expr(:(.), LoopVectorization, QuoteNode(reduction_to_scalar(instr))) # :IfElseReducer
  # else
  #   ifelse_reduction(:IfElseReducer,opp) do opv
  #     throw(LoopError("Does not support storing mirrored ifelse-reductions yet"))
  #   end
  # end
end

function reduce_expr_u₂(toreduct::Symbol, op::Operation, u₂::Int, suffix::Symbol)
  t = Expr(:tuple)
  for u ∈ 0:u₂-1
    push!(t.args, Symbol(toreduct, u, suffix))
  end
  Expr(:call, lv(:reduce_tup), reduce_to_onevecunroll(op), t)
end
function reduce_expr!(q::Expr, toreduct::Symbol, op::Operation, u₁::Int, u₂::Int, isu₁unrolled::Bool, isu₂unrolled::Bool)
  if isu₂unrolled# u₂ != -1
    _toreduct = Symbol(toreduct, 0)
    push!(q.args, Expr(:(=), _toreduct, reduce_expr_u₂(toreduct, op, u₂, Symbol(""))))
  else#if u₂ == -1
    _toreduct = Symbol(toreduct, '_', u₁)
  end
  if (u₁ == 1) | (~isu₁unrolled)
    push!(q.args, Expr(:(=), Symbol(toreduct, "##onevec##"), _toreduct))
  elseif instruction(op).instr ≢ :ifelse
    push!(q.args, Expr(:(=), Symbol(toreduct, "##onevec##"), Expr(:call, reduction_to_single_vector(op), _toreduct)))
  else
    fifelse = let u₁=u₁
      ifelse_reduction(:IfElseCollapser,op) do opv
        Symbol(mangledvar(opv), '_', u₁), tuple()
      end
    end
    push!(q.args, Expr(:(=), Symbol(toreduct, "##onevec##"), Expr(:call, fifelse, _toreduct, staticexpr(1))))
  end
  nothing
end

function lower_store_collection!(
  q::Expr, ls::LoopSet, op::Operation, ua::UnrollArgs, mask::Bool, inds_calc_by_ptr_offset::Vector{Bool}
)
  omop = offsetloadcollection(ls)
  batchid, bopind = omop.batchedcollectionmap[identifier(op)]
  collectionid, copind = omop.opidcollectionmap[identifier(op)]
  opidmap = offsetloadcollection(ls).opids[collectionid]
  idsformap = omop.batchedcollections[batchid]

  @unpack u₁, u₁loop, u₁loopsym, u₂loopsym, vloopsym, vloop, u₂max, suffix = ua
  ops = operations(ls)
  # __u₂max = ls.unrollspecification.u₂
  nouter = length(idsformap)

  t = Expr(:tuple)
  for (i,(opid,_)) ∈ enumerate(idsformap)
    opp = first(parents(ops[opidmap[opid]]))

    isu₁, isu₂ = isunrolled_sym(opp, u₁loopsym, u₂loopsym, vloopsym, ls)#, __u₂max)
    u = Core.ifelse(isu₁, u₁, 1)
    mvar = Symbol(variable_name(opp, ifelse(isu₂, suffix, -1)), '_', u)
    # mvar = Symbol(variable_name(_op, suffix), '_', u)
    # mvar = Symbol(variable_name(_op, ifelse(isu₂, suffix, -1)), '_', u)
    push!(t.args, mvar)
  end
  offset_dummy_loop = Loop(first(getindices(op)), MaybeKnown(1), MaybeKnown(1024), MaybeKnown(1), Symbol(""), Symbol(""))
  unrollcurl₂ = unrolled_curly(op, nouter, offset_dummy_loop, vloop, mask, 1)
  inds = mem_offset_u(op, ua, inds_calc_by_ptr_offset, false, 0, ls, false)
  falseexpr = Expr(:call, lv(:False));
  aliasexpr = falseexpr;
  # trueexpr = Expr(:call, lv(:True));
  rs = staticexpr(reg_size(ls));
  manualunrollu₁ = if isu₁unrolled(op) && u₁ > 1 # both unrolled
    # unrollcurl₂ is unrolled along `first(getindices(op))` by factor of `nouter`
    # 
    # if isknown(step(u₁loop)) && sum(Base.Fix2(===,u₁loopsym), getindicesonly(op)) == 1
    if (isknown(step(u₁loop)) && sum(Base.Fix2(===,u₁loopsym), getindicesonly(op)) == 1)# && (isone(step(u₁loop)) | (first(getindices(op)) ≢ u₁loopsym))
      # if first(getindices(op)) === u₁loopsym#vloopsym
      #   interleaveval = -nouter
      # else
        interleaveval = 0
      # end
      unrollcurl₁ = unrolled_curly(op, u₁, u₁loop, vloop, mask, interleaveval)
      inds = Expr(:call, unrollcurl₁, inds)
      false
    else
      true
    end
  else
    false
  end
  uinds = Expr(:call, unrollcurl₂, inds)
  vp = vptr(op)
  sptrsym = sptr!(q, op)
  storeexpr = Expr(:call, lv(:_vstore!), sptrsym, Expr(:call, lv(:VecUnroll), t), uinds)
  # not using `add_memory_mask!(storeexpr, op, ua, mask, ls)` because we checked `isconditionalmemop` earlier in `lower_load_collection!`
  u₁vectorized = u₁loopsym === vloopsym
  if mask# && isvectorized(op))
    if !(manualunrollu₁ & u₁vectorized)
      push!(storeexpr.args, MASKSYMBOL)
    end
  end
  push!(storeexpr.args, falseexpr, aliasexpr, falseexpr, rs)
  if manualunrollu₁
    masklast = mask & u₁vectorized
    gf = GlobalRef(Core,:getfield)
    tv = Vector{Symbol}(undef, length(t.args))
    for i ∈ eachindex(tv)
      s = gensym!(ls, "##tmp##collection##store##")
      tv[i] = s
      push!(q.args, Expr(:(=), s, Expr(:call, gf, t.args[i], 1)))
    end
    for u ∈ 0:u₁-1
      lastiter = (u+1) == u₁
      storeexpr_tmp = if lastiter
        storeexpr
        (((u+1) == u₁) & masklast) && insert!(storeexpr.args, length(storeexpr.args)-3, MASKSYMBOL) # 3 for falseexpr, aliasexpr, falseexpr
        storeexpr
      else
        copy(storeexpr)
      end
      vut = Expr(:tuple)
      for i ∈ eachindex(tv)
        push!(vut.args, Expr(:call, gf, tv[i], u+1, false))
      end
      storeexpr_tmp.args[3] = Expr(:call, lv(:VecUnroll), vut)
      if u ≠ 0
        storeexpr_tmp.args[4] = Expr(:call, unrollcurl₂, mem_offset_u(op, ua, inds_calc_by_ptr_offset, false, u, ls, false))
      end
      push!(q.args, storeexpr_tmp)
    end
  else
    push!(q.args, storeexpr)
  end
  nothing
end
gf(s::Symbol, n::Int) = Expr(:call, GlobalRef(Core,:getfield), s, n, false)
function lower_store!(
  q::Expr, ls::LoopSet, op::Operation, ua::UnrollArgs, mask::Bool,
  reductfunc::Symbol = storeinstr_preprend(op, ua.vloop.itersymbol), inds_calc_by_ptr_offset = indices_calculated_by_pointer_offsets(ls, op.ref)
)
  @unpack u₁, u₁loopsym, u₂loopsym, vloopsym, vloop, u₂max, suffix = ua
  omop = offsetloadcollection(ls)
  batchid, opind = omop.batchedcollectionmap[identifier(op)]
  if ((batchid ≠ 0) && isvectorized(op)) && (!rejectinterleave(op))
    (opind == 1) && lower_store_collection!(q, ls, op, ua, mask, inds_calc_by_ptr_offset)
    return
  end
  falseexpr = Expr(:call, lv(:False));
  aliasexpr = falseexpr;
  # trueexpr = Expr(:call, lv(:True));
  rs = staticexpr(reg_size(ls));
  opp = first(parents(op))
  if ((opp.instruction.instr === reductfunc) || (opp.instruction.instr === :identity))
    parents_opp = parents(opp)
    opppstate = Base.iterate(parents_opp)
    if opppstate ≢ nothing
      oppp, state = opppstate
      if (Base.iterate(parents_opp, state) === nothing) && isu₂unrolled(op) == isu₂unrolled(oppp)
        opp = oppp
      end
    end
  end
  # __u₂max = ls.unrollspecification.u₂
  isu₁, isu₂ = isunrolled_sym(opp, u₁loopsym, u₂loopsym, vloopsym, ls)#, __u₂max)
  u = isu₁ ? u₁ : 1
  mvar = Symbol(variable_name(opp, ifelse(isu₂, suffix, -1)), '_', u)
  if all(op.ref.loopedindex)
    inds = unrolledindex(op, ua, mask, inds_calc_by_ptr_offset, ls)
    storeexpr = if reductfunc === Symbol("")
      Expr(:call, lv(:_vstore!), sptr(op), mvar, inds)
    else
      Expr(:call, lv(:_vstore!), lv(reductfunc), sptr(op), mvar, inds)
    end
    add_memory_mask!(storeexpr, op, ua, mask, ls, 0)
    push!(storeexpr.args, falseexpr, aliasexpr, falseexpr, rs)
    push!(q.args, storeexpr)
  else
    parents_op = parents(op)
    data_u₁ = isu₁ & (u₁ > 1)
    
    indices_u₁ = data_u₁
    if !data_u₁ & (length(parents_op) > 1)
      indices_u₁ = first(isunrolled_sym(op, u₁loopsym, u₂loopsym, vloopsym, ls))
    end
    if indices_u₁
      mvard = Symbol(mvar, "##data##")
      # isu₁ &&
      data_u₁ && push!(q.args, Expr(:(=), mvard, Expr(:call, lv(:data), mvar)))
      sptrsym = sptr!(q, op)
      for u ∈ 1:u₁
        inds = mem_offset_u(op, ua, inds_calc_by_ptr_offset, true, u-1, ls, false)
        storeexpr = if data_u₁
          if reductfunc === Symbol("")
            Expr(:call, lv(:_vstore!), sptrsym, gf(mvard,u), inds)
          else
            Expr(:call, lv(:_vstore!), lv(reductfunc), sptrsym, gf(mvard,u), inds)
          end
        elseif reductfunc === Symbol("")
          Expr(:call, lv(:_vstore!), sptrsym, mvar, inds)
        else
          Expr(:call, lv(:_vstore!), lv(reductfunc), sptrsym, mvar, inds)
        end
        domask = mask && (isvectorized(op) & ((u == u₁) | (vloopsym !== u₁loopsym)))
        add_memory_mask!(storeexpr, op, ua, domask, ls, u)# & ((u == u₁) | isvectorized(op)))
        push!(storeexpr.args, falseexpr, aliasexpr, falseexpr, rs)
        push!(q.args, storeexpr)
      end
    else
      inds = mem_offset_u(op, ua, inds_calc_by_ptr_offset, true, 0, ls, false)
      storeexpr = if reductfunc === Symbol("")
        Expr(:call, lv(:_vstore!), sptr(op), mvar, inds)
      else
        Expr(:call, lv(:_vstore!), lv(reductfunc), sptr(op), mvar, inds)
      end
      add_memory_mask!(storeexpr, op, ua, mask, ls, 0)
      push!(storeexpr.args, falseexpr, aliasexpr, falseexpr, rs)
      push!(q.args, storeexpr)
    end
  end
  nothing
end

function lower_tiled_store!(
    blockq::Expr, opsv1::Vector{Operation}, opsv2::Vector{Operation}, ls::LoopSet, unrollsyms::UnrollSymbols, u₁::Int, u₂::Int, mask::Bool
)
    ua = UnrollArgs(ls, u₁, unrollsyms, u₂, 0)
    for opsv ∈ (opsv1, opsv2)
        for op ∈ opsv
            lower_tiled_store!(blockq, op, ls, ua, u₁, u₂, mask)
        end
    end
end

function donot_tile_store(ls::LoopSet, op::Operation, vloop::Loop, reductfunc::Symbol, u₂::Int)
    ((!((reductfunc === Symbol("")) && all(op.ref.loopedindex))) || (u₂ ≤ 1) || isconditionalmemop(op)) && return true
    rejectcurly(op) && return true
    omop = offsetloadcollection(ls)
    batchid, opind = omop.batchedcollectionmap[identifier(op)]
    return ((batchid ≠ 0) && isvectorized(op)) && (!rejectinterleave(op))
end

# VectorizationBase implements optimizations for certain grouped stores
# thus we group stores together here to allow for these possibilities.
# (In particular, it tries to replace scatters with shuffles when there are groups
#   of stores offset from one another.)
function lower_tiled_store!(blockq::Expr, op::Operation, ls::LoopSet, ua::UnrollArgs, u₁::Int, u₂::Int, mask::Bool)
    @unpack u₁loopsym, u₂loopsym, vloopsym, u₁loop, u₂loop, vloop = ua
    reductfunc = storeinstr_preprend(op, vloopsym)
    inds_calc_by_ptr_offset = indices_calculated_by_pointer_offsets(ls, op.ref)

    if donot_tile_store(ls, op, vloop, reductfunc, u₂)
        # If we have a reductfunc, we're using a reducing store instead of a contiuguous or shuffle store anyway
        # so no benefit to being able to handle that case here, vs just calling the default `lower_store!` method
        @unpack u₁, u₂max = ua
        for t ∈ 0:u₂-1
            unrollargs = UnrollArgs(u₁loop, u₂loop, vloop, u₁, u₂max, t)
            lower_store!(blockq, ls, op, unrollargs, mask, reductfunc, inds_calc_by_ptr_offset)
        end
        return
    end
    opp = first(parents(op))
    if (opp.instruction.instr === reductfunc) && isone(length(parents(opp)))
        throw(LoopError("Operation $opp's instruction is $reductfunc, shouldn't be able to reach here."))
        # opp = only(parents(opp))
    end
    isu₁, isu₂ = isunrolled_sym(opp, u₁loopsym, u₂loopsym, vloopsym, ls)#, u₂)
    # It's reasonable for this to be `!isu₁`
    u = Core.ifelse(isu₁, u₁, 1)
    tup = Expr(:tuple)
    for t ∈ 0:u₂-1
        push!(tup.args, Symbol(variable_name(opp, ifelse(isu₂, t, -1)), '_', u))
    end
    vut = Expr(:call, lv(:VecUnroll), tup) # `VecUnroll` of `VecUnroll`s
    inds = mem_offset_u(op, ua, inds_calc_by_ptr_offset, false, 0, ls, false)
    unrollcurl₂ = unrolled_curly(op, u₂, u₂loop, vloop, mask)
    falseexpr = Expr(:call, lv(:False));
    aliasexpr = falseexpr;
    # trueexpr = Expr(:call, lv(:True));
    rs = staticexpr(reg_size(ls));
    if isu₁ && u₁ > 1 # both unrolled
        unrollcurl₁ = unrolled_curly(op, u₁, u₁loop, vloop, mask)
        inds = Expr(:call, unrollcurl₁, inds)
    end
    uinds = Expr(:call, unrollcurl₂, inds)
    storeexpr = Expr(:call, lv(:_vstore!), sptr(op), vut, uinds)
    if mask && isvectorized(op)
        # add_memory_mask!(storeexpr, op, ua, mask, ls)
        # we checked for `isconditionalmemop` earlier, so we skip this check
        # and just directly take the branch in `add_memory_mask!`
        push!(storeexpr.args, MASKSYMBOL)
    end
    push!(storeexpr.args, falseexpr, aliasexpr, falseexpr, rs)
    push!(blockq.args, storeexpr)
    nothing
end
