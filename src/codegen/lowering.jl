
# the `lowernonstore` and `lowerstore` options are there as a means of lowering all non-store operations before lowering the stores.
function lower!(
  q::Expr,
  ops::AbstractVector{Operation},
  ls::LoopSet,
  unrollsyms::UnrollSymbols,
  u₁::Int,
  u₂::Int,
  suffix::Int,
  mask::Bool,
  lowernonstore::Bool,
  lowerstore::Bool
)
  ua = UnrollArgs(ls, u₁, unrollsyms, u₂, suffix)
  for op ∈ ops
    if isstore(op)
      lowerstore && lower_store!(q, ls, op, ua, mask)
    else
      lowernonstore || continue
      if isconstant(op)
        zerotyp = zerotype(ls, op)
        if zerotyp == INVALID
          lower_constant!(q, op, ls, ua)
        else
          lower_zero!(q, op, ls, ua, zerotyp)
        end
      elseif isload(op)
        lower_load!(q, op, ls, ua, mask)
      elseif iscompute(op)
        lower_compute!(q, op, ls, ua, mask)
      end
    end
  end
end
function isu₂invalidstorereorder(ls::LoopSet, us::UnrollSpecification)
  us.u₂ == -1 ? false : ls.validreorder[ls.loopordermap[us.u₂loopnum]] ≠ 0x03
end
function lower_block(
  ls::LoopSet,
  us::UnrollSpecification,
  n::Int,
  mask::Bool,
  UF::Int
)
  @unpack u₁loopnum, u₂loopnum, vloopnum, u₁, u₂ = us
  ops = oporder(ls)
  order = names(ls)
  u₁loopsym = order[u₁loopnum]
  u₂loopsym = order[u₂loopnum]
  vectorized = order[vloopnum]
  unrollsyms = UnrollSymbols(u₁loopsym, u₂loopsym, vectorized)
  u₁ = n == u₁loopnum ? UF : u₁
  dontmaskfirsttiles = mask && vloopnum == u₂loopnum
  blockq = Expr(:block)
  cannot_reorder_u₂ = isu₂invalidstorereorder(ls, us)
  for prepost ∈ 1:2
    # !u₁ && !u₂
    lower!(
      blockq,
      ops[1, 1, prepost, n],
      ls,
      unrollsyms,
      u₁,
      u₂,
      -1,
      mask,
      true,
      true
    )
    # isu₁unrolled, isu₂unrolled, after_loop, n
    opsv1 = ops[1, 2, prepost, n]
    opsv2 = ops[2, 2, prepost, n]
    if length(opsv1) + length(opsv2) > 0
      nstores = 0
      iszero(length(opsv1)) || (nstores += sum(isstore, opsv1))
      iszero(length(opsv2)) || (nstores += sum(isstore, opsv2))
      # if nstores
      if (length(opsv1) + length(opsv2) == nstores) && u₂ > 1 # all_u₂_ops_store
        lower!(
          blockq,
          ops[2, 1, prepost, n],
          ls,
          unrollsyms,
          u₁,
          u₂,
          -1,
          mask,
          true,
          true
        ) # for u ∈ 0:u₁-1
        lower_tiled_store!(blockq, opsv1, opsv2, ls, unrollsyms, u₁, u₂, mask)
      else
        for store ∈ (false, true)
          if cannot_reorder_u₂
            nstores = 0# break
            lowernonstore = lowerstore = true
          else
            lowerstore = store
            lowernonstore = !store
          end
          for t ∈ 0:u₂-1
            # !u₁ &&  u₂
            lower!(
              blockq,
              opsv1,
              ls,
              unrollsyms,
              u₁,
              u₂,
              t,
              mask & !(dontmaskfirsttiles & (t < u₂ - 1)),
              lowernonstore,
              lowerstore
            )
            if iszero(t) && !store #  u₁ && !u₂
              # for u ∈ 0:u₁-1
              lower!(
                blockq,
                ops[2, 1, prepost, n],
                ls,
                unrollsyms,
                u₁,
                u₂,
                -1,
                mask,
                true,
                true
              )
              # end
            end
            #  u₁ && u₂
            # for u ∈ 0:u₁-1
            lower!(
              blockq,
              opsv2,
              ls,
              unrollsyms,
              u₁,
              u₂,
              t,
              mask & !(dontmaskfirsttiles & (t < u₂ - 1)),
              lowernonstore,
              lowerstore
            )
            # end
          end
          nstores == 0 && break
        end
      end
    elseif cannot_reorder_u₂
      lower!(
        blockq,
        ops[2, 1, prepost, n],
        ls,
        unrollsyms,
        u₁,
        u₂,
        -1,
        mask,
        true,
        true
      )
    else
      # for u ∈ 0:u₁-1     #  u₁ && !u₂
      lower!(
        blockq,
        ops[2, 1, prepost, n],
        ls,
        unrollsyms,
        u₁,
        u₂,
        -1,
        mask,
        true,
        false
      )
      lower!(
        blockq,
        ops[2, 1, prepost, n],
        ls,
        unrollsyms,
        u₁,
        u₂,
        -1,
        mask,
        false,
        true
      )
      # end
    end
    if n > 1 && prepost == 1
      push!(blockq.args, lower_unrolled_dynamic(ls, us, n - 1, mask))
    end
  end
  incrementloopcounter!(blockq, ls, us, n, UF)
  blockq
end

assume(ex) = Expr(:call, GlobalRef(VectorizationBase, :assume), ex)
function loopiteratesatleastonce!(ls, loop::Loop)
  start = first(loop)
  stop = last(loop)
  (isknown(start) & isknown(stop)) && return loop
  comp = Expr(:call, :≥)
  pushexpr!(comp, last(loop))
  pushexpr!(comp, first(loop))
  pushpreamble!(ls, assume(comp))
  return loop
end

function check_full_conv_kernel(ls, us, N)
  loop₁ = getloop(ls, us.u₁loopnum)
  (isstaticloop(loop₁) && length(loop₁) == us.u₁) && return true
  loop₂ = getloop(ls, us.u₂loopnum)
  (isstaticloop(loop₂) && length(loop₂) == us.u₂) && return true
  false
end
function allinteriorunrolled(ls::LoopSet, us::UnrollSpecification, N)
  if ls.loadelimination
    check_full_conv_kernel(ls, us, N) || return false
  end
  unroll_total = 1
  for n ∈ 1:N-1
    loop = getloop(ls, n)
    nisvectorized = isvectorized(us, n)
    W = nisvectorized ? ls.vector_width : 1
    ((length(loop) ≤ 8W) && (isstaticloop(loop) & (!iszero(W)))) || return false
    unroll_total *= cld(length(loop), W)
  end
  if us.u₁loopnum > N
    unroll_total *= us.u₁
  end
  if us.u₂loopnum > N
    unroll_total *= us.u₂
  end
  unroll_total ≤ 16
end

function lower_no_unroll(
  ls::LoopSet,
  us::UnrollSpecification,
  n::Int,
  inclmask::Bool
)
  nisvectorized = isvectorized(us, n)
  loop = getloop(ls, n)
  tc = terminatecondition(ls, us, n, inclmask, 1)
  body = lower_block(ls, us, n, inclmask, 1)
  loopisstatic = isstaticloop(loop)
  W = nisvectorized ? ls.vector_width : 1
  loopisstatic &= (!iszero(W))
  completely_unrolled = false
  length_loop = loopisstatic ? length(loop) : 0
  if loopisstatic &&
     (!ls.loadelimination) &&
     (
       isone(length_loop ÷ W) ||
       (n ≤ 3 && length_loop ≤ 8W && allinteriorunrolled(ls, us, n)) ||
       (length_loop ≤ W)
     )
    completely_unrolled = true
    q = Expr(:block)
    for _ ∈ 1:(length_loop÷W)
      push!(q.args, body)
    end
  elseif nisvectorized
    q = Expr(:block, Expr(:while, tc, body))
  else
    termcond = gensym(:maybeterm)
    push!(body.args, Expr(:(=), termcond, tc))
    q = Expr(:block, Expr(:(=), termcond, true), Expr(:while, termcond, body))
  end
  if nisvectorized && !(loopisstatic && iszero(length_loop & (W - 1)))
    body = lower_block(ls, us, n, true, 1)
    if isone(num_loops(ls))
      pushfirst!(body.args, definemask(loop))
    end
    if loopisstatic
      push!(q.args, body)
    else
      tc = terminatecondition(ls, us, n, true, 1)
      push!(q.args, Expr(:if, tc, body))
    end
  end
  Expr(:let, startloop(ls, us, n, completely_unrolled), q)
end
function lower_unrolled_dynamic(
  ls::LoopSet,
  us::UnrollSpecification,
  n::Int,
  inclmask::Bool
)
  UF = unrollfactor(us, n)
  isone(UF) && return lower_no_unroll(ls, us, n, inclmask)
  @unpack u₁loopnum, vloopnum, u₁, u₂ = us
  order = names(ls)
  loop = getloop(ls, n)
  vectorized = order[vloopnum]
  nisunrolled = isunrolled1(us, n)
  nisvectorized = isvectorized(us, n)
  W = nisvectorized ? ls.vector_width : 1
  UFW = UF * W
  looplength = length(loop)
  if W ≠ 0 & isknown(first(loop)) & isknown(step(loop))
    loopisstatic = isknown(last(loop))
    # something other than the default hint currently means an UpperBoundedInteger was passed as an argument
    loopisbounded =
      (looplength < UFW) & (loopisstatic | (gethint(last(loop)) ≠ 1024))
  else
    loopisstatic = false
    loopisbounded = false
  end
  Ureduct = ((n == num_loops(ls) && (u₂ == -1))) ? ureduct(ls) : -1
  # for now, require loopisstatic or !Ureduct-ing for reducing UF
  if loopisbounded & (loopisstatic | (Ureduct < 0))
    UFWnew = cld(looplength, cld(looplength, UFW))
    UF = cld(UFWnew, W)
    UFW = UF * W
    us =
      nisunrolled ? UnrollSpecification(us, UF, u₂) :
      UnrollSpecification(us, u₁, UF)
  end
  remmask = inclmask | nisvectorized
  sl = startloop(ls, us, n, false)
  UFt = loopisstatic ? cld(looplength % UFW, W) : 1
  # Don't place remainder first if we're going to have to mask this loop (i.e., if this loop is vectorized)
  remfirst =
    loopisstatic &
    (!nisvectorized) &
    (UFt > 0) &
    !(unsigned(Ureduct) < unsigned(UF))
  tc = terminatecondition(ls, us, n, inclmask, remfirst ? 1 : UF)
  # Don't need to create the body if loop is dynamic and bounded
  dynamicbounded = ((!loopisstatic) & loopisbounded)
  body = dynamicbounded ? tc : lower_block(ls, us, n, inclmask, UF)
  if loopisstatic
    iters = length(loop) ÷ UFW
    if (iters ≤ 1) || (iters * UF ≤ 16 && allinteriorunrolled(ls, us, n))# Let's set a limit on total unrolling
      q = Expr(:block)
      for _ ∈ 1:iters
        push!(q.args, body)
      end
    else
      q = Expr(:while, tc, body)
    end
    remblock = Expr(:block)
    (nisvectorized && (UFt > 0) && isone(num_loops(ls))) &&
      push!(remblock.args, definemask(loop))
  else
    remblock = init_remblock(loop, ls.lssm, n)#loopsym)
    # unroll_cleanup = Ureduct > 0 || (nisunrolled ? (u₂ > 1) : (u₁ > 1))
    q = if loopisbounded
      Expr(:block)
    elseif unsigned(Ureduct) < unsigned(UF)
      termcond = gensym(:maybeterm)
      push!(body.args, Expr(:(=), termcond, tc))
      Expr(:block, Expr(:(=), termcond, true), Expr(:while, termcond, body))
    else
      Expr(:while, tc, body)
    end
  end
  q = if unsigned(Ureduct) < unsigned(UF) # unsigned(-1) == typemax(UInt); 
    add_cleanup = Core.ifelse(loopisstatic, !nisvectorized, true)# true
    if isone(Ureduct)
      UF_cleanup = 1
      if nisvectorized
        blockhead = :while
      else
        blockhead = if UF == 2
          if loopisstatic
            # add_cleanup = UFt == 1
            :block
          else
            :if
          end
        else
          :while
        end
        UFt = 0
      end
    elseif 2Ureduct < UF
      UF_cleanup = 2
      blockhead = :while
    else
      UF_cleanup = UF - Ureduct
      blockhead = :if
    end
    _q = if dynamicbounded
      initialize_outer_reductions!(q, ls, Ureduct)
      q
    elseif loopisstatic
      blockhead = :block
      if length(loop) < UF * W
        Expr(:block)
      else
        UFt -= Ureduct
        Expr(
          :block,
          add_upper_outer_reductions(ls, q, Ureduct, UF, loop, nisvectorized)
        )
      end
    else
      Expr(
        :block,
        add_upper_outer_reductions(ls, q, Ureduct, UF, loop, nisvectorized)
      )
    end
    if add_cleanup
      cleanup_expr = Expr(blockhead)
      blockhead === :block || push!(
        cleanup_expr.args,
        terminatecondition(ls, us, n, inclmask, UF_cleanup)
      )
      us_cleanup =
        nisunrolled ? UnrollSpecification(us, UF_cleanup, u₂) :
        UnrollSpecification(us, u₁, UF_cleanup)
      push!(
        cleanup_expr.args,
        lower_block(ls, us_cleanup, n, inclmask, UF_cleanup)
      )
      push!(_q.args, cleanup_expr)
    end
    UFt > 0 && push!(_q.args, remblock)
    _q
  elseif remfirst
    numiters = length(loop) ÷ UF
    if numiters > 2
      Expr(:block, remblock, q)
    else
      q = Expr(:block, remblock)
      for i ∈ 1:numiters
        push!(q.args, body)
      end
      q
    end
  elseif iszero(UFt)
    Expr(:block, q)
  elseif !nisvectorized && !loopisstatic && UF ≥ 10
    rem_uf = UF - 1
    UF = rem_uf >> 1
    UFt = rem_uf - UF
    ust =
      nisunrolled ? UnrollSpecification(us, UFt, u₂) :
      UnrollSpecification(us, u₁, UFt)
    newblock = lower_block(ls, ust, n, remmask, UFt)
    # comparison = unrollremcomparison(ls, loop, UFt, n, nisvectorized, remfirst)
    comparison = terminatecondition(ls, us, n, inclmask, UFt)
    UFt = 1
    UF += 1 - iseven(rem_uf)
    Expr(
      :block,
      q,
      Expr(iseven(rem_uf) ? :while : :if, comparison, newblock),
      remblock
    )
  else
    Expr(:block, q, remblock)
  end
  if !iszero(UFt)
    # if unroll_cleanup
    iforelseif = :if
    while true
      ust =
        nisunrolled ? UnrollSpecification(us, UFt, u₂) :
        UnrollSpecification(us, u₁, UFt)
      newblock = lower_block(ls, ust, n, remmask, UFt)
      if (UFt ≥ UF - 1 + nisvectorized) || UFt == Ureduct || loopisstatic
        if isone(num_loops(ls)) && isone(UFt) && isone(Ureduct)
          newblock = Expr(:block, definemask(loop), newblock)
        end
        push!(remblock.args, newblock)
        break
      end
      comparison =
        unrollremcomparison(ls, loop, UFt, n, nisvectorized, remfirst)
      if isone(num_loops(ls)) && isone(UFt)
        remblocknew = Expr(:if, comparison, newblock)
        push!(
          remblock.args,
          Expr(:block, Expr(:let, definemask(loop), remblocknew))
        )
        remblock = remblocknew
      else
        remblocknew = Expr(iforelseif, comparison, newblock)
        # remblocknew = Expr(:elseif, comparison, newblock)
        push!(remblock.args, remblocknew)
        remblock = remblocknew
        iforelseif = :elseif
      end
      UFt += 1
    end
    # else
    #     ust = nisunrolled ? UnrollSpecification(us, 1, u₂) : UnrollSpecification(us, u₁, 1)
    #     # newblock = lower_block(ls, ust, n, remmask, 1)
    #     push!(remblock.args, lower_no_unroll(ls, ust, n, inclmask, false, UF-1))
    # end
  end
  if (length(ls.outer_reductions) > 0) && (2 ≤ n < length(ls.loops))
    pre, post = reinit_and_update_tiled_outer_reduct!(
      sl,
      q,
      ls,
      order[u₁loopnum],
      order[us.u₂loopnum],
      vectorized
    )
    Expr(:block, pre, Expr(:let, sl, q), post)
  else
    Expr(:block, Expr(:let, sl, q))
  end
end
function unrollremcomparison(
  ls::LoopSet,
  loop::Loop,
  UFt::Int,
  n::Int,
  nisvectorized::Bool,
  remfirst::Bool
)
  termind = ls.lssm.terminators[n]
  if iszero(termind)
    loopvarremcomparison(loop, UFt, nisvectorized, remfirst)
  else
    pointerremcomparison(ls, termind, UFt, n, nisvectorized, remfirst, loop)
  end
end
function loopvarremcomparison(
  loop::Loop,
  UFt::Int,
  nisvectorized::Bool,
  remfirst::Bool
)
  loopsym = loop.itersymbol
  loopstep = loop.step
  if nisvectorized
    offset = mulexpr(VECTORWIDTHSYMBOL, UFt, loopstep)
    itercount = subexpr(last(loop), offset)
    Expr(:call, GlobalRef(Base, :>), loopsym, itercount)
  elseif remfirst # requires `isstaticloop(loop)`
    Expr(
      :call,
      GlobalRef(Base, :<),
      loopsym,
      gethint(first(loop)) + UFt * gethint(loopstep) - 1
    )
  elseif isknown(last(loop))
    if isknown(loopstep)
      Expr(
        :call,
        GlobalRef(Base, :>),
        loopsym,
        gethint(last(loop)) - UFt * gethint(loopstep)
      )
    elseif isone(UFt)
      Expr(
        :call,
        GlobalRef(Base, :>),
        loopsym,
        subexpr(gethint(last(loop)), getsym(loopstep))
      )
    else
      Expr(
        :call,
        GlobalRef(Base, :>),
        loopsym,
        subexpr(gethint(last(loop)), mulexpr(getsym(loopstep), UFt))
      )
    end
  else
    if isknown(loopstep)
      Expr(
        :call,
        GlobalRef(Base, :>),
        loopsym,
        Expr(:call, lv(:vsub_nsw), getsym(last(loop)), UFt * gethint(loopstep))
      )
    elseif isone(UFt)
      Expr(
        :call,
        GlobalRef(Base, :>),
        loopsym,
        Expr(:call, lv(:vsub_nsw), getsym(last(loop)), getsym(loopstep))
      )
    else
      Expr(
        :call,
        GlobalRef(Base, :>),
        loopsym,
        Expr(
          :call,
          lv(:vsub_nsw),
          getsym(last(loop)),
          mulexpr(getsym(loopstep), UFt)
        )
      )
    end
  end
end
function pointerremcomparison(
  ls::LoopSet,
  termind::Int,
  UFt::Int,
  n::Int,
  nisvectorized::Bool,
  remfirst::Bool,
  loop::Loop
)
  lssm = ls.lssm
  termar = lssm.incrementedptrs[n][termind]
  ptrdef = lssm.incrementedptrs[n][termind]
  ptr = vptr(termar)
  ptroff = vptr_offset(ptr)
  if remfirst
    cmp = GlobalRef(VectorizationBase, :vlt)
    Expr(
      :call,
      cmp,
      ptroff,
      pointermax(ls, ptrdef, n, 1 - UFt, nisvectorized, loop),
      ptr
    )
  else
    cmp = GlobalRef(VectorizationBase, :vge)
    Expr(:call, cmp, ptroff, maxsym(ptr, UFt), ptr)
  end
end

@generated function of_same_size(
  ::Type{T},
  ::Type{S},
  ::StaticInt{R}
) where {T,S,R}
  sizeof_S = sizeof(S)
  sizeof_T = sizeof(T)
  if T <: Integer
    # max(..., 4) to maybe demote Int64 -> Int32
    # but otherwise, we're giving up too much with the demotion.
    if T === Bool || sizeof_S < 8
      # HACK: T === Bool, makes code work.
      sizeof_S *= 8
    else
      sizeof_S *= max(8 ÷ R, 4)
    end
    # multiply by 8 for sake of following `==` comparison
    sizeof_T *= 8
  end
  sizeof_T == sizeof_S && return T
  # Tfloat = T <: Union{Float32,Float64}
  if T <: Union{Float32,Float64}
    sizeof_S ≥ 8 ? Float64 : Float32
  elseif T <: Signed
    Symbol(:Int, sizeof_S)
  elseif (T <: Unsigned) | (T === Bool)
    Symbol(:UInt, sizeof_S)
  else
    S
  end
end
@inline function of_same_size(::Type{T}, ::Type{S}) where {T,S}
  of_same_size(
    T,
    S,
    VectorizationBase.register_size() ÷
    VectorizationBase.simd_integer_register_size()
  )
end
function outer_reduction_zero(
  op::Operation,
  u₁u::Bool,
  Umax::Int,
  reduct_class::Float64,
  rs::Union{Expr,StaticInt}
)
  isifelse = instruction(op).instr === :ifelse
  reduct_zero = if isifelse
    Symbol(name(op), "##BASE##EXTRACT##")
    # Expr(:call, lv(:vbroadcast), VECTORWIDTHSYMBOL,
  else
    reduction_zero(reduct_class)
  end
  # Tsym = outer_reduct_init_typename(op)
  # Tsym = ELTYPESYMBOL
  Tsym =
    Expr(:call, lv(:of_same_size), outer_reduct_init_typename(op), ELTYPESYMBOL)
  if isvectorized(op)
    if Umax == 1 || !u₁u
      if reduct_zero === :zero
        Expr(:call, lv(:_vzero), VECTORWIDTHSYMBOL, Tsym, rs)
      elseif isifelse
        Expr(:call, lv(:_vbroadcast), VECTORWIDTHSYMBOL, reduct_zero, rs)
      else
        Expr(
          :call,
          lv(:_vbroadcast),
          VECTORWIDTHSYMBOL,
          Expr(:call, lv(reduct_zero), Tsym),
          rs
        )
      end
    else
      if reduct_zero === :zero
        Expr(
          :call,
          lv(:zero_vecunroll),
          staticexpr(Umax),
          VECTORWIDTHSYMBOL,
          Tsym,
          rs
        )
      elseif isifelse
        Expr(
          :call,
          lv(:vbroadcast_vecunroll),
          staticexpr(Umax),
          VECTORWIDTHSYMBOL,
          reduct_zero,
          rs
        )
      else
        Expr(
          :call,
          lv(:vbroadcast_vecunroll),
          staticexpr(Umax),
          VECTORWIDTHSYMBOL,
          Expr(:call, reduct_zero, Tsym),
          rs
        )
      end
    end
  elseif isifelse
    Expr(:call, identity, reduct_zero) # type stability within LV
  else
    Expr(:call, reduct_zero, Tsym)
  end
end

# TODO: handle tiled outer reductions; they will require a suffix arg
function initialize_outer_reductions!(
  q::Expr,
  ls::LoopSet,
  op::Operation,
  _Umax::Int,
  us::UnrollSpecification,
  rs::Union{Expr,StaticInt}
)
  @unpack u₁, u₂ = us
  Umax = u₂ == -1 ? _Umax : u₁

  u₁u, u₂u = isunrolled_sym(
    op,
    getloop(ls, us.u₁loopnum).itersymbol,
    getloop(ls, us.u₂loopnum).itersymbol,
    getloop(ls, us.vloopnum).itersymbol,
    ls
  )#, u₂)
  z = outer_reduction_zero(
    op,
    u₁u,
    Umax,
    reduction_instruction_class(instruction(op)),
    rs
  )
  mvar = variable_name(op, -1)
  if (u₂ == -1)
    push!(q.args, Expr(:(=), Symbol(mvar, '_', _Umax), z))
  elseif u₁u
    push!(q.args, Expr(:(=), Symbol(mvar, '_', u₁), z))
  elseif isu₂unrolled(op) # we unroll u₂
    for u ∈ 0:_Umax-1
      push!(q.args, Expr(:(=), Symbol(mvar, u), z))
    end
  else
    push!(q.args, Expr(:(=), Symbol(mvar, '_', 1), z))
  end
  nothing
end
function initialize_outer_reductions!(
  q::Expr,
  ls::LoopSet,
  Umax::Union{Int,StaticInt}
)
  rs = staticexpr(reg_size(ls))
  us = ls.unrollspecification
  for or ∈ ls.outer_reductions
    initialize_outer_reductions!(q, ls, ls.operations[or], Umax, us, rs)
  end
end
initialize_outer_reductions!(ls::LoopSet, Umax::Int) =
  initialize_outer_reductions!(ls.preamble, ls, Umax)
function add_upper_comp_check(unrolledloop, loopbuffer)
  if isstaticloop(unrolledloop)
    Expr(:call, Base.GlobalRef(Base, :≥), length(unrolledloop), loopbuffer)
  elseif isknown(first(unrolledloop))
    if isone(first(unrolledloop))
      Expr(
        :call,
        Base.GlobalRef(Base, :≥),
        getsym(last(unrolledloop)),
        loopbuffer
      )
    else
      Expr(
        :call,
        Base.GlobalRef(Base, :≥),
        getsym(last(unrolledloop)),
        addexpr(loopbuffer, gethint(first(unrolledloop)) - 1)
      )
    end
  elseif isknown(last(unrolledloop))
    Expr(
      :call,
      Base.GlobalRef(Base, :≥),
      Expr(
        :call,
        lv(:vsub_nsw),
        gethint(last(unrolledloop)) + 1,
        getsym(first(unrolledloop))
      ),
      loopbuffer
    )
  else# both are given by symbols
    Expr(
      :call,
      Base.GlobalRef(Base, :>),
      Expr(
        :call,
        lv(:vsub_nsw),
        getsym(last(unrolledloop)),
        Expr(:call, lv(:vsub_nsw), getsym(first(unrolledloop)), staticexpr(1))
      ),
      loopbuffer
    )
  end
end
function add_upper_outer_reductions(
  ls::LoopSet,
  loopq::Expr,
  Ulow::Int,
  Uhigh::Int,
  unrolledloop::Loop,
  reductisvectorized::Bool
)
  ifq = Expr(:block)
  ifqlet = Expr(:block)
  initialize_outer_reductions!(ifqlet, ls, Uhigh)
  push!(ifq.args, loopq)
  t = Expr(:tuple)
  mvartu = Expr(:tuple)
  mvartl = Expr(:tuple)
  for or ∈ ls.outer_reductions
    op = ls.operations[or]
    # var = name(op)
    mvar = Symbol(mangledvar(op), '_', Uhigh)
    if instruction(op).instr ≢ :ifelse
      f = reduce_number_of_vectors(op)
      push!(t.args, Expr(:call, f, mvar, staticexpr(Ulow)))
    else
      fifelse = let Uhigh = Uhigh
        ifelse_reduction(:IfElseCollapser, op) do opv
          Symbol(mangledvar(opv), '_', Uhigh), tuple()
        end
      end
      push!(t.args, Expr(:call, fifelse, mvar, staticexpr(Ulow)))
    end
    push!(mvartu.args, mvar)
    push!(mvartl.args, Symbol(mangledvar(op), '_', Ulow))
  end
  push!(ifq.args, t)
  ifqfull = Expr(:let, ifqlet, ifq)
  if isstaticloop(unrolledloop)
    W = Core.ifelse(reductisvectorized, ls.vector_width, 1)
    if Uhigh * W * gethint(step(unrolledloop)) ≤ length(unrolledloop)
      return Expr(:(=), mvartl, ifqfull)
    end
  end
  ncomparison = if reductisvectorized
    add_upper_comp_check(
      unrolledloop,
      mulexpr(VECTORWIDTHSYMBOL, Uhigh, step(unrolledloop))
    )
  elseif isknown(step(unrolledloop))
    add_upper_comp_check(unrolledloop, Uhigh * gethint(step(unrolledloop)))
  else
    add_upper_comp_check(unrolledloop, mulexpr(Uhigh, getsym(step(unrolledloop))))
  end
  elseq = Expr(:block)
  initialize_outer_reductions!(elseq, ls, Ulow)
  push!(elseq.args, mvartl)
  Expr(:(=), mvartl, Expr(:if, ncomparison, ifqfull, elseq))
end
## This performs reduction to one `Vec`
function reduce_expr!(q::Expr, ls::LoopSet, U::Int)
  us = ls.unrollspecification
  if us.u₂ == -1
    u₁f = ifelse(U == -1, us.u₁, U)
    u₂f = -1
  else
    u₁f = us.u₁
    u₂f = U
  end
  # u₁loop, u₂loop = getunrolled(ls)
  u₁loop = getloop(ls, us.u₁loopnum).itersymbol
  u₂loop = getloop(ls, us.u₂loopnum).itersymbol
  vloop = getloop(ls, us.vloopnum).itersymbol
  for or ∈ ls.outer_reductions
    op = ls.operations[or]
    var = name(op)
    mvar = mangledvar(op)
    u₁u, u₂u = isunrolled_sym(op, u₁loop, u₂loop, vloop, ls)#, u₂f)
    reduce_expr!(q, mvar, op, u₁f, u₂f, u₁u, u₂u)
    if length(ls.opdict) ≠ 0
      if (isu₁unrolled(op) | isu₂unrolled(op))
        if instruction(op).instr ≢ :ifelse
          push!(
            q.args,
            Expr(
              :(=),
              var,
              Expr(
                :call,
                reduction_scalar_combine(op),
                Symbol(mvar, "##onevec##"),
                var
              )
            )
          )
        else
          reductexpr = ifelse_reduction(:IfElseReduced, op) do opv
            Symbol(mangledvar(opv), "##onevec##"), (name(opv),)
          end
          push!(
            q.args,
            Expr(
              :(=),
              var,
              Expr(:call, reductexpr, Symbol(mvar, "##onevec##"), var)
            )
          )
        end
      else
        push!(q.args, Expr(:(=), var, mvar))
      end
    end
  end
end
function reinit_push_preblockpost!(
  letblock::Expr,
  pre::Expr,
  block::Expr,
  post::Expr,
  z::Expr,
  s::Symbol,
  reduct::Symbol
)
  push!(letblock.args, Expr(:(=), s, z))
  tempsym = gensym(s) # placeholder
  push!(pre.args, Expr(:(=), tempsym, s))
  push!(block.args, Expr(:(=), tempsym, Expr(:call, lv(reduct), tempsym, s)))
  push!(post.args, Expr(:(=), s, tempsym))
  nothing
end
function reinit_and_update_tiled_outer_reduct!(
  letblock::Expr,
  block::Expr,
  ls::LoopSet,
  u₁loopsym::Symbol,
  u₂loopsym::Symbol,
  vloopsym::Symbol
)
  rs = staticexpr(reg_size(ls))
  usorig = ls.unrollspecification
  Umax = ureduct(ls)
  pre = Expr(:block)
  post = Expr(:block)
  for or ∈ ls.outer_reductions
    op = ls.operations[or]
    instr = instruction(op).instr
    instr === :ifelse && continue # FIXME - skipping this will result in bad performance
    u₁u, u₂u = isunrolled_sym(op, u₁loopsym, u₂loopsym, vloopsym, ls)
    reduct_class::Float64 = reduction_instruction_class(instr)
    z = outer_reduction_zero(op, u₁u, Umax, reduct_class, rs)
    reduct = reduce_to_onevecunroll(reduct_class)
    mvar = variable_name(op, -1)
    if u₁u # it's u₁unrolled
      reinit_push_preblockpost!(
        letblock,
        pre,
        block,
        post,
        z,
        Symbol(mvar, '_', usorig.u₁),
        reduct
      )
    else # it's u₂unrolled
      for u ∈ 0:Umax-1
        reinit_push_preblockpost!(
          letblock,
          pre,
          block,
          post,
          z,
          Symbol(mvar, u),
          reduct
        )
      end
    end
    initialize_outer_reductions!(
      letblock,
      ls,
      ls.operations[or],
      ureduct(ls),
      usorig,
      rs
    )
  end
  pre, post
end

function gc_preserve(ls::LoopSet, q::Expr)
  length(ls.opdict) == 0 && return q
  q2 = Expr(:block)
  gcp = Expr(:gc_preserve, q)
  for array ∈ ls.includedactualarrays
    pb = gensym(array)
    push!(q2.args, Expr(:(=), pb, Expr(:call, lv(:preserve_buffer), array)))
    push!(gcp.args, pb)
  end
  q.head === :block && push!(q.args, nothing)
  push!(q2.args, gcp)
  q2
end
function push_outer_reduct_types!(pt::Expr, ls::LoopSet, ortypdefined::Bool)
  for j ∈ ls.outer_reductions
    oreducop = ls.operations[j]
    if ortypdefined
      push!(pt.args, eltype_expr(oreducop))
    else
      push!(pt.args, outer_reduct_init_typename(oreducop))
    end
  end
end
function determine_eltype(ls::LoopSet, ortypdefined::Bool)::Union{Symbol,Expr}
  narrays = length(ls.includedactualarrays)
  noreduc = length(ls.outer_reductions)
  ntyp = narrays + noreduc
  if ntyp == 0
    return Expr(:call, lv(:typeof), 0)
  elseif ntyp == 1
    if narrays == 1
      return Expr(:call, lv(:eltype), first(ls.includedactualarrays))
    else
      oreducop = ls.operations[ls.outer_reductions[1]]
      if ortypdefined
        return eltype_expr(oreducop)
      else
        return outer_reduct_init_typename(oreducop)
      end
    end
  end
  pt = Expr(:call, lv(:promote_type))
  for array ∈ ls.includedactualarrays
    push!(pt.args, Expr(:call, lv(:eltype), array))
  end
  push_outer_reduct_types!(pt, ls, ortypdefined)
  return pt
end
@inline _eltype(x) = eltype(x)
@inline _eltype(::BitArray) = VectorizationBase.Bit
function determine_width(ls::LoopSet, vectorized::Union{Symbol,Nothing})
  vwidth_q = Expr(:call, lv(:pick_vector_width))
  if vectorized ≢ nothing
    vloop = getloop(ls, vectorized)
    if isstaticloop(vloop)
      push!(vwidth_q.args, Expr(:call, Expr(:curly, :Val, length(vloop))))
    end
  end
  push!(vwidth_q.args, ELTYPESYMBOL)
  vwidth_q
end
function init_remblock(unrolledloop::Loop, lssm::LoopStartStopManager, n::Int)#u₁loop::Symbol = unrolledloop.itersymbol)
  termind = lssm.terminators[n]
  if iszero(termind)
    rangesym = unrolledloop.rangesym
    if rangesym === Symbol("")
      condition = Expr(
        :call,
        lv(:cmpend),
        unrolledloop.itersymbol,
        staticloopexpr(unrolledloop)
      )
    else
      condition = Expr(:call, lv(:cmpend), unrolledloop.itersymbol, rangesym)
    end
  else
    termar = lssm.incrementedptrs[n][termind]
    ptr = vptr(termar)
    ptroff = vptr_offset(ptr)
    condition = Expr(
      :call,
      GlobalRef(VectorizationBase, :vlt),
      ptroff,
      maxsym(ptr, 0),
      ptr
    )
  end
  Expr(:if, condition)
end

maskexpr(looplimit) =
  Expr(:(=), MASKSYMBOL, Expr(:call, lv(:mask), VECTORWIDTHSYMBOL, looplimit))
@inline idiv_fast(a::I, b::I) where {I<:Base.BitInteger} = Base.udiv_int(a, b)
@inline idiv_fast(a, b) = idiv_fast(Int(a), Int(b))
function definemask(loop::Loop)
  isstaticloop(loop) && return maskexpr(length(loop))
  # W = 4
  # loop iterates 3, step 2
  # (1, 3, 5), 7
  start = first(loop)
  incr = step(loop)
  stop = last(loop)
  if isone(start) & isone(incr)
    isknown(stop) ? maskexpr(gethint(stop)) : maskexpr(getsym(stop))
  elseif loop.lensym !== Symbol("")
    maskexpr(loop.lensym)
  elseif isone(incr)
    if isknown(start) & isknown(stop)
      maskexpr(1 + gethint(stop) - gethint(start))
    else
      lexpr = if isknown(start)
        subexpr(stop, gethint(start) - 1)
      elseif isknown(stop)
        subexpr(gethint(stop) + 1, start)
      else
        subexpr(stop, subexpr(start, 1))
      end
      maskexpr(lexpr)
    end
  else
    lenexpr = Expr(:call, lv(:idiv_fast), subexpr(stop, start))
    pushexpr!(lenexpr, incr)
    maskexpr(addexpr(lenexpr, 1))
  end
end
function define_eltype_vec_width!(
  q::Expr,
  ls::LoopSet,
  vectorized,
  ortypdefined::Bool
)
  push!(q.args, Expr(:(=), ELTYPESYMBOL, determine_eltype(ls, ortypdefined)))
  push!(q.args, Expr(:(=), VECTORWIDTHSYMBOL, determine_width(ls, vectorized)))
  nothing
end
function setup_preamble!(ls::LoopSet, us::UnrollSpecification, Ureduct::Int)
  @unpack u₁loopnum, u₂loopnum, vloopnum, u₁, u₂ = us
  order = names(ls)
  vectorized = order[vloopnum]
  set_vector_width!(ls, vectorized)
  iszero(length(ls.includedactualarrays) + length(ls.outer_reductions)) ||
    define_eltype_vec_width!(ls.preamble, ls, vectorized, false)
  lower_licm_constants!(ls)
  isone(num_loops(ls)) || pushpreamble!(ls, definemask(getloop(ls, vectorized)))#, u₁ > 1 && u₁loopnum == vloopnum))
  if (Ureduct == u₁) || (u₂ != -1) || (Ureduct == -1)
    initialize_outer_reductions!(ls, ifelse(Ureduct == -1, u₁, Ureduct)) # TODO: outer reducts?
  elseif length(ls.outer_reductions) > 0
    decl = Expr(:local)
    for or ∈ ls.outer_reductions
      push!(decl.args, Symbol(mangledvar(ls.operations[or]), '_', Ureduct))
    end
    pushpreamble!(ls, decl)
  end
  for op ∈ operations(ls)
    if (iszero(length(loopdependencies(op))) && iscompute(op))
      ua = UnrollArgs(
        getloop(ls, us.u₁loopnum),
        getloop(ls, us.u₂loopnum),
        getloop(ls, us.vloopnum),
        u₁,
        u₂,
        -1
      )
      lower_compute!(ls.preamble, op, ls, ua, false)
    end
  end
end
lsexpr(ls::LoopSet, q) = Expr(:block, ls.preamble, q)

function isanouterreduction(ls::LoopSet, op::Operation)
  opname = name(op)
  for or ∈ ls.outer_reductions
    name(ls.operations[or]) === opname && return true
  end
  false
end

# tiled_outerreduct_unroll(ls::LoopSet) = tiled_outerreduct_unroll(ls.unrollspecification)
# function tiled_outerreduct_unroll(us::UnrollSpecification)
#     @unpack u₁, u₂ = us
#     unroll = u₁ ≥ 8 ? 1 : 8 ÷ u₁
#     cld(u₂, cld(u₂, unroll))
# end
function calc_Ureduct!(ls::LoopSet, us::UnrollSpecification)
  @unpack u₁loopnum, u₁, u₂, vloopnum = us
  ur = if iszero(length(ls.outer_reductions))
    -1
  elseif u₂ == -1
    if u₁loopnum == num_loops(ls)
      u₁loop = getloop(ls, u₁loopnum)
      loopisstatic = isstaticloop(u₁loop)
      loopisstatic &= ((vloopnum != u₁loopnum) | (!iszero(ls.vector_width)))
      # loopisstatic ? u₁ : min(u₁, 4) # much worse than the other two options, don't use this one
      if loopisstatic
        W = Core.ifelse(vloopnum == u₁loopnum, ls.vector_width, 1)
        UFt = cld(length(u₁loop) % (W * u₁), W)
        Core.ifelse(UFt == 0, u₁, UFt)
        # rem = length(u₁loop) - 
        # max(1, cld(rem, u₁))
      else
        Core.ifelse(get_cpu_name() === "znver1", 1, Core.ifelse(u₁ ≥ 4, 2, 1))
      end
    else
      -1
    end
  else
    u₁ui = u₂ui = -1
    u₁loopsym = getloop(ls, u₁loopnum).itersymbol
    u₂loopsym = getloop(ls, us.u₂loopnum).itersymbol
    vloopsym = getloop(ls, vloopnum).itersymbol
    for or ∈ ls.outer_reductions
      op = ls.operations[or]
      u₁u, u₂u = isunrolled_sym(op, u₁loopsym, u₂loopsym, vloopsym, us)
      if u₁ui == -1
        u₁ui = Int(u₁u)
        u₂ui = Int(u₁u)
      elseif !((u₁ui == Int(u₁u)) & (u₂ui == Int(u₁u)))
        throw(
          ArgumentError(
            "Doesn't currently handle differently unrolled reductions yet, please file an issue with an example."
          )
        )
      end
    end
    if u₁ui % Bool
      u₁
    else
      u₂
    end
  end
  ls.ureduct = ur
end
ureduct(ls::LoopSet) = ls.ureduct
function lower_unrollspec(ls::LoopSet)
  us = ls.unrollspecification
  @unpack vloopnum, u₁, u₂ = us
  init_loop_map!(ls)
  Ureduct = calc_Ureduct!(ls, us)
  setup_preamble!(ls, us, Ureduct)
  initgesps = add_loop_start_stop_manager!(ls)
  q =
    Expr(:let, initgesps, lower_unrolled_dynamic(ls, us, num_loops(ls), false))
  q = gc_preserve(ls, Expr(:block, q))
  reduce_expr!(q, ls, Ureduct)
  lsexpr(ls, q)
end

function lower(
  ls::LoopSet,
  order,
  u₁loop,
  u₂loop,
  vectorized,
  u₁,
  u₂,
  inline::Bool
)
  cacheunrolled!(ls, u₁loop, u₂loop, vectorized)
  fillorder!(ls, order, u₁loop, u₂loop, u₂, vectorized)
  ls.unrollspecification =
    UnrollSpecification(ls, u₁loop, u₂loop, vectorized, u₁, u₂)
  q = lower_unrollspec(ls)
  inline && pushfirst!(q.args, Expr(:meta, :inline))
  q
end

function lower(ls::LoopSet, inline::Int = -1)
  fill_offset_memop_collection!(ls)
  order, u₁loop, u₂loop, vectorized, u₁, u₂, c, shouldinline =
    choose_order_cost(ls)
  lower(
    ls,
    order,
    u₁loop,
    u₂loop,
    vectorized,
    u₁,
    u₂,
    inlinedecision(inline, shouldinline)
  )
end
function lower(ls::LoopSet, u₁::Int, u₂::Int, v::Int, inline::Int)
  fill_offset_memop_collection!(ls)
  if u₂ > 1
    @assert num_loops(ls) > 1 "There is only $(num_loops(ls)) loop, but specified blocking parameter u₂ is $u₂."
    order, u₁loop, u₂loop, vectorized, _u₁, _u₂, c, shouldinline =
      choose_tile(ls, store_load_deps(operations(ls)), v)
    copyto!(ls.loop_order.bestorder, order)
  elseif u₁ > 0
    u₂ = -1
    order, vectorized, c =
      choose_unroll_order(ls, Inf, store_load_deps(operations(ls)), v)
    u₁loop = first(order)
    u₂loop = Symbol("##undefined##")
    shouldinline = true
    copyto!(ls.loop_order.bestorder, order)
  else
    order, u₁loop, u₂loop, vectorized, u₁, u₂, c, shouldinline =
      choose_order_cost(ls, v)
  end
  doinline = inlinedecision(inline, shouldinline)
  lower(ls, order, u₁loop, u₂loop, vectorized, u₁, u₂, doinline)
end

# Base.convert(::Type{Expr}, ls::LoopSet) = lower(ls)
Base.show(io::IO, ls::LoopSet) = println(io, lower(ls))

function search_children_for_self(op::Operation, target::Symbol = name(op))
  for opc ∈ children(op)
    name(opc) === target && return opc
  end
  for opc ∈ children(op)#breadth first
    opcc = search_children_for_self(opc, target)
    opcc === opc || return opcc
  end
  op
end

# TODO: this is no longer how I generate code...
"""
This function is normally called
isunrolled_sym(op, u₁loop)
isunrolled_sym(op, u₁loop, u₂loop)

It returns `true`/`false` for each loop, indicating whether they're unrolled.

If there is a third argument, it will avoid unrolling that symbol along reductions if said symbol is part of the reduction chain.
"""
function isunrolled_sym(
  op::Operation,
  u₁loop::Symbol,
  u₂loop::Symbol,
  vloop::Symbol,
  (u₁ild, u₂ild)::Tuple{Bool,Bool} = (isu₁unrolled(op), isu₂unrolled(op))
)
  (accesses_memory(op) | isloopvalue(op)) && return (u₁ild, u₂ild)
  if isconstant(op)
    if length(loopdependencies(op)) == 0
      newop = search_children_for_self(op)
      newop === op || return isunrolled_sym(newop, u₁loop, u₂loop, vloop)
    end
    if !u₁ild
      u₁ild = u₁loop ∈ reducedchildren(op)
    end
    if !u₂ild
      u₂ild = u₂loop ∈ reducedchildren(op)
    end
  end
  (u₁ild & u₂ild) || return u₁ild, u₂ild
  reductops = isconstant(op) ? reducedchildren(op) : reduceddependencies(op)
  iszero(length(reductops)) && return true, true
  u₁reduced = u₁loop ∈ reductops
  u₂reduced = u₂loop ∈ reductops
  # If they're being reduced, we want to only unroll the reduced variable along one of the two loops.
  if u₂reduced
    if u₁reduced# if both are reduced, we unroll u₁
      if vloop === u₁loop
        false, true
      else
        true, false
      end
    else
      true, false
    end
  elseif u₁reduced
    false, true
  else
    true, true
  end
end
function isunrolled_sym(
  op::Operation,
  u₁loop::Symbol,
  u₂loop::Symbol,
  vloop::Symbol,
  ls::LoopSet
)
  us = ls.unrollspecification
  isunrolled_sym(op, u₁loop, u₂loop, vloop, us)
end
function isunrolled_sym(
  op::Operation,
  u₁loop::Symbol,
  u₂loop::Symbol,
  vloop::Symbol,
  us::UnrollSpecification
)
  @unpack u₁, u₂ = us
  u₁u = (u₁ > 1) & isu₁unrolled(op)
  u₂u = (u₂ > 1) & isu₂unrolled(op)
  ((u₂ > 1) | accesses_memory(op)) ?
  isunrolled_sym(op, u₁loop, u₂loop, vloop, (u₁u, u₂u)) :
  (isunrolled_sym(op, u₁loop, u₁u), false)
end

isunrolled_sym(op::Operation, u₁loop::Symbol, ls::LoopSet) =
  isunrolled_sym(op, u₁loop, ls.unrollspecification)
function isunrolled_sym(op::Operation, u₁loop::Symbol, us::UnrollSpecification)
  u₁u = (us.u₁ > 1) & isu₁unrolled(op)
  isunrolled_sym(op, u₁loop, u₁u)
end
function isunrolled_sym(
  op::Operation,
  u₁loop::Symbol,
  u₁u::Bool = isu₁unrolled(op)
)
  u₁u || (isconstant(op) & (u₁loop ∈ reducedchildren(op)))
end

function isunrolled_sym(
  op::Operation,
  u₁loop::Symbol,
  u₂loop::Symbol,
  vloop::Symbol,
  u₂max::Int
)
  ((u₂max > 1) | accesses_memory(op)) ?
  isunrolled_sym(op, u₁loop, u₂loop, vloop) :
  (isunrolled_sym(op, u₁loop), false)
end

function variable_name(op::Operation, suffix::Int)
  mvar = mangledvar(op)
  suffix == -1 ? mvar : Symbol(mvar, suffix, :_)
end

function variable_name_and_unrolled(
  op::Operation,
  u₁loop::Symbol,
  u₂loop::Symbol,
  vloop::Symbol,
  u₂iter::Int,
  ls::LoopSet
)
  u₁op, u₂op = isunrolled_sym(op, u₁loop, u₂loop, vloop, ls)
  mvar = u₂op ? variable_name(op, u₂iter) : mangledvar(op)
  mvar, u₁op, u₂op
end
