

# the `lowernonstore` and `lowerstore` options are there as a means of lowering all non-store operations before lowering the stores.
function lower!(
    q::Expr, ops::AbstractVector{Operation}, ls::LoopSet, unrollsyms::UnrollSymbols, u₁::Int, u₂::Int,
    suffix::Int, mask::Bool, lowernonstore::Bool, lowerstore::Bool
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

function lower_block(
    ls::LoopSet, us::UnrollSpecification, n::Int, mask::Bool, UF::Int
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
    for prepost ∈ 1:2
        # !u₁ && !u₂
        lower!(blockq, ops[1,1,prepost,n], ls, unrollsyms, u₁, u₂, -1, mask, true, true)
        # isu₁unrolled, isu₂unrolled, after_loop, n
        opsv1 = ops[1,2,prepost,n]
        opsv2 = ops[2,2,prepost,n]
        if length(opsv1) + length(opsv2) > 0
            nstores = 0
            iszero(length(opsv1)) || (nstores += sum(isstore, opsv1))
            iszero(length(opsv2)) || (nstores += sum(isstore, opsv2))
            # if nstores
            if (length(opsv1) + length(opsv2) == nstores) && u₂ > 1 # all_u₂_ops_store
                lower!(blockq, ops[2,1,prepost,n], ls, unrollsyms, u₁, u₂, -1, mask, true, true) # for u ∈ 0:u₁-1
                lower_tiled_store!(blockq, opsv1, opsv2, ls, unrollsyms, u₁, u₂, mask)
            else
                for store ∈ (false,true)
                    for t ∈ 0:u₂-1
                        # !u₁ &&  u₂
                        lower!(blockq, opsv1, ls, unrollsyms, u₁, u₂, t, mask & !(dontmaskfirsttiles & (t < u₂ - 1)), !store, store)
                        if iszero(t) && !store #  u₁ && !u₂
                            # for u ∈ 0:u₁-1
                            lower!(blockq, ops[2,1,prepost,n], ls, unrollsyms, u₁, u₂, -1, mask, true, true)
                            # end
                        end
                        #  u₁ && u₂
                        # for u ∈ 0:u₁-1
                        lower!(blockq, opsv2, ls, unrollsyms, u₁, u₂, t, mask & !(dontmaskfirsttiles & (t < u₂ - 1)), !store, store)
                        # end
                    end
                    nstores == 0 && break
                end
            end
        else
            # for u ∈ 0:u₁-1     #  u₁ && !u₂
            lower!(blockq, ops[2,1,prepost,n], ls, unrollsyms, u₁, u₂, -1, mask, true, false)
            lower!(blockq, ops[2,1,prepost,n], ls, unrollsyms, u₁, u₂, -1, mask, false, true)
            # end
        end
        if n > 1 && prepost == 1
            push!(blockq.args, lower_unrolled_dynamic(ls, us, n-1, mask))
        end
    end
    loopsym = order[n]
    # if n > 1 || iszero(ls.align_loops[])
    incrementloopcounter!(blockq, ls, us, n, UF)
    # else
    #     loopsym = names(ls)[n]
    #     push!(blockq.args, Expr(:(=), loopsym, Expr(:call, lv(:vadd_fast), loopsym, Symbol("##ALIGNMENT#STEP##"))))
    # end
    blockq
end

assume(ex) = Expr(:call, GlobalRef(VectorizationBase, :assume), ex)
function expect(ex)
  # Expr(:call, GlobalRef(VectorizationBase, :expect), ex)
    ex
end
function loopiteratesatleastonce!(ls, loop::Loop)
  start = first(loop); stop = last(loop)
  (isknown(start) & isknown(stop)) && return loop
  comp = Expr(:call, :≥)
  pushexpr!(comp, last(loop))
  pushexpr!(comp, first(loop))
  pushpreamble!(ls, assume(comp))
  return loop
end
# @inline step_to_align(x, ::Val{W}) where {W} = step_to_align(pointer(x), Val{W}())
# @inline step_to_align(x::Ptr{T}, ::Val{W}) where {W,T} = vsub_fast(W, reinterpret(Int, x) & (W - 1))
# function align_inner_loop_expr(ls::LoopSet, us::UnrollSpecification, loop::Loop)
#     alignincr = Symbol("##ALIGNMENT#STEP##")
#     looplength = gensym(:inner_loop_length)
#     pushpreamble!(ls, Expr(:(=), looplength, looplengthexpr(loop)))
#     vp = vptr(operations(ls)[ls.align_loops[]])
#     align_step = Expr(:call, :min, Expr(:call, lv(:step_to_align), vp, VECTORWIDTHSYMBOL), looplength)
#     Expr(
#         :block,
#         Expr(:(=), alignincr, align_step),
#         maskexpr(alignincr),
#         lower_block(ls, us, 1, true, 1)
#     )
# end

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
        unroll_total *= cld(length(loop),W)
    end
    if us.u₁loopnum > N
        unroll_total *= us.u₁
    end
    if us.u₂loopnum > N
        unroll_total *= us.u₂
    end
    unroll_total ≤ 16
end

function lower_no_unroll(ls::LoopSet, us::UnrollSpecification, n::Int, inclmask::Bool)#, initialize::Bool = true, maxiters::Int=-1)
    usorig = ls.unrollspecification
    nisvectorized = isvectorized(us, n)
    loopsym = names(ls)[n]
    loop = getloop(ls, n)
    # if !nisvectorized && !inclmask && isone(n) && !iszero(ls.lssm.terminators[1]) && !ls.loadelimination && (us.u₁ > 1) && (usorig.u₁ == us.u₁) && (usorig.u₂ == us.u₂) && length(loop) > 15
    #     return lower_unroll_for_throughput(ls, us, loop, loopsym)
    #     # return lower_llvm_unroll(ls, us, n, loop)
    # end
    tc = terminatecondition(ls, us, n, inclmask, 1)
    body = lower_block(ls, us, n, inclmask, 1)
    # align_loop = isone(n) & (ls.align_loops[] > 0)
    loopisstatic = isstaticloop(loop)
    if !loopisstatic && (usorig.u₁ == us.u₁) && (usorig.u₂ == us.u₂) && !inclmask
        tc = expect(tc)
    end
    W = nisvectorized ? ls.vector_width : 1
    loopisstatic &= (!iszero(W))
    # q = if align_loop
    #     Expr(:block, align_inner_loop_expr(ls, us, loop), Expr(:while, tc, body))
    # elseif nisvectorized
    if loopisstatic && (!ls.loadelimination) && (isone(length(loop) ÷ W) || (n ≤ 3 && length(loop) ≤ 8W && allinteriorunrolled(ls, us, n)))
        q = Expr(:block)
        for _ ∈ 1:(length(loop) ÷ W)
            push!(q.args, body)
        end
    elseif nisvectorized
            # Expr(:block, loopiteratesatleastonce(loop, true), Expr(:while, expect(tc), body))
        # q = Expr(:block, Expr(maxiters == 1 ? :if : :while, tc, body))
        q = Expr(:block, Expr(:while, tc, body))
    else
        # push!(body.args, Expr(:(||), tc, Expr(:break)))
        # q = Expr(:block, Expr(:while, true, body))
        # using the `termcond` variable leads to better code gen.
        termcond = gensym(:maybeterm)
        push!(body.args, Expr(:(=), termcond, tc))
        q = Expr(:block, Expr(:(=), termcond, true), Expr(:while, termcond, body))
    end
    if nisvectorized && !(loopisstatic && iszero(length(loop) & (W - 1)))
        # tc = terminatecondition(loop, us, n, loopsym, true, 1)
        body = lower_block(ls, us, n, true, 1)
        if isone(num_loops(ls))
            pushfirst!(body.args, definemask(loop))
        # elseif align_loop
        #     pushfirst!(body.args, definemask_for_alignment_cleanup(loop))
        end
        if loopisstatic
            push!(q.args, body)
        else
            tc = terminatecondition(ls, us, n, true, 1)
            push!(q.args, Expr(:if, tc, body))
        end
    end
    # if initialize
    Expr(:let, startloop(ls, us, n), q)
    # else
    #     q
    # end
end
function lower_unrolled_dynamic(ls::LoopSet, us::UnrollSpecification, n::Int, inclmask::Bool)
    UF = unrollfactor(us, n)
    isone(UF) && return lower_no_unroll(ls, us, n, inclmask)
    @unpack u₁loopnum, vloopnum, u₁, u₂ = us
    order = names(ls)
    loopsym = order[n]
    loop = getloop(ls, n)
    vectorized = order[vloopnum]
    nisunrolled = isunrolled1(us, n)
    nisvectorized = isvectorized(us, n)
    W = nisvectorized ? ls.vector_width : 1
    loopisstatic = isstaticloop(loop) & (!iszero(W))
    UFW = UF * W
    looplength = length(loop)
    if loopisstatic & (UFW > looplength)
        UFWnew = cld(looplength, cld(looplength, UFW))
        UF = cld(UFWnew, W)
        UFW = UF * W
        us = nisunrolled ? UnrollSpecification(us, UF, u₂) : UnrollSpecification(us, u₁, UF)
    end
    remmask = inclmask | nisvectorized
    Ureduct = (n == num_loops(ls) && (u₂ == -1)) ? ureduct(ls) : -1
    sl = startloop(ls, us, n)
    UFt = loopisstatic ? cld(looplength % UFW, W) : 1
    # Don't place remainder first if we're going to have to mask this loop (i.e., if this loop is vectorized)
    remfirst = loopisstatic & (!nisvectorized) & (UFt > 0) & !(unsigned(Ureduct) < unsigned(UF))
    tc = terminatecondition(ls, us, n, inclmask, remfirst ? 1 : UF)
    # usorig = ls.unrollspecification
    # tc = (usorig.u₁ == us.u₁) && (usorig.u₂ == us.u₂) && !loopisstatic && !inclmask && !ls.loadelimination ? expect(tc) : tc
    body = lower_block(ls, us, n, inclmask, UF)
    if loopisstatic
        iters = length(loop) ÷ UFW
        if (iters ≤ 1) || (iters*UF ≤ 16 && allinteriorunrolled(ls, us, n))# Let's set a limit on total unrolling
            q = Expr(:block)
            for _ ∈ 1:iters
                push!(q.args, body)
            end
        else
            q = Expr(:while, tc, body)
        end
        remblock = Expr(:block)
        (nisvectorized && (UFt > 0) && isone(num_loops(ls))) && push!(remblock.args, definemask(loop))
        unroll_cleanup = true
    else
        remblock = init_remblock(loop, ls.lssm, n)#loopsym)
        # unroll_cleanup = Ureduct > 0 || (nisunrolled ? (u₂ > 1) : (u₁ > 1))
        # remblock = unroll_cleanup ? init_remblock(loop, ls.lssm, n)#loopsym) : Expr(:block)
        q = if unsigned(Ureduct) < unsigned(UF)
            # push!(body.args, Expr(:(||), tc, Expr(:break)))
            # Expr(:while, true, body)
            termcond = gensym(:maybeterm)
            push!(body.args, Expr(:(=), termcond, tc))
            Expr(:block, Expr(:(=), termcond, true), Expr(:while, termcond, body))
        else
            Expr(:while, tc, body)
        end
    end
    q = if unsigned(Ureduct) < unsigned(UF) # unsigned(-1) == typemax(UInt); 
        add_cleanup = true
        if isone(Ureduct)
            UF_cleanup = 1
            if nisvectorized
                blockhead = :while
            else
                blockhead = if UF == 2
                    if loopisstatic
                        add_cleanup = UFt == 1
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
        _q = Expr(:block, add_upper_outer_reductions(ls, q, Ureduct, UF, loop, vectorized, nisvectorized))
        if add_cleanup
            cleanup_expr = Expr(blockhead)
            blockhead === :block || push!(cleanup_expr.args, terminatecondition(ls, us, n, inclmask, UF_cleanup))
            us_cleanup = nisunrolled ? UnrollSpecification(us, UF_cleanup, u₂) : UnrollSpecification(us, u₁, UF_cleanup)
            push!(cleanup_expr.args, lower_block(ls, us_cleanup, n, inclmask, UF_cleanup))
            push!(_q.args, cleanup_expr)
        end
        UFt > 0 && push!(_q.args, remblock)
        _q
    elseif remfirst
        numiters = length(loop) ÷ UF
        if numiters > 2
            Expr( :block, remblock, q )
        else
            q = Expr(:block, remblock)
            for i ∈ 1:numiters
                push!(q.args, body)
            end
            q
        end
    elseif iszero(UFt)
        Expr( :block, q )
    elseif !nisvectorized && !loopisstatic && UF ≥ 10
        rem_uf = UF - 1
        UF = rem_uf >> 1
        UFt = rem_uf - UF
        ust = nisunrolled ? UnrollSpecification(us, UFt, u₂) : UnrollSpecification(us, u₁, UFt)
        newblock = lower_block(ls, ust, n, remmask, UFt)
        # comparison = unrollremcomparison(ls, loop, UFt, n, nisvectorized, remfirst)
        comparison = terminatecondition(ls, us, n, inclmask, UFt)
        UFt = 1
        UF += 1 - iseven(rem_uf)
        Expr( :block, q, Expr(iseven(rem_uf) ? :while : :if, comparison, newblock), remblock )
    else
        # if (usorig.u₁ == us.u₁) && (usorig.u₂ == us.u₂) && !isstaticloop(loop) && !inclmask# && !ls.loadelimination
        #     # Expr(:block, sl, assumeloopiteratesatleastonce(loop), Expr(:while, tc, body))
        #     Expr(:block, sl, expect(tc), q, remblock)
        # else
        #     Expr(:block, sl, q, remblock)
        # end
        Expr( :block, q, remblock )
    end
    if !iszero(UFt)
        # if unroll_cleanup
        iforelseif = :if
        while true
            ust = nisunrolled ? UnrollSpecification(us, UFt, u₂) : UnrollSpecification(us, u₁, UFt)
            newblock = lower_block(ls, ust, n, remmask, UFt)
            if (UFt ≥ UF - 1 + nisvectorized) || UFt == Ureduct || loopisstatic
                if isone(num_loops(ls)) && isone(UFt) && isone(Ureduct)
                    newblock = Expr(:block, definemask(loop), newblock)
                end
                push!(remblock.args, newblock)
                break
            end
            comparison = unrollremcomparison(ls, loop, UFt, n, nisvectorized, remfirst)
            if isone(num_loops(ls)) && isone(UFt)
                remblocknew = Expr(:if, comparison, newblock)
                push!(remblock.args, Expr(:block, Expr(:let, definemask(loop), remblocknew)))
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
    Expr(:block, Expr(:let, sl, q))
end
function unrollremcomparison(ls::LoopSet, loop::Loop, UFt::Int, n::Int, nisvectorized::Bool, remfirst::Bool)
    termind = ls.lssm.terminators[n]
    if iszero(termind)
        loopvarremcomparison(loop, UFt, nisvectorized, remfirst)
    else
        pointerremcomparison(ls, termind, UFt, n, nisvectorized, remfirst, loop)
    end
end
function loopvarremcomparison(loop::Loop, UFt::Int, nisvectorized::Bool, remfirst::Bool)
    loopsym = loop.itersymbol
    loopstep = loop.step
    if nisvectorized
        offset = mulexpr(VECTORWIDTHSYMBOL, UFt, loopstep)
        itercount = subexpr(last(loop), offset)
        Expr(:call, GlobalRef(Base,:>), loopsym, itercount)
    elseif remfirst # requires `isstaticloop(loop)`
        Expr(:call, GlobalRef(Base,:<), loopsym, gethint(first(loop)) + UFt*gethint(loopstep) - 1)
    elseif isknown(last(loop))
        if isknown(loopstep)
            Expr(:call, GlobalRef(Base,:>), loopsym, gethint(last(loop)) - UFt*gethint(loopstep))
        elseif isone(UFt)
            Expr(:call, GlobalRef(Base,:>), loopsym, subexpr(gethint(last(loop)), getsym(loopstep)))
        else
            Expr(:call, GlobalRef(Base,:>), loopsym, subexpr(gethint(last(loop)),  mulexpr(getsym(loopstep), UFt)))
        end
    else
        if isknown(loopstep)
            Expr(:call, GlobalRef(Base,:>), loopsym, Expr(:call, lv(:vsub_fast), getsym(last(loop)), UFt*gethint(loopstep)))
        elseif isone(UFt)
            Expr(:call, GlobalRef(Base,:>), loopsym, Expr(:call, lv(:vsub_fast), getsym(last(loop)), getsym(loopstep)))
        else
            Expr(:call, GlobalRef(Base,:>), loopsym, Expr(:call, lv(:vsub_fast), getsym(last(loop)), mulexpr(getsym(loopstep), UFt)))
        end
    end
end
function pointerremcomparison(ls::LoopSet, termind::Int, UFt::Int, n::Int, nisvectorized::Bool, remfirst::Bool, loop::Loop)
    lssm = ls.lssm
    termar = lssm.incrementedptrs[n][termind]
    ptrdef = lssm.incrementedptrs[n][termind]
    ptr = vptr(termar)
    if remfirst
        Expr(:call, :<, ptr, pointermax(ls, ptrdef, n, 1 - UFt, nisvectorized, loop))
    else
        Expr(:call, :≥, ptr, maxsym(ptr, UFt))
    end
end

# TODO: handle tiled outer reductions; they will require a suffix arg
function initialize_outer_reductions!(
    q::Expr, ls::LoopSet, op::Operation, _Umax::Int, vectorized::Symbol, us::UnrollSpecification, rs::Expr
)
    @unpack u₁, u₂ = us
    Umax = u₂ == -1 ? _Umax : u₁
    reduct_zero = reduction_zero(op.instruction)
    typeTr = ELTYPESYMBOL
    u₁u, u₂u = isunrolled_sym(op, getloop(ls, us.u₁loopnum).itersymbol, getloop(ls, us.u₂loopnum).itersymbol, getloop(ls, us.vloopnum).itersymbol, ls)#, u₂)
    z = if isvectorized(op)
        if Umax == 1 || !u₁u
            if reduct_zero === :zero
                Expr(:call, lv(:_vzero), VECTORWIDTHSYMBOL, typeTr, rs)
            else
                Expr(:call, lv(:_vbroadcast), VECTORWIDTHSYMBOL, Expr(:call, reduct_zero, typeTr), rs)
            end
        else
            if reduct_zero === :zero
                Expr(:call, lv(:zero_vecunroll), staticexpr(Umax), VECTORWIDTHSYMBOL, typeTr, rs)
            else
                Expr(:call, lv(:vbroadcast_vecunroll), staticexpr(Umax), VECTORWIDTHSYMBOL, Expr(:call, reduct_zero, typeTr), rs)
            end
        end
    else
        Expr(:call, reduct_zero, typeTr)
    end
    mvar = variable_name(op, -1)
    # @show u₁, u₂, u₁u, _Umax
    if u₂ == -1
        push!(q.args, Expr(:(=), Symbol(mvar, '_', _Umax), z))
    else#if isu₂unrolled(op) #& (us.vloopnum ≠ us.u₂loopnum) # tiled outer reduction, u₂unrolled
        # TODO: add `(us.vloopnum ≠ us.u₂loopnum)` check to avoid unrolling and vectorizing a reduction along the same axis
        if u₁u
            push!(q.args, Expr(:(=), Symbol(mvar, '_', u₁), z))
        else
            for u ∈ 0:_Umax-1
                # push!(q.args, Expr(:(=), Symbol(mvar, '_', u), z))
                push!(q.args, Expr(:(=), Symbol(mvar, u), z))
            end
        end
    end
    nothing
end
function initialize_outer_reductions!(q::Expr, ls::LoopSet, Umax::Int, vectorized::Symbol)
    rs = staticexpr(reg_size(ls))
    us = ls.unrollspecification
    for or ∈ ls.outer_reductions
        initialize_outer_reductions!(q, ls, ls.operations[or], Umax, vectorized, us, rs)
    end
end
initialize_outer_reductions!(ls::LoopSet, Umax::Int, vectorized::Symbol) = initialize_outer_reductions!(ls.preamble, ls, Umax, vectorized)
function add_upper_comp_check(unrolledloop, loopbuffer)
  
    if isstaticloop(unrolledloop)
        Expr(:call, Base.GlobalRef(Base,:≥), length(unrolledloop), loopbuffer)
    elseif isknown(first(unrolledloop))
        if isone(first(unrolledloop))
            Expr(:call, Base.GlobalRef(Base,:≥), getsym(last(unrolledloop)), loopbuffer)
        else
            Expr(:call, Base.GlobalRef(Base,:≥), getsym(last(unrolledloop)), addexpr(loopbuffer,gethint(first(unrolledloop))-1))
        end
    elseif isknown(last(unrolledloop))
        Expr(:call, Base.GlobalRef(Base,:≥), Expr(:call, lv(:vsub_fast), gethint(last(unrolledloop))+1, getsym(first(unrolledloop))), loopbuffer)
    else# both are given by symbols
        Expr(:call, Base.GlobalRef(Base,:>), Expr(:call, lv(:vsub_fast), getsym(last(unrolledloop)), Expr(:call,lv(:vsub_fast), getsym(first(unrolledloop)), staticexpr(1))), loopbuffer)
    end
end
function add_upper_outer_reductions(ls::LoopSet, loopq::Expr, Ulow::Int, Uhigh::Int, unrolledloop::Loop, vectorized::Symbol, reductisvectorized::Bool)
    ifq = Expr(:block)
    ifqlet = Expr(:block)
    initialize_outer_reductions!(ifqlet, ls, Uhigh, vectorized)
    # @show loopq
    push!(ifq.args, loopq)
    t = Expr(:tuple)
    mvartu = Expr(:tuple)
    mvartl = Expr(:tuple)
    for or ∈ ls.outer_reductions
        op = ls.operations[or]
        # var = name(op)
        mvar = Symbol(mangledvar(op), '_', Uhigh)
        instr = instruction(op)
        f = reduce_number_of_vectors(instr)
        push!(t.args, Expr(:call, lv(f), mvar, staticexpr(Ulow)))
        push!(mvartu.args, mvar)
        push!(mvartl.args, Symbol(mangledvar(op), '_', Ulow))
    end
    push!(ifq.args, t)
    ifqfull = Expr(:let, ifqlet, ifq)
    ncomparison = if reductisvectorized
        add_upper_comp_check(unrolledloop, mulexpr(VECTORWIDTHSYMBOL, Uhigh, step(unrolledloop)))
    elseif isknown(step(unrolledloop))
        add_upper_comp_check(unrolledloop, Uhigh*gethint(step(unrolledloop)))
    else
        add_upper_comp_check(unrolledloop, mulexpr(Uhigh, getsym(step(unrolledloop))))
    end
    elseq = Expr(:block)
    initialize_outer_reductions!(elseq, ls, Ulow, vectorized)
    push!(elseq.args, mvartl)
    Expr(:(=), mvartl, Expr(:if, ncomparison, ifqfull, elseq))
end
## This performs reduction to one `Vec`
function reduce_expr!(q::Expr, ls::LoopSet, U::Int)
    us = ls.unrollspecification
    u₁f, u₂f = if us.u₂ == -1
        ifelse(U == -1, us.u₁, U), -1
    else
        us.u₁, U
    end
    # u₁loop, u₂loop = getunrolled(ls)
    u₁loop = getloop(ls, us.u₁loopnum).itersymbol
    u₂loop = getloop(ls, us.u₂loopnum).itersymbol
    vloop  = getloop(ls,  us.vloopnum).itersymbol
    for or ∈ ls.outer_reductions
        op = ls.operations[or]
        var = name(op)
        mvar = mangledvar(op)
        instr = instruction(op)
        u₁u, u₂u = isunrolled_sym(op, u₁loop, u₂loop, vloop, ls)#, u₂f)
        reduce_expr!(q, mvar, instr, u₁f, u₂f, u₁u, u₂u)
        if !iszero(length(ls.opdict))
            if (isu₁unrolled(op) | isu₂unrolled(op))
                push!(q.args, Expr(:(=), var, Expr(:call, lv(reduction_scalar_combine(instr)), Symbol(mvar, "##onevec##"), var)))
            else
                push!(q.args, Expr(:(=), var, mvar))
            end
        end
    end
end


function gc_preserve(ls::LoopSet, q::Expr)
    length(ls.opdict) == 0 && return q
    q2 = Expr(:block)
    gcp = Expr(:gc_preserve, q)
    # gcp = Expr(:macrocall, Expr(:(.), :GC, QuoteNode(Symbol("@preserve"))), LineNumberNode(@__LINE__, Symbol(@__FILE__)))
    for array ∈ ls.includedactualarrays
        pb = gensym(array);
        push!(q2.args, Expr(:(=), pb, Expr(:call, lv(:preserve_buffer), array)))
        push!(gcp.args, pb)
    end
    q.head === :block && push!(q.args, nothing)
    # push!(gcp.args, q)
    push!(q2.args, gcp)
    q2
    # Expr(:block, gcp)
end

function typeof_outer_reduction_init(ls::LoopSet, op::Operation)
    opid = identifier(op)
    for (id, sym) ∈ ls.preamble_symsym
        opid == id && return Expr(:call, :typeof, sym)
    end
    for (id,(intval,intsz,signed)) ∈ ls.preamble_symint
        if opid == id
            if intsz == 1
                return :Bool
            elseif signed
                return Symbol(:Int,intsz)
            else
                return Symbol(:UInt,intsz)
            end
        end
    end
    for (id,floatval) ∈ ls.preamble_symfloat
        opid == id && return :Float64
    end
    for (id,typ) ∈ ls.preamble_zeros
        instruction(ops[id]) === LOOPCONSTANT || continue
        opid == id || continue
        if typ == IntOrFloat
            return :Float64
        elseif typ == HardInt
            return :Int
        else#if typ == HardFloat
            return :Float64
        end
    end
    throw("Could not find initializing constant.")
end
function typeof_outer_reduction(ls::LoopSet, op::Operation)
    for opp ∈ operations(ls)
        opp === op && continue
        name(op) === name(opp) && return typeof_outer_reduction_init(ls, opp)
    end
    throw("Could not find initialization op.")
end

function determine_eltype(ls::LoopSet)::Union{Symbol,Expr}
    if length(ls.includedactualarrays) == 0
        if length(ls.outer_reductions) == 0
            return Expr(:call, lv(:typeof), 0)
        elseif length(ls.outer_reductions) == 1
            op = ls.operations[only(ls.outer_reductions)]
            return typeof_outer_reduction(ls, op)
        else
            pt = Expr(:call, lv(:promote_type))
            for j ∈ ls.outer_reductions
                push!(pt.args, typeof_outer_reduction(ls, ls.operations[j]))
            end
            return pt
        end
    elseif length(ls.includedactualarrays) == 1
        return Expr(:call, lv(:eltype), first(ls.includedactualarrays))
    end
    promote_q = Expr(:call, lv(:promote_type))
    for array ∈ ls.includedactualarrays
        push!(promote_q.args, Expr(:call, lv(:eltype), array))
    end
    promote_q
end
@inline _eltype(x) = eltype(x)
@inline _eltype(::BitArray) = VectorizationBase.Bit
function determine_width(
    ls::LoopSet, vectorized::Union{Symbol,Nothing}
)
    vwidth_q = Expr(:call, lv(:pick_vector_width))
    if !(vectorized === nothing)
        vloop = getloop(ls, vectorized)
        if isstaticloop(vloop)
            push!(vwidth_q.args, Expr(:call, Expr(:curly, :Val, length(vloop))))
        end
    end
    # push!(vwidth_q.args, ls.T)
    if length(ls.includedactualarrays) < 2
        push!(vwidth_q.args, ELTYPESYMBOL)
    else
        for array ∈ ls.includedactualarrays
            push!(vwidth_q.args, Expr(:call, lv(:_eltype), array))
        end
    end
    vwidth_q
end
function init_remblock(unrolledloop::Loop, lssm::LoopStartStopManager, n::Int)#u₁loop::Symbol = unrolledloop.itersymbol)
  termind = lssm.terminators[n]
  if iszero(termind)
    condition = Expr(:call, lv(:cmpend), unrolledloop.itersymbol, unrolledloop.rangesym)
  else
    termar = lssm.incrementedptrs[n][termind]
    ptr = vptr(termar)
    condition = Expr(:call, GlobalRef(Base,:<), ptr, maxsym(ptr, 0))
  end
  Expr(:if, condition)
end

maskexpr(looplimit) = Expr(:(=), MASKSYMBOL, Expr(:call, lv(:mask), VECTORWIDTHSYMBOL, looplimit))
@inline idiv_fast(a::I, b::I) where {I <: Base.BitInteger} = Base.udiv_int(a, b)
@inline idiv_fast(a, b) = idiv_fast(Int(a), Int(b))
# @inline idiv_fast(a, b) = idiv_fast(@show(Int(a)), @show(Int(b)))
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
                subexpr(stop, gethint(start)-1)
            elseif isknown(stop)
                subexpr(gethint(stop) + 1, start)
            else
                subexpr(stop, subexpr(start,1))
            end
            maskexpr(lexpr)
        end
    else
        lenexpr = Expr(:call, lv(:idiv_fast), subexpr(stop, start))
        pushexpr!(lenexpr, incr)
        maskexpr(addexpr(lenexpr, 1))
    end
end
function define_eltype_vec_width!(q::Expr, ls::LoopSet, vectorized)
    push!(q.args, Expr(:(=), ELTYPESYMBOL, determine_eltype(ls)))
    push!(q.args, Expr(:(=), VECTORWIDTHSYMBOL, determine_width(ls, vectorized)))
    nothing
end
function setup_preamble!(ls::LoopSet, us::UnrollSpecification, Ureduct::Int)
    @unpack u₁loopnum, u₂loopnum, vloopnum, u₁, u₂ = us
    order = names(ls)
    u₁loopsym = order[u₁loopnum]
    u₂loopsym = order[u₂loopnum]
    vectorized = order[vloopnum]
    set_vector_width!(ls, vectorized)
    iszero(length(ls.includedactualarrays) + length(ls.outer_reductions)) || define_eltype_vec_width!(ls.preamble, ls, vectorized)
    lower_licm_constants!(ls)
    isone(num_loops(ls)) || pushpreamble!(ls, definemask(getloop(ls, vectorized)))#, u₁ > 1 && u₁loopnum == vloopnum))
    if (Ureduct == u₁) || (u₂ != -1) || (Ureduct == -1)
        initialize_outer_reductions!(ls, ifelse(Ureduct == -1, u₁, Ureduct), vectorized) # TODO: outer reducts?
    elseif length(ls.outer_reductions) > 0
        decl = Expr(:local)
        for or ∈ ls.outer_reductions
            push!(decl.args, Symbol(mangledvar(ls.operations[or]), '_', Ureduct))
        end
        pushpreamble!(ls, decl)
    end
    for op ∈ operations(ls)
        if (iszero(length(loopdependencies(op))) && iscompute(op))
            ua = UnrollArgs(getloop(ls, us.u₁loopnum), getloop(ls, us.u₂loopnum), getloop(ls, us.vloopnum), u₁, u₂, -1)
            lower_compute!(ls.preamble, op, ls, ua, false)
        end
    end
end
function lsexpr(ls::LoopSet, q)
    Expr(:block, ls.preamble, q)
end

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
            loopisstatic = isstaticloop(getloop(ls, u₁loopnum))
            loopisstatic &= ((vloopnum != u₁loopnum) | (!iszero(ls.vector_width)))
            # loopisstatic ? u₁ : min(u₁, 4) # much worse than the other two options, don't use this one
            if Sys.CPU_NAME === "znver1"
                loopisstatic ? u₁ : 1
            else
                loopisstatic ? u₁ : (u₁ ≥ 4 ? 2 : 1)
            end
        else
            -1
        end
    else
        u₁ui = u₂ui = -1
        u₁loopsym = getloop(ls,    u₁loopnum).itersymbol
        u₂loopsym = getloop(ls, us.u₂loopnum).itersymbol
        vloopsym  = getloop(ls,     vloopnum).itersymbol
        for or ∈ ls.outer_reductions
            op = ls.operations[or]
            u₁u, u₂u = isunrolled_sym(op, u₁loopsym, u₂loopsym, vloopsym, us)
            if u₁ui == -1
                u₁ui = Int(u₁u)
                u₂ui = Int(u₁u)
            elseif !((u₁ui == Int(u₁u)) & (u₂ui == Int(u₁u)))
                throw(ArgumentError("Doesn't currenly handle differently unrolled reductions yet, please file an issue with an example."))
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
    # @show u₁, u₂
    order = names(ls)
    init_loop_map!(ls)
    vectorized = order[vloopnum]
    Ureduct = calc_Ureduct!(ls, us)
    setup_preamble!(ls, us, Ureduct)
    initgesps = add_loop_start_stop_manager!(ls)
    q = Expr(:let, initgesps, lower_unrolled_dynamic(ls, us, num_loops(ls), false))
    q = gc_preserve( ls, Expr(:block, q) )
    reduce_expr!(q, ls, Ureduct)
    lsexpr(ls, q)
end

function lower(ls::LoopSet, order, u₁loop, u₂loop, vectorized, u₁, u₂, inline::Bool)
    cacheunrolled!(ls, u₁loop, u₂loop, vectorized)
    fillorder!(ls, order, u₁loop, u₂loop, u₂, vectorized)
    ls.unrollspecification = UnrollSpecification(ls, u₁loop, u₂loop, vectorized, u₁, u₂)
    q = lower_unrollspec(ls)
    inline && pushfirst!(q.args, Expr(:meta, :inline))
    q
end

function lower(ls::LoopSet, inline::Int = -1)
    fill_offset_memop_collection!(ls)
    order, u₁loop, u₂loop, vectorized, u₁, u₂, c, shouldinline = choose_order_cost(ls)
    lower(ls, order, u₁loop, u₂loop, vectorized, u₁, u₂, inlinedecision(inline, shouldinline))
end
function lower(ls::LoopSet, u₁::Int, u₂::Int, inline::Int)
    fill_offset_memop_collection!(ls)
    if u₂ > 1
        @assert num_loops(ls) > 1 "There is only $(num_loops(ls)) loop, but specified blocking parameter u₂ is $u₂."
        order, u₁loop, u₂loop, vectorized, _u₁, _u₂, c, shouldinline = choose_tile(ls)
        copyto!(ls.loop_order.bestorder, order)
    else
        u₂ = -1
        order, vectorized, c = choose_unroll_order(ls, Inf)
        u₁loop = first(order); u₂loop = Symbol("##undefined##"); shouldinline = true
        copyto!(ls.loop_order.bestorder, order)
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
    op::Operation, u₁loop::Symbol, u₂loop::Symbol, vloop::Symbol,
    (u₁ild,u₂ild)::Tuple{Bool,Bool} = (isu₁unrolled(op), isu₂unrolled(op))
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
    # @show u₁reduced, u₂reduced
    if u₂reduced
        if u₁reduced# if both are reduced, we unroll u₁
            if vloop === u₁loop
                false, true
            else
                true, false
            end
        else
            true,false
        end
    elseif u₁reduced
        false, true
        # true, false
    else
        true, true
    end
end
function isunrolled_sym(op::Operation, u₁loop::Symbol, u₂loop::Symbol, vloop::Symbol, ls::LoopSet)
    us = ls.unrollspecification
    isunrolled_sym(op, u₁loop, u₂loop, vloop, us)
end
function isunrolled_sym(op::Operation, u₁loop::Symbol, u₂loop::Symbol, vloop::Symbol, us::UnrollSpecification)
    @unpack u₁, u₂ = us
    u₁u = (u₁ > 1) & isu₁unrolled(op)
    u₂u = (u₂ > 1) & isu₂unrolled(op)
    ((u₂ > 1) | accesses_memory(op)) ? isunrolled_sym(op, u₁loop, u₂loop, vloop, (u₁u,u₂u)) : (isunrolled_sym(op, u₁loop, u₁u), false)
end

isunrolled_sym(op::Operation, u₁loop::Symbol, ls::LoopSet) = isunrolled_sym(op, u₁loop, ls.unrollspecification)
function isunrolled_sym(op::Operation, u₁loop::Symbol, us::UnrollSpecification)
    u₁u = (us.u₁ > 1) & isu₁unrolled(op)
    isunrolled_sym(op, u₁loop, u₁u)
end
function isunrolled_sym(op::Operation, u₁loop::Symbol, u₁u::Bool = isu₁unrolled(op))
    u₁u || (isconstant(op) & (u₁loop ∈ reducedchildren(op)))
end

function isunrolled_sym(op::Operation, u₁loop::Symbol, u₂loop::Symbol, vloop::Symbol, u₂max::Int)
    ((u₂max > 1) | accesses_memory(op)) ? isunrolled_sym(op, u₁loop, u₂loop, vloop) : (isunrolled_sym(op, u₁loop), false)
end



function variable_name(op::Operation, suffix::Int)
    mvar = mangledvar(op)
    suffix == -1 ? mvar : Symbol(mvar, suffix, :_)
end

function variable_name_and_unrolled(op::Operation, u₁loop::Symbol, u₂loop::Symbol, vloop::Symbol, u₂iter::Int, ls::LoopSet)
    # u₁op, u₂op = isunrolled_sym(op, u₁loop, u₂loop, vloop, Core.ifelse(u₂iter == -1, 1, u₂max))
    u₁op, u₂op = isunrolled_sym(op, u₁loop, u₂loop, vloop, ls)#, u₂max)
    mvar = u₂op ? variable_name(op, u₂iter) : mangledvar(op)
    mvar, u₁op, u₂op
end
