
# struct TileDescription
    # vectorized::Symbol
    # u₁loop::Symbol
    # u₂loop::Symbol
    # U::Int
    # T::Int
# end


function lower!(
    q::Expr, op::Operation, ls::LoopSet, unrollargs::UnrollArgs, mask::Union{Nothing,Symbol,Unsigned}, ::Nothing
)
    if isconstant(op)
        zerotyp = zerotype(ls, op)
        if zerotyp == INVALID
            lower_constant!(q, op, ls, unrollargs)
        else
            lower_zero!(q, op, ls, unrollargs, zerotyp)
        end
    elseif isload(op)
        lower_load!(q, op, ls, unrollargs, mask)
    elseif iscompute(op)
        lower_compute!(q, op, unrollargs, mask)
    elseif isstore(op)
        lower_store!(q, ls, op, unrollargs, mask)
    # elseif isloopvalue(op)
    end
end
function lower!(
    q::Expr, op::Operation, ls::LoopSet, unrollargs::UnrollArgs, mask::Union{Nothing,Symbol,Unsigned}, filterstore::Bool
)
    if filterstore
        if isstore(op)
            lower_store!(q, ls, op, unrollargs, mask)
        end
    else
        if isconstant(op)
            zerotyp = zerotype(ls, op)
            if zerotyp == INVALID
                lower_constant!(q, op, ls, unrollargs)
            else
                lower_zero!(q, op, ls, unrollargs, zerotyp)
            end
        elseif isload(op)
            lower_load!(q, op, ls, unrollargs, mask)
        elseif iscompute(op)
            lower_compute!(q, op, unrollargs, mask)
        end
    end
end
function lower!(
    q::Expr, ops::AbstractVector{Operation}, ls::LoopSet, unrollsyms::UnrollSymbols, u₁::Int, u₂::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned}, filterstore = nothing
)
    foreach(op -> lower!(q, op, ls, UnrollArgs(u₁, unrollsyms, u₂, suffix), mask, filterstore), ops)
end

function lower_block(ls::LoopSet, us::UnrollSpecification, n::Int, inclmask::Bool, UF)
    if inclmask
        lower_block(ls, us, n, Symbol("##mask##"), UF)
    else
        lower_block(ls, us, n, nothing, UF)
    end
end


function lower_block(
    ls::LoopSet, us::UnrollSpecification, n::Int, mask::Union{Nothing,Symbol}, UF::Int
)
    @unpack u₁loopnum, u₂loopnum, vectorizedloopnum, u₁, u₂ = us
    ops = oporder(ls)
    order = names(ls)
    u₁loopsym = order[u₁loopnum]
    u₂loopsym = order[u₂loopnum]
    vectorized = order[vectorizedloopnum]
    unrollsyms = UnrollSymbols(u₁loopsym, u₂loopsym, vectorized)
    u₁ = n == u₁loopnum ? UF : u₁
    dontmaskfirsttiles = !isnothing(mask) && vectorizedloopnum == u₂loopnum
    blockq = Expr(:block)
    delay_u₁ = true
    # delay_u₁ = false
    dontdelayat = nothing#4
    for prepost ∈ 1:2
        # !u₁ && !u₂
        lower!(blockq, ops[1,1,prepost,n], ls, unrollsyms, u₁, u₂, nothing, mask)
        if !delay_u₁ || u₁ == dontdelayat
            lower!(blockq, ops[2,1,prepost,n], ls, unrollsyms, u₁, u₂, nothing, mask)
        end
        opsv1 = ops[1,2,prepost,n]
        opsv2 = ops[2,2,prepost,n]
        if length(opsv1) + length(opsv2) > 0
            # if u₁ == 3
                # lower!(blockq, ops[2,1,prepost,n], vectorized, ls, u₁loop, u₂loop, u₁, u₂, nothing, mask)
            # end
            for store ∈ (false,true)
                # let store = nothing
                nstores = 0
                iszero(length(opsv1)) || (nstores += sum(isstore, opsv1))
                iszero(length(opsv2)) || (nstores += sum(isstore, opsv2))
                if delay_u₁ && !store && length(opsv1) + length(opsv2) == nstores
                    u₁ != dontdelayat && lower!(blockq, ops[2,1,prepost,n], ls, unrollsyms, u₁, u₂, nothing, mask) # for u ∈ 0:u₁-1     
                    continue
                end
                for t ∈ 0:u₂-1
                    # if t == 0
                    #     push!(blockq.args, Expr(:(=), u₂loop, tiledsym(u₂loop)))
                    # elseif u₂loopnum == vectorizedloopnum
                    #     push!(blockq.args, Expr(:(=), u₂loop, Expr(:call, lv(:valadd), VECTORWIDTHSYMBOL, u₂loop)))
                    # else
                    #     push!(blockq.args, Expr(:+=, u₂loop, 1))
                    # end
                    if dontmaskfirsttiles && t < u₂ - 1 # !u₁ &&  u₂
                        lower!(blockq, opsv1, ls, unrollsyms, u₁, u₂, t, nothing, store)
                    else # !u₁ &&  u₂
                        lower!(blockq, opsv1, ls, unrollsyms, u₁, u₂, t, mask, store)
                    end
                    if delay_u₁ && iszero(t) && !store && u₁ != dontdelayat #  u₁ && !u₂
                        # for u ∈ 0:u₁-1     
                        lower!(blockq, ops[2,1,prepost,n], ls, unrollsyms, u₁, u₂, nothing, mask)
                        # end
                    end
                    if dontmaskfirsttiles && t < u₂ - 1 #  u₁ && u₂
                        # for u ∈ 0:u₁-1
                        lower!(blockq, opsv2, ls, unrollsyms, u₁, u₂, t, nothing, store)
                        # end
                    else #  u₁ && u₂
                        # for u ∈ 0:u₁-1 
                        lower!(blockq, opsv2, ls, unrollsyms, u₁, u₂, t, mask, store)
                        # end
                    end
                end
                nstores == 0 && break
            end
        elseif delay_u₁ && u₁ != dontdelayat
            # for u ∈ 0:u₁-1     #  u₁ && !u₂
            lower!(blockq, ops[2,1,prepost,n], ls, unrollsyms, u₁, u₂, nothing, mask)
            # end
        end
        if n > 1 && prepost == 1
            push!(blockq.args, lower_unrolled_dynamic(ls, us, n-1, !isnothing(mask)))
        end
    end
    # loopsym = mangletiledsym(order[n], us, n)
    loopsym = order[n]
    # push!(blockq.args, incrementloopcounter(us, n, loopsym, UF))
    push!(blockq.args, incrementloopcounter(ls, us, n, UF))
    blockq
end

# function lower_llvm_unroll(ls::LoopSet, us::UnrollSpecification, n::Int, loop::Loop)
#     loopsym = names(ls)[n]
#     loop = getloop(ls, loopsym)
#     # loopsym = mangletiledsym(loopsym, us, n)
#     nisvectorized = false#isvectorized(us, n)
#     sl = startloop(loop, nisvectorized, loopsym)
#     # tc = terminatecondition(loop, us, n, loopsym, inclmask, 1)
#     looprange = if loop.startexact
#         if loop.stopexact
#             Expr(:(=), loopsym, Expr(:call, :(:), loop.starthint-1, loop.stophint-1))
#         else
#             # expectation = expect(Expr(:call, :(>), loop.stopsym, loop.starthint-5))
#             Expr(:(=), loopsym, Expr(:call, :(:), loop.starthint-1, Expr(:call, lv(:staticm1), loop.stopsym)))
#         end
#     elseif loop.stopexact
#         # expectation = expect(Expr(:call, :(>), loop.stophint+5, loop.startsym))
#         Expr(:(=), loopsym, Expr(:call, :(:), Expr(:call, lv(:staticm1), loop.startsym), loop.stophint - 1))
#     else
#         # expectation = expect(Expr(:call, :(>), loop.stopsym, Expr(:call, lv(:vsub), loop.startsym, 5)))
#         Expr(:(=), loopsym, Expr(:call, :(:), Expr(:call, lv(:staticm1), loop.startsym), Expr(:call, lv(:staticm1), loop.stopsym)))
#     end
#     body = lower_block(ls, us, n, false, 1)
#     push!(body.args, Expr(:loopinfo, (Symbol("llvm.loop.unroll.count"), 4)))
#     q = Expr(:for, looprange, body)
#     # if nisvectorized
#     #     tc = terminatecondition(loop, us, n, loopsym, true, 1)
#     #     body = lower_block(ls, us, n, true, 1)
#     #     push!(q.args, Expr(:if, tc, body))
#     # end
#     if loop.startexact && loop.stopexact
#         q
#     else
#         # Expr(:block, assumption, expectation, q)
#         q
#         Expr(:block, loopiteratesatleastonce(loop), q)
#     end
# end
# function lower_unroll_for_throughput(ls::LoopSet, us::UnrollSpecification, n::Int, loop::Loop, loopsym::Symbol)
#     sl = startloop(loop, false, loopsym)
#     UF = 4
#     tcc = terminatecondition(loop, us, n, loopsym, false, 1)
#     tcu = terminatecondition(loop, us, n, loopsym, false, UF)
#     body = lower_block(ls, us, n, false, 1)
#     # loopisstatic = isstaticloop(loop)
#     unrolledlabel = gensym(:unrolled)
#     cleanuplabel = gensym(:cleanup)
#     gotounrolled = Expr(:macrocall, Symbol("@goto"), LineNumberNode(@__LINE__, Symbol(@__FILE__)), unrolledlabel)
#     gotocleanup = Expr(:macrocall, Symbol("@goto"), LineNumberNode(@__LINE__, Symbol(@__FILE__)), cleanuplabel)
#     branch = Expr(:if, tcu, gotounrolled, gotocleanup)
#     unrolled = Expr(:block, Expr(:macrocall, Symbol("@label"), LineNumberNode(@__LINE__, Symbol(@__FILE__)), unrolledlabel))
#     foreach(_ -> push!(unrolled.args, body), 1:UF)
#     push!(unrolled.args, Expr(:if, tcu, gotounrolled, Expr(:if, tcc, gotocleanup)))
#     cleanup = Expr(:block, Expr(:macrocall, Symbol("@label"), LineNumberNode(@__LINE__, Symbol(@__FILE__)), cleanuplabel), body, Expr(:if, tcc, gotocleanup))
#     Expr(:let, sl, Expr(:block, branch, unrolled, cleanup))
# end

function assume(ex)
    Expr(:call, Expr(:(.), Expr(:(.), :LoopVectorization, QuoteNode(:SIMDPirates)), QuoteNode(:assume)), ex)
end
function expect(ex)
    use_expect() || return ex
    Expr(:call, Expr(:(.), Expr(:(.), :LoopVectorization, QuoteNode(:SIMDPirates)), QuoteNode(:expect)), ex)
    # ex
end
function loopiteratesatleastonce(loop::Loop, as::Bool = true)
    comp = if loop.startexact # requires !loop.stopexact
        Expr(:call, :>, loop.stopsym, loop.starthint - 1)
    elseif loop.stopexact # requires !loop.startexact
        Expr(:call, :>, loop.stopexact + 1, loop.startsym)
    else
        Expr(:call, :>, loop.stopsym, Expr(:call, lv(:vsub), loop.startsym, 1))
    end
    # as ? assume(comp) : expect(comp)
    assume(comp)
end
function lower_no_unroll(ls::LoopSet, us::UnrollSpecification, n::Int, inclmask::Bool)
    usorig = ls.unrollspecification[]
    nisvectorized = isvectorized(us, n)
    loopsym = names(ls)[n]
    loop = getloop(ls, loopsym)
    # if !nisvectorized && !inclmask && isone(n) && !ls.loadelimination[] && (us.u₁ > 1) && (usorig.u₁ == us.u₁) && (usorig.u₂ == us.u₂) && length(loop) > 7
    #     # return lower_unroll_for_throughput(ls, us, n, loop, loopsym)
    #     return lower_llvm_unroll(ls, us, n, loop)
    # end
    # sl = startloop(loop, nisvectorized, loopsym)
    sl = startloop(ls, us, n)
    tc = terminatecondition(ls, us, n, inclmask, 1)
    body = lower_block(ls, us, n, inclmask, 1)
    isstatic = isstaticloop(loop)
    q = if nisvectorized
            # Expr(:block, loopiteratesatleastonce(loop, true), Expr(:while, expect(tc), body))
        Expr(:block, Expr(:while, isstatic ? tc : expect(tc), body))
    elseif isstatic && length(loop) ≤ 8
        Expr(:block, (body for _ ∈ 1:length(loop))...)
    else
        termcond = gensym(:maybeterm)
        push!(body.args, Expr(:(=), termcond, isstatic ? tc : expect(tc)))
        Expr(:block, Expr(:(=), termcond, true), Expr(:while, termcond, body))
        # Expr(:block, Expr(:while, expect(tc), body))
        # Expr(:block, assume(tc), Expr(:while, tc, body))
        # push!(body.args, Expr(:&&, expect(Expr(:call, :!, tc)), Expr(:break)))
        # Expr(:block, assume(tc), Expr(:while, true, body))
        # push!(body.args, Expr(:||, expect(tc), Expr(:break)))
        # Expr(:block, Expr(:while, true, body))
    end

    if nisvectorized
        # tc = terminatecondition(loop, us, n, loopsym, true, 1)
        tc = terminatecondition(ls, us, n, true, 1)
        body = lower_block(ls, us, n, true, 1)
        isone(num_loops(ls)) && pushfirst!(body.args, definemask(loop))
        push!(q.args, Expr(:if, tc, body))
    end
    Expr(:block, Expr(:let, sl, q))
end
function lower_unrolled_dynamic(ls::LoopSet, us::UnrollSpecification, n::Int, inclmask::Bool)
    UF = unrollfactor(us, n)
    UF == 1 && return lower_no_unroll(ls, us, n, inclmask)
    @unpack u₁loopnum, vectorizedloopnum, u₁, u₂ = us
    order = names(ls)
    loopsym = order[n]
    loop = getloop(ls, loopsym)
    vectorized = order[vectorizedloopnum]
    nisunrolled = isunrolled1(us, n)
    nisvectorized = isvectorized(us, n)
    loopisstatic = isstaticloop(loop) & (!nisvectorized)


    remmask = inclmask | nisvectorized
    Ureduct = (n == num_loops(ls) && (u₂ == -1)) ? calc_Ureduct(ls, us) : -1
    # sl = startloop(loop, nisvectorized, loopsym)
    sl = startloop(ls, us, n)

    UFt = if loopisstatic
        length(loop) % UF
    else
        1
    end
    remfirst = loopisstatic & (UFt > 0) & !(unsigned(Ureduct) < unsigned(UF))
    tc = terminatecondition(ls, us, n, inclmask, remfirst ? 1 : UF)
    usorig = ls.unrollspecification[]

    # tc = (usorig.u₁ == us.u₁) && (usorig.u₂ == us.u₂) && !loopisstatic && !inclmask && !ls.loadelimination[] ? expect(tc) : tc

    body = lower_block(ls, us, n, inclmask, UF)
    q = Expr(:while, tc, body)
    remblock = init_remblock(loop, ls.lssm[], n)#loopsym)
    q = if unsigned(Ureduct) < unsigned(UF) # unsigned(-1) == typemax(UInt); is logic relying on twos-complement bad?
        UF_cleanup = UF - Ureduct
        us_cleanup = nisunrolled ? UnrollSpecification(us, UF_cleanup, u₂) : UnrollSpecification(us, u₁, UF_cleanup)
        Expr(
            :block,
            add_upper_outer_reductions(ls, q, Ureduct, UF, loop, vectorized),
            Expr(
                # :if, terminatecondition(loop, us, n, loopsym, inclmask, UF_cleanup),
                :if, terminatecondition(ls, us, n, inclmask, UF_cleanup),
                lower_block(ls, us_cleanup, n, inclmask, UF_cleanup)
            ),
            remblock
        )
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
    else
        # if (usorig.u₁ == us.u₁) && (usorig.u₂ == us.u₂) && !isstaticloop(loop) && !inclmask# && !ls.loadelimination[]
        #     # Expr(:block, sl, assumeloopiteratesatleastonce(loop), Expr(:while, tc, body))
        #     Expr(:block, sl, expect(tc), q, remblock)
        # else
        #     Expr(:block, sl, q, remblock)
        # end
        Expr( :block, q, remblock )
    end
    if !iszero(UFt)
        while true
            ust = nisunrolled ? UnrollSpecification(us, UFt, u₂) : UnrollSpecification(us, u₁, UFt)
            newblock = lower_block(ls, ust, n, remmask, UFt)
            if (UFt ≥ UF - 1 + nisvectorized) || UFt == Ureduct || loopisstatic
                push!(remblock.args, newblock)
                break
            end
            comparison = unrollremcomparison(ls, loop, UFt, n, nisvectorized, remfirst)
            if isone(num_loops(ls)) && isone(UFt)
                remblocknew = Expr(:if, comparison, newblock)
                push!(remblock.args, Expr(:block, definemask(loop), remblocknew))
                remblock = remblocknew
            else
                remblocknew = Expr(:elseif, comparison, newblock)
                push!(remblock.args, remblocknew)
                remblock = remblocknew
            end
            UFt += 1
        end
    end
    Expr(:block, Expr(:let, sl, q))
end
function unrollremcomparison(ls::LoopSet, loop::Loop, UFt::Int, n::Int, nisvectorized::Bool, remfirst::Bool)
    termind = ls.lssm[].terminators[n]
    if iszero(termind)
        loopvarremcomparison(loop, UFt, nisvectorized, remfirst)
    else
        pointerremcomparison(ls, termind, UFt, n, nisvectorized, remfirst, loop)
    end
end
function loopvarremcomparison(loop::Loop, UFt::Int, nisvectorized::Bool, remfirst::Bool)
    loopsym = loop.itersymbol
    if nisvectorized
        itercount = if loop.stopexact
            Expr(:call, lv(:vsub), loop.stophint - 1, Expr(:call, lv(:valmul), VECTORWIDTHSYMBOL, UFt))
        else
            Expr(:call, lv(:vsub), loop.stopsym, Expr(:call, lv(:valmuladd), VECTORWIDTHSYMBOL, UFt, 1))
        end
        Expr(:call, :>, loopsym, itercount)
    elseif remfirst
        Expr(:call, :<, loopsym, loop.starthint + UFt - 1)
    elseif loop.stopexact
        Expr(:call, :>, loopsym, loop.stophint - UFt - 1)
    else
        Expr(:call, :>, loopsym, Expr(:call, lv(:vsub), loop.stopsym, UFt + 1))
    end
end
function pointerremcomparison(ls::LoopSet, termind::Int, UFt::Int, n::Int, nisvectorized::Bool, remfirst::Bool, loop::Loop)
    lssm = ls.lssm[]
    termar = lssm.incrementedptrs[n][termind]
    ptrdef = lssm.incrementedptrs[n][termind]
    ptr = vptr(termar)
    ptrex = callpointerforcomparison(ptr)
    if remfirst
        Expr(:call, :<, ptrex, pointermax(ls, ptrdef, n, 1 - UFt, nisvectorized, loop.startexact))
    else
        # Expr(:call, :≥, ptrex, pointermax(ls, ptrdef, n, UFt, nisvectorized, loop))
        Expr(:call, :≥, ptrex, maxsym(ptr, UFt))
    end
end

function initialize_outer_reductions!(
    q::Expr, op::Operation, Umin::Int, Umax::Int, vectorized::Symbol, suffix::Union{Symbol,Nothing} = nothing
)
    reduct_zero = reduction_zero(op.instruction)
    isvectorized = vectorized ∈ reduceddependencies(op)
    # typeTr = Symbol("##TYPEOF##", name(op))
    typeTr = Expr(:call, :typeof, mangledvar(op))
    z = if isvectorized
        if reduct_zero === :zero
            Expr(:call, lv(:vzero), VECTORWIDTHSYMBOL, typeTr)
        else
            Expr(:call, lv(:vbroadcast), VECTORWIDTHSYMBOL, Expr(:call, reduct_zero, typeTr))
        end
    else
        Expr(:call, reduct_zero, typeTr)
    end
    mvar = variable_name(op, suffix)
    for u ∈ Umin:Umax-1
        push!(q.args, Expr(:(=), Symbol(mvar, u), z))
    end
    nothing
end
function initialize_outer_reductions!(q::Expr, ls::LoopSet, Umin::Int, Umax::Int, vectorized::Symbol, suffix::Union{Symbol,Nothing} = nothing)
    foreach(or -> initialize_outer_reductions!(q, ls.operations[or], Umin, Umax, vectorized, suffix), ls.outer_reductions)
end
function initialize_outer_reductions!(ls::LoopSet, Umin::Int, Umax::Int, vectorized::Symbol, suffix::Union{Symbol,Nothing} = nothing)
    initialize_outer_reductions!(ls.preamble, ls, Umin, Umax, vectorized, suffix)
end
function add_upper_outer_reductions(ls::LoopSet, loopq::Expr, Ulow::Int, Uhigh::Int, unrolledloop::Loop, vectorized::Symbol)
    ifq = Expr(:block)
    initialize_outer_reductions!(ifq, ls, Ulow, Uhigh, vectorized)
    push!(ifq.args, loopq)
    reduce_range!(ifq, ls, Ulow, Uhigh)
    comparison = if isstaticloop(unrolledloop)
        Expr(:call, lv(:scalar_less), length(unrolledloop), Expr(:call, lv(:valmul), VECTORWIDTHSYMBOL, Uhigh))
    elseif unrolledloop.starthint == 1
        Expr(:call, lv(:scalar_less), unrolledloop.stopsym, Expr(:call, lv(:valmul), VECTORWIDTHSYMBOL, Uhigh))
    elseif unrolledloop.startexact
        Expr(:call, lv(:scalar_less), Expr(:call, lv(:vsub), unrolledloop.stopsym, unrolledloop.starthint-1), Expr(:call, lv(:valmul), VECTORWIDTHSYMBOL, Uhigh))
    elseif unrolledloop.stopexact
        Expr(:call, lv(:scalar_less), Expr(:call, lv(:vsub), unrolledloop.stophint+1, unrolledloop.sartsym), Expr(:call, lv(:valmul), VECTORWIDTHSYMBOL, Uhigh))
    else# both are given by symbols
        Expr(:call, lv(:scalar_less), Expr(:call, lv(:vsub), unrolledloop.stopsym, Expr(:call,lv(:vsub),unrolledloop.startsym)), Expr(:call, lv(:valmul), VECTORWIDTHSYMBOL, Uhigh))
    end
    ncomparison = Expr(:call, :!, comparison)
    Expr(:if, ncomparison, ifq)
end
function reduce_expr!(q::Expr, ls::LoopSet, U::Int)
    us = ls.unrollspecification[]
    # u₁loop, u₂loop = getunrolled(ls)
    for or ∈ ls.outer_reductions
        op = ls.operations[or]
        var = name(op)
        mvar = mangledvar(op)
        instr = instruction(op)
        reduce_expr!(q, mvar, instr, U)
        if !iszero(length(ls.opdict))
            if (isu₁unrolled(op) | isu₂unrolled(op))
                push!(q.args, Expr(:(=), var, Expr(:call, lv(reduction_scalar_combine(instr)), var, Symbol(mvar, 0))))
            else
                push!(q.args, Expr(:(=), var, mvar))
            end
        end
    end
end
function gc_preserve(ls::LoopSet, q::Expr)
    length(ls.opdict) == 0 && return q
    gcp = Expr(:macrocall, Expr(:(.), :GC, QuoteNode(Symbol("@preserve"))), LineNumberNode(@__LINE__, Symbol(@__FILE__)))
    for array ∈ ls.includedactualarrays
        push!(gcp.args, array)
    end
    q.head === :block && push!(q.args, nothing)
    push!(gcp.args, q)
    Expr(:block, gcp)
end
function determine_eltype(ls::LoopSet)
    if length(ls.includedactualarrays) == 0
        return Expr(:call, :typeof, 0)
    elseif length(ls.includedactualarrays) == 1
        return Expr(:call, :eltype, first(ls.includedactualarrays))
    end
    promote_q = Expr(:call, :promote_type)
    for array ∈ ls.includedactualarrays
        push!(promote_q.args, Expr(:call, :eltype, array))
    end
    promote_q
end
function determine_width(
    ls::LoopSet, vectorized::Symbol
)
    vloop = getloop(ls, vectorized)
    vwidth_q = Expr(:call, lv(:pick_vector_width_val))
    if isstaticloop(vloop)
        push!(vwidth_q.args, Expr(:call, Expr(:curly, :Val, length(vloop))))
    end
    # push!(vwidth_q.args, ls.T)
    if length(ls.includedactualarrays) < 2
        push!(vwidth_q.args, ELTYPESYMBOL)
    else
        for array ∈ ls.includedactualarrays
            push!(vwidth_q.args, Expr(:call, :eltype, array))
        end
    end
    vwidth_q
end
function init_remblock(unrolledloop::Loop, lssm::LoopStartStopManager, n::Int)#u₁loop::Symbol = unrolledloop.itersymbol)
    termind = lssm.terminators[n]
    condition = if iszero(termind)
        loopsym = unrolledloop.itersymbol
        if unrolledloop.stopexact
            Expr(:call, :<, loopsym, unrolledloop.stophint)
        else
            Expr(:call, :<, loopsym, unrolledloop.stopsym)
        end
    else
        termar = lssm.incrementedptrs[n][termind]
        ptr = vptr(termar)
        Expr(:call, :<, callpointerforcomparison(ptr), maxsym(ptr, 0))
    end
    Expr(:if, condition)
end

function maskexpr(looplimit)
    Expr(:(=), Symbol("##mask##"), Expr(:call, lv(:mask), VECTORWIDTHSYMBOL, looplimit))
    # rem = Expr(:call, lv(:valrem), W, looplimit)
    # Expr(:(=), Symbol("##mask##"), Expr(:call, lv(:masktable), W, rem))
end
function definemask(loop::Loop)
    if isstaticloop(loop)
        maskexpr(length(loop))
    elseif loop.startexact && loop.starthint == 1
        maskexpr(loop.stopsym)
    else
        lexpr = if loop.startexact
            Expr(:call, lv(:vsub), loop.stopsym, loop.starthint - 1)
        elseif loop.stopexact
            Expr(:call, lv(:vsub), loop.stophint + 1, loop.startsym)
        else
            Expr(:call, lv(:vsub), Expr(:call, lv(:vadd), loop.stopsym, 1), loop.startsym)
        end
        maskexpr(lexpr)
    end
end

function setup_preamble!(ls::LoopSet, us::UnrollSpecification)
    @unpack u₁loopnum, u₂loopnum, vectorizedloopnum, u₁, u₂ = us
    order = names(ls)
    u₁loopsym = order[u₁loopnum]
    u₂loopsym = order[u₂loopnum]
    vectorized = order[vectorizedloopnum]
    if length(ls.includedactualarrays) > 0
        push!(ls.preamble.args, Expr(:(=), ELTYPESYMBOL, determine_eltype(ls)))
        push!(ls.preamble.args, Expr(:(=), VECTORWIDTHSYMBOL, determine_width(ls, vectorized)))
    end
    lower_licm_constants!(ls)
    isone(num_loops(ls)) || pushpreamble!(ls, definemask(getloop(ls, vectorized)))#, u₁ > 1 && u₁loopnum == vectorizedloopnum))
    for op ∈ operations(ls)
        (iszero(length(loopdependencies(op))) && iscompute(op)) && lower_compute!(ls.preamble, op, UnrollArgs(u₁, u₁loopsym, u₂loopsym, vectorized, u₂, nothing), nothing)
    end
    # define_remaining_ops!( ls, vectorized, W, u₁loop, u₂loop, u₁ )
end
function lsexpr(ls::LoopSet, q)
    Expr(:block, ls.preamble, q)
end

function calc_Ureduct(ls::LoopSet, us::UnrollSpecification)
    @unpack u₁loopnum, u₁, u₂ = us
    if iszero(length(ls.outer_reductions))
        -1
    elseif u₂ == -1
        min(u₁, 4)
    else
        4#u₁
    # elseif num_loops(ls) == u₁loopnum
    #     min(u₁, 4)
    # else
    #     # u₂ == -1 ? u₁ : 4
    #     u₁
    end
end
function lower_unrollspec(ls::LoopSet)
    us = ls.unrollspecification[]
    @unpack vectorizedloopnum, u₁, u₂ = us
    order = names(ls)
    vectorized = order[vectorizedloopnum]
    setup_preamble!(ls, us)
    Ureduct = calc_Ureduct(ls, us)
    initialize_outer_reductions!(ls, 0, Ureduct, vectorized)
    initgesps = add_loop_start_stop_manager!(ls)
    q = Expr(:let, initgesps, lower_unrolled_dynamic(ls, us, num_loops(ls), false))
    q = gc_preserve( ls, Expr(:block, q) )
    reduce_expr!(q, ls, Ureduct)
    lsexpr(ls, q)
end

function lower(ls::LoopSet, order, u₁loop, u₂loop, vectorized, u₁, u₂, inline::Bool)
    cacheunrolled!(ls, u₁loop, u₂loop, vectorized)
    fillorder!(ls, order, u₁loop, u₂loop, u₂ != -1, vectorized)
    ls.unrollspecification[] = UnrollSpecification(ls, u₁loop, u₂loop, vectorized, u₁, u₂)
    q = lower_unrollspec(ls)
    inline && pushfirst!(q.args, Expr(:meta, :inline))
    q
end

function lower(ls::LoopSet, inline::Int = -1)
    order, u₁loop, u₂loop, vectorized, u₁, u₂, c, shouldinline = choose_order_cost(ls)
    lower(ls, order, u₁loop, u₂loop, vectorized, u₁, u₂, inlinedecision(inline, shouldinline))
end
function lower(ls::LoopSet, u₁::Int, u₂::Int, inline::Int)
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

Base.convert(::Type{Expr}, ls::LoopSet) = lower(ls)
Base.show(io::IO, ls::LoopSet) = println(io, lower(ls))



"""
This function is normally called
isunrolled_sym(op, u₁loop)
isunrolled_sym(op, u₁loop, u₂loop)

It returns `true`/`false` for each loop, indicating whether they're unrolled.

If there is a third argument, it will avoid unrolling that symbol along reductions if said symbol is part of the reduction chain.

"""
function isunrolled_sym(op::Operation, u₁loop::Symbol, u₂loop::Symbol)
    u₁ild = u₁loop ∈ loopdependencies(op)
    u₂ild = u₂loop ∈ loopdependencies(op)
    (accesses_memory(op) | isloopvalue(op)) && return (u₁ild, u₂ild)
    if isconstant(op)
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
    # We want to only unroll one of them.
    # We prefer not to unroll a reduced loop
    if u₂reduced # if both are reduced, we unroll u₁
        true, false
    elseif u₁reduced
        false, true
    else
        true, true
    end
end

function isunrolled_sym(op::Operation, u₁loop::Symbol)
    u₁loop ∈ loopdependencies(op) || (isconstant(op) && (u₁loop ∈ reducedchildren(op)))
end

isunrolled_sym(op::Operation, u₁loop::Symbol, u₂loop::Symbol, ::Nothing) = (isunrolled_sym(op, u₁loop), false)
isunrolled_sym(op::Operation, u₁loop::Symbol, u₂loop::Symbol, ::Int) = isunrolled_sym(op, u₁loop, u₂loop)

variable_name(op::Operation, ::Nothing) = mangledvar(op)
variable_name(op::Operation, suffix) = Symbol(mangledvar(op), suffix, :_)

function variable_name_and_unrolled(op::Operation, u₁loop::Symbol, u₂loop::Symbol, ::Nothing)
    mangledvar(op), isunrolled_sym(op, u₁loop), false
end
function variable_name_and_unrolled(op::Operation, u₁loop::Symbol, u₂loop::Symbol, u₂iter::Int)
    u₁op, u₂op = isunrolled_sym(op, u₁loop, u₂loop)
    mvar = u₂op ? variable_name(op, u₂iter) : mangledvar(op)
    mvar, u₁op, u₂op
end

