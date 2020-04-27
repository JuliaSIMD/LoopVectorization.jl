
# struct TileDescription
    # vectorized::Symbol
    # u₁loop::Symbol
    # u₂loop::Symbol
    # U::Int
    # T::Int
# end


function lower!(
    q::Expr, op::Operation, vectorized::Symbol, ls::LoopSet, u₁loop::Symbol, u₂loop::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned}, ::Nothing
)
    if isconstant(op)
        zerotyp = zerotype(ls, op)
        if zerotyp == INVALID
            lower_constant!(q, op, vectorized, ls, u₁loop, u₂loop, U, suffix)
        else
            lower_zero!(q, op, vectorized, ls, u₁loop, u₂loop, U, suffix, zerotyp)
        end
    elseif isload(op)
        lower_load!(q, op, vectorized, ls, u₁loop, u₂loop, U, suffix, mask)
    elseif iscompute(op)
        lower_compute!(q, op, vectorized, u₁loop, u₂loop, U, suffix, mask)
    elseif isstore(op)
        lower_store!(q, op, vectorized, u₁loop, u₂loop, U, suffix, mask)
    # elseif isloopvalue(op)
    end
end
function lower!(
    q::Expr, op::Operation, vectorized::Symbol, ls::LoopSet, u₁loop::Symbol, u₂loop::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned}, filterstore::Bool
)
    if filterstore
        if isstore(op)
            lower_store!(q, op, vectorized, u₁loop, u₂loop, U, suffix, mask)
        end
    else
        if isconstant(op)
            zerotyp = zerotype(ls, op)
            if zerotyp == INVALID
                lower_constant!(q, op, vectorized, ls, u₁loop, u₂loop, U, suffix)
            else
                lower_zero!(q, op, vectorized, ls, u₁loop, u₂loop, U, suffix, zerotyp)
            end
        elseif isload(op)
            lower_load!(q, op, vectorized, ls, u₁loop, u₂loop, U, suffix, mask)
        elseif iscompute(op)
            lower_compute!(q, op, vectorized, u₁loop, u₂loop, U, suffix, mask)
        end
    end
end
function lower!(
    q::Expr, ops::AbstractVector{Operation}, vectorized::Symbol, ls::LoopSet, u₁loop::Symbol, u₂loop::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned}, filterstore = nothing
)
    foreach(op -> lower!(q, op, vectorized, ls, u₁loop, u₂loop, U, suffix, mask, filterstore), ops)
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
    u₁loop = order[u₁loopnum]
    u₂loop = order[u₂loopnum]
    vectorized = order[vectorizedloopnum]
    u₁ = n == u₁loopnum ? UF : u₁
    dontmaskfirsttiles = !isnothing(mask) && vectorizedloopnum == u₂loopnum
    blockq = Expr(:block)
    for prepost ∈ 1:2
        # !u₁ && !u₂
        lower!(blockq, ops[1,1,prepost,n], vectorized, ls, u₁loop, u₂loop, u₁, nothing, mask)
        if u₁ == 4
            lower!(blockq, ops[2,1,prepost,n], vectorized, ls, u₁loop, u₂loop, u₁, nothing, mask)
        end
        opsv1 = ops[1,2,prepost,n]
        opsv2 = ops[2,2,prepost,n]
        if length(opsv1) + length(opsv2) > 0
            # if u₁ == 3
                # lower!(blockq, ops[2,1,prepost,n], vectorized, ls, u₁loop, u₂loop, u₁, nothing, mask)
            # end
            for store ∈ (false,true)
                # let store = nothing
                nstores = 0
                iszero(length(opsv1)) || (nstores += sum(isstore, opsv1))
                iszero(length(opsv2)) || (nstores += sum(isstore, opsv2))
                if !store && length(opsv1) + length(opsv2) == nstores
                    u₁ != 4 && lower!(blockq, ops[2,1,prepost,n], vectorized, ls, u₁loop, u₂loop, u₁, nothing, mask) # for u ∈ 0:u₁-1     
                    continue
                end
                for t ∈ 0:u₂-1
                    if t == 0
                        push!(blockq.args, Expr(:(=), u₂loop, tiledsym(u₂loop)))
                    elseif u₂loopnum == vectorizedloopnum
                        push!(blockq.args, Expr(:(=), u₂loop, Expr(:call, lv(:valadd), VECTORWIDTHSYMBOL, u₂loop)))
                    else
                        push!(blockq.args, Expr(:+=, u₂loop, 1))
                    end
                    if dontmaskfirsttiles && t < u₂ - 1 # !u₁ &&  u₂
                        lower!(blockq, opsv1, vectorized, ls, u₁loop, u₂loop, u₁, t, nothing, store)
                    else # !u₁ &&  u₂
                        lower!(blockq, opsv1, vectorized, ls, u₁loop, u₂loop, u₁, t, mask, store)
                    end
                    if iszero(t) && !store && u₁ != 4 #  u₁ && !u₂
                        # for u ∈ 0:u₁-1     
                        lower!(blockq, ops[2,1,prepost,n], vectorized, ls, u₁loop, u₂loop, u₁, nothing, mask)
                        # end
                    end
                    if dontmaskfirsttiles && t < u₂ - 1 #  u₁ && u₂
                        # for u ∈ 0:u₁-1
                        lower!(blockq, opsv2, vectorized, ls, u₁loop, u₂loop, u₁, t, nothing, store)
                        # end
                    else #  u₁ && u₂
                        # for u ∈ 0:u₁-1 
                        lower!(blockq, opsv2, vectorized, ls, u₁loop, u₂loop, u₁, t, mask, store)
                        # end
                    end
                end
                nstores == 0 && break
            end
        elseif u₁ != 4
            # for u ∈ 0:u₁-1     #  u₁ && !u₂
            lower!(blockq, ops[2,1,prepost,n], vectorized, ls, u₁loop, u₂loop, u₁, nothing, mask)
            # end
        end
        if n > 1 && prepost == 1
            push!(blockq.args, lower_unrolled_dynamic(ls, us, n-1, !isnothing(mask)))
        end
    end
    loopsym = mangletiledsym(order[n], us, n)
    push!(blockq.args, incrementloopcounter(us, n, loopsym, UF))
    blockq
end
tiledsym(s::Symbol) = Symbol("##outer##", s, "##outer##")
mangletiledsym(s::Symbol, us::UnrollSpecification, n::Int) = isunrolled2(us, n) ? tiledsym(s) : s
function lower_no_unroll(ls::LoopSet, us::UnrollSpecification, n::Int, inclmask::Bool)
    loopsym = names(ls)[n]
    loop = getloop(ls, loopsym)
    loopsym = mangletiledsym(loopsym, us, n)
    nisvectorized = isvectorized(us, n)

    sl = startloop(loop, nisvectorized, loopsym)
    tc = terminatecondition(loop, us, n, loopsym, inclmask, 1)
    body = lower_block(ls, us, n, inclmask, 1)
    q = Expr( :block, sl, Expr(:while, tc, body))
    if nisvectorized
        tc = terminatecondition(loop, us, n, loopsym, true, 1)
        body = lower_block(ls, us, n, true, 1)
        push!(q.args, Expr(:if, tc, body))
    end
    q
end
function lower_unrolled_dynamic(ls::LoopSet, us::UnrollSpecification, n::Int, inclmask::Bool)
    UF = unrollfactor(us, n)
    UF == 1 && return lower_no_unroll(ls, us, n, inclmask)
    @unpack u₁loopnum, vectorizedloopnum, u₁, u₂ = us
    order = names(ls)
    loopsym = order[n]
    loop = getloop(ls, loopsym)
    loopsym = mangletiledsym(loopsym, us, n)
    vectorized = order[vectorizedloopnum]
    nisunrolled = isunrolled1(us, n)
    nisvectorized = isvectorized(us, n)
    loopisstatic = isstaticloop(loop) & (!nisvectorized)


    remmask = inclmask | nisvectorized
    Ureduct = (n == num_loops(ls) && (u₂ == -1)) ? calc_Ureduct(ls, us) : -1
    sl = startloop(loop, nisvectorized, loopsym)

    remfirst = loopisstatic & !(unsigned(Ureduct) < unsigned(UF))
    if remfirst
        tc = Expr(:call, lv(:scalar_less), loopsym, loop.stophint + 1)
    else
        tc = terminatecondition(loop, us, n, loopsym, inclmask, UF)
    end
    body = lower_block(ls, us, n, inclmask, UF)
    q = Expr(:while, tc, body)
    remblock = init_remblock(loop, loopsym)
    UFt = if loopisstatic
        length(loop) % UF
    else
        1
    end
    q = if unsigned(Ureduct) < unsigned(UF) # unsigned(-1) == typemax(UInt); is logic relying on twos-complement bad?
        Expr(
            :block, sl,
            add_upper_outer_reductions(ls, q, Ureduct, UF, loop, vectorized),
            Expr(
                :if, terminatecondition(loop, us, n, loopsym, inclmask, UF - Ureduct),
                lower_block(ls, us, n, inclmask, UF - Ureduct)
            ),
            remblock
        )
    elseif remfirst
        numiters = length(loop) ÷ UF
        if numiters > 2
            Expr( :block, sl, remblock, q )
        else
            q = Expr(:block, sl, remblock)
            for i ∈ 1:numiters
                push!(q.args, body)
            end
            q
        end
    else
        Expr( :block, sl, q, remblock )
    end
    UFt = if loopisstatic
        length(loop) % UF
    else
        1
    end
    while !iszero(UFt)
        comparison = if nisvectorized
            itercount = if loop.stopexact
                Expr(:call, :-, loop.stophint, Expr(:call, lv(:valmul), VECTORWIDTHSYMBOL, UFt))
            else
                Expr(:call, :-, loop.stopsym, Expr(:call, lv(:valmul), VECTORWIDTHSYMBOL, UFt))
            end
            Expr(:call, lv(:scalar_greater), loopsym, itercount)
        elseif remfirst
            Expr(:call, lv(:scalar_less), loopsym, loop.starthint + UFt)
        elseif loop.stopexact
            Expr(:call, lv(:scalar_greater), loopsym, loop.stophint - UFt)
        else
            Expr(:call, lv(:scalar_greater), loopsym, Expr(:call, :-, loop.stopsym, UFt))
        end
        ust = nisunrolled ? UnrollSpecification(us, UFt, u₂) : UnrollSpecification(us, u₁, UFt)
        remblocknew = Expr(:elseif, comparison, lower_block(ls, ust, n, remmask, UFt))
        push!(remblock.args, remblocknew)
        remblock = remblocknew
        if !(UFt < UF - 1 + nisvectorized) || UFt == Ureduct || loopisstatic
            break
        else
            UFt += 1
        end
    end
    q
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
        Expr(:call, lv(:scalar_less), Expr(:call, :-, unrolledloop.stopsym, unrolledloop.starthint-1), Expr(:call, lv(:valmul), VECTORWIDTHSYMBOL, Uhigh))
    elseif unrolledloop.stopexact
        Expr(:call, lv(:scalar_less), Expr(:call, :-, unrolledloop.stophint+1, unrolledloop.sartsym), Expr(:call, lv(:valmul), VECTORWIDTHSYMBOL, Uhigh))
    else# both are given by symbols
        Expr(:call, lv(:scalar_less), Expr(:call, :-, unrolledloop.stopsym, Expr(:call,:-,unrolledloop.startsym)), Expr(:call, lv(:valmul), VECTORWIDTHSYMBOL, Uhigh))
    end
    ncomparison = Expr(:call, :!, comparison)
    Expr(:if, ncomparison, ifq)
end
function reduce_expr!(q::Expr, ls::LoopSet, U::Int)
    for or ∈ ls.outer_reductions
        op = ls.operations[or]
        var = name(op)
        mvar = mangledvar(op)
        instr = instruction(op)
        reduce_expr!(q, mvar, instr, U)
        length(ls.opdict) == 0 || push!(q.args, Expr(:(=), var, Expr(:call, lv(reduction_scalar_combine(instr)), var, Symbol(mvar, 0))))
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
function init_remblock(unrolledloop::Loop, u₁loop::Symbol = unrolledloop.itersymbol)
    condition = if unrolledloop.stopexact
        Expr(:call, lv(:scalar_greater), u₁loop, unrolledloop.stophint)
    else
        Expr(:call, lv(:scalar_greater), u₁loop, unrolledloop.stopsym)
    end
    Expr(:if, condition, nothing)
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
            Expr(:call, :-, loop.stopsym, loop.starthint - 1)
        elseif loop.stopexact
            Expr(:call, :-, loop.stophint + 1, loop.startsym)
        else
            Expr(:call, :-, Expr(:call, :+, loop.stopsym, 1), loop.startsym)
        end
        maskexpr(lexpr)
    end
end

function setup_preamble!(ls::LoopSet, us::UnrollSpecification)
    @unpack u₁loopnum, u₂loopnum, vectorizedloopnum, u₁, u₂ = us
    order = names(ls)
    u₁loop = order[u₁loopnum]
    u₂loop = order[u₂loopnum]
    vectorized = order[vectorizedloopnum]
    if length(ls.includedactualarrays) > 0
        push!(ls.preamble.args, Expr(:(=), ELTYPESYMBOL, determine_eltype(ls)))
        push!(ls.preamble.args, Expr(:(=), VECTORWIDTHSYMBOL, determine_width(ls, vectorized)))
    end
    lower_licm_constants!(ls)
    pushpreamble!(ls, definemask(getloop(ls, vectorized)))#, u₁ > 1 && u₁loopnum == vectorizedloopnum))
    for op ∈ operations(ls)
        (iszero(length(loopdependencies(op))) && iscompute(op)) && lower_compute!(ls.preamble, op, vectorized, u₁loop, u₂loop, u₁, nothing, nothing)
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
    elseif num_loops(ls) == u₁loopnum
        min(u₁, 4)
    else
        u₂ == -1 ? u₁ : 4#u₂
    end
end
function lower(ls::LoopSet, us::UnrollSpecification)
    @unpack vectorizedloopnum, u₁, u₂ = us
    order = names(ls)
    vectorized = order[vectorizedloopnum]
    setup_preamble!(ls, us)
    Ureduct = calc_Ureduct(ls, us)
    initialize_outer_reductions!(ls, 0, Ureduct, vectorized)
    
    q = gc_preserve( ls, lower_unrolled_dynamic(ls, us, num_loops(ls), false) )
    reduce_expr!(q, ls, Ureduct)
    lsexpr(ls, q)
end

function lower(ls::LoopSet, order, u₁loop, u₂loop, vectorized, u₁, u₂)
    fillorder!(ls, order, u₁loop, u₂loop, u₂ != -1)
    q = lower(ls, UnrollSpecification(ls, u₁loop, u₂loop, vectorized, u₁, u₂))
    iszero(length(ls.opdict)) && pushfirst!(q.args, Expr(:meta, :inline))
    q
end

function lower(ls::LoopSet)#, prependinlineORorUnroll = 0)
    order, u₁loop, u₂loop, vectorized, u₁, u₂ = choose_order(ls)
    lower(ls, order, u₁loop, u₂loop, vectorized, u₁, u₂)
end
function lower(ls::LoopSet, u₁, u₂)#, prependinlineORorUnroll = 0)
    if u₂ > 1
        @assert num_loops(ls) > 1 "There is only $(num_loops(ls)) loop, but specified blocking parameter u₂ is $u₂."
        order, u₁loop, u₂loop, vectorized, _u₁, _u₂, c = choose_tile(ls)
    else
        u₂ = -1
        order, vectorized, c = choose_unroll_order(ls, Inf)
        u₁loop = first(order); u₂loop = Symbol("##undefined##")
    end
    lower(ls, order, u₁loop, u₂loop, vectorized, u₁, u₂)
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

