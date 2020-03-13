
# struct TileDescription
    # vectorized::Symbol
    # unrolled::Symbol
    # tiled::Symbol
    # U::Int
    # T::Int
# end


function lower!(
    q::Expr, op::Operation, vectorized::Symbol, ls::LoopSet, unrolled::Symbol, tiled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned}
)
    W = ls.W
    if isconstant(op)
        zerotyp = zerotype(ls, op)
        if zerotyp == INVALID
            lower_constant!(q, op, vectorized, ls, unrolled, U, suffix)
        else
            lower_zero!(q, op, vectorized, ls, unrolled, U, suffix, zerotyp)
        end
    elseif isload(op)
        lower_load!(q, op, vectorized, W, unrolled, tiled, U, suffix, mask)
    elseif iscompute(op)
        lower_compute!(q, op, vectorized, W, unrolled, tiled, U, suffix, mask)
    elseif isstore(op)
        lower_store!(q, op, vectorized, W, unrolled, tiled, U, suffix, mask)
    # elseif isloopvalue(op)
    end
end
function lower!(
    q::Expr, ops::AbstractVector{Operation}, vectorized::Symbol, ls::LoopSet, unrolled::Symbol, tiled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned}
)
    foreach(op -> lower!(q, op, vectorized, ls, unrolled, tiled, U, suffix, mask), ops)
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
    @unpack unrolledloopnum, tiledloopnum, vectorizedloopnum, U, T = us
    ops = oporder(ls)
    order = names(ls)
    unrolled = order[unrolledloopnum]
    tiled = order[tiledloopnum]
    vectorized = order[vectorizedloopnum]
    U = n == unrolledloopnum ? UF : U
    dontmaskfirsttiles = !isnothing(mask) && vectorizedloopnum == tiledloopnum
    blockq = Expr(:block)
    for prepost ∈ 1:2
        # !U && !T
        lower!(blockq, ops[1,1,prepost,n], vectorized, ls, unrolled, tiled, U, nothing, mask)
        # for u ∈ 0:U-1     #  U && !T
        lower!(blockq, ops[2,1,prepost,n], vectorized, ls, unrolled, tiled, U, nothing, mask)
        # end
        if length(ops[1,2,prepost,n]) + length(ops[2,2,prepost,n]) > 0
            for t ∈ 0:T-1
                if t == 0
                    push!(blockq.args, Expr(:(=), tiled, tiledsym(tiled)))
                elseif tiledloopnum == vectorizedloopnum
                    push!(blockq.args, Expr(:(=), tiled, Expr(:call, lv(:valadd), ls.W, tiled)))
                else
                    push!(blockq.args, Expr(:+=, tiled, 1))
                end
                # !U &&  T
                if dontmaskfirsttiles && t < T - 1
                    lower!(blockq, ops[1,2,prepost,n], vectorized, ls, unrolled, tiled, U, t, nothing)
                    # for u ∈ 0:U-1 #  U &&  T
                    lower!(blockq, ops[2,2,prepost,n], vectorized, ls, unrolled, tiled, U, t, nothing)
                    # end
                else
                    lower!(blockq, ops[1,2,prepost,n], vectorized, ls, unrolled, tiled, U, t, mask)
                    # for u ∈ 0:U-1 #  U &&  T
                    lower!(blockq, ops[2,2,prepost,n], vectorized, ls, unrolled, tiled, U, t, mask)
                    # end
                end
            end
        end
        if n > 1 && prepost == 1
            push!(blockq.args, lower_unrolled_dynamic(ls, us, n-1, !isnothing(mask)))
        end
    end
    loopsym = mangletiledsym(order[n], us, n)
    push!(blockq.args, incrementloopcounter(us, n, ls.W, loopsym, UF))
    blockq
end
tiledsym(s::Symbol) = Symbol("##outer##", s, "##outer##")
mangletiledsym(s::Symbol, us::UnrollSpecification, n::Int) = istiled(us, n) ? tiledsym(s) : s
function lower_no_unroll(ls::LoopSet, us::UnrollSpecification, n::Int, inclmask::Bool)
    loopsym = names(ls)[n]
    loop = getloop(ls, loopsym)
    loopsym = mangletiledsym(loopsym, us, n)
    nisvectorized = isvectorized(us, n)

    sl = startloop(loop, nisvectorized, ls.W, loopsym)
    tc = terminatecondition(loop, us, n, ls.W, loopsym, inclmask, 1)
    body = lower_block(ls, us, n, inclmask, 1)
    q = Expr( :block, sl, Expr(:while, tc, body))
    if nisvectorized
        tc = terminatecondition(loop, us, n, ls.W, loopsym, true, 1)
        body = lower_block(ls, us, n, true, 1)
        push!(q.args, Expr(:if, tc, body))
    end
    q
end
function lower_unrolled_dynamic(ls::LoopSet, us::UnrollSpecification, n::Int, inclmask::Bool)
    UF = unrollfactor(us, n)
    UF == 1 && return lower_no_unroll(ls, us, n, inclmask)
    @unpack unrolledloopnum, vectorizedloopnum, U, T = us
    order = names(ls)
    loopsym = order[n]
    loop = getloop(ls, loopsym)
    loopsym = mangletiledsym(loopsym, us, n)
    vectorized = order[vectorizedloopnum]
    nisunrolled = isunrolled(us, n)
    nisvectorized = isvectorized(us, n)

    remmask = inclmask | nisvectorized
    Ureduct = (n == num_loops(ls)) ? calc_Ureduct(ls, us) : -1
    sl = startloop(loop, nisvectorized, ls.W, loopsym)
    tc = terminatecondition(loop, us, n, ls.W, loopsym, inclmask, UF)
    body = lower_block(ls, us, n, inclmask, UF)

    q = Expr(:while, tc, body)
    q = if unsigned(Ureduct) < unsigned(UF) # unsigned(-1) == typemax(UInt); is logic relying on twos-complement bad?
        Expr(
            :block, sl,
            add_upper_outer_reductions(ls, q, Ureduct, UF, loop, vectorized),
            Expr(
                :if, terminatecondition(loop, us, n, ls.W, loopsym, inclmask, UF - Ureduct),
                lower_block(ls, us, n, inclmask, UF - Ureduct)
            )
        )
    else
        Expr( :block, sl, q )
    end
    remblock = init_remblock(loop, loopsym)
    push!(q.args, remblock)
    UFt = 1
    while true
        comparison = if nisvectorized
            itercount = if loop.stopexact
                Expr(:call, :-, loop.stophint, Expr(:call, lv(:valmul), ls.W, UFt))
            else
                Expr(:call, :-, loop.stopsym, Expr(:call, lv(:valmul), ls.W, UFt))
            end
            Expr(:call, :>, loopsym, itercount)
        elseif loop.stopexact
            Expr(:call, :>, loopsym, loop.stophint - UFt)
        else
            Expr(:call, :>, loopsym, Expr(:call, :-, loop.stopsym, UFt))
        end
        ust = nisunrolled ? UnrollSpecification(us, UFt, T) : UnrollSpecification(us, U, UFt)
        remblocknew = Expr(:elseif, comparison, lower_block(ls, ust, n, remmask, UFt))
        push!(remblock.args, remblocknew)
        remblock = remblocknew
        if !(UFt < UF - 1 + nisvectorized) || UFt == Ureduct
            break
        else
            UFt += 1
        end
    end
    q
end

function initialize_outer_reductions!(
    q::Expr, op::Operation, Umin::Int, Umax::Int, vectorized::Symbol, W::Symbol, suffix::Union{Symbol,Nothing} = nothing
)
    reduct_zero = reduction_zero(op.instruction)
    isvectorized = vectorized ∈ reduceddependencies(op)
    # typeTr = Symbol("##TYPEOF##", name(op))
    typeTr = Expr(:call, :typeof, mangledvar(op))
    z = if isvectorized
        if reduct_zero === :zero
            Expr(:call, lv(:vzero), W, typeTr)
        else
            Expr(:call, lv(:vbroadcast), W, Expr(:call, reduct_zero, typeTr))
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
    foreach(or -> initialize_outer_reductions!(q, ls.operations[or], Umin, Umax, vectorized, ls.W, suffix), ls.outer_reductions)
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
        Expr(:call, :<, length(unrolledloop), Expr(:call, lv(:valmul), ls.W, Uhigh))
    elseif unrolledloop.starthint == 1
        Expr(:call, :<, unrolledloop.stopsym, Expr(:call, lv(:valmul), ls.W, Uhigh))
    elseif unrolledloop.startexact
        Expr(:call, :<, Expr(:call, :-, unrolledloop.stopsym, unrolledloop.starthint-1), Expr(:call, lv(:valmul), ls.W, Uhigh))
    elseif unrolledloop.stopexact
        Expr(:call, :<, Expr(:call, :-, unrolledloop.stophint+1, unrolledloop.sartsym), Expr(:call, lv(:valmul), ls.W, Uhigh))
    else# both are given by symbols
        Expr(:call, :<, Expr(:call, :-, unrolledloop.stopsym, Expr(:call,:-,unrolledloop.startsym)), Expr(:call, lv(:valmul), ls.W, Uhigh))
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
function determine_width(ls::LoopSet, vectorized::Symbol)
    vloop = getloop(ls, vectorized)
    vwidth_q = Expr(:call, lv(:pick_vector_width_val))
    if isstaticloop(vloop)
        push!(vwidth_q.args, Expr(:call, Expr(:curly, :Val, length(vloop))))
    end
    # push!(vwidth_q.args, ls.T)
    if length(ls.includedactualarrays) < 2
        push!(vwidth_q.args, ls.T)
    else
        for array ∈ ls.includedactualarrays
            push!(vwidth_q.args, Expr(:call, :eltype, array))
        end
    end
    vwidth_q
end
function init_remblock(unrolledloop::Loop, unrolled::Symbol = unrolledloop.itersymbol)
    condition = if unrolledloop.stopexact
        Expr(:call, :(>), unrolled, unrolledloop.stophint)
    else
        Expr(:call, :(>), unrolled, unrolledloop.stopsym)
    end
    Expr(:if, condition, nothing)
end

function maskexpr(W::Symbol, looplimit, all_on::Bool)
    rem = Expr(:call, lv(:valrem), W, looplimit)
    Expr(:(=), Symbol("##mask##"), Expr(:call, lv(all_on ? :masktable : :mask), W, rem))
end
function definemask(loop::Loop, W::Symbol, allon::Bool)
    if isstaticloop(loop)
        maskexpr(W, length(loop), allon)
    elseif loop.startexact && loop.starthint == 1
        maskexpr(W, loop.stopsym, allon)
    else
        lexpr = if loop.startexact
            Expr(:call, :-, loop.stopsym, loop.starthint - 1)
        elseif loop.stopexact
            Expr(:call, :-, loop.stophint + 1, loop.startsym)
        else
            Expr(:call, :-, Expr(:call, :+, loop.stopsym, 1), loop.startsym)
        end
        maskexpr(W, lexpr, allon)
    end
end

function setup_preamble!(ls::LoopSet, us::UnrollSpecification)
    @unpack unrolledloopnum, tiledloopnum, vectorizedloopnum, U, T = us
    order = names(ls)
    unrolled = order[unrolledloopnum]
    tiled = order[tiledloopnum]
    vectorized = order[vectorizedloopnum]
    # println("Setup preamble")
    W = ls.W; typeT = ls.T
    length(ls.includedarrays) == 0 || push!(ls.preamble.args, Expr(:(=), typeT, determine_eltype(ls)))
    push!(ls.preamble.args, Expr(:(=), W, determine_width(ls, vectorized)))
    lower_licm_constants!(ls)
    pushpreamble!(ls, definemask(getloop(ls, vectorized), W, U > 1 && unrolledloopnum == vectorizedloopnum))
    for op ∈ operations(ls)
        (iszero(length(loopdependencies(op))) && iscompute(op)) && lower_compute!(ls.preamble, op, vectorized, ls.W, unrolled, tiled, U, nothing, nothing)
    end
    # define_remaining_ops!( ls, vectorized, W, unrolled, tiled, U )
end
function lsexpr(ls::LoopSet, q)
    Expr(:block, ls.preamble, q)
end

function calc_Ureduct(ls::LoopSet, us::UnrollSpecification)
    @unpack unrolledloopnum, U, T = us
    if iszero(length(ls.outer_reductions))
        -1
    elseif num_loops(ls) == unrolledloopnum
        min(U, 4)
    else
        T == -1 ? U : T
    end
end
function lower(ls::LoopSet, us::UnrollSpecification)
    @unpack vectorizedloopnum, U, T = us
    order = names(ls)
    vectorized = order[vectorizedloopnum]
    setup_preamble!(ls, us)
    Ureduct = calc_Ureduct(ls, us)
    initialize_outer_reductions!(ls, 0, Ureduct, vectorized)
    
    q = gc_preserve( ls, lower_unrolled_dynamic(ls, us, num_loops(ls), false) )
    reduce_expr!(q, ls, Ureduct)
    lsexpr(ls, q)
end

function lower(ls::LoopSet, order, unrolled, tiled, vectorized, U, T)
    fillorder!(ls, order, unrolled, tiled, T != -1)
    q = lower(ls, UnrollSpecification(ls, unrolled, tiled, vectorized, U, T))
    iszero(length(ls.opdict)) && pushfirst!(q.args, Expr(:meta, :inline))
    q
end

function lower(ls::LoopSet)#, prependinlineORorUnroll = 0)
    order, unrolled, tiled, vectorized, U, T = choose_order(ls)
    lower(ls, order, unrolled, tiled, vectorized, U, T)
end
function lower(ls::LoopSet, U, T)#, prependinlineORorUnroll = 0)
    num_loops(ls) == 1 && @assert T == -1
    if T == -1
        order, vectorized, c = choose_unroll_order(ls, Inf)
        unrolled = first(order); tiled = Symbol("##undefined##")
    else
        order, unrolled, tiled, vectorized, _U, _T, c = choose_tile(ls)
    end
    lower(ls, order, unrolled, tiled, vectorized, U, T)
end

Base.convert(::Type{Expr}, ls::LoopSet) = lower(ls)
Base.show(io::IO, ls::LoopSet) = println(io, lower(ls))

