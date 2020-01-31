
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
        if identifier(op) ∈ ls.preamble_zeros
            lower_zero!(q, op, vectorized, W, unrolled, U, suffix, ls.T)
        else
            lower_constant!(q, op, vectorized, W, unrolled, U, suffix, ls.T)
        end
    elseif isload(op)
        lower_load!(q, op, vectorized, W, unrolled, tiled, U, suffix, mask)
    elseif iscompute(op)
        lower_compute!(q, op, vectorized, W, unrolled, tiled, U, suffix, mask)
    else#if isstore(op)
        lower_store!(q, op, vectorized, W, unrolled, tiled, U, suffix, mask)
    end
end
function lower!(
    q::Expr, ops::AbstractVector{Operation}, vectorized::Symbol, ls::LoopSet, unrolled::Symbol, tiled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned}
)
    foreach(op -> lower!(q, op, vectorized, ls, unrolled, tiled, U, suffix, mask), ops)
end

tiledsym(s::Symbol) = Symbol("##outer##", s, "##outer##")
function lower_nest(
    ls::LoopSet, n::Int, vectorized::Symbol, U::Int, T::Int, loopq_old::Union{Expr,Nothing},
    loopstart::Union{Int,Symbol}, W::Symbol,
    mask::Union{Nothing,Symbol,Unsigned} = nothing, exprtype::Symbol = :while
)
    ops = oporder(ls)
    order = names(ls)
    istiled = T != -1
    loopsym = order[n]
    nloops = num_loops(ls)
    outer_reduce = length(ls.outer_reductions) > 0
    nisvectorized = loopsym === vectorized
    nisunrolled = false
    nistiled = false
    if istiled
        if n == nloops
            loopsym = tiledsym(loopsym)
            nistiled = true
        elseif n == nloops - 1
            nisunrolled = true
        end
        unrolled = order[end-1]
    else
        unrolled = last(order)
        nisunrolled = n == nloops
    end
    blockq = Expr(:block)
    if n > 1
        looptoadd = order[n-1]
        push!(blockq.args, startloop(getloop(ls, looptoadd), looptoadd === vectorized, W, looptoadd))
    end
    loopq = if exprtype === :block
        blockq
    else
        @assert exprtype === :while || exprtype === :if "Expression type $exprtype not recognized."
        # if we have a mask, we use 1 as the incremement to pass to looprange, so that we get a iter < maxiter condition.
        lr = terminatecondition(getloop(ls, loopsym), W, U, T, nisvectorized, nisunrolled, nistiled, loopsym, mask)
        Expr(exprtype, lr, blockq)
    end
    for prepost ∈ 1:2
        # !U && !T
        lower!(blockq, ops[1,1,prepost,n], vectorized, ls, unrolled, last(order), U, nothing, mask)
        # for u ∈ 0:U-1     #  U && !T
        lower!(blockq, ops[2,1,prepost,n], vectorized, ls, unrolled, last(order), U, nothing, mask)
        # end
        if length(ops[1,2,prepost,n]) + length(ops[2,2,prepost,n]) > 0
            for t ∈ 0:T-1
                if t == 0
                    push!(blockq.args, Expr(:(=), last(order), tiledsym(last(order))))
                else
                    push!(blockq.args, Expr(:+=, last(order), 1))
                end
                # !U &&  T
                lower!(blockq, ops[1,2,prepost,n], vectorized, ls, unrolled, last(order), U, t, mask)
                # for u ∈ 0:U-1 #  U &&  T
                lower!(blockq, ops[2,2,prepost,n], vectorized, ls, unrolled, last(order), U, t, mask)
                # end
            end
        end
        if loopq_old !== nothing && n > 1 && prepost == 1
            push!(blockq.args, loopq_old)
        end
    end
    if nisvectorized
        if nisunrolled
            push!(blockq.args, Expr(:+=, loopsym, Expr(:call, lv(:valmul), W, U)))
        else
            push!(blockq.args, Expr(:(=), loopsym, Expr(:call, lv(:valadd), W, loopsym)))
        end
    else
        push!(blockq.args, Expr(:+=, loopsym, nisunrolled ? U : (nistiled ? T : 1)))
    end
    loopq
end
# Calculates nested loop set,
# if tiled, it will not lower the tiled iteration.
function lower_set(ls::LoopSet, vectorized::Symbol, U::Int, T::Int, W::Symbol, ::Nothing, Uexprtype::Symbol)
    loopstart = 0
    istiled = T != -1
    order = names(ls)
    unrolled = order[end - istiled]
    unrolled === vectorized && return lower_set_unrolled_is_vectorized(ls, vectorized, U, T, W, nothing, Uexprtype)
    ns = 1
    nl = num_loops(ls) - istiled
    exprtype = nl == ns ? Uexprtype : :while
    nomaskq = lower_nest(ls, 1, vectorized, U, T, nothing, loopstart, W, nothing, exprtype)
    maskq = lower_nest(ls, 1, vectorized, U, T, nothing, loopstart, W, Symbol("##mask##"), order[ns] === vectorized ? :if : exprtype)
    while order[ns] !== vectorized
        ns += 1
        exprtype = nl == ns ? Uexprtype : :while
        nomaskq = lower_nest(ls, ns, vectorized, U, T, nomaskq, loopstart, W, nothing, exprtype)
        maskq = lower_nest(ls, ns, vectorized, U, T, maskq, loopstart, W, Symbol("##mask##"), order[ns] === vectorized ? :if : exprtype)
    end
    ns += 1
    loopq = Expr(:block, nomaskq, maskq)
    for n ∈ ns:nl
        exprtype = n == nl ? Uexprtype : :while
        loopq = lower_nest(ls, n, vectorized, U, T, loopq, loopstart, W, nothing, exprtype)
        # loopq = add_vec_rem_iter( ls, n, vectorized, U, T, loopq, 0, W, exprtype, order )
    end
    loopq
end
function lower_set(ls::LoopSet, vectorized::Symbol, U::Int, T::Int, W::Symbol, mask::Symbol, Uexprtype::Symbol)
    lower_set_unrolled_is_vectorized(ls, vectorized, U, T, W, mask, Uexprtype)
end
function lower_set_unrolled_is_vectorized(ls::LoopSet, vectorized::Symbol, U::Int, T::Int, W::Symbol, mask, Uexprtype::Symbol)
    nl = num_loops(ls) - (T != -1)
    loopq = lower_nest(ls, 1, vectorized, U, T, nothing, 0, W, mask, nl == 1 ? Uexprtype : :while)
    for n ∈ 2:nl
        exprtype = n == nl ? Uexprtype : :while
        loopq = lower_nest(ls, n, vectorized, U, T, loopq, 0, W, mask, exprtype)
    end
    loopq
end
function initialize_outer_reductions!(
    q::Expr, op::Operation, Umin::Int, Umax::Int, W::Symbol, typeT::Symbol, vectorized::Symbol, suffix::Union{Symbol,Nothing} = nothing
)
    reduct_zero = reduction_zero(op.instruction)
    isvectorized = vectorized ∈ reduceddependencies(op)
    z = if isvectorized
        if reduct_zero === :zero
            Expr(:call, lv(:vzero), W, typeT)
        else
            Expr(:call, lv(:vbroadcast), W, Expr(:call, reduct_zero, typeT))
        end
    else
        Expr(:call, reduct_zero, typeT)
    end
    mvar = variable_name(op, suffix)
    for u ∈ Umin:Umax-1
        push!(q.args, Expr(:(=), Symbol(mvar, u), z))
    end
    nothing
end
function initialize_outer_reductions!(q::Expr, ls::LoopSet, Umin::Int, Umax::Int, W::Symbol, typeT::Symbol, vectorized::Symbol, suffix::Union{Symbol,Nothing} = nothing)
    foreach(or -> initialize_outer_reductions!(q, ls.operations[or], Umin, Umax, W, typeT, vectorized, suffix), ls.outer_reductions)
end
function initialize_outer_reductions!(ls::LoopSet, Umin::Int, Umax::Int, W::Symbol, typeT::Symbol, vectorized::Symbol, suffix::Union{Symbol,Nothing} = nothing)
    initialize_outer_reductions!(ls.preamble, ls, Umin, Umax, W, typeT, vectorized, suffix)
end
function add_upper_outer_reductions(ls::LoopSet, loopq::Expr, Ulow::Int, Uhigh::Int, W::Symbol, typeT::Symbol, unrolledloop::Loop, vectorized::Symbol)
    ifq = Expr(:block)
    initialize_outer_reductions!(ifq, ls, Ulow, Uhigh, W, typeT, vectorized)
    push!(ifq.args, loopq)
    reduce_range!(ifq, ls, Ulow, Uhigh)
    comparison = if isstaticloop(unrolledloop)
        Expr(:call, :<, length(unrolledloop), Expr(:call, lv(:valmul), W, Uhigh))
    elseif unrolledloop.starthint == 0
        Expr(:call, :<, unrolledloop.stopsym, Expr(:call, lv(:valmul), W, Uhigh))
    elseif unrolledloop.startexact
        Expr(:call, :<, Expr(:call, :-, unrolledloop.stopsym, unrolledloop.starthint), Expr(:call, lv(:valmul), W, Uhigh))
    elseif unrolledloop.stopexact
        Expr(:call, :<, Expr(:call, :-, unrolledloop.stophint, unrolledloop.sartsym), Expr(:call, lv(:valmul), W, Uhigh))
    else# both are given by symbols
        Expr(:call, :<, Expr(:call, :-, unrolledloop.stopsym, unrolledloop.startsym), Expr(:call, lv(:valmul), W, Uhigh))
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
    length(ls.includedactualarrays) == 0 && return q
    gcp = Expr(:macrocall, Expr(:(.), :GC, QuoteNode(Symbol("@preserve"))), LineNumberNode(@__LINE__, @__FILE__))
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
function determine_width(ls::LoopSet, typeT::Symbol, unrolled::Symbol)
    unrolledloop = getloop(ls, unrolled)
    if isstaticloop(unrolledloop)
        Expr(:call, lv(:pick_vector_width_val), Expr(:call, Expr(:curly, :Val, length(unrolledloop))), typeT)
    else
        Expr(:call, lv(:pick_vector_width_val), typeT)
    end
end
function lower_unrolled!(
    q::Expr, ls::LoopSet, vectorized::Symbol, U::Int, T::Int, W::Symbol, typeT::Symbol, unrolledloop::Loop
)
    Ureduct = min(U,4)
    q = lower_unrolled_dynamic!( q, ls, vectorized, U, T, W, typeT, unrolledloop )
    if T == -1
        q = gc_preserve(ls, q)
    end
    manageouterreductions = T == -1 && length(ls.outer_reductions) > 0
    manageouterreductions && reduce_expr!(q, ls, Ureduct)
    q
end

function lower_unrolled_dynamic!(
    q::Expr, ls::LoopSet, vectorized::Symbol, U::Int, T::Int, W::Symbol, typeT::Symbol, unrolledloop::Loop
)
    unrolled = unrolledloop.itersymbol
    unrolled_stopsym = unrolledloop.stopsym
    Urem = 0
    Uexprtype = :while
    manageouterreductions = T == -1 && length(ls.outer_reductions) > 0
    if manageouterreductions
        Ureduct = U > 6 ? 4 : U
        initialize_outer_reductions!(q, ls, 0, Ureduct, W, typeT, vectorized)#last(names(ls)))
    else
        Ureduct = -1
    end
    Ut = U
    vecisunrolled = unrolled === vectorized
    local remblock::Expr
    firstiter = true
    while true
        if firstiter # first iter
            loopq = lower_set(ls, vectorized, Ut, T, W, nothing, Uexprtype)
            if T == -1 && manageouterreductions && U > 4
                loopq = add_upper_outer_reductions(ls, loopq, Ureduct, U, W, typeT, unrolledloop, vectorized)
            end
            push!(q.args, loopq)
        elseif U == 1 #
            if vecisunrolled
                push!(remblock.args, lower_set(ls, vectorized, Ut, T, W, Symbol("##mask##"), :block))
            else
                push!(remblock.args, lower_set(ls, vectorized, Ut, T, W, nothing, :block))
            end
        else
            remblocknew = if vecisunrolled
                itercount = if isstaticloop(unrolledloop)
                    Expr(:call, :-, length(unrolledloop), Expr(:call, lv(:valmuladd), W, Ut, 1))
                else
                    Expr(:call, :-, unrolled_stopsym, Expr(:call, lv(:valmuladd), W, Ut, 1))
                end
                comparison = Expr(:call, :>, unrolled, itercount)
                Expr(Ut == 1 ? :if : :elseif, comparison, lower_set(ls, vectorized, Ut, T, W, Symbol("##mask##"), :block))
            else
                comparison = if isstaticloop(unrolledloop)
                    Expr(:call, :>, unrolled, length(unrolledloop) - (Ut + 1))
                else
                    Expr(:call, :>, unrolled, Expr(:call, :-, unrolled_stopsym, Ut + 1))
                end
                Expr(Ut == 1 ? :if : :elseif, comparison, lower_set(ls, vectorized, Ut, T, W, nothing, :block))
                # Expr(Ut == 1 ? :if : :elseif, comparison, lower_set(ls, vectorized, Ut, T, W, Symbol("##mask##"), :block))
            end
            push!(remblock.args, remblocknew)
            remblock = remblocknew
        end
        if firstiter
            firstiter = false
            if manageouterreductions && Ureduct < U
                Udiff = U - Ureduct
                loopq = lower_set(ls, vectorized, Udiff, T, W, nothing, :if)
                push!(q.args, loopq)
            end
            Ut = 1
            # setup for branchy remainder calculation
            comparison = if isstaticloop(unrolledloop)
                Expr(:call, :(!=), length(unrolledloop), unrolled)
            else
                Expr(:call, :(!=), unrolled_stopsym, unrolled)
            end
            remblock = Expr(:block)
            push!(q.args, Expr(:if, comparison, remblock))
        elseif !(Ut < U - 1 + vecisunrolled) || Ut == Ureduct
            break
        else
            Ut += 1
        end
    end
    q
end
function maskexpr(W::Symbol, looplimit, allon::Bool)
    rem = Expr(:call, lv(:valrem), W, looplimit)
    Expr(:(=), Symbol("##mask##"), Expr(:call, lv(allon ? :masktable : :mask), W, rem))
end
function definemask(loop::Loop, W::Symbol, allon::Bool)
    if isstaticloop(loop)
        maskexpr(W, length(loop), allon)
    elseif loop.starthint == 0
        maskexpr(W, loop.stopsym, allon)
    else
        lexpr = if loop.startexact
            Expr(:call, :-, loop.stopsym, loop.starthint)
        elseif loop.stopexact
            Expr(:call, :-, loop.stophint, loop.startsym)
        else
            Expr(:call, :-, loop.stopsym, loop.startsym)
        end
        maskexpr(W, lexpr, allon)
    end
end
@inline sizeequivalentfloat(::Type{T}, x::T) where {T} = x
@inline sizeequivalentfloat(::Type{Int64}, x::Float64) = x
@inline sizeequivalentfloat(::Type{Int64}, x::Float32) = Float64(x)
@inline sizeequivalentfloat(::Type{Int64}, x::Float16) = Float64(x)
@inline sizeequivalentfloat(::Type{Int32}, x::Float64) = Float32(x)
@inline sizeequivalentfloat(::Type{Int32}, x::Float32) = x
@inline sizeequivalentfloat(::Type{Int32}, x::Float16) = Float32(x)
@inline sizeequivalentfloat(::Type{Int16}, x::Float64) = Float16(x)
@inline sizeequivalentfloat(::Type{Int16}, x::Float32) = Float16(x)
@inline sizeequivalentfloat(::Type{Int16}, x::Float16) = x
@inline sizeequivalentfloat(::Type{Float64}, x::Float32) = Float64(x)
@inline sizeequivalentfloat(::Type{Float64}, x::Float16) = Float64(x)
@inline sizeequivalentfloat(::Type{Float32}, x::Float64) = Float32(x)
@inline sizeequivalentfloat(::Type{Float32}, x::Float16) = Float32(x)
@inline sizeequivalentfloat(::Type{Float16}, x::Float64) = Float16(x)
@inline sizeequivalentfloat(::Type{Float16}, x::Float32) = Float16(x)
@inline sizeequivalentint(::Type{T}, x::T) where {T} = x
@inline sizeequivalentint(::Type{Int64}, x::Int64) = x
@inline sizeequivalentint(::Type{Int64}, x::Int32) = Int64(x)
@inline sizeequivalentint(::Type{Int64}, x::Int16) = Int64(x)
@inline sizeequivalentint(::Type{Int32}, x::Int64) = Int32(x)
@inline sizeequivalentint(::Type{Int32}, x::Int32) = x
@inline sizeequivalentint(::Type{Int32}, x::Int16) = Int32(x)
@inline sizeequivalentint(::Type{Int16}, x::Int64) = Int16(x)
@inline sizeequivalentint(::Type{Int16}, x::Int32) = Int16(x)
@inline sizeequivalentint(::Type{Int16}, x::Int16) = x
@inline sizeequivalentint(::Type{Float64}, x::Int32) = Int64(x)
@inline sizeequivalentint(::Type{Float64}, x::Int16) = Int64(x)
@inline sizeequivalentint(::Type{Float32}, x::Int64) = Int32(x)
@inline sizeequivalentint(::Type{Float32}, x::Int16) = Int32(x)
@inline sizeequivalentint(::Type{Float16}, x::Int64) = Int16(x)
@inline sizeequivalentint(::Type{Float16}, x::Int32) = Int16(x)


function setup_preamble!(ls::LoopSet, W::Symbol, typeT::Symbol, vectorized::Symbol, unrolled::Symbol, tiled::Symbol, U::Int)
    # println("Setup preamble")
    length(ls.includedarrays) == 0 || push!(ls.preamble.args, Expr(:(=), typeT, determine_eltype(ls)))
    push!(ls.preamble.args, Expr(:(=), W, determine_width(ls, typeT, unrolled)))
    lower_licm_constants!(ls)
    pushpreamble!(ls, definemask(getloop(ls, vectorized), W, U > 1 && unrolled === vectorized))
    for op ∈ operations(ls)
        (iszero(length(loopdependencies(op))) && iscompute(op)) && lower_compute!(ls.preamble, op, vectorized, ls.W, unrolled, tiled, U, nothing, nothing)
    end
    # define_remaining_ops!( ls, vectorized, W, unrolled, tiled, U )
end
function lsexpr(ls::LoopSet, q)
    Expr(:block, ls.preamble, q)
end
function lower_tiled(ls::LoopSet, vectorized::Symbol, U::Int, T::Int)
    order = ls.loop_order.loopnames
    tiled    = order[end]
    unrolled = order[end-1]
    mangledtiled = tiledsym(tiled)
    W = ls.W
    typeT = ls.T
    setup_preamble!(ls, W, typeT, vectorized, unrolled, tiled, U)
    tiledloop = getloop(ls, tiled)
    static_tile = isstaticloop(tiledloop)
    unrolledloop = getloop(ls, unrolled)
    initialize_outer_reductions!(ls, 0, 4, W, typeT, vectorized)#unrolled)
    q = Expr(:block, startloop(tiledloop, false, W, mangledtiled))
    # we build up the loop expression.
    Trem = Tt = T
    nloops = num_loops(ls);
    firstiter = true
    mangledtiled = tiledsym(tiled)
    local qifelse::Expr
    while Tt > 0
        tiledloopbody = Expr(:block)
        lower_unrolled!(tiledloopbody, ls, vectorized, U, Tt, W, typeT, unrolledloop)
        tiledloopbody = lower_nest(ls, nloops, vectorized, U, Tt, tiledloopbody, 0, W, nothing, :block)
        if firstiter
            looptermcon = terminatecondition(tiledloop, W, U, Tt, false, false, true, mangledtiled, nothing)
            push!(q.args, (static_tile && length(tiledloop) < 2T) ? tiledloopbody : Expr(:while, looptermcon, tiledloopbody))
        elseif static_tile
            push!(q.args, tiledloopbody)
        else # not static, not firstiter
            comparison = Expr(:call, :(==), mangledtiled, Expr(:call, :-, tiledloop.stopsym, Tt))
            qifelsenew = Expr(:elseif, comparison, tiledloopbody)
            push!(qifelse.args, qifelsenew)
            qifelse = qifelsenew
        end
        if static_tile
            if Tt == T
                Texprtype = :block
                Tt = length(tiledloop) % T
                # Recalculate U
                U = solve_tilesize_constT(ls, Tt)
            else
                Tt = 0 # terminate
            end
            nothing
        else
            if firstiter
                comparison = if tiledloop.stopexact
                    Expr(:call, :(==), mangledtiled, tiledloop.stophint)
                else
                    Expr(:call, :(==), mangledtiled, tiledloop.stopsym)
                end
                qifelse = Expr(:if, comparison, Expr(:block)) # do nothing
                push!(q.args, qifelse)
                Tt = 0
            end
            Tt += 1 # we start counting up by 1
            if Tt == T # terminate on Tt = T
                Tt = 0
            end
            nothing
        end
        firstiter = false
    end
    q = gc_preserve(ls, q)
    reduce_expr!(q, ls, 4)
    lsexpr(ls, q)
end
function lower_unrolled(ls::LoopSet, vectorized::Symbol, U::Int)
    order = ls.loop_order.loopnames
    unrolled = last(order)
    # W = VectorizationBase.pick_vector_width(ls, unrolled)
    W = ls.W
    typeT = ls.T
    setup_preamble!(ls, W, typeT, vectorized, unrolled, last(order), U)
    initunrolledcounter = startloop(getloop(ls, unrolled), unrolled === vectorized, W, unrolled)
    q = lower_unrolled!(Expr(:block, initunrolledcounter), ls, vectorized, U, -1, W, typeT, getloop(ls, unrolled))
    lsexpr(ls, q)
end



function maybeinline!(q, ls, istiled, prependinlineORorUnroll)
    if prependinlineORorUnroll == 1
        if !istiled | length(ls.outer_reductions) > 1
            pushfirst!(q.args, Expr(:meta, :inline))
        end
    elseif prependinlineORorUnroll == 2
        pushfirst!(q.args, Expr(:meta, :inline))
    elseif prependinlineORorUnroll == -1
        pushfirst!(q.args, Expr(:meta, :noinline))
    end
    q
end
# Here, we have to figure out how to convert the loopset into a vectorized expression.
# This must traverse in a parent -> child pattern
# but order is also dependent on which loop inds they depend on.
# Requires sorting
# values for prependinlineORorUnroll:
# -1 : force @noinline
# 0 : nothing
# 1 : inline if length(ls.outer_reductions) > 1
# 2 : force inline
function lower(ls::LoopSet, prependinlineORorUnroll = 0)
    order, vectorized, U, T = choose_order(ls)
    istiled = T != -1
    fillorder!(ls, order, istiled)
    q = istiled ? lower_tiled(ls, vectorized, U, T) : lower_unrolled(ls, vectorized, U)
    maybeinline!(q, ls, istiled, prependinlineORorUnroll)
end
function lower(ls::LoopSet, U, T, prependinlineORorUnroll = 0)
    num_loops(ls) == 1 && @assert T == -1
    order, vectorized, _U, _T = choose_order(ls)
    istiled = T != -1
    fillorder!(ls, order, istiled)
    q = istiled ? lower_tiled(ls, vectorized, Int(U), Int(T)) : lower_unrolled(ls, vectorized, Int(U))
    maybeinline!(q, ls, istiled, prependinlineORorUnroll)
end

Base.convert(::Type{Expr}, ls::LoopSet) = lower(ls)
Base.show(io::IO, ls::LoopSet) = println(io, lower(ls))

