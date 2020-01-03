
# function unitstride(op::Operation, sym::Symbol)
    # (first(op.symbolic_metadata) === sym) && (first(op.numerical_metadata) == 1)
# end
function append_deps!(ret, deps)
    if first(deps) === Symbol("##DISCONTIGUOUSSUBARRAY##")
        append!(ret.args, @view(deps[2:end]))
    else
        append!(ret.args, deps)
    end
    ret
end

function mem_offset(op::Operation)
    @assert accesses_memory(op) "Computing memory offset only makes sense for operations that access memory."
    append_deps!(Expr(:tuple), op.ref.ref)
end
function mem_offset(op::Operation, incr::Int, mul::Symbol)
    @assert accesses_memory(op) "Computing memory offset only makes sense for operations that access memory."
    ret = Expr(:tuple)
    deps = op.ref.ref
    if incr == 0
        append!(ret.args, deps)
    else
        dep = first(deps)
        push!(ret.args, Expr(:call, :+, dep, Expr(:call, lv(:valmul), mul, incr)))
         for n ∈ 2:length(deps)
            push!(ret.args, deps[n])
        end
    end
    ret
end
function mem_offset(op::Operation, incr::Int, mul::Symbol, unrolled::Symbol)
    @assert accesses_memory(op) "Computing memory offset only makes sense for operations that access memory."
    ret = Expr(:tuple)
    deps = op.ref.ref
    if incr == 0
        append_deps!(ret, deps)
    else
        for n ∈ 1:length(deps)
            dep = deps[n]
            n == 1 && dep === Symbol("##DISCONTIGUOUSSUBARRAY##") && continue
            if dep === unrolled
                push!(ret.args, Expr(:call, :+, dep, Expr(:call, lv(:valmul), mul, incr)))
            else
                push!(ret.args, dep)
            end
        end
    end
    ret
end

# function add_expr(q, incr)
#     if q.head === :call && q.args[2] === :+
#         qc = copy(q)
#         push!(qc.args, incr)
#         qc
#     else
#         Expr(:call, :+, q, incr)
#     end
# end
function pushscalarload!(q::Expr, op, var, u, U)
    ptr = refname(op)
    push!(q.args, Expr(:(=), Symbol("##", var), Expr(:call, lv(:load),  ptr, mem_offset(op))))    
end
function pushvectorload!(q::Expr, op, var, u, U, W)
    ptr = refname(op)
    instrcall = Expr(:call, lv(:vload), W, ptr, mem_offset(op, u, W))
    if mask !== nothing && u == U - 1
        push!(instrcall.args, mask)
    end
    push!(q.args, Expr(:(=), Symbol("##",var,:_,u), instrcall))
end
function lower_load_scalar!( 
    q::Expr, op::Operation, vectorized::Symbol, W::Symbol, unrolled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    loopdeps = loopdependencies(op)
    @assert vectorized ∉ loopdeps
    var = op.variable
    if suffix !== nothing
        var = Symbol(var, :_, suffix)
    end
    ptr = refname(op)
    if unrolled ∈ loopdeps
        for u ∈ 0:U-1
            push!(q.args, Expr(:(=), Symbol("##", var,:_, u), Expr(:call, lv(:load),  ptr, mem_offset(op, u))))
        end
    else
        push!(q.args, Expr(:(=), Symbol("##", var), Expr(:call, lv(:load),  ptr, mem_offset(op))))
    end
    nothing
end
function lower_load_unrolled!(
    q::Expr, op::Operation, vectorized::Symbol, W::Symbol, unrolled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    loopdeps = loopdependencies(op)
    @assert vectorized ∈ loopdeps
    var = op.variable
    if suffix !== nothing
        var = Symbol(var, :_, suffix)
    end
    ptr = refname(op)
    if first(loopdependencies(op)) === unrolled # vload
        for u ∈ 0:U-1
            instrcall = Expr(:call, lv(:vload), W, ptr, mem_offset(op, u, W))
            if mask !== nothing && u == U - 1
                push!(instrcall.args, mask)
            end
            push!(q.args, Expr(:(=), Symbol("##",var,:_,u), instrcall))
        end
    else
        sn = findfirst(x -> x === unrolled, loopdependencies(op))::Int
        ustrides = Expr(:call, lv(:vmul), Expr(:call, :stride, ptr, sn), Expr(:call, lv(:vrange), W))
        ustride = gensym(:ustride)
        push!(q.args, Expr(:(=), ustride, ustrides))
        for u ∈ 0:U-1
            instrcall = Expr(:call, lv(:gather), ptr, mem_offset(op, u, W, unrolled), ustride)
            if mask !== nothing && u == U - 1
                push!(instrcall.args, mask)
            end
            push!(q.args, Expr(:(=), Symbol("##",var,:_,u), instrcall))
        end
    end
    nothing
end

# TODO: this code should be rewritten to be more "orthogonal", so that we're just combining separate pieces.
# Using sentinel values (eg, T = -1 for non tiling) in part to avoid recompilation.
function lower_load!(
    q::Expr, op::Operation, vectorized::Symbol, W::Symbol, unrolled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    # @show op.instruction
    # @show unrolled, loopdependencies(op)
    if vectorized ∈ loopdependencies(op)
        lower_load_unrolled!(q, op, vectorized, W, unrolled, U, suffix, mask)
    else
        lower_load_scalar!(q, op, vectorized, W, unrolled, U, suffix, mask)
    end
end
function reduce_range!(q::Expr, toreduct::Symbol, instr::Symbol, Uh::Int, Uh2::Int)
    instr = lv(instr)
    for u ∈ 0:Uh-1
        tru = Symbol("##",toreduct,:_,u)
        push!(q.args, Expr(:(=), tru, Expr(:call, instr, tru, Symbol("##",toreduct,:_,u + Uh))))
    end
    for u ∈ 2Uh:Uh2-1
        tru = Symbol("##",toreduct,:_, u + 1 - 2Uh)
        push!(q.args, Expr(:(=), tru, Expr(:call, instr, tru, Symbol("##",toreduct,:_,u))))
    end
end
function reduce_range!(q::Expr, ls::LoopSet, Ulow::Int, Uhigh::Int)
    for or ∈ ls.outer_reductions
        op = ls.operations[or]
        var = op.variable
        temp = gensym(var)
        instr = op.instruction
        instr = get(REDUCTION_TRANSLATION, instr, instr)
        reduce_range!(q, var, instr, Ulow, Uhigh)
    end
end

function reduce_expr!(q::Expr, toreduct::Symbol, instr::Symbol, U::Int)
    U == 1 && return nothing
    instr = get(REDUCTION_TRANSLATION, instr, instr)
    Uh2 = U
    iter = 0
    while true # combine vectors
        Uh = Uh2 >> 1
        reduce_range!(q, toreduct, instr, Uh, Uh2)
        Uh == 1 && break
        # @show Uh
        Uh2 = Uh
        iter += 1; iter > 5 && throw("Oops! This seems to be excessive unrolling.")
    end
    # reduce last vector
    # push!(q.args, Expr(:(=), assignto, Expr(:call, reductfunc, Symbol(toreduct,:_0))))
    nothing
 end

function lower_store_reduction!(
    q::Expr, op::Operation, vectorized::Symbol, W::Symbol, unrolled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    loopdeps = loopdependencies(op)
    # @assert unrolled ∉ loopdeps
    var = first(parents(op)).variable
    if suffix !== nothing
        var = Symbol(var, :_, suffix)
    end
    ptr = refname(op)
    # need to find out reduction type
    instr = first(parents(op)).instruction
    reduce_expr!(q, var, instr, U) # assigns reduction to storevar
    reducedname = Symbol("##", var, :_0)
    storevar = Expr(:call, lv(CORRESPONDING_REDUCTION[instr]), reducedname)
    push!(q.args, Expr(:call, lv(:store!), ptr, storevar, mem_offset(op))) # store storevar
    nothing
end
function lower_store_scalar!(
    q::Expr, op::Operation, vectorized::Symbol, W::Symbol, unrolled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    var = first(parents(op)).variable
    if suffix !== nothing
        var = Symbol(var, :_, suffix)
    end
    ptr = refname(op)
    push!(q.args, Expr(:call, lv(:store!), ptr, Symbol("##", var), mem_offset(op)))
    nothing
end
function lower_store_unrolled!(
    q::Expr, op::Operation, vectorized::Symbol, W::Symbol, unrolled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    loopdeps = loopdependencies(op)
    @assert unrolled ∈ loopdeps
    var = first(parents(op)).variable
    if suffix !== nothing
        var = Symbol(var, :_, suffix)
    end
    ptr = refname(op)
    if first(loopdependencies(op)) === unrolled # vstore!
        for u ∈ 0:U-1
            instrcall = Expr(:call,lv(:vstore!), ptr, Symbol("##",var,:_,u), mem_offset(op, u, W))
            if mask !== nothing && u == U - 1
                push!(instrcall.args, mask)
            end
            push!(q.args, instrcall)
        end
    else
        sn = findfirst(x -> x === unrolled, loopdependencies(op))::Int
        ustrides = Expr(:call, lv(:vmul), Expr(:call, :stride, ptr, sn), Expr(:call, lv(:vrange), W))
        for u ∈ 0:U-1
            instrcall = Expr(:call, lv(:scatter!), ptr, mem_offset(op,u,W,unrolled), ustrides, Symbol("##",var,:_,u))
            if mask !== nothing && u == U - 1
                push!(instrcall.args, mask)
            end
            push!(q.args, instrcall)
        end
    end
    nothing
end
function lower_store!(
    q::Expr, op::Operation, vectorized::Symbol, W::Symbol, unrolled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    if unrolled ∈ reduceddependencies(op)
        lower_store_reduction!(q, op, W, unrolled, U, suffix, mask)
    elseif unrolled ∈ loopdependencies(op)
        lower_store_unrolled!(q, op, W, unrolled, U, suffix, mask)
    else
        lower_store_scalar!(q, op, W, unrolled, U, suffix, mask)
    end
end
# A compute op needs to know the unrolling and tiling status of each of its parents.
#
function lower_compute_scalar!(
    q::Expr, op::Operation, vectorized::Symbol, W::Symbol, unrolled::Symbol, tiled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    lower_compute!(q, op, W, unrolled, tiled, U, suffix, mask, false)
end
function lower_compute_unrolled!(
    q::Expr, op::Operation, vectorized::Symbol, W::Symbol, unrolled::Symbol, tiled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    lower_compute!(q, op, W, unrolled, tiled, U, suffix, mask, true)
end
function lower_compute!(
    q::Expr, op::Operation, vectorized::Symbol, W::Symbol, unrolled::Symbol, tiled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing,
    opunrolled = unrolled ∈ loopdependencies(op)
)

    var = op.variable
    parents_op = parents(op)
    nparents = length(parents_op)
    if opunrolled
        parentsunrolled = Vector{Bool}(undef, nparents)
        for (p,opp) ∈ enumerate(parents_op)
            # if op is an inner reduction, one of its parents will be the initialization of op
            # They will share the same `variable` field. The initialization may not have
            # unrolled in its loop dependencies, but (if opunrolled) op itself is, so we return true
            parentsunrolled[p] = var === opp.variable ? true : (unrolled ∈ loopdependencies(opp))
        end
    else # maybe skip allocating this?
        parentsunrolled = fill(false, nparents)
    end
    parentstiled = if suffix === nothing
        optiled = false
        tiledouterreduction = -1
        fill(false, nparents)
    else
        tiledouterreduction = isouterreduction(op)
        if tiledouterreduction == -1
            var = Symbol(var, :_, suffix)
        end
        optiled = true
        [tiled ∈ loopdependencies(opp) for opp ∈ parents_op]
    end
    instr = op.instruction
    # cache unroll and tiling check of parents
    # not broadcasted, because we use frequent checks of individual bools
    # making BitArrays inefficient.
    # @show instr parentsunrolled
    # parentsyms = [opp.variable for opp ∈ parents(op)]
    Uiter = opunrolled ? U - 1 : 0
    maskreduct = mask !== nothing && isreduction(op) && any(opp -> opp.variable === var, parents_op)
    # if a parent is not unrolled, the compiler should handle broadcasting CSE.
    # because unrolled/tiled parents result in an unrolled/tiled dependendency,
    # we handle both the tiled and untiled case here.
    # bajillion branches that go the same way on each iteration
    # but smaller function is probably worthwhile. Compiler could theoreically split anyway
    # but I suspect that the branches are so cheap compared to the cost of everything else going on
    # that smaller size is more advantageous.
    modsuffix = 0
    for u ∈ 0:Uiter
        instrcall = callfun(instr) # Expr(:call, instr)
        varsym = if tiledouterreduction > 0 # then suffix !== nothing
            modsuff = ((u+suffix*U) & 3)
            Symbol("##",var,:_, modsuffix)
        elseif opunrolled
            Symbol("##",var,:_,u)
        else
            Symbol("##",var)
        end
        for n ∈ 1:nparents
            parent = parents_op[n].variable
            if n == tiledouterreduction
                parent = Symbol(parent,:_,modsuffix)
            else
                if parentstiled[n]
                    parent = Symbol(parent,:_,suffix)
                end
                if parentsunrolled[n]
                    parent = Symbol(parent,:_,u)
                end
            end
            push!(instrcall.args, Symbol("##", parent))
        end
        if maskreduct && u == Uiter # only mask last
            push!(q.args, Expr(:(=), varsym, Expr(:call, lv(:vifelse), mask, instrcall, varsym)))
        else
            push!(q.args, Expr(:(=), varsym, instrcall))
        end
    end
end
function lower!(
    q::Expr, op::Operation, vectorized::Symbol, W::Symbol, unrolled::Symbol, tiled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    if isload(op)
        lower_load!(q, op, vectorized, W, unrolled, U, suffix, mask)
    elseif isstore(op)
        lower_store!(q, op, vectorized, W, unrolled, tiled, U, suffix, mask)
    elseif iscompute(op)
        lower_compute!(q, op, vectorized, W, unrolled, U, suffix, mask)
    else
        lower_constant!(q, op, vectorized, W, unrolled, U, suffix, mask)
    end
end
function lower_constant!(
    q::Expr, op::Operation, vectorized::Symbol, W::Symbol, unrolled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Any = nothing
)
    @unpack variable, instruction = op
    if suffix !== nothing
        variable = Symbol(variable, :_, suffix)
    end
    # store parent's reduction deps
    # @show op.instruction, loopdependencies(op), reduceddependencies(op), unrolled, unrolled ∈ loopdependencies(op)
    if unrolled ∈ loopdependencies(op) || unrolled ∈ reduceddependencies(op)
        call = Expr(:call, lv(:vbroadcast), W, instruction)
        for u ∈ 0:U-1
            push!(q.args, Expr(:(=), Symbol("##", variable, :_, u), call))
        end
    else
        for u ∈ 0:U-1
            push!(q.args, Expr(:(=), Symbol("##", variable, :_, u), instruction))
        end
    end
    nothing
end
function lower!(
    q::Expr, ops::AbstractVector{Operation}, vectorized::Symbol, W::Symbol, unrolled::Symbol, tiled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    foreach(op -> lower!(q, op, vectorized, W, unrolled, tiled, U, suffix, mask), ops)
end
function lower_load!(
    q::Expr, ops::AbstractVector{Operation}, vectorized::Symbol, W::Symbol, unrolled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    foreach(op -> lower_load!(q, op, vectorized, W, unrolled, U, suffix, mask), ops)
end
function lower_compute!(
    q::Expr, ops::AbstractVector{Operation}, vectorized::Symbol, W::Symbol, unrolled::Symbol, tiled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    foreach(op -> lower_compute!(q, op, vectorized, W, unrolled, tiled::Symbol, U, suffix, mask), ops)
end
function lower_store!(
    q::Expr, ops::AbstractVector{Operation}, vectorized::Symbol, W::Symbol, unrolled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    foreach(op -> lower_store!(q, op, vectorized, W, unrolled, U, suffix, mask), ops)
end
function lower_constant!(
    q::Expr, ops::AbstractVector{Operation}, vectorized::Symbol, W::Symbol, unrolled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    foreach(op -> lower_constant!(q, op, vectorized, W, unrolled, U, suffix, mask), ops)
end

function lower!(
    q::Expr, ops::AbstractVector{<:AbstractVector{Operation}}, vectorized::Symbol, W::Symbol, unrolled::Symbol, tiled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    @assert length(ops) == 4
    @inbounds begin
        foreach(op -> lower_constant!(q, op, vectorized, W, unrolled, U, suffix, mask), ops[1])
        foreach(op -> lower_load!(q, op, vectorized, W, unrolled, U, suffix, mask), ops[2])
        foreach(op -> lower_compute!(q, op, vectorized, W, unrolled, tiled, U, suffix, mask), ops[3])
        foreach(op -> lower_store!(q, op, vectorized, W, unrolled, U, suffix, mask), ops[4])
    end
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
    # @show unrolled, order
    blockq = Expr(:block)
    n == 1 || push!(blockq.args, Expr(:(=), order[n-1], loopstart))
    loopq = if exprtype === :block
        blockq
    else
        @assert exprtype === :while || exprtype === :if "Expression type $exprtype not recognized."
        # if we have a mask, we use 1 as the incremement to pass to looprange, so that we get a iter < maxiter condition.
        lr = if mask !== nothing # on mask, we do evaluate the rest of the loop, so decrement only 1
            looprange(ls, loopsym, 1)
        elseif nisvectorized
            vec_looprange(ls, iter, isunrolled, W, U) # may not be tiled
        else
            looprange(ls, loopsym, nisunrolled ? U : (nistiled ? T : 1))
        end
        Expr(exprtype, lr, blockq)
    end
    for prepost ∈ 1:2
        # !U && !T
        lower!(blockq, @view(ops[:,1,1,prepost,n]), vectorized, W, unrolled, last(order), U, nothing, mask)
        # for u ∈ 0:U-1     #  U && !T
        lower!(blockq, @view(ops[:,2,1,prepost,n]), vectorized, W, unrolled, last(order), U, nothing, mask)
        # end
        if sum(length, @view(ops[:,:,2,prepost,n])) > 0
            for t ∈ 0:T-1
                if t == 0
                    push!(blockq.args, Expr(:(=), last(order), tiledsym(last(order))))
                else
                    push!(blockq.args, Expr(:+=, last(order), 1))
                end
                # !U &&  T
                lower!(blockq, @view(ops[:,1,2,prepost,n]), vectorized, W, unrolled, last(order), U, t, mask)
                # for u ∈ 0:U-1 #  U &&  T
                lower!(blockq, @view(ops[:,2,2,prepost,n]), vectorized, W, unrolled, last(order), U, t, mask)
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
            push!(blockq.args, Expr(:=, loopsym, Expr(:call, lv(:valadd), W, loopsym)))
        end
    else
        push!(blockq.args, Expr(:+=, loopsym, nisunrolled ? U : (nistiled ? T : 1)))
    end
    loopq
end
# Calculates nested loop set,
# if tiled, it will not lower the tiled iteration.
function add_vec_rem_iter(
    ls::LoopSet, n::Int, vectorized::Symbol, U::Int, T::Int, loopqold, loopstart, W, Uexprtype, order
)
    loopq = lower_nest(ls, n, vectorized, U, T, loopqold, loopstart, W, nothing, Uexprtype)
    if order[n] === vectorized
        vecloop = ls.loops[vectorized]
        comparison = if vecloop.hintexact
            Expr(:call, :(!=), vectorized, vecloop.rangehint)
        else
            Expr(:call, :(!=), vectorized, vecloop.rangesym)
        end
        loopq = Expr(
            :block, loopq,
            lower_nest(ls, n, vectorized, U, T, loopqold, loopstart, W, Symbol("##mask##"), :if)
        )
    end
    loopq
end
function lower_set(ls::LoopSet, vectorized::Symbol, U::Int, T::Int, W::Symbol, ::Nothing, Uexprtype::Symbol)
    # @show U, T, W
    istiled = T != -1
    order = names(ls)
    unrolled = order[end - istiled]
    unrolled === vectorized && return lower_set_unrolled_is_vectorized(ls, vectorized, U, T, W, nothing, Uexprtype)
    nl = num_loops(ls) - istiled
    loopq = add_vec_rem_iter( ls, 1, vectorized, U, T, nothing, 0, W, nl == 1 ? Uexprtype : :while, order )
    for n ∈ 2:nl
        exprtype = n == nl ? Uexprtype : :while
        loopq = add_vec_rem_iter( ls, n, vectorized, U, T, loopq, 0, W, exprtype, order )
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
    q::Expr, op::Operation, Umin::Int, Umax::Int, W::Symbol, typeT::Symbol, unrolled::Symbol, suffix::Union{Symbol,Nothing} = nothing
)
    # T = op.elementbytes == 8 ? :Float64 : :Float32
    var = op.variable
    instr = op.instruction # maybe just replace op instead?
    z = Expr(:call, REDUCTION_ZERO[instr], typeT)
    if unrolled ∈ reduceddependencies(op)
        z = Expr(:call, lv(:vbroadcast), W, z)
    end
    if suffix !== nothing
        var = Symbol(var, :_, suffix)
    end
    for u ∈ Umin:Umax-1
        push!(q.args, Expr(:(=), Symbol("##", var, :_, u), z))
    end
    nothing
end
function initialize_outer_reductions!(q::Expr, ls::LoopSet, Umin::Int, Umax::Int, W::Symbol, typeT::Symbol, unrolled::Symbol, suffix::Union{Symbol,Nothing} = nothing)
    foreach(or -> initialize_outer_reductions!(q, ls.operations[or], Umin, Umax, W, typeT, unrolled, suffix), ls.outer_reductions)
end
function initialize_outer_reductions!(ls::LoopSet, Umin::Int, Umax::Int, W::Symbol, typeT::Symbol, unrolled::Symbol, suffix::Union{Symbol,Nothing} = nothing)
    initialize_outer_reductions!(ls.preamble, ls, Umin, Umax, W, typeT, unrolled, suffix)
end
function add_upper_outer_reductions(ls::LoopSet, loopq::Expr, Ulow::Int, Uhigh::Int, W::Symbol, typeT::Symbol, unrolledloop::Loop)
    unrolled = unrolledloop.itersymbol
    ifq = Expr(:block)
    initialize_outer_reductions!(ifq, ls, Ulow, Uhigh, W, typeT, unrolled)
    push!(ifq.args, loopq)
    reduce_range!(ifq, ls, Ulow, Uhigh)
    comparison = Expr(:call, :!, Expr(:call, :<, unrolledloop.rangesym, Expr(:call, lv(:valmul), W, Uhigh)))
    Expr(:if, comparison, ifq)
end
function reduce_expr!(q::Expr, ls::LoopSet, U::Int)
    for or ∈ ls.outer_reductions
        op = ls.operations[or]
        var = op.variable
        instr = op.instruction
        reduce_expr!(q, var, instr, U)
        push!(q.args, Expr(:(=), var, Expr(:call, REDUCTION_SCALAR_COMBINE[instr], var, Symbol("##", var, :_0))))
    end
end
function gc_preserve(ls::LoopSet, q::Expr)
    length(ls.includedarrays) == 0 && return q # is this even possible?
    gcp = Expr(:macrocall, Expr(:(.), :GC, QuoteNode(Symbol("@preserve"))), LineNumberNode(@__LINE__, @__FILE__))
    for (array,_) ∈ ls.includedarrays
        push!(gcp.args, array)
    end
    push!(q.args, nothing)
    push!(gcp.args, q)
    Expr(:block, gcp)
end
function determine_eltype(ls::LoopSet)
    # length(ls.includedarrays) == 0 && return REGISTER_SIZE >>> 3
    if length(ls.includedarrays) == 1
        return Expr(:call, :eltype, first(first(ls.includedarrays)))
    end
    promote_q = Expr(:call, :promote_type)
    for (array,_) ∈ ls.includedarrays
        push!(promote_q.args, Expr(:call, :eltype, array))
    end
    promote_q
end
function determine_width(ls::LoopSet, typeT::Symbol, unrolled::Symbol)
    unrolledloop = ls.loops[unrolled]
    if unrolledloop.hintexact
        Expr(:call, lv(:pick_vector_width_val), Expr(:call, Expr(:curly, :Val, unrolledloop.rangehint)), typeT)
    else
        Expr(:call, lv(:pick_vector_width_val), typeT)
    end
end
function lower_unrolled!(
    q::Expr, ls::LoopSet, vectorized::Symbol, U::Int, T::Int, W::Symbol, typeT::Symbol, unrolledloop::Loop
)
    # q = if unrolledloop.hintexact
        # Ureduct = U
        # lower_unrolled_static!( q, ls, U, T, W, typeT, unrolledloop )
    # else
    Ureduct = min(U,4)
    q = lower_unrolled_dynamic!( q, ls, vectorized, U, T, W, typeT, unrolledloop )
    # end
    if T == -1
        q = gc_preserve(ls, q)
    end
    manageouterreductions = T == -1 && length(ls.outer_reductions) > 0
    manageouterreductions && reduce_expr!(q, ls, Ureduct)
    q
end
#=
function lower_unrolled_static!(
    q::Expr, ls::LoopSet, U::Int, T::Int, W::Int, unrolledloop::Loop
)
    unrolled = unrolledloop.itersymbol
    unrolled_numiter = unrolledloop.rangehint
    Urem = unrolled_numiter
    # if static, we use Urem to indicate remainder.
    if unrolled_numiter ≥ 2U*W # we need at least 2 iterations
        Uexprtype = :while
    elseif unrolled_numiter ≥ U*W # complete unroll
        Uexprtype = :block
    else# we have only a single block
        Uexprtype = :skip
    end
    manageouterreductions = T == -1 && length(ls.outer_reductions) > 0
    if manageouterreductions
        # Umax = (!static_unroll && U > 2) ? U >> 1 : U
        Ureduct = U
        initialize_outer_reductions!(q, ls, 0, Ureduct, W, last(names(ls)))
    else
        Ureduct = -1
    end
    Wt = W
    Ut = U
    while true
        if Uexprtype !== :skip
            loopq = if Urem == unrolled_numiter || Urem == -1 # static, no mask
                lower_set(ls, Ut, T, Wt, nothing, Uexprtype)
            else # static, need mask
                lower_set(ls, Ut, T, Wt, VectorizationBase.unstable_mask(Wt, Urem), Uexprtype)
            end
            push!(q.args, loopq)
        end
        if Urem == unrolled_numiter
            remUiter = unrolled_numiter % (U*W)
            if remUiter == 0 # no remainder, we're done with the unroll
                break
            else # remainder, requires another iteration; what size?
                Ut, Urem = divrem(remUiter, W)
                if Urem == 0 # Ut iters of W
                    Urem = -1 
                else
                    if Ut == 0 # if Urem == unrolled_numiter, we may already be done, othererwise, we may be able to shrink Wt
                        if Urem == unrolled_numiter && Uexprtype !== :skip
                            break
                        else
                            Wt = VectorizationBase.nextpow2(Urem)
                            if Wt == Urem # no mask needed
                                Urem = -1
                            end
                        end
                    end
                    # because initial Urem > 0 (it either still is, or we shrunk Wt and made it a complete iter)
                    # we must increment Ut (to perform masked or shrunk complete iter)
                    Ut += 1
                end
                Uexprtype = :block
            end
        else
            break
        end
    end
    q
end
=#
function lower_unrolled_dynamic!(
    q::Expr, ls::LoopSet, vectorized::Symbol, U::Int, T::Int, W::Symbol, typeT::Symbol, unrolledloop::Loop
)
    unrolled = unrolledloop.itersymbol
    unrolled_numitersym = unrolledloop.rangesym
    Urem = 0
    Uexprtype = :while
    manageouterreductions = T == -1 && length(ls.outer_reductions) > 0
    if manageouterreductions
        # Umax = (!static_unroll && U > 2) ? U >> 1 : U
        Ureduct = U > 6 ? 4 : U
        initialize_outer_reductions!(q, ls, 0, Ureduct, W, typeT, last(names(ls)))
    else
        Ureduct = -1
    end
    Ut = U
    local remblock::Expr
    firstiter = true
    while true
        if firstiter # first iter
            loopq = lower_set(ls, vectorized, Ut, T, W, nothing, Uexprtype)
            if T == -1 && manageouterreductions && U > 4
                loopq = add_upper_outer_reductions(ls, loopq, Ureduct, U, W, typeT, unrolledloop)
            end
            push!(q.args, loopq)
        elseif U == 1 #
            if unrolled === vectorized
                push!(remblock.args, lower_set(ls, vectorized, Ut, T, W, Symbol("##mask##"), :block))
            else
                push!(remblock.args, lower_set(ls, vectorized, Ut, T, W, nothing, :block))
            end
        else
            remblocknew = if unrolled === vectorized
                comparison = Expr(:call, :>, unrolled, Expr(:call, :-, unrolled_numitersym, Expr(:call, lv(:valmuladd), W, Ut, 1)))
                Expr(Ut == 1 ? :if : :elseif, comparison, lower_set(ls, vectorized, Ut, T, W, Symbol("##mask##"), :block))
            else
                comparison = Expr(:call, :>, unrolled, Expr(:call, :-, unrolled_numitersym, Ut + 1))
                Expr(Ut == 1 ? :if : :elseif, comparison, lower_set(ls, vectorized, Ut, T, W, nothing, :block))
            end
            push!(remblock.args, remblocknew)
            remblock = remblocknew
        end
        if Ut == U || Ut == Ureduct
            firstiter || break
            firstiter = false
            if manageouterreductions && Ureduct < U
                Udiff = U - Ureduct
                loopq = lower_set(ls, vectorized, Udiff, T, W, nothing, :if)
                push!(q.args, loopq)
            end
            Ut = 1
            # setup for branchy remainder calculation
            comparison = Expr(:call, :(!=), unrolled_numitersym, unrolled)
            remblock = Expr(:block)
            push!(q.args, Expr(:if, comparison, remblock))
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
function definemask(q::Expr, loop, W::Symbol, allon::Bool)
    if loop.hintexact
        maskexpr(W, loop.rangehint, allon)
    else
        maskexpr(W, loop.rangesym, allon)
    end
end
function setup_mainblock(ls::LoopSet, W::Symbol, typeT::Symbol, vectorized::Symbol, unrolled::Symbol, U::Int, q::Expr)
    preambleW = Expr(
        :block,
        Expr(:(=), typeT, determine_eltype(ls)),
        Expr(:(=), W, determine_width(ls, typeT, unrolled)),
        definemask(ls.loops[vectorized], W, U > 1 && unrolled !== vectorized)
    )
    Expr(:block, ls.preamble, preambleW, q)
end
function lower_tiled(ls::LoopSet, vectorized::Symbol, U::Int, T::Int)
    order = ls.loop_order.loopnames
    tiled    = order[end]
    unrolled = order[end-1]
    mangledtiled = tiledsym(tiled)
    W = gensym(:W)
    typeT = gensym(:T)
    # W = VectorizationBase.pick_vector_width(ls, unrolled)
    tiledloop = ls.loops[tiled]
    static_tile = tiledloop.hintexact
    unrolledloop = ls.loops[unrolled]
    initialize_outer_reductions!(ls, 0, 4, W, typeT, unrolled)
    q = Expr(:block, Expr(:(=), mangledtiled, 0))
    # we build up the loop expression.
    Trem = Tt = T
    nloops = num_loops(ls);
    # addtileonly = sum(length, @view(oporder(ls)[:,:,:,:,end])) > 0
    # Texprtype = (static_tile && tiled_iter < 2T) ? :block : :while
    firstiter = true
    mangledtiled = tiledsym(tiled)
    local qifelse::Expr
    while Tt > 0
        tiledloopbody = Expr(:block)
        lower_unrolled!(tiledloopbody, ls, vectorized, U, Tt, W, typeT, unrolledloop)
        tiledloopbody = lower_nest(ls, nloops, vectorized, U, Tt, tiledloopbody, 0, W, nothing, :block)
        if firstiter
            push!(q.args, (static_tile && tiled_iter < 2T) ? tiledloopbody : Expr(:while, looprange(ls, tiled, Tt, mangledtiled, tiledloop), tiledloopbody))
        elseif static_tile
            push!(q.args, tiledloopbody)
        else # not static, not firstiter
            comparison = Expr(:call, :(==), mangledtiled, Expr(:call, :-, tiledloop.rangesym, Tt))
            qifelsenew = Expr(:elseif, comparison, tiledloopbody)
            push!(qifelse.args, qifelsenew)
            qifelse = qifelsenew
        end
        if static_tile
            if Tt == T
                # push!(tiledloopbody.args, Expr(:+=, mangledtiled, Tt))
                Texprtype = :block
                Tt = looprangehint(ls, tiled) % T
                # Recalculate U
                U = solve_tilesize_constT(ls, Tt)
            else
                Tt = 0 # terminate
            end
            nothing
        else
            if firstiter
                comparison = Expr(:call, :(==), mangledtiled, tiledloop.rangesym)
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
    reduce_expr!(q, ls, U)
    setup_mainblock(ls, W, typeT, vectorized, unrolled, U, q)
end
function lower_unrolled(ls::LoopSet, vectorized::Symbol, U::Int)
    order = ls.loop_order.loopnames
    # @show order
    unrolled = last(order)
    # W = VectorizationBase.pick_vector_width(ls, unrolled)
    W = gensym(:W)
    typeT = gensym(:T)
    q = lower_unrolled!(Expr(:block, Expr(:(=), unrolled, 0)), ls, vectorized, U, -1, W, typeT, ls.loops[unrolled])
    setup_mainblock(ls, W, typeT, vectorized, unrolled, U, q)
end



# Here, we have to figure out how to convert the loopset into a vectorized expression.
# This must traverse in a parent -> child pattern
# but order is also dependent on which loop inds they depend on.
# Requires sorting 
function lower(ls::LoopSet)
    order, vectorized, U, T = choose_order(ls)
    # @show order, U, T
    # @show ls.loop_order.loopnames
    istiled = T != -1
    fillorder!(ls, order, istiled)
    # @show order, ls.loop_order.loopnames
    istiled ? lower_tiled(ls, vectorized, U, T) : lower_unrolled(ls, vectorized, U)
end

Base.convert(::Type{Expr}, ls::LoopSet) = lower(ls)
Base.show(io::IO, ls::LoopSet) = println(io, lower(ls))

