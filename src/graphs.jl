# using LightGraphs


isdense(::Type{<:DenseArray}) = true

# """
# ShortVector{T} simply wraps a Vector{T}, but uses a different hash function that is faster for short vectors to support using it as the keys of a Dict.
# This hash function scales O(N) with length of the vectors, so it is slow for long vectors.
# """
# struct ShortVector{T} <: DenseVector{T}
#     data::Vector{T}
# end
# Base.@propagate_inbounds Base.getindex(x::ShortVector, I...) = x.data[I...]
# Base.@propagate_inbounds Base.setindex!(x::ShortVector, v, I...) = x.data[I...] = v
# @inbounds Base.length(x::ShortVector) = length(x.data)
# @inbounds Base.size(x::ShortVector) = size(x.data)
# @inbounds Base.strides(x::ShortVector) = strides(x.data)
# @inbounds Base.push!(x::ShortVector, v) = push!(x.data, v)
# @inbounds Base.append!(x::ShortVector, v) = append!(x.data, v)
# function Base.hash(x::ShortVector, h::UInt)
#     @inbounds for n ∈ eachindex(x)
#         h = hash(x[n], h)
#     end
#     h
# end



@enum OperationType begin
    memload
    memstore
    compute_new
    compute_update
    # accumulator
end

# const ID = Threads.Atomic{UInt}(0)

# TODO: can some computations be cached in the operations?
"""
if ooperation_type == memstore || operation_type == memstore# || operation_type == compute_new || operation_type == compute_update
symbolic metadata contains info on direct dependencies / placement within loop.

if accesses_memory(op)
Symbol(:vptr_, op.variable)
is how we access the memory.
If numerical_metadata[i] == -1
Symbol(:stride_, op.variable, :_, op.symbolic_metadata[i])
is the stride for loop index
symbolic_metadata[i]
"""
struct Operation
    identifier::UInt
    variable::Symbol
    elementbytes::Int
    instruction::Symbol
    node_type::OperationType
    # dependencies::Vector{Symbol}
    dependencies::Set{Symbol}
    reduced_deps::Set{Symbol}
    # dependencies::Set{Symbol}
    parents::Vector{Operation}
    children::Vector{Operation}
    numerical_metadata::Vector{Int} # stride of -1 indicates dynamic
    symbolic_metadata::Vector{Symbol}
    # strides::Dict{Symbol,Union{Symbol,Int}}
    function Operation(
        identifier,
        elementbytes,
        instruction,
        node_type,
        variable = gensym()
    )
        new(
            identifier, variable, elementbytes, instruction, node_type,
            Set{Symbol}(), Operation[], Operation[], Int[], Symbol[]#, Dict{Symbol,Union{Symbol,Int}}()
        )
    end
end



function isreduction(op::Operation)
    (op.node_type == memstore) && (length(op.symbolic_metadata) < length(op.dependencies))# && issubset(op.symbolic_metadata, op.dependencies)
end
isload(op::Operation) = op.node_type == memload
isstore(op::Operation) = op.node_type == memstore
accesses_memory(op::Operation) = isload(op) | isstore(op)
elsize(op::Operation) = op.elementbytes
dependson(op::Operation, sym::Symbol) = sym ∈ op.dependencies
parents(op::Operation) = op.parents
children(op::Operation) = op.children
loopdependencies(op::Operation) = op.dependencies
reduceddependencies(op::Operation) = op.reduced_deps
identifier(op::Operation) = op.identifier
name(op::Operation) = op.variable
instruction(op::Operation) = op.instruction

function hasintersection(s1::Set{T}, s2::Set{T}) where {T}
    for x ∈ s1
        x ∈ s2 && return true
    end
    false
end

function symposition(op::Operation, sym::Symbol)
    findfirst(s -> s === sym, op.symbolic_metadata)
end
function stride(op::Operation, sym::Symbol)
    @assert accesses_memory(op) "This operation does not access memory!"
    # access stride info?
    op.numerical_metadata[symposition(op,sym)]
end
# function
function unitstride(op::Operation, sym::Symbol)
    (first(op.symbolic_metadata) === sym) && (first(op.numerical_metadata) == 1)
end
function mem_offset(op::Operation, incr::Int = 0)::Union{Symbol,Expr}
    @assert accesses_memory(op) "Computing memory offset only makes sense for operations that access memory."
    @unpack numerical_metadata, symbolic_metadata = op
    if incr == 0 && length(numerical_metadata) == 1
        firstsym = first(symbolic_metadata)
        if first(numerical_metadata) == 1
            return firstsym
        elseif first(numerical_metadata) == -1
            return Expr(:call, :*,  Symbol(:stride_, op.variable, :_, firstsym), firstsym)
        else
            return Expr(:call, :*,  first(numerical_metadata), firstsym)
        end
    end
    ret = Expr(:call, :+, )
    for i ∈ eachindex(numerical_metadata)
        sym = symbolic_metadata[i]; num = numerical_metadata[i]
        if num == 1
            push!(ret.args, sym)
        elseif num == -1
            push!(ret.args, Expr(:call, :*, Symbol(:stride_, op.variable, :_, firstsym), sym))
        else
            push!(ret.args, Expr(:call, :*, num, sym))
        end        
    end
    incr == 0 || push!(ret.args, incr)
    ret
end


struct Loop
    itersymbol::Symbol
    rangehint::Int
    rangesym::Symbol
    hintexact::Bool # if true, rangesym ignored and rangehint used for final lowering
end
function Loop(itersymbol::Symbol, rangehint::Int)
    Loop( itersymbol, rangehint, Symbol("##UNDEFINED##"), true )
end
function Loop(itersymbol::Symbol, rangesym::Symbol, rangehint::Int = 1_024)
    Loop( itersymbol, rangehint, rangesym, false )
end

# load/compute/store × isunroled × istiled × pre/post loop × Loop number
struct LoopOrder <: AbstractArray{Vector{Operation},5}
    oporder::Array{Vector{Operation},5}
    loopnames::Vector{Symbol}
end
function LoopOrder(N::Int)
    LoopOrder( [ Operation[] for i ∈ 1:3, j ∈ 1:2, k ∈ 1:2, l ∈ 1:2, n ∈ 1:N ], Vector{Symbol}(undef, N) )
end
Base.empty!(lo::LoopOrder) = foreach(empty!, lo.oporder)
Base.size(lo::LoopOrder) = (3,2,2,2,size(lo.oporder,5))
Base.@propagate_inbounds Base.getindex(lo::LoopOrder, i...) = lo.oporder[i...]

# Must make it easy to iterate
struct LoopSet
    loops::Dict{Symbol,Loop} # sym === loops[sym].itersymbol
    # operations::Vector{Operation}
    loadops::Vector{Operation} # Split them to make it easier to iterate over just a subset
    computeops::Vector{Operation}
    storeops::Vector{Operation}
    inner_reductions::Set{UInt} # IDs of reduction operations nested within loops and stored.
    outer_reductions::Set{UInt} # IDs of reduction operations that need to be reduced at end.
    loop_order::LoopOrder
    # strideset::Vector{} 
end
num_loops(ls::LoopSet) = length(ls.loops)
isstaticloop(ls::LoopSet, s::Symbol) = ls.loops[s].hintexact
itersyms(ls::LoopSet) = keys(ls.loops)
function looprange(ls::LoopSet, s::Symbol)
    loop = ls.loops[s]
    Expr(:call, :<, s, loop.hintexact ? loop.rangehint : loop.rangesym)
end
function Base.length(ls::LoopSet, is::Symbol)
    ls.loops[is].rangehint
end
load_operations(ls::LoopSet) = ls.loadops
compute_operations(ls::LoopSet) = ls.computeops
store_operations(ls::LoopSet) = ls.storeops
function operations(ls::LoopSet)
    Base.Iterators.flatten((
        load_operations(ls),
        compute_operations(ls),
        store_operations(ls)
    ))
end

function fillorder!(ls::LoopSet, order::Vector{Symbol})
    lo = ls.loop_order
    copyto!(lo.names, order)
    empty!(lo)
    nloops = length(order)
    unrolled = first(order)
    tiled = nloops > 1 ? order[2] : Symbol("##UNDEFINED##")
    included_vars = fill(false, length(operations(ls)))
    # to go inside out, we just have to include all those not-yet included depending on the current sym
    for _n ∈ 1:nloops
        n = 1 + nloops - _n
        loopsym = order[n]
        for (id,op) ∈ enumerate(operations(ls))
            included_vars[id] && continue
            loopsym ∈ dependencies(op) || continue
            included_vars[id] = true
            isunrolled = (unrolled ∈ loopdependencies(op)) + 1
            istiled = (nloops == 1 ? false : (tiled ∈ loopdependencies(op))) + 1
            optype = if isload(op)
                1
            elseif isstore(op)
                3
            else#if compute
                2
            end
            after_loop = (length(reduceddependencies(op)) > 0) + 1
            push!(lo[optype,isunrolled,istiled,after_loop,_n], op)
        end
    end    
end


function cost(op::Operation, unrolled::Symbol, Wshift::Int, size_T::Int)
    # Wshift == dependson(op, unrolled) ? Wshift : 0
    # c = first(cost(instruction(op), Wshift, size_T))::Int
    instr = instruction(op)
    opisunrolled = dependson(op, unrolled)
    srt, sl, srp = opisunrolled ? vector_cost(instr, Wshift, size_T) : scalar_cost(instr)
    if accesses_memory(op)
        # either vbroadcast/reductionstore, vmov(a/u)pd, or gather/scatter
        if opisunrolled
            if !unitstride(op, unrolled)# || !isdense(op) # need gather/scatter
                r = (1 << Wshift)
                c *= r
                sl *= r
            # else # vmov(a/u)pd
            end
        elseif instr === :setindex! # broadcast or reductionstore; if store we want to penalize reduction
            c *= 2
            sl *= 2
        end
    end
    srt, sl, srp
end

    # Base._return_type()

function biggest_type_size(ls::LoopSet)
    maximum(elsize, operations(ls))
end
function VectorizationBase.pick_vector_width(ls::LoopSet, u::Symbol)
    VectorizationBase.pick_vector_width(length(ls, u), biggest_type_size(ls))
end
function VectorizationBase.pick_vector_width_shift(ls::LoopSet, u::Symbol)
    VectorizationBase.pick_vector_width_shift(length(ls, u), biggest_type_size(ls))
end


# evaluates cost of evaluating loop in given order
# heuristically, could simplify analysis by just unrolling outer loop?
function evaluate_cost_unroll(
    ls::LoopSet, order::Vector{Symbol}, max_cost = typemax(Float64), unrolled::Symbol = first(order)
)
    # included_vars = Set{UInt}()
    included_vars = fill(false, length(operations(ls)))
    nested_loop_syms = Set{Symbol}()
    total_cost = 0.0
    iter = 1.0
    # Need to check if fusion is possible
    size_T = biggest_type_size(ls)
    W, Wshift = VectorizationBase.pick_vector_width_shift(length(ls, unrolled), size_T)::Tuple{Int,Int}
    for itersym ∈ order
        # Add to set of defined symbles
        push!(nested_loop_syms, itersym)
        liter = Float64(length(ls, itersym))
        if itersym === unrolled
            liter /= W
        end
        iter *= liter
        # check which vars we can define at this level of loop nest
        for op ∈ operations(ls)
            # won't define if already defined...
            id = identifier(op)
            included_vars[id] && continue
            # it must also be a subset of defined symbols
            loopdependencies(op) ⊆ nested_loop_syms || continue
            hasintersection(reduceddependencies(op), nested_loop_syms) && return Inf
            included_vars[id] = true
            
            total_cost += iter * first(cost(op, unrolled, Wshift, size_T))
            total_cost > max_cost && return total_cost # abort if more expensive; we only want to know the cheapest
        end
    end
    total_cost
end

# only covers unrolled ops; everything else considered lifted?
function depchain_cost!(
    skip::Vector{Bool}, op::Operation, unrolled::Symbol, Wshift::Int, size_T::Int, sl::Int = 0, rt::Float64 = 0.0
)
    skip[identifier(op)] = true
    # depth first search
    for opp ∈ parents(op)
        skip[identifier(opp)] && continue
        sl, rt = depchain_cost!(skip, opp, unrolled, Wshift, size_T, sl, rt)
    end
    # Basically assuming memory and compute don't conflict, but everything else does
    # Ie, ignoring the fact that integer and floating point operations likely don't either
    if accesses_memory(op)
        return sl, rt
    end
    slᵢ, rtᵢ = cost(op, 1 << Wshift, Wshift, unrolled)
    sl + slᵢ, rt + rtᵢ
end
   
function determine_unroll_factor(
    ls::LoopSet, order::Vector{Symbol}, unrolled::Symbol = first(order)
)
    size_T = biggest_type_size(ls)
    W, Wshift = VectorizationBase.pick_vector_width_shift(length(ls, unrolled), size_T)::Tuple{Int,Int}

    # The strategy is to use an unroll factor of 1, unless there appears to be loop carried dependencies (ie, num_reductions > 0)
    # The assumption here is that unrolling provides no real benefit, unless it is needed to enable OOO execution by breaking up these dependency chains
    num_reductions = sum(isreduction, operations(ls))
    iszero(num_reductions) && return 1
    # So if num_reductions > 0, we set the unroll factor to be high enough so that the CPU can be kept busy
    # if there are, U = max(1, round(Int, max(latency) * throughput / num_reductions)) = max(1, round(Int, latency / (recip_throughput * num_reductions)))
    # We also make sure register pressure is not too high.
    latency = 0
    recip_throughput = 0.0
    visited_nodes = fill(false, length(operations(ls)))
    for op ∈ operations(ls)
        if isreduction(op) && dependson(op, unrolled)
            sl, rt = depchain_cost!(visited_nodes, instruction(op), unrolled, Wshift, size_T)
            latency = max(sl, latency)
            recip_throughput += rt
        end
    end
    max(1, round(Int, latency / (recip_throughput * num_reductions) ) )  
end

function tile_cost(X, U, T)
    X[1] + X[4] + X[2] / T + X[3] / U
end
function solve_tilesize(X, R)
    # We use lagrange multiplier to finding floating point values for U and T
    # first solving for U via quadratic formula
    RR = VectorizationBase.REGISTER_COUNT - R[3] - R[4]
    a = (R[1])^2*X[2] - (R[2])^2*R[1]*X[3]/RR
    b = 2*R[1]*R[2]*X[3]
    c = -RR*R[1]*X[3]
    Ufloat = (sqrt(b^2 - 4a*c) - b) / (2a)
    Tfloat = (RR - Ufloat*R[2])/(Ufloat*R[1])
    Ufloat, Tfloat
    
    Ulow = max(1, floor(Int, Ufloat)) # must be at least 1
    Uhigh = Ulow + 1 #ceil(Int, Ufloat)
    Tlow = max(1, floor(Int, Tfloat)) # must be at least 1
    Thigh = Tlow + 1 #ceil(Int, Tfloat)

    U, T = Ulow, Tlow
    tcost = tile_cost(X, Ulow, Tlow)
    if RR > Ulow*Thigh*R[1] + Ulow*R[2]
        tcost_temp = tile_cost(X, Ulow, Thigh)
        if tcost_temp < tcost
            tcost = tcost_temp
            U, T = Ulow, Thigh
        end
    end
    if RR > Uhigh*Tlow*R[1] + Uhigh*R[2]
        tcost_temp = tile_cost(X, Uhigh, Tlow)
        if tcost_temp < tcost
            tcost = tcost_temp
            U, T = Uhigh, Tlow
        end
    end
    if RR > Uhigh*Thigh*R[1] + Uhigh*R[2]
        throw("Something when wrong when solving for Tfloat and Ufloat.")
    end
    U, T, tcost
end


# Just tile outer two loops?
# But optimal order within tile must still be determined
# as well as size of the tiles.
function evaluate_cost_tile(
    ls::LoopSet, order::Vector{Symbol}
)
    N = length(order)
    @assert N ≥ 2 "Cannot tile merely $N loops!"
    tiled = order[1]
    unrolled = order[2]
    included_vars = fill(false, length(operations(ls)))
    nested_loop_syms = Set{Symbol}()
    iter = 1.0
    # Need to check if fusion is possible
    size_T = biggest_type_size(ls)
    W, Wshift = VectorizationBase.pick_vector_width_shift(length(ls, unrolled), size_T)::Tuple{Int,Int}
    # costs = 
    # cost_mat[1] / ( unrolled * tiled)
    # cost_mat[2] / ( tiled)
    # cost_mat[3] / ( unrolled)
    # cost_mat[4] 
    cost_vec = zeros(Float64, 4)
    reg_pressure = zeros(Int, 4)
    for n ∈ 1:N
        itersym = order[n]
        # Add to set of defined symbles
        push!(nested_loop_syms, itersym)
        liter = Float64(length(ls, itersym))
        if n == 2 # unrolled
            liter /= W
        # elseif n == 1 # tiled
            # liter
        end
        iter *= liter
        # check which vars we can define at this level of loop nest
        for (id, op) ∈ enumerate(operations(ls))
            @assert id == identifier(op) # testing, for now
            # won't define if already defined...
            included_vars[id] && continue
            # it must also be a subset of defined symbols
            loopdependencies(op) ⊆ nested_loop_syms || continue
            hasintersection(reduceddependencies(op), nested_loop_syms) && return 0,0,Inf
            included_vars[id] = true
            rt, lat, rp = cost(op, unrolled, Wshift, size_T)
            rt *= iter
            isunrolled = unrolled ∈ loopdependencies(op)
            istiled = tiled ∈ loopdependencies(op)
            if isunrolled && istiled # no cost decrease; cost must be repeated
                cost_vec[1] = rt
                reg_pressure[1] = rp
            elseif isunrolled # cost decreased by tiling
                cost_vec[2] = rt
                reg_pressure[2] = rp
            elseif istiled # cost decreased by unrolling
                cost_vec[3] = rt
                reg_pressure[3] = rp
            else# neither unrolled or tiled
                cost_vec[4] = rt
                reg_pressure[4] = rp
            end
        end
    end
    solve_tilesize(cost_vec, reg_pressure)
end


struct LoopOrders
    syms::Vector{Symbol}
    buff::Vector{Symbol}
end
function LoopOrders(ls::LoopSet)
    syms = [s for s ∈ keys(ls.loops)]
    LoopOrders(syms, similar(buff))
end
function Base.iterate(lo::LoopOrders)
    lo.syms, zeros(Int, length(lo.syms))# - 1)
end

function swap!(x, i, j)
    xᵢ, xⱼ = x[i], x[j]
    x[j], x[i] = xᵢ, xⱼ
end
function advance_state!(state)
    N = length(state)
    for n ∈ 1:N
        sₙ = state[n]
        if sₙ == N - n
            if n == N
                return false
            else
                state[n] = 0
            end
        else
            state[n] = sₙ + 1
            break
        end
    end
    true
end
# I doubt this is the most efficient algorithm, but it's the simplest thing
# that I could come up with.
function Base.iterate(lo::LoopOrders, state)
    advance_state!(state) || return nothing
    # @show state
    syms = copy!(lo.buff, lo.syms)
    for i ∈ eachindex(state)
        sᵢ = state[i]
        sᵢ == 0 || swap!(syms, i, i + sᵢ)
    end
    syms, state
end
function choose_unroll_order(ls::LoopSet, lowest_cost::Float64 = Inf)
    lo = LoopOrders(ls)
    best_order = lo.syms
    new_order, state = iterate(lo) # right now, new_order === best_order
    while true
        cost_temp = evaluate_cost_unroll(ls, new_order, lowest_cost)
        if cost_temp < lowest_cost
            lowest_cost = cost_temp
            best_order = new_order
        end
        iter = iterate(lo, state)
        iter === nothing && return best_order, lowest_cost
        new_order, state = iter
    end    
end
function choose_tile(ls::LoopSet)
    lo = LoopOrders(ls)
    best_order = lo.syms
    new_order, state = iterate(lo) # right now, new_order === best_order
    U, T, lowest_cost = 0, 0, Inf
    while true
        U_temp, T_temp, cost_temp = evaluate_cost_tile(ls, new_order)
        if cost_temp < lowest_cost
            lowest_cost = cost_temp
            U, T = U_temp, T_temp
            best_order = new_order
        end
        iter = iterate(lo, state)
        iter === nothing && return best_order, U, T, lowest_cost
        new_order, state = iter
    end
end
function choose_order(ls::LoopSet)
    if num_loops(ls) > 1
        torder, tU, tT, tc = choose_tile(ls)
    else
        tc = Inf
    end
    uorder, uc = choose_unroll_order(ls, tc)
    if num_loops(ls) <= 1 || tc > uc # if tc == uc, then that probably means we want tc, and no unrolled managed to beat the tiled cost
        return uorder, determine_unroll_factor(ls, uorder), -1
    else
        return torder, tU, tT
    end
end

function depends_on_assigned(op::Operation, assigned::Vector{Bool})
    for p ∈ parents(op)
        p === op && continue # don't fall into recursive loop when we have updates, eg a = a + b
        assigned[identifier(op)] && return true
        depends_on_assigned(p, assigned) && return true
    end
    false
end
# ind gets increased across tiles / unroll, so we need steps.
function replace_ind_in_offset!(offset::Vector, op::Operation, ind::Int, t)
    t == 0 && return nothing
    var = op.variable
    siter = op.symbolic_metadata[ind]
    striden = op.numerical_metadata[ind]
    strides = Symbol(:stride_, var)
    offset[ind] = if tstriden == -1
        Expr(:call, :*, Expr(:call, :+, strides, t), siter)
    else
        Expr(:call, :*, striden + t, siter)
    end
    nothing
end

function lower_load_scalar!(
    q::Expr, op::Operation, W::Int, unrolled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing
)

    loopdeps = loopdependencies(op)
    @assert unrolled ∉ loopdeps
    var = op.variable
    if suffix !== nothing
        var = Symbol(var, :_, suffix)
    end
    ptr = Symbol(:vptr_, var)
    memoff = mem_offset(op)
    push!(q.args, Expr(:(=), var, Expr(:call, :load,  ptr, memoff)))
    nothing
end
function lower_load_unrolled!(
    q::Expr, op::Operation, W::Int, unrolled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    loopdeps = loopdependencies(op)
    @assert unrolled ∈ loopdeps
    var = op.variable
    if suffix !== nothing
        var = Symbol(var, :_, suffix)
    end
    ptr = Symbol(:vptr_, var)
    memoff = mem_offset(op)
    upos = symposition(op, unrolled)
    ustride = op.numerical_metadata[upos]
    if ustride == 1 # vload
        if U == 1
            push!(q.args, Expr(:(=), var, Expr(:call,:vload,ptr,memoff)))
        else
            for u ∈ 0:U-1
                instrcall = Expr(:call,:vload, Val{W}(), ptr, u == 0 ? memoff : push!(copy(memoff), W*u))
                mask === nothing || push!(instrcall.args, mask)
                push!(q.args, Expr(:(=), Symbol(var,:_,u), instrcall))
            end
        end
    else
        # ustep = ustride > 1 ? ustride : op.symbolic_metadata[upos]
        ustrides = Expr(:tuple, (ustride > 1 ? [Core.VecElement{Int}(ustride*w) for w ∈ 0:W-1] : [:(Core.VecElement{Int}($(op.symbolic_metadata[upos])*$w)) for w ∈ 0:W-1])...)
        if U == 1 # we gather, no tile, no extra unroll
            instrcall = Expr(:call,:gather,ptr,Expr(:call,:vadd,memoff,ustrides))
            mask === nothing || push!(instrcall.args, mask)
            push!(q.args, Expr(:(=), var, instrcall))
        else # we gather, no tile, but extra unroll
            for u ∈ 0:U-1
                memoff2 = u == 0 ? memoff : push!(copy(memoff), ustride > 1 ? u*W*ustride : Expr(:call,:*,op.symbolic_metadata[upos],u*W) )
                instrcall = Expr(:call, :gather, ptr, Expr(:call,:vadd,memoff2,ustrides))
                mask === nothing || push!(instrcall.args, mask)
                push!(q.args, Expr(:(=), Symbol(var,:_,u), instrcall))
            end
        end
    end
    nothing
end

# TODO: this code should be rewritten to be more "orthogonal", so that we're just combining separate pieces.
# Using sentinel values (eg, T = -1 for non tiling) in part to avoid recompilation.
function lower_load!(
    q::Expr, op::Operation, W::Int, unrolled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    if unrolled ∈ loopdependencies(op)
        lower_load_unrolled!(q, op, W, unrolled, U, suffix, mask)
    else
        lower_load_scalar!(q, op, W, unrolled, U, suffix, mask)
    end
end

function lower_store_scalar!(
    q::Expr, op::Operation, W::Int, unrolled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    loopdeps = loopdependencies(op)
    @assert unrolled ∉ loopdeps
    var = first(parents(op)).variable
    if suffix !== nothing
        var = Symbol(var, :_, suffix)
    end
    ptr = Symbol(:vptr_, op.variable)
    memoff = mem_offset(op)
    # need to find out reduction type
    reduct = CORRESPONDING_REDUCTION[first(parents(op)).instruction]
    storevar = Expr(:call, reduct, var)
    push!(q.args, Expr(:call, :store!, ptr, storevar, memoff))
    nothing
end
function lower_store_unrolled!(
    q::Expr, op::Operation, W::Int, unrolled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    loopdeps = loopdependencies(op)
    @assert unrolled ∈ loopdeps
    var = first(parents(op)).variable
    if suffix !== nothing
        var = Symbol(var, :_, suffix)
    end
    ptr = Symbol(:vptr_, op.variable)
    memoff = mem_offset(op)
    upos = symposition(op, unrolled)
    ustride = op.numerical_metadata[upos]
    if ustride == 1 # vload
        if U == 1
            push!(q.args, Expr(:(=), var, Expr(:call,:vload,ptr,memoff)))
        else
            for u ∈ 0:U-1
                instrcall = Expr(:call,:vstore!, ptr, Symbol(var,:_,u), u == 0 ? memoff : push!(copy(memoff), W*u))
                mask === nothing || push!(instrcall.args, mask)
                push!(q.args, instrcall)
            end
        end
    else
        # ustep = ustride > 1 ? ustride : op.symbolic_metadata[upos]
        ustrides = Expr(:tuple, (ustride > 1 ? [Core.VecElement{Int}(ustride*w) for w ∈ 0:W-1] : [:(Core.VecElement{Int}($(op.symbolic_metadata[upos])*$w)) for w ∈ 0:W-1])...)
        if U == 1 # we gather, no tile, no extra unroll
            instrcall = Expr(:call,:scatter!,ptr, var, Expr(:call,:vadd,memoff,ustrides))
            mask === nothing || push!(instrcall.args, mask)
            push!(q.args, instrcall)
        else # we gather, no tile, but extra unroll
            for u ∈ 0:U-1
                memoff2 = u == 0 ? memoff : push!(copy(memoff), ustride > 1 ? u*W*ustride : Expr(:call,:*,op.symbolic_metadata[upos],u*W) )
                instrcall = Expr(:call, :scatter!, ptr, Symbol(var,:_,u), Expr(:call,:vadd,memoff2,ustrides))
                mask === nothing || push!(instrcall.args, mask)
                push!(q.args, instrcall)
            end
        end
    end
    nothing
end
function lower_store!(
    q::Expr, op::Operation, W::Int, unrolled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    if unrolled ∈ loopdependencies(op)
        lower_store_unrolled!(q, op, W, unrolled, U, suffix, mask)
    else
        lower_store_scalar!(q, op, W, unrolled, U, suffix, mask)
    end
end
# A compute op needs to know the unrolling and tiling status of each of its parents.
#
function lower_compute_scalar!(
    q::Expr, op::Operation, W::Int, unrolled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    lower_compute!(q, op, W, unrolled, U, suffix, mask, false)
end
function lower_compute_unrolled!(
    q::Expr, op::Operation, W::Int, unrolled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    lower_compute!(q, op, W, unrolled, U, suffix, mask, true)
end
function lower_compute!(
    q::Expr, op::Operation, W::Int, unrolled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing,
    opunrolled = unrolled ∈ loopdependencies(op)
)

    var = op.variable
    if suffix === nothing
        optiled = false
    else
        var = Symbol(var, :_, suffix)
        optiled = true
    end
    instr = op.instruction
    
    # cache unroll and tiling check of parents
    # not broadcasted, because we use frequent checks of individual bools
    # making BitArrays inefficient.
    parents_op = parents(op)
    nparents = length(parents_op)
    parentsunrolled = opunrolled ? [unrolled ∈ loopdependencies(opp) for opp ∈ parents_op] : fill(false, nparents)
    parentstiled = optiled ? [tiled ∈ loopdependencies(opp) for opp ∈ parents_op] : fill(false, nparents)
    # parentsyms = [opp.variable for opp ∈ parents(op)]
    Uiter = opunrolled ? U - 1 : 0
    maskreduct = mask !== nothing && any(opp -> opp.variable === var, parents_op)
    # if a parent is not unrolled, the compiler should handle broadcasting CSE.
    # because unrolled/tiled parents result in an unrolled/tiled dependendency,
    # we handle both the tiled and untiled case here.
    # bajillion branches that go the same way on each iteration
    # but smaller function is probably worthwhile. Compiler could theoreically split anyway
    # but I suspect that the branches are so cheap compared to the cost of everything else going on
    # that smaller size is more advantageous.
    for u ∈ 0:Uiter
        intrcall = Expr(:call, instr)
        for n ∈ 1:nparents
            parent = parents_op.variable
            if parentsunrolled[n]
                parent = Symbol(parent,:_,u)
            end
            if parentstiled[n]
                parent = Symbol(parent,:_,t)
            end
            push!(intrcall.args, parent)
        end
        varsym = var
        if optiled
            varsym = Symbol(varsym,:_,suffix)
        end
        if opunrolled
            varsym = Symbol(varsym,:_,u)
        end
        if maskreduct
            push!(q.args, Expr(:(=), varsym, Expr(:call, :vifelse, mask, varsym, instrcall)))
        else
            push!(q.args, Expr(:(=), varsym, instrcall))
        end
    end
end
function lower!(
    q::Expr, op::Operation, W::Int, unrolled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    if isload(op)
        lower_load!(q, op, W, unrolled, U, T, tiled, mask)
    elseif isstore(op)
        lower_store!(q, op, W, unrolled, U, T, tiled, mask)
    else
        lower_compute!(q, op, W, unrolled, U, T, tiled, mask)
    end
end
function lower!(
    q::Expr, ops::AbstractVector{Operation}, W::Int, unrolled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    foreach(op -> lower!(q, op, W, unrolled, U, suffix, mask), ops)
end
function lower!(
    q::Expr, op::Operation, W::Int, unrolled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    foreach(op -> lower!(q, op, W, unrolled, U, suffix, mask), ops)
end



# construction requires ops inserted into operations vectors in dependency order.
function lower_unroll(ls::LoopSet, order::Vector{Symbol}, U::Int)
    if isstaticloop(ls, first(order))
        lower_unroll_static(ls, order, U)
    else
        lower_unroll_dynamic(ls, order, U)
    end
end
function lower_inner_block(ls::LoopSet, U::Int, T::Int, peel::Int = 1)
    @assert peel ≥ 0
    lo = ls.loop_order
    order = lo.loopnames
    W = VectorizationBase.pick_vector_width(ls, first(order))
    unrolled = first(order)
    nloops = length(order)
    istiled = nloops > 1 && T > 0
    if istiled
        Titer = T - 1
        tiled = order[2]
    else
        Titer = 0
        tiled = Symbol("##UNDEFINED##")
    end
    local loopq_old::Expr
    for n ∈ 1:nloops - peel
        loopsym = order[n]
        blockq = if n == 1
            Expr(:block, )
        else
            Expr(:block, Expr(:=, order[n-1], 0))
        end
        loopq = Expr(:while, looprange(ls, loopsym), blockq)
        for prepost ∈ 1:2
            # !U && !T
            lower_scalar!(blockq, @view(ops[:,1,1,prepost,n]), W, unrolled, U, nothing, mask)
            for u ∈ 0:U-1     #  U && !T
                lower_unrolled!(blockq, @view(ops[:,2,1,prepost,n]), W, unrolled, U, nothing, mask)
            end
            for t ∈ 0:Titer   # !U &&  T
                lower_scalar!(blockq, @view(ops[:,1,2,prepost,n]), W, unrolled, U, t, mask)
                for u ∈ 0:U-1 #  U &&  T
                    lower_unrolled!(blockq, @view(ops[:,2,2,prepost,n]), W, unrolled, U, t, mask)
                end
            end
            if n > 1 && prepost == 1
                push!(blockq.args, loopq_old)
            end
        end
        loopq_old = loopq
    end

    @assert peel ≥ 0
    # this function create the inner block
    # args = Any[]
    nloops = length(order)
    unrolled = first(order)
    # included_syms = Set( (unrolled,) )
    included_vars = fill(false, length(operations(ls)))
    # to go inside out, we just have to include all those not-yet included depending on the current sym
    n = 0
    loopsym = last(order)
    blockq = Expr(:block, )#Expr(:(=), loopsym, 0))
    loopq = Expr(:while, looprange(ls, loopsym), blockq)
    for (id,op) ∈ enumerate(operations(ls))
        # We add an op the first time all loop dependencies are met
        # when working through loops backwords, that equates to the first time we encounter a loop dependency
        loopsym ∈ dependencies(op) || continue
        included_vars[id] = true
        lower!(blockq, op, unrolled, U)
    end
    for n ∈ 1:nloops - 1 - peel
        blockq = Expr(:block, Expr(:(=), loopsym, 0)) # sets old loopsym to 0
        loopsym = order[nloops - n]
        postloop = Expr(:block, )
        for (id,op) ∈ enumerate(operations(ls))
            included_vars[id] && continue
            # We add an op the first time all loop dependencies are met
            # when working through loops backwords, that equates to the first time we encounter a loop dependency
            loopsym ∈ dependencies(op) || continue
            included_vars[id] = true
            
            after_loop = depends_on_assigned(op, included_vars)
            after_loop || lower!(blockq, op, unrolled, U)
            after_loop && lower!(postloop, op, unrolled, U)
        end
        push!(blockq.args, loopq_old); append!(blockq.args, postloop.args)
        push!(blockq, Expr(:+=, loopsym, 1))
        loopq = Expr(:while, looprange(ls, loopsym), blockq)
    end
    Expr(:block, Expr(:=, order[1 + peel], 0), loopq), included_vars
end
function lower_unroll_static(ls::LoopSet, order::Vector{Symbol}, U::Int)

end
function lower_unroll_dynamic(ls::LoopSet, order::Vector{Symbol}, U::Int)

    
    unrolled = first(order)
    q = Expr(:block, )

    # we repeatedly break into smaller chunks.
    while U > 0
        inner_block, included_vars = lower_unroll_inner_block(ls, order, U, 1)

    end
    
    Uispow2 = VectorizationBase.ispow2(U)
    looprange(ls, loopsym)
    
    loop = ls.loops[s]
    Expr(:(:), 0, loop.hintexact ? loop.rangehint - 1 : Expr(:call, :(-), loop.rangesym, 1))

    if U == 1 # no unrolling needed
        
    elseif Uispow2 # we use shifts and bitwise &
        log2U = VectorizationBase.intlog2(U)
        
    else

    end
    
    # now must repeat inner block
    
    nested_loop_syms = Set{Symbol}()
    # included_vars = Set{UInt}()
    included_vars = fill(false, length(operations(ls)))
    q = quote end #Expr(:block,)
    # rely on compiler to simplify integer indices
    for s ∈ itersyms(ls)
        push!(q.args, Expr(:(=), s, 0))
    end
    lastqargs = q.args
    postloop_reduction = false
    num_loops = length(order)
    unrolled = first(order)

    for n ∈ 2:num_loops
        itersym = order[n]
        # Add to set of defined symbols
        push!(nested_loop_syms, itersym)
        # check which vars we can define at this level of loop nest
        if itersym === first(order)
            
        else
            loopq = looprange(ls::LoopSet, s::Symbol)
        end
        blockq = Expr(:block, )

        loopq = Expr(:for, Expr(:(=), itersym, looprange), blockq)
        for op ∈ operations(ls)
            # won't define if already defined...
            id = identifier(op)
            id ∈ included_vars && continue
            # it must also be a subset of defined symbols
            if loopdependencies(op) ⊈ nested_loop_syms
                if isreduction(op) && Set(op.symbolic_metadata) ⊆ nested_loop_syms
                    postloop_reduction = true
                else
                    continue
                end
            else
                postloop_reduction = false
            end
            push!(included_vars, id)


            
        end
    end
    q    
end
function lower_tile(ls::LoopSet, order::Vector{Symbol}, U::Int, T::Int)

end


# Here, we have to figure out how to convert the loopset into a vectorized expression.
# This must traverse in a parent -> child pattern
# but order is also dependent on which loop inds they depend on.
# Requires sorting 
function lower(ls::LoopSet)
    order, U, T = choose_order(ls)
    if T == -1
        lower_unroll(ls, order, U)
    else
        lower_tile(ls, order, U, T)
    end
end

function Base.convert(::Type{Expr}, ls::LoopSet)
    lower(ls)
end




