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



@enum NodeType begin
    memload
    memstore
    compute_new
    compute_update
    # accumulator
end

# const ID = Threads.Atomic{UInt}(0)

"""
if node_type == memstore || node_type == compute_new || node_type == compute_store
symbolic metadata contains info on direct dependencies / placement within loop.


"""
struct Operation
    identifier::UInt
    variable::Symbol
    elementbytes::Int
    instruction::Symbol
    node_type::NodeType
    # dependencies::Vector{Symbol}
    dependencies::Set{Symbol}
    # dependencies::Set{Symbol}
    parents::Vector{Operation}
    children::Vector{Operation}
    numerical_metadata::Vector{Int}
    symbolic_metadata::Vector{Symbol}
    function Operation(
        elementbytes,
        instruction,
        node_type,
        identifier,
        variable = gensym()
    )
        # identifier = Threads.atomic_add!(ID, one(UInt))
        new(
            identifier, variable, elementbytes, instruction, node_type,
            Set{Symbol}(), Operation[], Operation[], Int[], Symbol[]
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
identifier(op::Operation) = op.identifier
name(op::Operation) = op.variable
instruction(op::Operation) = op.instruction

function stride(op::Operation, sym::Symbol)
    @assert accesses_memory(op) "This operation does not access memory!"
    # access stride info?
    op.numerical_metadata[findfirst(s -> s === sym, op.symbolic_metadata)]
end
# function
function unitstride(op::Operation, sym::Symbol)
    (first(op.symbolic_metadata) === sym) && (first(op.numerical_metadata) == 1)
end


struct Loop
    itersymbol::Symbol
    rangehint::Int
    rangesym::Symbol
    hintexact::Bool # if true, rangesym ignored and rangehint used for final lowering
end
function Loop(itersymbol::Symbol, rangehint::Int)
    Loop( itersymbol, rangehint, :undef, true )
end
function Loop(itersymbol::Symbol, rangesym::Symbol, rangehint::Int = 1_024)
    Loop( itersymbol, rangehint, rangesym, false )
end

# Must make it easy to iterate
struct LoopSet
    loops::Dict{Symbol,Loop} # sym === loops[sym].itersymbol
    # operations::Vector{Operation}
    loadops::Vector{Operation} # Split them to make it easier to iterate over just a subset
    computeops::Vector{Operation}
    storeops::Vector{Operation}
    reductions::Set{UInt} # IDs of reduction operations that need to be reduced at end.
    strideset::Vector{} 
end
num_loops(ls::LoopSet) = length(ls.loops)
isstaticloop(ls::LoopSet, s::Symbol) = ls.loops[s].hintexact
itersyms(ls::LoopSet) = keys(ls.loops)
function looprange(ls::LoopSet, s::Symbol)
    loop = ls.loops[s]
    Expr(:(:), 0, loop.hintexact ? loop.rangehint - 1 : Expr(:call, :(-), loop.rangesym, 1))
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
    lo = LoopOrder(ls)
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
    lo = LoopOrder(ls)
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
        assigned[identifier(op)] && return true
        depends_on_assigned(p, assigned) && return true
    end
    false
end
function lower_load!(q::Expr, op::Operation, unrolled::Symbol, U, Umax, T = nothing, Tmax = nothing)
    loopdeps = loopdependencies(op)
    if unrolled ∈ loopdeps # we need a vector
        if unitstride(op, unrolled) # vload
            
        else # gather
            
        end
    else # load scalar; promotion should broadcast as/when neccesary
        Expr(:call, :(VectorizationBase.load),  )
    end
end
function lower_store!(q::Expr, op::Operation, unrolled::Symbol, U, T = 1)

end
function lower_compute!(q::Expr, op::Operation, unrolled::Symbol, U, T = 1)
    for t ∈ T, u ∈ U
        
    end
end
function lower!(q::Expr, op::Operation, unrolled::Symbol, U, T = 1)
    if isload(op)
        lower_load!(q, op, unrolled, U, T)
    elseif isstore(op)
        lower_store!(q, op, unrolled, U, T)
    else
        lower_compute!(q, op, unrolled, U, T)
    end
end

# construction requires ops inserted into operations vectors in dependency order.
function lower_unroll(ls::LoopSet, order::Vector{Symbol}, U::Int)
    if isstaticloop(ls, first(order))
        lower_unroll_static(ls, order, U)
    else
        lower_unroll_dynamic(ls, order, U)
    end
end
function lower_unroll_inner_block(ls::LoopSet, order::Vector{Symbol}, U::Int)
    # this function create the inner block
    args = Any[]
    nloops = length(order)
    unrolled = first(order)
    # included_syms = Set( (unrolled,) )
    included_vars = fill(false, length(operations(ls)))
    # to go inside out, we just have to include all those not-yet included depending on the current sym

    n = 0
    loopsym = last(order)
    blockq = Expr(:block, )
    loopq = Expr(:for, Expr(:(=), itersym, looprange(ls, loopsym)), blockq)
    for (id,op) ∈ enumerate(operations(ls))
        # We add an op the first time all loop dependencies are met
        # when working through loops backwords, that equates to the first time we encounter a loop dependency
        loopsym ∈ dependencies(op) || continue
        included_vars[id] = true
        lower!(blockq, op, unrolled, U)
    end
    for n ∈ 1:nloops - 2
        loopsym = order[nloops - n]
        blockq = Expr(:block, )
        loopq = Expr(:for, Expr(:(=), itersym, looprange), blockq)
        for (id,op) ∈ enumerate(operations(ls))
            included_vars[id] && continue
            # We add an op the first time all loop dependencies are met
            # when working through loops backwords, that equates to the first time we encounter a loop dependency
            loopsym ∈ dependencies(op) || continue
            included_vars[id] = true

            after_loop = depends_on_assigned(op, included_vars)

            
        end
    end
end
function lower_unroll_static(ls::LoopSet, order::Vector{Symbol}, U::Int)

end
function lower_unroll_dynamic(ls::LoopSet, order::Vector{Symbol}, U::Int)
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




using BenchmarkTools, LoopVectorization, SLEEF
θ = randn(1000); c = randn(1000);
function sumsc_vectorized(θ::AbstractArray{Float64}, coef::AbstractArray{Float64})
    s, c = 0.0, 0.0
    @vvectorize for i ∈ eachindex(θ, coef)
        sinθᵢ, cosθᵢ = sincos(θ[i])
        s += coef[i] * sinθᵢ
        c += coef[i] * cosθᵢ
    end
    s, c
end
function sumsc_serial(θ::AbstractArray{Float64}, coef::AbstractArray{Float64})
    s, c = 0.0, 0.0
    @inbounds for i ∈ eachindex(θ, coef)
        sinθᵢ, cosθᵢ = sincos(θ[i])
        s += coef[i] * sinθᵢ
        c += coef[i] * cosθᵢ
    end
    s, c
end
function sumsc_sleef(θ::AbstractArray{Float64}, coef::AbstractArray{Float64})
    s, c = 0.0, 0.0
    @inbounds @simd for i ∈ eachindex(θ, coef)
        sinθᵢ, cosθᵢ = SLEEF.sincos_fast(θ[i])
        s += coef[i] * sinθᵢ
        c += coef[i] * cosθᵢ
    end
    s, c
end

@btime sumsc_serial($θ, $c)
@btime sumsc_sleef($θ, $c)
@btime sumsc_vectorized($θ, $c)


