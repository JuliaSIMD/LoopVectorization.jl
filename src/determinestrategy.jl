
# TODO: FIXME for general case
# wrong for transposed matrices, and certain views/SubArrays.
unitstride(op, s) = first(loopdependencies(op)) === s

function cost(op::Operation, unrolled::Symbol, Wshift::Int, size_T::Int = op.elementbytes)
    isconstant(op) && return 0.0, 0, 1
    # Wshift == dependson(op, unrolled) ? Wshift : 0
    # c = first(cost(instruction(op), Wshift, size_T))::Int
    instr = instruction(op)
    opisunrolled = dependson(op, unrolled)
    srt, sl, srp = opisunrolled ? vector_cost(instr, Wshift, size_T) : scalar_cost(instr)
    if accesses_memory(op)
        # either vbroadcast/reductionstore, vmov(a/u)pd, or gather/scatter
        # @show instr, unrolled, loopdependencies(op), unitstride(op, unrolled)
        if opisunrolled
            if !unitstride(op, unrolled)# || !isdense(op) # need gather/scatter
                r = (1 << Wshift)
                srt *= r
                sl *= r
            # else # vmov(a/u)pd
            end
        elseif instr === :setindex! # broadcast or reductionstore; if store we want to penalize reduction
            srt *= 2
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
function hasintersection(a, b)
    for aᵢ ∈ a, bᵢ ∈ b
        aᵢ === bᵢ && return true
    end
    false
end

# evaluates cost of evaluating loop in given order
# heuristically, could simplify analysis by just unrolling outer loop?
function evaluate_cost_unroll(
    ls::LoopSet, order::Vector{Symbol}, max_cost = typemax(Float64), unrolled::Symbol = first(order)
)
    # included_vars = Set{UInt}()
    included_vars = fill(false, length(operations(ls)))
    nested_loop_syms = Symbol[]#Set{Symbol}()
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
        for (id,op) ∈ enumerate(operations(ls))
            # won't define if already defined...
            # id = identifier(op)
            included_vars[id] && continue
            # it must also be a subset of defined symbols
            loopdependencies(op) ⊆ nested_loop_syms || continue
            # hasintersection(reduceddependencies(op), nested_loop_syms) && return Inf
            rd = reduceddependencies(op)
            hasintersection(rd, nested_loop_syms[1:end-length(rd)]) && return Inf
            included_vars[id] = true
            
            total_cost += iter * first(cost(op, unrolled, Wshift, size_T))
            total_cost > max_cost && return total_cost # abort if more expensive; we only want to know the cheapest
        end
    end
    total_cost
end

# only covers unrolled ops; everything else considered lifted?
function depchain_cost!(
    skip::Vector{Bool}, op::Operation, unrolled::Symbol, Wshift::Int, size_T::Int, rt::Float64 = 0.0, sl::Int = 0
)
    skip[identifier(op)] = true
    # depth first search
    for opp ∈ parents(op)
        skip[identifier(opp)] && continue
        rt, sl = depchain_cost!(skip, opp, unrolled, Wshift, size_T, rt, sl)
    end
    # Basically assuming memory and compute don't conflict, but everything else does
    # Ie, ignoring the fact that integer and floating point operations likely don't either
    if iscompute(op)
        rtᵢ, slᵢ = cost(op, unrolled, Wshift, size_T)
        rt += rtᵢ; sl += slᵢ
    end
    rt, sl
end
function parentsnotreduction(op::Operation)
    for opp ∈ parents(op)
        isreduction(opp) && return false
    end
    return true
end
function determine_unroll_factor(
    ls::LoopSet, order::Vector{Symbol}, unrolled::Symbol = first(order)
)
    size_T = biggest_type_size(ls)
    W, Wshift = VectorizationBase.pick_vector_width_shift(length(ls, unrolled), size_T)::Tuple{Int,Int}

    # The strategy is to use an unroll factor of 1, unless there appears to be loop carried dependencies (ie, num_reductions > 0)
    # The assumption here is that unrolling provides no real benefit, unless it is needed to enable OOO execution by breaking up these dependency chains
    num_reductions = 0#sum(isreduction, operations(ls))
    for op ∈ operations(ls)
        if isreduction(op) & iscompute(op) && parentsnotreduction(op)
            num_reductions += 1
        end
    end
    # @show num_reductions
    if iszero(num_reductions) # the 4 is a hack, based on the idea that there is some cost to moving through columns
        return length(order) == 1 ? 1 : 4
    end
    # So if num_reductions > 0, we set the unroll factor to be high enough so that the CPU can be kept busy
    # if there are, U = max(1, round(Int, max(latency) * throughput / num_reductions)) = max(1, round(Int, latency / (recip_throughput * num_reductions)))
    # We also make sure register pressure is not too high.
    latency = 0
    compute_recip_throughput = 0.0
    visited_nodes = fill(false, length(operations(ls)))
    load_recip_throughput = 0.0
    store_recip_throughput = 0.0
    for op ∈ operations(ls)
        dependson(op, unrolled) || continue
        if isreduction(op)
            rt, sl = depchain_cost!(visited_nodes, op, unrolled, Wshift, size_T)
            latency = max(sl, latency)
            compute_recip_throughput += rt
        elseif isload(op)
            load_recip_throughput += first(cost(op, unrolled, Wshift, size_T))
        elseif isstore(op)
            store_recip_throughput += first(cost(op, unrolled, Wshift, size_T))
        end
    end
    recip_throughput = max(
        compute_recip_throughput,
        load_recip_throughput,
        store_recip_throughput
    )
    max(1, round(Int, latency / (recip_throughput * num_reductions) ) )
end

function tile_cost(X, U, T)
    X[1] + X[4] + X[2] / T + X[3] / U
end
function solve_tilesize(X, R)
    @inbounds any(iszero, (R[1],R[2],R[3])) && return -1,-1,Inf #solve_smalltilesize(X, R, Umax, Tmax)
    # @inbounds any(iszero, (R[1],R[2],R[3])) && return -1,-1,Inf #solve_smalltilesize(X, R, Umax, Tmax)
    # We use lagrange multiplier to finding floating point values for U and T
    # first solving for U via quadratic formula
    # X is vector of costs, and R is of register pressures
    # @show X
    # @show R 
    RR = REGISTER_COUNT - R[3] - R[4]
    a = (R[1])^2*X[2] - (R[2])^2*R[1]*X[3]/RR
    b = 2*R[1]*R[2]*X[3]
    c = -RR*R[1]*X[3]
    Ufloat = (sqrt(b^2 - 4a*c) - b) / (2a)
    Tfloat = (RR - Ufloat*R[2])/(Ufloat*R[1])
    # @show Ufloat, Tfloat
    (isfinite(Tfloat) && isfinite(Ufloat)) || return -1,-1,Inf
    Ulow = max(1, floor(Int, Ufloat)) # must be at least 1
    Tlow = max(1, floor(Int, Tfloat)) # must be at least 1
    Uhigh = Ulow + 1 #ceil(Int, Ufloat)
    Thigh = Tlow + 1 #ceil(Int, Tfloat)

    RR = REGISTER_COUNT - R[3] - R[4]
    U, T = Ulow, Tlow
    tcost = tile_cost(X, Ulow, Tlow)
    # @show Ulow*Thigh*R[1] + Ulow*R[2]
    if RR ≥ Ulow*Thigh*R[1] + Ulow*R[2]
        tcost_temp = tile_cost(X, Ulow, Thigh)
        # @show tcost_temp, tcost
        if tcost_temp < tcost
            tcost = tcost_temp
            U, T = Ulow, Thigh
        end
    end
    # The RR + 1 is a hack to get it to favor Uhigh in more scenarios
    Tl = Tlow
    while RR < Uhigh*Tl*R[1] + Uhigh*R[2]
        Tl -= 1
    end
    tcost_temp = tile_cost(X, Uhigh, Tl)
    if tcost_temp < tcost
        tcost = tcost_temp
        U, T = Uhigh, Tl
    end
    if RR > Uhigh*Thigh*R[1] + Uhigh*R[2]
        throw("Something went wrong when solving for Tfloat and Ufloat.")
    end
    min(U,RR), min(T,RR), tcost
end
function solve_tilesize_constU(X, R, U)
    floor(Int, (REGISTER_COUNT - R[3] - R[4] - U*R[2]) / (U * R[1]))
end
function solve_tilesize_constT(X, R, T)
    floor(Int, (REGISTER_COUNT - R[3] - R[4]) / (T * R[1] + R[2]))
end
function solve_tilesize_constT(ls, T)
    R = @view ls.reg_pres[:,1]
    floor(Int, (REGISTER_COUNT - R[3] - R[4]) / (T * R[1] + R[2]))
end
# Tiling here is about alleviating register pressure for the UxT
function solve_tilesize(X, R, Umax, Tmax)
    first(R) == 0 && return -1,-1,Inf #solve_smalltilesize(X, R, Umax, Tmax)
    U, T, cost = solve_tilesize(X, R)
    U_too_large = U > Umax
    T_too_large = T > Tmax
    if U_too_large
        if T_too_large
            U = Umax
            T = Tmax
        else # U too large, resolve T
            U = Umax
            T = solve_tilesize_constU(X, R, U)
        end
    elseif T_too_large
        T = Tmax
        U = solve_tilesize_constT(X, R, T)
    end
    U, T, cost
end
function solve_tilesize(
    ls::LoopSet, unrolled::Symbol, tiled::Symbol,
    cost_vec::AbstractVector{Float64} = @view(ls.cost_vec[:,1]),
    reg_pressure::AbstractVector{Int} = @view(ls.reg_pres[:,1])
)
    maxT = isstaticloop(ls, tiled) ? looprangehint(ls, tiled) : REGISTER_COUNT
    maxU = isstaticloop(ls, unrolled) ? looprangehint(ls, unrolled) : REGISTER_COUNT
    solve_tilesize(cost_vec, reg_pressure, maxT, maxU)
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
    nested_loop_syms = Symbol[]# Set{Symbol}()
    iter = 1.0
    # Need to check if fusion is possible
    size_T = biggest_type_size(ls)
    W, Wshift = VectorizationBase.pick_vector_width_shift(length(ls, unrolled), size_T)::Tuple{Int,Int}
    # costs = 
    # cost_mat[1] / ( unrolled * tiled)
    # cost_mat[2] / ( tiled)
    # cost_mat[3] / ( unrolled)
    # cost_mat[4]
    # @show order
    cost_vec = cost_vec_buf(ls)
    reg_pressure = reg_pres_buf(ls)
    # @inbounds reg_pressure[2] = 1
    # @inbounds reg_pressure[3] = 1
    for n ∈ 1:N
        itersym = order[n]
        # Add to set of defined symbles
        push!(nested_loop_syms, itersym)
        if n == 1
            iter = length(ls, itersym) * length(ls, order[2]) / W
        elseif n > 2
            iter *= Float64(length(ls, itersym))
        end
        # check which vars we can define at this level of loop nest
        for (id, op) ∈ enumerate(operations(ls))
            # isconstant(op) && continue
            # @assert id == identifier(op)+1 # testing, for now
            # won't define if already defined...
            included_vars[id] && continue
            # it must also be a subset of defined symbols
            loopdependencies(op) ⊆ nested_loop_syms || continue
            # # @show nested_loop_syms
            # # @show reduceddependencies(op)
            rd = reduceddependencies(op)
            hasintersection(rd, nested_loop_syms[1:end-length(rd)]) && return 0,0,Inf
            included_vars[id] = true
            rt, lat, rp = cost(op, unrolled, Wshift, size_T)
            # @show instruction(op), rt, lat, rp, iter
            rt *= iter
            isunrolled = unrolled ∈ loopdependencies(op)
            istiled = tiled ∈ loopdependencies(op)
            # @show isunrolled, istiled
            if isunrolled && istiled # no cost decrease; cost must be repeated
                cost_vec[1] += rt
                reg_pressure[1] += rp
            elseif isunrolled # cost decreased by tiling
                cost_vec[2] += rt
                reg_pressure[2] += rp
            elseif istiled # cost decreased by unrolling
                cost_vec[3] += rt
                reg_pressure[3] += rp
            else# neither unrolled or tiled
                cost_vec[4] += rt
                reg_pressure[4] += rp
            end
        end
    end
    solve_tilesize(ls, unrolled, tiled, cost_vec, reg_pressure)
end


struct LoopOrders
    syms::Vector{Symbol}
    buff::Vector{Symbol}
end
function LoopOrders(ls::LoopSet)
    syms = [s for s ∈ keys(ls.loops)]
    LoopOrders(syms, similar(syms))
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
    # # @show state
    syms = copyto!(lo.buff, lo.syms)
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
    best_order = copyto!(ls.loop_order.bestorder, lo.syms)
    new_order, state = iterate(lo) # right now, new_order === best_order
    U, T, lowest_cost = 0, 0, Inf
    while true
        U_temp, T_temp, cost_temp = evaluate_cost_tile(ls, new_order)
        if cost_temp < lowest_cost
            lowest_cost = cost_temp
            U, T = U_temp, T_temp
            copyto!(best_order, new_order)
            save_tilecost!(ls)
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
        # copyto!(ls.loop_order.loopnames, uorder)
        return uorder, determine_unroll_factor(ls, uorder), -1
    else
        # copyto!(ls.loop_order.loopnames, torder)
        return torder, tU, tT
    end
end

