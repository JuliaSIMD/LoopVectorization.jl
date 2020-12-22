
const CACHELINE_SIZE = something(VectorizationBase.L₁CACHE.linesize, 64)

# function indexappearences(op::Operation, s::Symbol)
#     s ∉ loopdependencies(op) && return 0
#     appearences = 0
#     if isloopvalue(op)
#         return s === first(loopdependencies(op)) ? 1 : 0
#     elseif isload(op)
#         return 100
#     end
#     newapp = 0
#     for opp ∈ parents(op)
#         newapp += indexappearences(opp, s)
#     end
#     factor = instruction(op).instr ∈ (:+, :vadd, :add_fast, :evadd) ? 1 : 10
#     newapp * factor
# end
function findparent(ls::LoopSet, s::Symbol)#opdict isn't filled when reconstructing
    id = findfirst(op -> name(op) === s, operations(ls))
    id === nothing && throw("$s not found")
    operations(ls)[id]
end
function unitstride(ls::LoopSet, op::Operation, s::Symbol)
    inds = getindices(op)
    li = op.ref.loopedindex
    # The first index is allowed to be indexed by `s`
    fi = first(inds)
    if fi === Symbol("##DISCONTIGUOUSSUBARRAY##")
        return false
    # elseif !first(li)
    #     # We must check if this
    #     parent = findparent(ls, fi)
    #     indexappearences(parent, s) > 1 && return false
    end
    for i ∈ 2:length(inds)
        if li[i]
            s === inds[i] && return false
        else
            parent = findparent(ls, inds[i])
            s ∈ loopdependencies(parent) && return false
        end
    end
    true
end

# function register_pressure(op::Operation)
#     if isconstant(op) || isloopvalue(op)
#         0
#     else
#         instruction_cost(instruction(op)).register_pressure
#     end
# end
function cost(ls::LoopSet, op::Operation, vectorized::Symbol, Wshift::Int, size_T::Int = op.elementbytes)
    isconstant(op) && return 0.0, 0, Float64(length(loopdependencies(op)) > 0)
    isloopvalue(op) && return 0.0, 0, 0.0
    # Wshift == dependson(op, vectorized) ? Wshift : 0
    # c = first(cost(instruction(op), Wshift, size_T))::Int
    instr = instruction(op)
    # instr = instruction(op)
    if length(parents(op)) == 1
        if instr == Instruction(:-) || instr === Instruction(:vsub) || instr == Instruction(:+) || instr == Instruction(:vadd)
            return 0.0, 0, 0.0
        end
    elseif iscompute(op) && all(opp -> (isloopvalue(opp) | isconstant(opp)), parents(op))
        return 0.0, 0, 0.0
    end
    opisvectorized = dependson(op, vectorized)
    srt, sl, srp = opisvectorized ? vector_cost(instr, Wshift, size_T) : scalar_cost(instr)
    if accesses_memory(op)
        # either vbroadcast/reductionstore, vmov(a/u)pd, or gather/scatter
        if opisvectorized
            if !unitstride(ls, op, vectorized)# || !isdense(op) # need gather/scatter
                r = (1 << Wshift)
                srt *= r# * 2
                sl *= r
            elseif isload(op) & (length(loopdependencies(op)) > 1)# vmov(a/u)pd
                # penalize vectorized loads with more than 1 loopdep
                # heuristic; more than 1 loopdep means that many loads will not be aligned
                # Roughly corresponds to double-counting loads crossing cacheline boundaries
                # TODO: apparently the new ARM A64FX CPU (with 512 bit vectors) is NOT penalized for unaligned loads
                #       would be nice to add a check for this CPU, to see if such a penalty is still appropriate.
                #       Also, once more SVE (scalable vector extension) CPUs are released, would be nice to know if
                #       this feature is common to all of them.
                srt += 0.5VectorizationBase.REGISTER_SIZE / CACHELINE_SIZE
            end
        elseif isstore(op) # broadcast or reductionstore; if store we want to penalize reduction
            srt *= 3
            sl *= 3
        end
    end
    srt, sl, Float64(srp)
end

    # Base._return_type()

function biggest_type_size(ls::LoopSet)
    maximum(elsize, operations(ls))
end
# function VectorizationBase.pick_vector_width(ls::LoopSet, u::Symbol)
#     VectorizationBase.pick_vector_width(length(ls, u), biggest_type_size(ls))
# end
# function VectorizationBase.pick_vector_width_shift(ls::LoopSet, u::Symbol)
#     VectorizationBase.pick_vector_width_shift(length(ls, u), biggest_type_size(ls))
# end
function hasintersection(a, b)
    for aᵢ ∈ a, bᵢ ∈ b
        aᵢ === bᵢ && return true
    end
    false
end
const num_iterations = cld

function set_vector_width!(ls::LoopSet, vectorized::Symbol)
    W = ls.vector_width[]
    if !iszero(W)
        ls.vector_width[] = min(W, VectorizationBase.nextpow2(length(ls, vectorized)))
    end
    nothing
end
function lsvecwidthshift(ls::LoopSet, vectorized::Symbol, size_T = nothing)
    W = ls.vector_width[]
    lvec = length(ls, vectorized)
    if iszero(W)
        VectorizationBase.pick_vector_width_shift_from_size(lvec, isnothing(size_T) ? biggest_type_size(ls) : size_T)::Tuple{Int,Int}
        # VectorizationBase.pick_vector_width_shift(lvec, isnothing(size_T) ? biggest_type_size(ls) : size_T)::Tuple{Int,Int}
    else
        W = min(W, VectorizationBase.nextpow2(lvec))
        W, VectorizationBase.intlog2(W)
    end
end

# evaluates cost of evaluating loop in given order
function evaluate_cost_unroll(
    ls::LoopSet, order::Vector{Symbol}, vectorized::Symbol, max_cost = typemax(Float64)
)
    included_vars = fill!(resize!(ls.included_vars, length(operations(ls))), false)
    nested_loop_syms = Symbol[]#Set{Symbol}()
    total_cost = 0.0
    iter = 1.0
    size_T = biggest_type_size(ls)
    W, Wshift = lsvecwidthshift(ls, vectorized, size_T)
    # Need to check if fusion is possible
    for itersym ∈ order
        # Add to set of defined symbles
        push!(nested_loop_syms, itersym)
        looplength = length(ls, itersym)
        liter = itersym === vectorized ? num_iterations(looplength, W) : looplength
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
            hasintersection(rd, @view(nested_loop_syms[1:end-length(rd)])) && return Inf
            if isstore(op) #TODO: DRY (this is repeated in evaluate_cost_tile)
                loadstoredeps = store_load_deps(op)
                if !isnothing(loadstoredeps)
                    any(s -> (s ∉ loadstoredeps), nested_loop_syms) && return Inf
                end
            end
            included_vars[id] = true
            total_cost += iter * first(cost(ls, op, vectorized, Wshift, size_T))
            total_cost > max_cost && return total_cost # abort if more expensive; we only want to know the cheapest
        end
    end
    0.999total_cost + stride_penalty(ls, order) # 0.999 to place finger on scale in its favor
end

# only covers vectorized ops; everything else considered lifted?
function depchain_cost!(
    ls::LoopSet, skip::Vector{Bool}, op::Operation, vectorized::Symbol, Wshift::Int, size_T::Int, rt::Float64 = 0.0, sl::Int = 0
)
    skip[identifier(op)] = true
    # depth first search
    for opp ∈ parents(op)
        skip[identifier(opp)] && continue
        rt, sl = depchain_cost!(ls, skip, opp, vectorized, Wshift, size_T, rt, sl)
    end
    # Basically assuming memory and compute don't conflict, but everything else does
    # Ie, ignoring the fact that integer and floating point operations likely don't either
    if iscompute(op)
        rtᵢ, slᵢ = cost(ls, op, vectorized, Wshift, size_T)
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
# function roundpow2(i::Integer)
#     u = VectorizationBase.nextpow2(i)
#     l = u >>> 1
#     ud = u - i
#     ld = i - l
#     ud > ld ? l : u
# end
# function roundpow2(x::Float64)
    # 1 << round(Int, log2(x))
# end
function unroll_no_reductions(ls, order, vectorized)
    size_T = biggest_type_size(ls)
    W, Wshift = lsvecwidthshift(ls, vectorized, size_T)
    # W, Wshift = VectorizationBase.pick_vector_width_shift(length(ls, vectorized), size_T)::Tuple{Int,Int}

    compute_rt = load_rt = store_rt = 0.0
    unrolled = last(order)
    if unrolled === vectorized && length(order) > 1
        unrolled = order[end-1]
    end
    # latency not a concern, because no depchains
    for op ∈ operations(ls)
        dependson(op, unrolled) || continue
        if iscompute(op)
            compute_rt += first(cost(ls, op, vectorized, Wshift, size_T))
        elseif isload(op)
            load_rt += first(cost(ls, op, vectorized, Wshift, size_T))
        elseif isstore(op)
            store_rt += first(cost(ls, op, vectorized, Wshift, size_T))
        end
    end 
    # heuristic guess
    # roundpow2(min(4, round(Int, (compute_rt + load_rt + 1) / compute_rt)))
    memory_rt = load_rt + store_rt
    u = if compute_rt > memory_rt
        max(1, VectorizationBase.nextpow2( min( 4, round(Int, 8 / compute_rt) ) ))
    elseif iszero(compute_rt)
        4
    elseif iszero(load_rt)
        iszero(store_rt) ? 4 : max(1, min(4, round(Int, 2compute_rt / store_rt)))
    else
        max(1, min(4, round(Int, 2compute_rt / load_rt)))
    end
    # commented out here is to decide to align loops
    # if memory_rt > compute_rt && isone(u) && (length(order) > 1) && (last(order) === vectorized) && length(getloop(ls, last(order))) > 8W
    #     ls.align_loops[] = findfirst(operations(ls)) do op
    #         isstore(op) && dependson(op, unrolled)
    #     end
    # end
    if unrolled === vectorized
        u = demote_unroll_factor(ls, u, vectorized)
    end
    u, unrolled
    # rt = max(compute_rt, load_rt + store_rt)
    # # (iszero(rt) ? 4 : max(1, roundpow2( min( 4, round(Int, 16 / rt) ) ))), unrolled
    # (iszero(rt) ? 4 : max(1, VectorizationBase.nextpow2( min( 4, round(Int, 8 / rt) ) ))), unrolled
end
function determine_unroll_factor(
    ls::LoopSet, order::Vector{Symbol}, unrolled::Symbol, vectorized::Symbol
)
    size_T = biggest_type_size(ls)
    W, Wshift = lsvecwidthshift(ls, vectorized, size_T)

    # So if num_reductions > 0, we set the unroll factor to be high enough so that the CPU can be kept busy
    # if there are, U = max(1, round(Int, max(latency) * throughput / num_reductions)) = max(1, round(Int, latency / (recip_throughput * num_reductions)))
    # We also make sure register pressure is not too high.
    latency = 1.0
    # compute_recip_throughput_u = 0.0
    compute_recip_throughput = 0.0
    visited_nodes = fill(false, length(operations(ls)))
    load_recip_throughput = 0.0
    store_recip_throughput = 0.0
    for op ∈ operations(ls)
        # dependson(op, unrolled) || continue
        if isreduction(op)
            rt, sl = depchain_cost!(ls, visited_nodes, op, vectorized, Wshift, size_T)
            if isouterreduction(op) != -1 || unrolled ∉ reduceddependencies(op)
                latency = max(sl, latency)
            end
            # if unrolled ∈ loopdependencies(op)
            #     compute_recip_throughput_u += rt
            # else
            compute_recip_throughput += rt
            # end
        elseif isload(op)
            load_recip_throughput += first(cost(ls, op, vectorized, Wshift, size_T))
        elseif isstore(op)
            store_recip_throughput += first(cost(ls, op, vectorized, Wshift, size_T))
        end
    end
    recip_throughput = max(
        compute_recip_throughput,
        load_recip_throughput,
        store_recip_throughput
    )
    recip_throughput, latency
end
function count_reductions(ls::LoopSet)
    num_reductions = 0
    for op ∈ operations(ls)
        if isreduction(op) & iscompute(op) && parentsnotreduction(op)
            num_reductions += 1
        end
    end
    num_reductions
end

demote_unroll_factor(ls::LoopSet, UF, loop::Symbol) = demote_unroll_factor(ls, UF, getloop(ls, loop))
function demote_unroll_factor(ls::LoopSet, UF, loop::Loop)
    W = ls.vector_width[] 
    if !iszero(W) && isstaticloop(loop)
        UFW = maybedemotesize(UF*W, length(loop))
        UF = cld(UFW, W)
    end
    UF
end

function determine_unroll_factor(ls::LoopSet, order::Vector{Symbol}, vectorized::Symbol)
    num_reductions = count_reductions(ls)
    # The strategy is to use an unroll factor of 1, unless there appears to be loop carried dependencies (ie, num_reductions > 0)
    # The assumption here is that unrolling provides no real benefit, unless it is needed to enable OOO execution by breaking up these dependency chains
    if iszero(num_reductions)
        # if only 1 loop, no need to unroll
        # if more than 1 loop, there is some cost. Picking 2 here as a heuristic.
        return unroll_no_reductions(ls, order, vectorized)
    end
    innermost_loop = last(order)
    rt = Inf; rtcomp = Inf; latency = Inf; best_unrolled = Symbol("")
    for unrolled ∈ order
        rttemp, ltemp = determine_unroll_factor(ls, order, unrolled, vectorized)
        rtcomptemp = rttemp + (0.01 * ((vectorized === unrolled) + (unrolled === innermost_loop) - latency))
        if rtcomptemp < rtcomp
            rt = rttemp
            rtcomp = rtcomptemp
            latency = ltemp
            best_unrolled = unrolled
        end
    end
    # min(8, roundpow2(max(1, round(Int, latency / (rt * num_reductions) ) ))), best_unrolled
    UF = min(8, VectorizationBase.nextpow2(max(1, round(Int, latency / (rt * num_reductions) ) )))
    if best_unrolled === vectorized
        UF = demote_unroll_factor(ls, UF, vectorized)
    end
    UF, best_unrolled
end

function unroll_cost(X, u₁, u₂, u₁L, u₂L)
    u₂factor = (num_iterations(u₂L, u₂)/u₂L)
    u₁factor = (num_iterations(u₁L, u₁)/u₁L)
    # @show num_iterations(u₂L, u₂)/u₂L, u₂, u₂L
    # @show num_iterations(u₁L, u₁)/u₁L, u₁, u₁L
    # X[1]*u₂factor*u₁factor + X[4] + X[2] * u₂factor + X[3] * u₁factor
    X[1] + X[2] * u₂factor + X[3] * u₁factor + X[4] * u₁factor * u₂factor
end
# function itertilesize(X, u₁L, u₂L)
#     cb = Inf
#     u₁b = 1; u₂b = 1
#     for u₁ ∈ 1:4, u₂ ∈ 1:4
#         c = unroll_cost(X, u₁, u₂, u₁L, u₂L)
#         if cb > c
#             cb = c
#             u₁b = u₁; u₂b = u₂
#         end
#     end
#     u₁b, u₂b, cb
# end

function solve_unroll_iter(X, R, u₁L, u₂L, u₁range, u₂range)
    R₁, R₂, R₃, R₄, R₅ = R[1], R[2], R[3], R[4], R[5]
    RR = REGISTER_COUNT - R₃ - R₄
    u₁best, u₂best = 0, 0
    bestcost = Inf
    for u₁temp ∈ u₁range
        for u₂temp ∈ u₂range
            RR ≥ u₁temp*u₂temp*R₁ + u₁temp*R₂ + u₂temp*R₅ || continue
            tempcost = unroll_cost(X, u₁temp, u₂temp, u₁L, u₂L)
            # @show u₁temp, u₂temp, tempcost
            # @show u₁temp*u₂temp*R₁ + u₁temp*R₂ + u₂temp*R₅
            if tempcost ≤ bestcost
                bestcost = tempcost
                u₁best, u₂best = u₁temp, u₂temp
            end
        end
    end
    u₁best, u₂best, bestcost
end

function solve_unroll(X, R, u₁L, u₂L, u₁step, u₂step)
    X₁, X₂, X₃, X₄ = X[1], X[2], X[3], X[4]
    # If we don't have AVX512, masks occupy a vector register;
    # AVX512F is currently defined as `false` for non-x86 CPUs, but
    # should instead define generic constant `HAS_OPMASK_REGISTERS` in VectorizationBase.jl to use here instead.
    VectorizationBase.AVX512F || (R[3] += 1)
    R₁, R₂, R₃, R₄, R₅ = R[1], R[2], R[3], R[4], R[5]
    iszero(R₅) || return solve_unroll_iter(X, R, u₁L, u₂L, u₁step:u₁step:10, u₂step:u₂step:10)
    RR = REGISTER_COUNT - R₃ - R₄
    a = R₂^2*X₃ -R₁*X₄ * R₂ - R₁*X₂*RR
    b = R₁ * X₄ * RR - R₁ * X₄ * RR - 2X₃*RR*R₂
    c = X₃*RR^2
    discriminant = b^2 - 4a*c
    discriminant < 0 && return -1,-1,Inf
    u₁float = max(float(u₁step), (sqrt(discriminant) + b) / (-2a)) # must be at least 1
    u₂float = (RR - u₁float*R₂)/(u₁float*R₁)
    if !(isfinite(u₂float) & isfinite(u₁float)) # brute force
        u₁low = u₂low = 1
        u₁high = iszero(X₂) ? 2 : (REGISTER_COUNT == 32 ? 8 : 6)
        u₂high = iszero(X₃) ? 2 : (REGISTER_COUNT == 32 ? 8 : 6)
        return solve_unroll_iter(X, R, u₁L, u₂L, u₁low:u₁step:u₁high, u₂low:u₂step:u₂high)
    end
    u₁low = floor(Int, u₁float)
    u₂low = max(u₂step, floor(Int, 0.8u₂float)) # must be at least 1
    u₁high = solve_unroll_constT(R, u₂low) + u₁step
    u₂high = solve_unroll_constU(R, u₁low) + u₂step
    maxunroll = REGISTER_COUNT == 32 ? (((X₂ > 0) & (X₃ > 0)) ? 10 : 8) : 6
    u₁low = (min(u₁low, maxunroll) ÷ u₁step) * u₁step
    u₂low = (min(u₂low, maxunroll) ÷ u₂step) * u₂step
    u₁high = min(u₁high, maxunroll)
    u₂high = min(u₂high, maxunroll)
    solve_unroll_iter(X, R, u₁L, u₂L, reverse(u₁low:u₁step:u₁high), reverse(u₂low:u₂step:u₂high))
end

function solve_unroll_constU(R::AbstractVector, u₁::Int)
    denom = u₁ * R[1] + R[5]
    iszero(denom) && return 8
    floor(Int, (REGISTER_COUNT - R[3] - R[4] - u₁*R[2]) / denom)
end
function solve_unroll_constT(R::AbstractVector, u₂::Int)
    denom = u₂ * R[1] + R[2]
    iszero(denom) && return 8
    floor(Int, (REGISTER_COUNT - R[3] - R[4] - u₂*R[5]) / denom)
end
# function solve_unroll_constT(ls::LoopSet, u₂::Int)
#     R = @view ls.reg_pres[:,1]
#     denom = u₂ * R[1] + R[2]
#     iszero(denom) && return 8
#     floor(Int, (REGISTER_COUNT - R[3] - R[4] - u₂*R[5]) / (u₂ * R[1] + R[2]))
# end
# Tiling here is about alleviating register pressure for the UxT
function solve_unroll(X, R, u₁max, u₂max, u₁L, u₂L, u₁step, u₂step)
    # iszero(first(R)) && return -1,-1,Inf #solve_smalltilesize(X, R, u₁max, u₂max)
    u₁, u₂, cost = solve_unroll(X, R, u₁L, u₂L, u₁step, u₂step)
    # u₂ -= u₂ & 1
    # u₁ = min(u₁, u₂)
    u₁_too_large = u₁ > u₁max
    u₂_too_large = u₂ > u₂max
    if u₁_too_large
        u₁ = u₁max
        if u₂_too_large
            u₂ = u₂max
        else # u₁ too large, resolve u₂
            u₂ = min(u₂max, max(1,solve_unroll_constU(R, u₁)))
        end
        cost = unroll_cost(X, u₁, u₂, u₁L, u₂L)
    elseif u₂_too_large
        u₂ = u₂max
        u₁ = min(u₁max, max(1,solve_unroll_constT(R, u₂)))
        cost = unroll_cost(X, u₁, u₂, u₁L, u₂L)
    end
    u₁, u₂, cost
end
function maybedemotesize(U::Int, N::Int)
    num_iterations(N, num_iterations(N, U))
end
function maybedemotesize(u₂::Int, N::Int, U::Int, Uloop::Loop, maxu₂base::Int)
    u₂ > 1 || return 1
    u₂ == N && return u₂
    u₂ = maybedemotesize(u₂, N)
    if !(isstaticloop(Uloop) && length(Uloop) == U)
        if N % u₂ != 0
            u₂ = min(u₂, maxu₂base)
        end
    end
    u₂
end

function solve_unroll(
    ls::LoopSet, u₁loopsym::Symbol, u₂loopsym::Symbol,
    cost_vec::AbstractVector{Float64},
    reg_pressure::AbstractVector{Float64},
    W::Int, vectorized::Symbol, rounduᵢ::Int
)
    (u₁step, u₂step) = if rounduᵢ == 1 # max is to safeguard against some weird arch I've never heard of.
        (max(1,CACHELINE_SIZE ÷ VectorizationBase.REGISTER_SIZE), 1)
    elseif rounduᵢ == 2
        (1, max(1,CACHELINE_SIZE ÷ VectorizationBase.REGISTER_SIZE))
    else
        (1, 1)
    end
    u₁loop = getloop(ls, u₁loopsym)
    u₂loop = getloop(ls, u₂loopsym)
    solve_unroll(
        u₁loopsym, u₂loopsym, cost_vec, reg_pressure, W, vectorized, u₁loop, u₂loop, u₁step, u₂step
    )
end

function solve_unroll(
    u₁loopsym::Symbol, u₂loopsym::Symbol,
    cost_vec::AbstractVector{Float64},
    reg_pressure::AbstractVector{Float64},
    W::Int, vectorized::Symbol,
    u₁loop::Loop, u₂loop::Loop,
    u₁step::Int, u₂step::Int
)
    maxu₂base = maxu₁base = REGISTER_COUNT == 32 ? 10 : 6#8
    maxu₂ = maxu₂base#8
    maxu₁ = maxu₁base#8
    u₁L = length(u₁loop)
    u₂L = length(u₂loop)
    if isstaticloop(u₂loop)
        if u₂loopsym !== vectorized && u₂L ≤ 4
            u₁ = max(1, solve_unroll_constT(reg_pressure, u₂L))
            u₁ = isstaticloop(u₁loop) ? maybedemotesize(u₁, u₁L) : u₁
            return u₁, u₂L, unroll_cost(cost_vec, u₁, u₂L, u₁L, u₂L)
        end
        u₂Ltemp = u₂loopsym === vectorized ? cld(u₂L, W) : u₂L
        maxu₂ = min(4maxu₂, u₂Ltemp)
    end
    if isstaticloop(u₁loop)
        if u₁loopsym !== vectorized && u₁L ≤ 4
            u₂ = max(1, solve_unroll_constU(reg_pressure, u₁L))
            u₂ = isstaticloop(u₂loop) ? maybedemotesize(u₂, u₂L) : u₂
            return u₁L, u₂, unroll_cost(cost_vec, u₁L, u₂, u₁L, u₂L)
        end
        u₁Ltemp = u₁loopsym === vectorized ? cld(u₁L, W) : u₁L
        maxu₁ = min(4maxu₁, u₁Ltemp)
    end
    if u₁loopsym === vectorized
        u₁Lf = u₁L / W
    else
        u₁Lf = Float64(u₁L)
    end
    if u₂loopsym === vectorized
        u₂Lf = u₂L / W
    else
        u₂Lf = Float64(u₂L)
    end
    # @show u₁Lf, u₂Lf, u₁L, length(u₁loop)
    u₁, u₂, cost = solve_unroll(cost_vec, reg_pressure, maxu₁, maxu₂, u₁Lf, u₂Lf, u₁step, u₂step)
    # heuristic to more evenly divide small numbers of iterations
    if isstaticloop(u₂loop)
        u₂ = maybedemotesize(u₂, length(u₂loop), u₁, u₁loop, maxu₂base)
    end
    if isstaticloop(u₁loop)
        u₁ = maybedemotesize(u₁, length(u₁loop), u₂, u₂loop, maxu₁base)
    end
    u₁, u₂, cost
end

function set_upstream_family!(adal::Vector{T}, op::Operation, val::T) where {T}
    adal[identifier(op)] == val && return # must already have been set
    adal[identifier(op)] = val
    for opp ∈ parents(op)
        set_upstream_family!(adal, opp, val)
    end
end
function loopdepindices(ls::LoopSet, op::Operation)
    loopdeps = loopdependencies(op.ref)
    isdiscontig = first(loopdeps) === Symbol("##DISCONTIGUOUSSUBARRAY##")
    if !isdiscontig && all(op.ref.loopedindex)
        return loopdeps
    end
    loopdepsret = Symbol[]
    for i ∈ eachindex(op.ref.loopedindex)
        if op.ref.loopedindex[i]
            push!(loopdepsret, loopdeps[i+isdiscontig])
        else
            oploopdeps = loopdependencies(findparent(ls, loopdeps[i+isdiscontig]))
            for ld ∈ oploopdeps
                (ld ∉ loopdepsret) && push!(loopdepsret, ld)
            end
        end
    end
    loopdepsret
end
function stride_penalty(ls::LoopSet, op::Operation, order::Vector{Symbol}, loopfreqs)
    loopdeps = loopdepindices(ls, op)
    opstrides = Vector{Int}(undef, length(loopdeps))
    # very minor stride assumption here, because we don't really want to base optimization decisions on it...
    opstrides[1] = 1.0 + (first(loopdependencies(op.ref)) === Symbol("##DISCONTIGUOUSSUBARRAY##"))
    # loops = map(s -> getloop(ls, s), loopdeps)
    l = Float64(length(getloop(ls, first(loopdeps)))) 
    for i ∈ 2:length(loopdeps)
        looplength = length(getloop(ls, loopdeps[i-1]))
        opstrides[i] = opstrides[i-1] * looplength
        l *= looplength
        # opstrides[i] = opstrides[i-1] * length(loops[i-1])
    end
    penalty = 0.0
    for i ∈ eachindex(order)
        id = findfirst(isequal(order[i]), loopdeps)
        if !isnothing(id)
            penalty += loopfreqs[i] * opstrides[id]
        end
    end
    penalty * l
end
function stride_penalty(ls::LoopSet, order::Vector{Symbol})
    stridepenaltydict = Dict{Symbol,Vector{Float64}}()
    loopfreqs = Vector{Int}(undef, length(order))
    loopfreqs[1] = 1
    for i ∈ 2:length(order)
        loopfreqs[i] = loopfreqs[i-1] * length(getloop(ls, order[i]))
    end
    for op ∈ operations(ls)
        if accesses_memory(op)
            v = get!(() -> Float64[], stridepenaltydict, op.ref.ref.array)
            push!(v, stride_penalty(ls, op, order, loopfreqs))
        end
    end
    if iszero(length(values(stridepenaltydict)))
        0.0
    else # 1 / 1024 = 0.0009765625
        10.0sum(maximum, values(stridepenaltydict)) * Base.power_by_squaring(0.0009765625, length(order))
    end
end
function isoptranslation(ls::LoopSet, op::Operation, unrollsyms::UnrollSymbols)
    @unpack u₁loopsym, u₂loopsym, vectorized = unrollsyms
    (vectorized == u₁loopsym || vectorized == u₂loopsym) && return 0, false
    (isu₁unrolled(op) && isu₂unrolled(op)) || return 0, false
    istranslation = 0
    inds = getindices(op); li = op.ref.loopedindex
    translationplus = false
    for i ∈ eachindex(li)
        if !li[i]
            opp = findparent(ls, inds[i + (first(inds) === Symbol("##DISCONTIGUOUSSUBARRAY##"))])
            if instruction(opp).instr ∈ (:+, :-) && u₁loopsym ∈ loopdependencies(opp) && u₂loopsym ∈ loopdependencies(opp)
                istranslation = i
                translationplus = instruction(opp).instr === :+
            end
        end
    end
    istranslation, translationplus
end
function maxnegativeoffset(ls::LoopSet, op::Operation, u::Symbol)
    opmref = op.ref
    opref = opmref.ref
    mno = typemin(Int)
    id = 0
    for opp ∈ operations(ls)
        opp === op && continue
        oppmref = opp.ref
        oppref = oppmref.ref
        sameref(opref, oppref) || continue
        opinds = getindicesonly(op)
        oppinds = getindicesonly(opp)
        opoffs = opref.offsets
        oppoffs = oppref.offsets
        # oploopi = opmref.loopedindex
        # opploopi = oppmref.loopedindex
        mnonew = typemin(Int)
        for i ∈ eachindex(opinds)
            if opinds[i] === u
                mnonew = (opoffs[i] - oppoffs[i])
            elseif opoffs[i] != oppoffs[i]
                mnonew = 1
                break
            end
        end
        if mno < mnonew < 0
            mno = mnonew
            id = identifier(opp)
        end
    end
    mno, id
end
function maxnegativeoffset(ls::LoopSet, op::Operation, unrollsyms::UnrollSymbols)
    @unpack u₁loopsym, u₂loopsym, vectorized = unrollsyms
    mno = typemin(Int)
    i = 0
    if u₁loopsym !== vectorized
        mnou₁ = first(maxnegativeoffset(ls, op, u₁loopsym))
        if mnou₁ > mno
            i = 1
            mno = mnou₁
        end
    end
    if u₂loopsym !== vectorized
        mnou₂ = first(maxnegativeoffset(ls, op, u₂loopsym))
        if mnou₂ > mno
            i = 2
            mno = mnou₂
        end
    end
    mno, i
end
function load_elimination_cost_factor!(
    cost_vec, reg_pressure, choose_to_inline, ls::LoopSet, op::Operation, iters, unrollsyms::UnrollSymbols, Wshift, size_T
)
    @unpack u₁loopsym, u₂loopsym, vectorized = unrollsyms
    if !iszero(first(isoptranslation(ls, op, unrollsyms)))
        rt, lat, rp = cost(ls, op, vectorized, Wshift, size_T)
        rto = rt
        rt *= iters
            # rt *= factor1; rp *= factor2;
        choose_to_inline[] = true
        # for loop ∈ ls.loops
        #     # If another loop is short, assume that LLVM will unroll it, in which case
        #     # we want to be a little more conservative in terms of register pressure.
        #     #FIXME: heuristic hack to get some desired behavior.
        #     if isstaticloop(loop) && length(loop) ≤ 4
        #         itersym = loop.itersymbol
        #         if itersym !== u₁loopsym && itersym !== u₂loopsym
        #             return (0.25, REGISTER_COUNT == 32 ? 2.0 : 1.0)
        #             # return (0.25, 1.0)
        #             return true
        #         end
        #     end
        # end
        # # (0.25, REGISTER_COUNT == 32 ? 1.2 : 1.0)
        # (0.25, 1.0)
        cost_vec[1] -= 0.1looplengthprod(ls)
        reg_pressure[1] += 0.25rp
        cost_vec[2] += rt
        reg_pressure[2] += rp
        cost_vec[3] += rt
        # reg_pressure[3] += rp
        reg_pressure[5] += rp
        true
    else
        (1.0, 1.0)
        false
    end
end
function loadintostore(ls::LoopSet, op::Operation)
    isload(op) || return false # leads to bad behavior more than it helps
    for opp ∈ operations(ls)
        isstore(opp) && opp.ref == op.ref && return true
    end
    false
end
function store_load_deps!(deps::Vector{Symbol}, op::Operation, compref = op.ref)
    for opp ∈ parents(op)
        foreach(ld -> ((ld ∈ deps) || push!(deps, ld)), loopdependencies(opp))
        foreach(ld -> ((ld ∈ deps) || push!(deps, ld)), reduceddependencies(opp))
        if isload(opp)
            (opp.ref == compref) && return true
        else
            store_load_deps!(deps, opp, compref) && return true
        end
    end
    false
end
function store_load_deps(op::Operation)
    isstore(op) || return nothing
    deps = copy(loopdependencies(op))
    store_load_deps!(deps, op) ? deps : nothing
end
function add_constant_offset_load_elmination_cost!(
    X, R, choose_to_inline, ls::LoopSet, op::Operation, iters, unrollsyms::UnrollSymbols, u₁reduces::Bool, u₂reduces::Bool, Wshift::Int, size_T::Int, opisininnerloop::Bool
)
    @unpack u₁loopsym, u₂loopsym, vectorized = unrollsyms
    offset, uid = maxnegativeoffset(ls, op, unrollsyms)
    if -4 < offset < 0
        udependent_reduction = (-1 - offset) / 3
        uindependent_increase = (4 + offset) / 3
        rt, lat, rp = cost(ls, op, vectorized, Wshift, size_T)
        rt *= iters
        rp = opisininnerloop ? rp : zero(rp)
        # if loadintostore(ls, op) # For now, let's just avoid unrolling in this way...
        #     rt = Inf
        # end
        # u_uid is getting eliminated
        # we treat this as the unrolled loop getting eliminated is split into 2 parts:
        # 1 a non-cost-reduced part, with factor udependent_reduction
        # 2 a cost-reduced part, with factor uindependent_increase
        (r, i) = if uid == 1 # u₁reduces was false
            @assert !u₁reduces
            u₂reduces ? (4, 2) : (3, 1)
        elseif uid == 2 # u₂reduces was false
            @assert !u₂reduces
            u₁reduces ? (4, 3) : (2, 1)
        else
            throw("uid somehow did not return 1 or 2, even though offset > -4.")
        end
        X[r] += rt * uindependent_increase
        R[r] += rp * uindependent_increase
        X[i] += rt * udependent_reduction
        R[i] += rp * udependent_reduction
        choose_to_inline[] = true
        return true
    else
        return false
    end
end

function update_costs!(costs, cost, u₁reduces, u₂reduces)
    if u₁reduces & u₂reduces
        costs[4] += cost
    elseif u₂reduces # cost decreased by unrolling u₂loop
        costs[2] += cost
    elseif u₁reduces # cost decreased by unrolling u₁loop
        costs[3] += cost
    else # no cost decrease; cost must be repeated
        costs[1] += cost
    end
end

# Just tile outer two loops?
# But optimal order within tile must still be determined
# as well as size of the tiles.
function evaluate_cost_tile(
    ls::LoopSet, order::Vector{Symbol}, unrollsyms::UnrollSymbols
)
    N = length(order)
    @assert N ≥ 2 "Cannot tile merely $N loops!"
    @unpack u₁loopsym, u₂loopsym, vectorized = unrollsyms
    cacheunrolled!(ls, u₁loopsym, u₂loopsym, vectorized)
    # u₂loopsym = order[1]
    # u₁loopsym = order[2]
    ops = operations(ls)
    nops = length(ops)
    included_vars = fill!(resize!(ls.included_vars, nops), false)
    reduced_by_unrolling = fill(false, 2, 2, nops)
    descendentsininnerloop = fill!(resize!(ls.place_after_loop, nops), false)
    innerloop = last(order)
    iters = fill(-99.9, nops)
    nested_loop_syms = Symbol[]# Set{Symbol}()
    # Need to check if fusion is possible
    size_T = biggest_type_size(ls)
    W, Wshift = lsvecwidthshift(ls, vectorized, size_T)
    # W, Wshift = VectorizationBase.pick_vector_width_shift(length(ls, vectorized), size_T)::Tuple{Int,Int}
    # costs =
    # cost_mat[1] / ( unrolled * u₂loopsym)
    # cost_mat[2] / ( u₂loopsym)
    # cost_mat[3] / ( unrolled)
    # cost_mat[4]
    cost_vec = cost_vec_buf(ls)
    reg_pressure = reg_pres_buf(ls)
    # @inbounds reg_pressure[2] = 1
    # @inbounds reg_pressure[3] = 1
    iter::Float64 = 1.0
    u₁reached = u₂reached = false
    choose_to_inline = Ref(false)
    copyto!(names(ls), order); reverse!(names(ls))
    prefetch_good_idea = false
    for n ∈ 1:N
        itersym = order[n]
        if itersym == u₁loopsym
            u₁reached = true
        elseif itersym == u₂loopsym
            u₂reached = true
        end
        # Add to set of defined symbles
        push!(nested_loop_syms, itersym)
        looplength = length(ls, itersym)
        iter *= itersym === vectorized ? num_iterations(looplength, W) : looplength
        # check which vars we can define at this level of loop nest
        for (id, op) ∈ enumerate(ops)
            # isconstant(op) && continue
            # @assert id == identifier(op)+1 # testing, for now
            # won't define if already defined...
            included_vars[id] && continue
            # it must also be a subset of defined symbols
            all(ld -> ld ∈ nested_loop_syms, loopdependencies(op)) || continue
            rd = reduceddependencies(op)
            if hasintersection(rd, @view(nested_loop_syms[1:end-length(rd)]))
                return 0,0,Inf,false
            end
            if isstore(op)
                loadstoredeps = store_load_deps(op)
                if !isnothing(loadstoredeps)
                    any(s -> (s ∉ loadstoredeps), nested_loop_syms) && return 0,0,Inf,false
                end
            end
            included_vars[id] = true
            if isconstant(op)
                depends_on_u₁, depends_on_u₂ = isunrolled_sym(op, u₁loopsym, u₂loopsym)
                reduced_by_unrolling[1,1,id] = !depends_on_u₁
                reduced_by_unrolling[2,1,id] = !depends_on_u₂
            else
                depends_on_u₁ = isu₁unrolled(op)
                depends_on_u₂ = isu₂unrolled(op)
                reduced_by_unrolling[1,1,id] = (u₁reached) & !depends_on_u₁
                reduced_by_unrolling[2,1,id] = (u₂reached) & !depends_on_u₂
            end
            # cost is reduced by unrolling u₁ if it is interior to u₁loop (true if either u₁reached, or if depends on u₂ [or u₁]) and doesn't depend on u₁
            reduced_by_unrolling[1,2,id] = (u₁reached | depends_on_u₂) & !depends_on_u₁
            reduced_by_unrolling[2,2,id] = (u₂reached | depends_on_u₁) & !depends_on_u₂
            iters[id] = iter
            innerloop ∈ loopdependencies(op) && set_upstream_family!(descendentsininnerloop, op, true)
        end
    end
    irreducible_storecosts = 0.0
    for (id, op) ∈ enumerate(ops)
        iters[id] == -99.9 && continue
        opisininnerloop = descendentsininnerloop[id]
        
        u₁reducesrt, u₂reducesrt = reduced_by_unrolling[1,1,id], reduced_by_unrolling[2,1,id]
        u₁reducesrp, u₂reducesrp = reduced_by_unrolling[1,2,id], reduced_by_unrolling[2,2,id]
        if isload(op)
            if add_constant_offset_load_elmination_cost!(cost_vec, reg_pressure, choose_to_inline, ls, op, iters[id], unrollsyms, u₁reducesrp, u₂reducesrp, Wshift, size_T, opisininnerloop)
                continue
            elseif load_elimination_cost_factor!(cost_vec, reg_pressure, choose_to_inline, ls, op, iters[id], unrollsyms, Wshift, size_T)
                continue
            end
        #elseif isconstant(op)
        end
        rt, lat, rp = cost(ls, op, vectorized, Wshift, size_T)
        if isload(op)
            if !iszero(prefetchisagoodidea(ls, op, UnrollArgs(4, unrollsyms, 4, 0)))
                # rt += 0.5VectorizationBase.REGISTER_SIZE / CACHELINE_SIZE
                prefetch_good_idea = true
            end
        end
        # rp = (opisininnerloop && !(loadintostore(ls, op))) ? rp : zero(rp) # we only care about register pressure within the inner most loop
        rp = opisininnerloop ? rp : zero(rp) # we only care about register pressure within the inner most loop
        rto = rt
        rt *= iters[id]
        if isstore(op) & (!u₁reducesrt) & (!u₂reducesrt)
            irreducible_storecosts += rt
        end
        # iiter = convert(Int, iters[id]); @show u₁reducesrt, u₂reducesrt, op, rt, rto, rp, iiter
        update_costs!(cost_vec, rt, u₁reducesrt, u₂reducesrt)
        update_costs!(reg_pressure, rp, u₁reducesrp, u₂reducesrp)
    end
    # reg_pressure[1] = max(reg_pressure[1], length(ls.outer_reductions))
    # @inbounds ((cost_vec[4] > 0) || ((cost_vec[2] > 0) & (cost_vec[3] > 0))) || return 0,0,Inf,false
    costpenalty = (sum(reg_pressure) > REGISTER_COUNT) ? 2 : 1
    u₁v = vectorized === u₁loopsym; u₂v = vectorized === u₂loopsym
    round_uᵢ = prefetch_good_idea ? (u₁v ? 1 : (u₂v ? 2 : 0)) : 0
    if (irreducible_storecosts / sum(cost_vec) ≥ 0.25) && !any(op -> loadintostore(ls, op), operations(ls))
        u₁, u₂ = (1, 1)
        ucost = unroll_cost(cost_vec, 1, 1, length(getloop(ls, u₁loopsym)), length(getloop(ls, u₂loopsym)))
    else
        u₁, u₂, ucost = solve_unroll(ls, u₁loopsym, u₂loopsym, cost_vec, reg_pressure, W, vectorized, round_uᵢ)
    end
    outer_reduct_penalty = length(ls.outer_reductions) * (u₁ + isodd(u₁))
    favor_bigger_u₂ = u₁ - u₂
    # favor_smaller_vectorized = (u₁v ? u₁ : -u₁) + (u₂v ?  u₂ : -u₂)
    favor_smaller_vectorized = (u₁v ⊻ u₂v) ? (u₁v ? u₁ - u₂ : u₂ - u₁) : 0
    favor_u₁_vectorized = -0.2u₁v
    favoring_heuristics = favor_bigger_u₂ + 0.5favor_smaller_vectorized + favor_u₁_vectorized
    costpenalty = costpenalty * ucost + stride_penalty(ls, order) + outer_reduct_penalty + favoring_heuristics
    u₁, u₂, costpenalty, choose_to_inline[]
end


struct LoopOrders
    syms::Vector{Symbol}
    buff::Vector{Symbol}
end
function LoopOrders(ls::LoopSet)
    syms = copy(ls.loopsymbols)
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
        sn = state[n]
        if sn == N - n
            if n == N
                return false
            else
                state[n] = 0
            end
        else
            state[n] = sn + 1
            break
        end
    end
    true
end
# I doubt this is the most efficient algorithm, but it's the simplest thing
# that I could come up with.
function Base.iterate(lo::LoopOrders, state)
    advance_state!(state) || return nothing
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
    best_vec = first(new_order)
    while true
        for new_vec ∈ new_order
            cost_temp = evaluate_cost_unroll(ls, new_order, new_vec, lowest_cost)
            if cost_temp < lowest_cost
                lowest_cost = cost_temp
                copyto!(best_order, new_order)
                best_vec = new_vec
            end
        end
        iter = iterate(lo, state)
        iter === nothing && return best_order, best_vec, lowest_cost
        new_order, state = iter
    end
end


"""
This function searches for unrolling combinations that will cause LoopVectorization to generate invalid code.

Currently, it is only searching for one scenario, based on how `isunrolled_sym` and lowering currently work.
`isunrolledsym` tries to avoid the creation of excessive numbers of accumulation vectors in the case of reductions.
If an unrolled loop isn't reduced, it will need separate vectors.
But separate vectors for a reduced loop are not needed. Separate vectors will help to break up dependency chains,
so you want to unroll at least one of the loops. However, reductions demand combining all the separate vectors,
and each vector also eats a valuable register, so it's best to avoid excessive numbers these accumulation vectors.


If a reduced op depends on both unrolled loops (u1 and u2), it will check over which of these it is reduced. If...
neither: cannot avoid unrolling it along both
one of them: don't unroll the reduced loop
both of them: don't unroll along u2 (unroll along u1)

Now, a look at lowering:
It interleaves u1-unrolled operations in an effort to improve superscalar parallelism,
while u2-unrolled operations are lowered by block. E.g., op_u2id_u1id (as they're printed):

u2 = 0
opa_0_0 = fa(...)
opa_0_1 = fa(...)
opa_0_2 = fa(...)
opb_0_0 = fb(...)
opb_0_1 = fb(...)
opb_0_2 = fb(...)
u2 += 1
opa_1_0 = fa(...)
opa_1_1 = fa(...)
opa_1_2 = fa(...)
opb_1_0 = fb(...)
opb_1_1 = fb(...)
opb_1_2 = fb(...)

what if `opa` vectors were not replicated across u1?
opa_0_ = fa(...)
opa_0_ = fa(...)
opa_0_ = fa(...)

Then unless `fa` was taking the previous `opa_0_`s as an argument and updating them, this would be wrong, because it'd be overwriting the previous `opa_0_` values.
"""
function reject_candidate(op::Operation, u₁loopsym::Symbol, u₂loopsym::Symbol)
    if iscompute(op) && u₁loopsym ∈ reduceddependencies(op) && u₁loopsym ∈ loopdependencies(op)
        if u₂loopsym ∉ reduceddependencies(op) && !any(opp -> name(opp) === name(op), parents(op))
            return true
        end
    end
    false
end

function reject_candidate(ls::LoopSet, u₁loopsym::Symbol, u₂loopsym::Symbol)
    for op ∈ operations(ls)
        reject_candidate(op, u₁loopsym, u₂loopsym) && return true
    end
    false
end
inlinedecision(inline::Int, shouldinline::Bool) = iszero(inline) ? shouldinline : isone(inline)
function choose_tile(ls::LoopSet)
    lo = LoopOrders(ls)
    best_order = copyto!(ls.loop_order.bestorder, lo.syms)
    bestu₁ = bestu₂ = best_vec = first(best_order) # filler
    u₁ = u₂ = 0; lowest_cost = Inf; shouldinline = false
    for newu₂ ∈ lo.syms, newu₁ ∈ lo.syms#@view(new_order[nt+1:end])
        ((newu₁ == newu₂) || reject_candidate(ls, newu₁, newu₂)) && continue
        new_order, state = iterate(lo) # right now, new_order === best_order
        while true
            for new_vec ∈ new_order # view to skip first
                u₁temp, u₂temp, cost_temp, shouldinline_temp = evaluate_cost_tile(ls, new_order, UnrollSymbols(newu₁, newu₂, new_vec))
                # if cost_temp < lowest_cost # leads to 4 vmovapds
                if cost_temp ≤ lowest_cost # lead to 2 vmovapds
                    lowest_cost = cost_temp
                    u₁, u₂ = u₁temp, u₂temp
                    best_vec = new_vec
                    bestu₂ = newu₂
                    bestu₁ = newu₁
                    shouldinline = shouldinline_temp
                    copyto!(best_order, new_order)
                    save_tilecost!(ls)
                end
            end
            iter = iterate(lo, state)
            iter === nothing && break
            new_order, state = iter
        end
    end
    ls.loadelimination[] = shouldinline
    best_order, bestu₁, bestu₂, best_vec, u₁, u₂, lowest_cost, looplengthprod(ls) < 4097.0
end
# Last in order is the inner most loop
function choose_order_cost(ls::LoopSet)
    resize!(ls.loop_order, length(ls.loopsymbols))
    if num_loops(ls) > 1
        torder, tunroll, ttile, tvec, tU, tT, tc, shouldinline = choose_tile(ls)
    else
        torder = names(ls) # dummy
        tunroll = ttile = tvec = Symbol("##undefined##") # dummy
        tU = tT = 0 # dummy
        tc = Inf
    end
    uorder, uvec, uc = choose_unroll_order(ls, tc)
    if num_loops(ls) > 1 && tc ≤ uc
        copyto!(ls.loop_order.bestorder, torder)
        return torder, tunroll, ttile, tvec, tU, tT, tc, shouldinline
        # return torder, tvec, 4, 4#5, 5
    else
        copyto!(ls.loop_order.bestorder, uorder)
        UF, uunroll = determine_unroll_factor(ls, uorder, uvec)
        return uorder, uunroll, Symbol("##undefined##"), uvec, UF, -1, uc, true
    end
end
function choose_order(ls::LoopSet)
    order, unroll, tile, vec, u₁, u₂, c = choose_order_cost(ls)
    order, unroll, tile, vec, u₁, u₂
end

# function register_pressure(ls::LoopSet, u₁, u₂)
#     if u₂ == -1
#         sum(register_pressure, operations(ls))
#     else
#         rp = @view ls.reg_pres[:,1]
#         u₁ * u₂ * rp[1] + u₁ * rp[2] + rp[3] + rp[4]
#     end
# end
# function register_pressure(ls::LoopSet)
#     order, unroll, tile, vec, u₁, u₂ = choose_order(ls)
#     register_pressure(ls, u₁, u₂)
# end

