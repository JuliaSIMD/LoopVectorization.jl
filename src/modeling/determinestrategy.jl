function check_linear_parents(ls::LoopSet, op::Operation, s::Symbol)
  (s ∈ loopdependencies(op)) || return true
  if isload(op) # TODO: handle loading from ranges.
    return false
  elseif !iscompute(op)
    return true
  end
  Base.sym_in(instruction(op).instr, (:vadd_nsw, :vadd_nuw, :vadd_nw, :vsub_nsw, :vsub_nuw, :vsub_nw, :(+), :vadd, :add_fast, :(-), :vsub, :sub_fast)) || return false
  for opp ∈ parents(op)
    check_linear_parents(ls, opp, s) || return false
  end
  true
end

function findparent(ls::LoopSet, s::Symbol)#opdict isn't filled when reconstructing
    id = findfirst(Base.Fix2(===,s) ∘ name, operations(ls))
    id === nothing && throw("$s not found")
    operations(ls)[id]
end
function unitstride(ls::LoopSet, op::Operation, s::Symbol)
    inds = getindices(op)
    li = op.ref.loopedindex
    # The first index is allowed to be indexed by `s`
    fi = first(inds)
    if ((fi === DISCONTIGUOUS) | (fi === CONSTANTZEROINDEX)) || (first(getstrides(op)) ≠ 1) || !unitstep(getloop(ls,s))
        return false
    # elseif !first(li)
    #     # We must check if this
    #     parent = findparent(ls, fi)
    #     indexappearences(parent, s) > 1 && return false
    end
    if length(li) > 0 && !first(li)
        parent = findparent(ls, first(inds))
        check_linear_parents(ls, parent, s) || return false
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

function cost(ls::LoopSet, op::Operation, (u₁,u₂)::Tuple{Symbol,Symbol}, vloopsym::Symbol, Wshift::Int, size_T::Int = op.elementbytes)
  isconstant(op) && return 0.0, 0, 1.0#Float64(length(loopdependencies(op)) > 0)
  isloopvalue(op) && return 0.0, 0, 0.0
  instr = instruction(op)
  if length(parents(op)) == 1
    if instr == Instruction(:-) || instr === Instruction(:sub_fast) || instr == Instruction(:+) || instr == Instruction(:add_fast)
      return 0.0, 0, 0.0
    end
  elseif iscompute(op) &&
    (Base.sym_in(instruction(op).instr, (:vadd_nsw, :vsub_nsw, :(+), :(-), :add_fast, :sub_fast)) &&
    all(opp -> (isloopvalue(opp)), parents(op)))# || (reg_count(ls) == 32) && (instruction(op).instr === :ifelse))
    # all(opp -> (isloopvalue(opp) | isconstant(opp)), parents(op))
    return 0.0, 0, 0.0
  end
  opisvectorized = isvectorized(op)
  srt, sl, srp = opisvectorized ? vector_cost(instr, Wshift, size_T) : scalar_cost(instr)
  if accesses_memory(op)
    # either vbroadcast/reductionstore, vmov(a/u)pd, or gather/scatter
    if opisvectorized
      if !unitstride(ls, op, vloopsym)# || !isdense(op) # need gather/scatter
        indices = getindices(op)
        contigind = first(indices)
        shifter = max(2,Wshift)
        if rejectinterleave(op)
          offset = 0.0 # gather/scatter, alignment doesn't matter
        else
          shifter -= 1
          offset = 0.5reg_size(ls) / cache_lnsze(ls)
          if shifter > 1 &&
            (!rejectcurly(op) && (((contigind === CONSTANTZEROINDEX) && ((length(indices) > 1) && (indices[2] === u₁) || (indices[2] === u₂))) ||
            ((u₁ === contigind) | (u₂ === contigind))))

            shifter -= 1
            offset = 0.5reg_size(ls) / cache_lnsze(ls)
          end
        end
        r = 1 << shifter
        srt = srt*r + offset
        sl *= r
      elseif isload(op) & (length(loopdependencies(op)) > 1)# vmov(a/u)pd
        # penalize vectorized loads with more than 1 loopdep
        # heuristic; more than 1 loopdep means that many loads will not be aligned
        # Roughly corresponds to double-counting loads crossing cacheline boundaries
        # TODO: apparently the new ARM A64FX CPU (with 512 bit vectors) is NOT penalized for unaligned loads
        #       would be nice to add a check for this CPU, to see if such a penalty is still appropriate.
        #       Also, once more SVE (scalable vector extension) CPUs are released, would be nice to know if
        #       this feature is common to all of them.
        srt += 0.5reg_size(ls) / cache_lnsze(ls)
        # srt += 0.25reg_size(ls) / cache_lnsze(ls)
      end
    elseif isstore(op) # broadcast or reductionstore; if store we want to penalize reduction
      srt *= 3
      sl *= 3
    end
  end
  srt, sl, Float64(srp+1)
end

    # Base._return_type()

function biggest_type_size(ls::LoopSet)
    maximum(elsize, operations(ls))
end
function hasintersection(a, b)
    for aᵢ ∈ a, bᵢ ∈ b
        aᵢ === bᵢ && return true
    end
    false
end
const num_iterations = cld

function set_vector_width!(ls::LoopSet, vloopsym::Symbol)
    W = ls.vector_width
    if !iszero(W)
        ls.vector_width = min(W, VectorizationBase.nextpow2(length(ls, vloopsym)))
    end
    nothing
end
function lsvecwidthshift(ls::LoopSet, vloopsym::Symbol, size_T = nothing)
    W = ls.vector_width
    lvec = length(ls, vloopsym)
    W = if iszero(W)
        bytes = size_T === nothing ? biggest_type_size(ls) : size_T
        reg_size(ls) ÷ bytes
    else
        min(W, VectorizationBase.nextpow2(lvec))
    end
    W, VectorizationBase.intlog2(W)
end

# evaluates cost of evaluating loop in given order
function evaluate_cost_unroll(
  ls::LoopSet, order::Vector{Symbol}, vloopsym::Symbol, max_cost::Float64 = typemax(Float64), sld::Vector{Vector{Symbol}} = store_load_deps(operations(ls))
)
  included_vars = fill!(resize!(ls.included_vars, length(operations(ls))), false)
  nested_loop_syms = Symbol[]#Set{Symbol}()
  total_cost = 0.0
  iter = 1.0
  size_T = biggest_type_size(ls)
  W, Wshift = lsvecwidthshift(ls, vloopsym, size_T)
  # Need to check if fusion is possible
  for itersym ∈ order
    cacheunrolled!(ls, itersym, Symbol(""), vloopsym)
    # Add to set of defined symbles
    push!(nested_loop_syms, itersym)
    looplength = length(ls, itersym)
    liter = itersym === vloopsym ? num_iterations(looplength, W) : looplength
    iter *= liter
    # check which vars we can define at this level of loop nest
    for (id,op) ∈ enumerate(operations(ls))
      # won't define if already defined...
      # id = identifier(op)
      included_vars[id] && continue
      # it must also be a subset of defined symbols
      loopdependencies(op) ⊆ nested_loop_syms || continue
      # hasintersection(reduceddependencies(op), nested_loop_syms) && return Inf
      (isassigned(sld, id) && any(s -> (s ∉ sld[id]), nested_loop_syms)) && return Inf
      included_vars[id] = true
      # TODO: use actual unrolls here?
      c = first(cost(ls, op, (Symbol(""),Symbol("")), vloopsym, Wshift, size_T))
      total_cost += iter * c
      0.9total_cost > max_cost && return total_cost # abort if more expensive; we only want to know the cheapest
    end
  end
  0.9total_cost + stride_penalty(ls, order) # 0.999 to place finger on scale in its favor
end

# only covers vectorized ops; everything else considered lifted?
function depchain_cost!(
    ls::LoopSet, skip::Vector{Bool}, op::Operation, unrolled::Symbol, vloopsym::Symbol, Wshift::Int, size_T::Int, rt::Float64 = 0.0, sl::Int = 0
)
    skip[identifier(op)] = true
    # depth first search
    for opp ∈ parents(op)
      skip[identifier(opp)] && continue
        rt, sl = depchain_cost!(ls, skip, opp, unrolled, vloopsym, Wshift, size_T, rt, sl)
    end
    # Basically assuming memory and compute don't conflict, but everything else does
    # Ie, ignoring the fact that integer and floating point operations likely don't either
    if iscompute(op)
      rtᵢ, slᵢ = cost(ls, op, (unrolled,Symbol("")), vloopsym, Wshift, size_T)
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
function unroll_no_reductions(ls, order, vloopsym)
  size_T = biggest_type_size(ls)
  W, Wshift = lsvecwidthshift(ls, vloopsym, size_T)

  compute_rt = load_rt = store_rt = 0.0
  unrolled = last(order)
  i = 0
  while reject_reorder(ls, unrolled, false)
    i += 1
    unrolled = order[end-i]
  end
  if unrolled === vloopsym && length(order) > 1
    unrolled_candidate = order[end-1]
    if !reject_reorder(ls, unrolled_candidate, false)
      unrolled = unrolled_candidate
    end
  end
    # latency not a concern, because no depchains
  compute_l = 0.0
  rpp = 0 # register pressure proportional to unrolling
  rpc = 0 # register pressure independent of unroll factor
  for op ∈ operations(ls)
    isu₁unrolled(op) || continue
    rt, sl, rpop = cost(ls, op, (unrolled,Symbol("")), vloopsym, Wshift, size_T)
    if iscompute(op)
      compute_rt += rt
      compute_l += sl
      rpc += max(zero(rpop),rpop - one(rpop)) # constant loads for special functions reused with unrolling
    elseif isload(op)
      load_rt += rt
      rpp += rpop # loads are proportional to unrolling
    elseif isstore(op)
      store_rt += rt
    end
  end
  # heuristic guess
  # roundpow2(min(4, round(Int, (compute_rt + load_rt + 1) / compute_rt)))
  memory_rt = load_rt + store_rt
  u = if compute_rt ≤ 1
    4
  elseif compute_rt > memory_rt
    clamp(round(Int, compute_l / compute_rt), 1, 4)
    # max(1, VectorizationBase.nextpow2( min( 4, round(Int, 8 / compute_rt) ) ))
  elseif iszero(load_rt)
    iszero(store_rt) ? 4 : max(1, min(4, round(Int, 2compute_rt / store_rt)))
  else
    max(1, min(4, round(Int, 1.75compute_rt / load_rt)))
  end
  # @show load_rt, store_rt, compute_rt, compute_l, u, rpc, rpp
  # u = min(u, max(1, (reg_count(ls) ÷ max(1,round(Int,rp)))))
  # commented out here is to decide to align loops
  # if memory_rt > compute_rt && isone(u) && (length(order) > 1) && (last(order) === vloopsym) && length(getloop(ls, last(order))) > 8W
  #     ls.align_loops[] = findfirst(operations(ls)) do op
  #         isstore(op) && isu₁unrolled(op)
  #     end
  # end
  if unrolled === vloopsym
    u = demote_unroll_factor(ls, u, vloopsym)
  end
  remaining_reg = max(8, (reg_count(ls) - round(Int,rpc))) # spilling a few consts isn't so bad
  if compute_l ≥ 4compute_rt ≥ 4rpp
    # motivation for skipping division by loads here: https://github.com/microhh/stencilbuilder/blob/master/julia/stencil_julia_4th.jl
    # Some values:
    # (load_rt, store_rt, compute_rt, compute_l, u, rpc, rpp) = (52.0, 3.0, 92.0, 736.0, 4, 0.0, 52.0)
    # This is fastest when `u = 4`, but `reg_constraint` was restricting it to 1.
    # Obviously, this limitation on number of registers didn't seem so important in practice.
    # So, heuristically I check if compute latency dominates the problem, in which case unrolling could be expected to benefit us.
    # Ideally, we'd count the number of loads that actually have to be live at a given time. But this heuristic is hopefully okay for now.
    reg_constraint = max(1, remaining_reg)
  else
    reg_constraint = max(1, remaining_reg ÷ max(1,round(Int,rpp)))
  end
  clamp(u, 1, reg_constraint), unrolled
    # rt = max(compute_rt, load_rt + store_rt)
    # # (iszero(rt) ? 4 : max(1, roundpow2( min( 4, round(Int, 16 / rt) ) ))), unrolled
    # (iszero(rt) ? 4 : max(1, VectorizationBase.nextpow2( min( 4, round(Int, 8 / rt) ) ))), unrolled
end
function determine_unroll_factor(
    ls::LoopSet, order::Vector{Symbol}, unrolled::Symbol, vloopsym::Symbol
)
    cacheunrolled!(ls, unrolled, Symbol(""), vloopsym)
    size_T = biggest_type_size(ls)
    W, Wshift = lsvecwidthshift(ls, vloopsym, size_T)

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
        if isreduction(op)
            rt, sl = depchain_cost!(ls, visited_nodes, op, unrolled, vloopsym, Wshift, size_T)
            if isouterreduction(ls, op) ≠ -1 || unrolled ∉ reduceddependencies(op)
                latency = max(sl, latency)
            end
            # if unrolled ∈ loopdependencies(op)
            #     compute_recip_throughput_u += rt
            # else
            compute_recip_throughput += rt
            # end
        elseif isload(op)
            load_recip_throughput += first(cost(ls, op, (unrolled,Symbol("")), vloopsym, Wshift, size_T))
        elseif isstore(op)
            store_recip_throughput += first(cost(ls, op, (unrolled,Symbol("")), vloopsym, Wshift, size_T))
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
    W = ls.vector_width
    if !iszero(W) && isstaticloop(loop)
        UFW = maybedemotesize(UF*W, length(loop))
        UF = cld(UFW, W)
    end
    UF
end

function determine_unroll_factor(ls::LoopSet, order::Vector{Symbol}, vloopsym::Symbol)
    num_reductions = count_reductions(ls)
    # The strategy is to use an unroll factor of 1, unless there appears to be loop carried dependencies (ie, num_reductions > 0)
    # The assumption here is that unrolling provides no real benefit, unless it is needed to enable OOO execution by breaking up these dependency chains
    loopindexesbit = ls.loopindexesbit
    if iszero(length(loopindexesbit)) || ((!loopindexesbit[getloopid(ls, vloopsym)]))
        if iszero(num_reductions)
            return unroll_no_reductions(ls, order, vloopsym)
        else
            return determine_unroll_factor(ls, order, vloopsym, num_reductions)
        end
    elseif iszero(num_reductions) # handle `BitArray` loops w/out reductions
        return 8 ÷ ls.vector_width, vloopsym
    else # handle `BitArray` loops with reductions
        rttemp, ltemp = determine_unroll_factor(ls, order, vloopsym, vloopsym)
        UF = min(8, VectorizationBase.nextpow2(max(1, round(Int, ltemp / (rttemp) ) )))
        UFfactor = 8 ÷ ls.vector_width
        cld(UF, UFfactor)*UFfactor, vloopsym
    end
end
# function scale_unrolled()
# end
function determine_unroll_factor(ls::LoopSet, order::Vector{Symbol}, vloopsym::Symbol, num_reductions::Int)
    innermost_loop = last(order)
    rt = Inf; rtcomp = Inf; latency = Inf; best_unrolled = Symbol("")
    for unrolled ∈ order
        reject_reorder(ls, unrolled, false) && continue
        rttemp, ltemp = determine_unroll_factor(ls, order, unrolled, vloopsym)
        rtcomptemp = rttemp + (0.01 * ((vloopsym === unrolled) + (unrolled === innermost_loop) - latency))
        if rtcomptemp < rtcomp
            rt = rttemp
            rtcomp = rtcomptemp
            latency = ltemp
            best_unrolled = unrolled
        end
    end
    # min(8, roundpow2(max(1, round(Int, latency / (rt * num_reductions) ) ))), best_unrolled
    lrtratio  = latency / rt
    if lrtratio ≥ 7.0
        UF = 8
    else
        UF = VectorizationBase.nextpow2(round(Int, clamp(lrtratio, 1.0, 4.0)))
    end
    if best_unrolled === vloopsym
        UF = demote_unroll_factor(ls, UF, vloopsym)
    end
    UF, best_unrolled
end

@inline function unroll_cost(X, u₁, u₂, u₁L, u₂L)
    u₂factor = (num_iterations(u₂L, u₂)/u₂L)
    u₁factor = (num_iterations(u₁L, u₁)/u₁L)
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
    R₁, R₂, R₃, R₄ = R[1], R[2], R[3], R[4]
    RR = R₄
    u₁best, u₂best = 0, 0
    bestcost = Inf
    for u₁temp ∈ u₁range
        for u₂temp ∈ u₂range
            RR ≥ u₁temp*u₂temp*R₁ + u₁temp*R₂ + u₂temp*R₃ || continue
            tempcost = unroll_cost(X, u₁temp, u₂temp, u₁L, u₂L)
            if tempcost ≤ bestcost
                bestcost = tempcost
                u₁best, u₂best = u₁temp, u₂temp
            end
        end
    end
    u₁best, u₂best, bestcost
end

function solve_unroll_lagrange(X, R, u₁L, u₂L, u₁step::Int, u₂step::Int, atleast32registers::Bool)
    X₁, X₂, X₃, X₄ = X[1], X[2], X[3], X[4]
    # If we don't have opmask registers, masks probably occupy a vector register (e.g., on CPUs with AVX but not AVX512)
    R₁, R₂, R₃, R₄ = R[1], R[2], R[3], R[4]
    iszero(R₃) || return solve_unroll_iter(X, R, u₁L, u₂L, u₁step:u₁step:10, u₂step:u₂step:10)
    RR = R₄
    a = R₂^2*X₃ -R₁*X₄ * R₂ - R₁*X₂*RR
    b = R₁ * X₄ * RR - R₁ * X₄ * RR - 2X₃*RR*R₂
    c = X₃*RR^2
    discriminant = b^2 - 4a*c
    discriminant < 0 && return -1,-1,Inf
    u₁float = max((sqrt(discriminant) + b) / (-2a), float(u₁step)) # must be at least 1
    u₂float = (RR - u₁float*R₂)/(u₁float*R₁)
    if !(isfinite(u₂float) & isfinite(u₁float)) # brute force
        u₁low = u₂low = 1
        u₁high = iszero(X₂) ? 2 : (atleast32registers ? 8 : 6)
        u₂high = iszero(X₃) ? 2 : (atleast32registers ? 8 : 6)
        return solve_unroll_iter(X, R, u₁L, u₂L, u₁low:u₁step:u₁high, u₂low:u₂step:u₂high)
    end
    u₁low = floor(Int, u₁float)
    u₂low = max(u₂step, floor(Int, 0.8u₂float)) # must be at least 1
    u₁high = solve_unroll_constT(R, u₂low) + u₁step
    u₂high = solve_unroll_constU(R, u₁low) + u₂step
    if u₁low ≥ u₁high
        u₁low = solve_unroll_constT(R, u₂high)
    end
    if u₂low ≥ u₂high
        u₂low = solve_unroll_constU(R, u₁high)
    end
    maxunroll = atleast32registers ? (((X₂ > 0) & (X₃ > 0)) ? 10 : 8) : 6
    u₁low = (clamp(u₁low, 1, maxunroll) ÷ u₁step) * u₁step
    u₂low = (clamp(u₂low, 1, maxunroll) ÷ u₂step) * u₂step
    u₁high = clamp(u₁high, 1, maxunroll)
    u₂high = clamp(u₂high, 1, maxunroll)
    solve_unroll_iter(X, R, u₁L, u₂L, reverse(u₁low:u₁step:u₁high), reverse(u₂low:u₂step:u₂high))
end

function solve_unroll_constU(R::AbstractVector, u₁::Int)
    denom = u₁ * R[1] + R[3]
    iszero(denom) && return 8
    floor(Int, (R[4] - u₁*R[2]) / denom)
end
function solve_unroll_constT(R::AbstractVector, u₂::Int)
    denom = u₂ * R[1] + R[2]
    iszero(denom) && return 8
    floor(Int, (R[4] - u₂*R[3]) / denom)
end
# function solve_unroll_constT(ls::LoopSet, u₂::Int)
#     R = @view ls.reg_pres[:,1]
#     denom = u₂ * R[1] + R[2]
#     iszero(denom) && return 8
#     floor(Int, (dynamic_register_count() - R[3] - R[4] - u₂*R[5]) / (u₂ * R[1] + R[2]))
# end
# Tiling here is about alleviating register pressure for the UxT
function solve_unroll(X, R, u₁max, u₂max, u₁L, u₂L, u₁step, u₂step, atleast32registers::Bool)
    # iszero(first(R)) && return -1,-1,Inf #solve_smalltilesize(X, R, u₁max, u₂max)
    u₁, u₂, cost = solve_unroll_lagrange(X, R, u₁L, u₂L, u₁step, u₂step, atleast32registers)
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
    W::Int, vloopsym::Symbol, rounduᵢ::Int
)
    (u₁step, u₂step) = if rounduᵢ == 1 # max is to safeguard against some weird arch I've never heard of.
        (clamp(cache_lnsze(ls) ÷ reg_size(ls), 1, 4), 1)
    elseif rounduᵢ == 2
        (1, clamp(cache_lnsze(ls) ÷ reg_size(ls), 1, 4))
    elseif rounduᵢ == -1
        (8 ÷ ls.vector_width, 1)
    elseif rounduᵢ == -2
        (1, 8 ÷ ls.vector_width)
    else
        (1, 1)
    end
    u₁loop = getloop(ls, u₁loopsym)
    u₂loop = getloop(ls, u₂loopsym)
    solve_unroll(
        u₁loopsym, u₂loopsym, cost_vec, reg_pressure, W, vloopsym, u₁loop, u₂loop, u₁step, u₂step, reg_count(ls) ≥ 32
    )
end

function solve_unroll(
    u₁loopsym::Symbol, u₂loopsym::Symbol,
    cost_vec::AbstractVector{Float64},
    reg_pressure::AbstractVector{Float64},
    W::Int, vloopsym::Symbol,
    u₁loop::Loop, u₂loop::Loop,
    u₁step::Int, u₂step::Int,
    atleast32registers::Bool
)
    maxu₂base = maxu₁base = atleast32registers ? 10 : 6#8
    maxu₂ = maxu₂base#8
    maxu₁ = maxu₁base#8
    u₁L = length(u₁loop)
    u₂L = length(u₂loop)
    if isstaticloop(u₂loop)
      if u₂loopsym !== vloopsym && u₂L ≤ 4
        if isstaticloop(u₁loop)
          u₁ = max(solve_unroll_constT(reg_pressure, u₂L), 1)
          u₁ = maybedemotesize(u₁, u₁loopsym === vloopsym ? cld(u₁L,W) : u₁L)
        else
          u₁ = clamp(solve_unroll_constT(reg_pressure, u₂L), 1, 8)
        end
        return u₁, u₂L, unroll_cost(cost_vec, u₁, u₂L, u₁L, u₂L)
      end
      u₂Ltemp = u₂loopsym === vloopsym ? cld(u₂L, W) : u₂L
      maxu₂ = min(4maxu₂, u₂Ltemp)
    end
    if isstaticloop(u₁loop)
      if u₁loopsym !== vloopsym && u₁L ≤ 4
        if isstaticloop(u₂loop)
          u₂ = max(solve_unroll_constU(reg_pressure, u₁L), 1)
          u₂ = maybedemotesize(u₂, u₂loopsym === vloopsym ? cld(u₂L,W) : u₂L)
        else
          u₂ = clamp(solve_unroll_constU(reg_pressure, u₁L), 1, 8)
        end
        return u₁L, u₂, unroll_cost(cost_vec, u₁L, u₂, u₁L, u₂L)
      end
      u₁Ltemp = u₁loopsym === vloopsym ? cld(u₁L, W) : u₁L
      maxu₁ = min(4maxu₁, u₁Ltemp)
    end
    if u₁loopsym === vloopsym
        u₁Lf = u₁L / W
    else
        u₁Lf = Float64(u₁L)
    end
    if u₂loopsym === vloopsym
        u₂Lf = u₂L / W
    else
        u₂Lf = Float64(u₂L)
    end
    u₁, u₂, cost = solve_unroll(cost_vec, reg_pressure, maxu₁, maxu₂, u₁Lf, u₂Lf, u₁step, u₂step, atleast32registers)
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
    isdiscontig = first(loopdeps) === DISCONTIGUOUS
    # isdiscontig = isdiscontiguous(op.ref)
    loopedindex = op.ref.loopedindex
    if !isdiscontig && all(loopedindex) && !(any(==(CONSTANTZEROINDEX), loopdeps))
        return loopdeps
    end
    loopdepsret = Symbol[]
    for i ∈ eachindex(loopedindex)
        if loopedindex[i]
            loopdeps[i+isdiscontig] === CONSTANTZEROINDEX || push!(loopdepsret, loopdeps[i+isdiscontig])
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
    opstrides[1] = 1.0 + (first(loopdependencies(op.ref)) === DISCONTIGUOUS) + (first(loopdependencies(op.ref)) === CONSTANTZEROINDEX)
    l = Float64(length(getloop(ls, first(loopdeps))))
    for i ∈ 2:length(loopdeps)
        looplength = length(getloop(ls, loopdeps[i-1]))
        opstrides[i] = opstrides[i-1] * looplength
        l *= looplength
        # opstrides[i] = opstrides[i-1] * length(loops[i-1])
    end
    penalty = 0.0
    for i ∈ eachindex(order)
        id = findfirst(Base.Fix2(===,order[i]), loopdeps)
        if !(id === nothing)
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
    @unpack u₁loopsym, u₂loopsym, vloopsym = unrollsyms
    (vloopsym == u₁loopsym || vloopsym == u₂loopsym) && return 0, 0x00
    (isu₁unrolled(op) && isu₂unrolled(op)) || return 0, 0x00
    u₁step = step(getloop(ls, u₁loopsym))
    u₂step = step(getloop(ls, u₂loopsym))
    (isknown(u₁step) & isknown(u₂step)) || return 0, 0x00
    abs(gethint(u₁step)) == abs(gethint(u₂step)) || return 0, 0x00

    istranslation = 0
    inds = getindices(op); li = op.ref.loopedindex
    for i ∈ eachindex(li)
        if !li[i]
            opp = findparent(ls, inds[i + (first(inds) === DISCONTIGUOUS)])
            if isu₁unrolled(opp) && isu₂unrolled(opp)
                if Base.sym_in(instruction(opp).instr, (:vadd_nsw, :(+)))
                    return i, 0x03 # 00000011 - both positive
                elseif Base.sym_in(instruction(opp).instr, (:vsub_nsw, :(-)))
                    oppp1 = parents(opp)[1]
                    if isu₁unrolled(oppp1)
                        return i, 0x01 # 00000001 - u₁ positive, u₂ negative
                    else#isu₂unrolled(oppp1)
                        return i, 0x02 # 00000010 - u₂ positive, u₁ negative
                    end
                end
            end
        end
    end
    0, 0x00
end
# function maxnegativeoffset_old(ls::LoopSet, op::Operation, u::Symbol)
#     opmref = op.ref
#     opref = opmref.ref
#     mno = typemin(Int)
#     id = 0
#     opoffs = opref.offsets
#     for opp ∈ operations(ls)
#         opp === op && continue
#         oppmref = opp.ref
#         oppref = oppmref.ref
#         sameref(opref, oppref) || continue
#         opinds = getindicesonly(op)
#         oppinds = getindicesonly(opp)
#         oppoffs = oppref.offsets
#         # oploopi = opmref.loopedindex
#         # opploopi = oppmref.loopedindex
#         mnonew = typemin(Int)
#         for i ∈ eachindex(opinds)
#             if opinds[i] === u
#                 mnonew = (opoffs[i] - oppoffs[i])
#             elseif opoffs[i] != oppoffs[i]
#                 mnonew = 1
#                 break
#             end
#         end
#         if mno < mnonew < 0
#             mno = mnonew
#             id = identifier(opp)
#         end
#     end
#     mno, id
# end
function maxnegativeoffset(ls::LoopSet, op::Operation, u::Symbol)
    mno::Int = typemin(Int)
    id = 0
    isknown(step(getloop(ls, u))) || return mno, id
    omop = offsetloadcollection(ls)
    collectionid, opind = omop.opidcollectionmap[identifier(op)]
    collectionid == 0 && return mno, id
    @unpack opids = omop

    # offsetcol = offsets[collectionid]
    opidcol = opids[collectionid]
    opid = identifier(op)
    # opoffs = offsetcol[opind]
    opoffs = getoffsets(op)
    ops = operations(ls)
    opindices = getindicesonly(op)
    for (i,oppid) ∈ enumerate(opidcol)
        opid == oppid && continue
        opp = ops[oppid]
        oppoffs = getoffsets(opp)
        mnonew::Int = typemin(Int)
        for i ∈ eachindex(opindices)
            if opindices[i] === u
                mnonew = ((opoffs[i] % Int) - (oppoffs[i] % Int))
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
    @unpack u₁loopsym, u₂loopsym, vloopsym = unrollsyms
    mno = typemin(Int)
    i = 0
    if u₁loopsym !== vloopsym
        mnou₁ = first(maxnegativeoffset(ls, op, u₁loopsym))
        if mnou₁ > mno
            i = 1
            mno = mnou₁
        end
    end
    if u₂loopsym !== vloopsym
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
    @unpack u₁loopsym, u₂loopsym, vloopsym = unrollsyms
    if !iszero(first(isoptranslation(ls, op, unrollsyms)))
        rt, lat, rp = cost(ls, op, (u₁loopsym, u₂loopsym), vloopsym, Wshift, size_T)
        # rt = Core.ifelse(isvectorized(op), 0.5rt, rt)
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
        #             return (0.25, dynamic_register_count() == 32 ? 2.0 : 1.0)
        #             # return (0.25, 1.0)
        #             return true
        #         end
        #     end
        # end
        # # (0.25, dynamic_register_count() == 32 ? 1.2 : 1.0)
        # (0.25, 1.0)
        # cost_vec[1] -= rt
        # cost_vec[1] -= 0.5625 * iters
      # cost_vec[1] -= 0.5625 * iters / 2
      # @show rto, 0.8rt, op
        reg_pressure[1] += 0.25rp
        cost_vec[2] += rt
        reg_pressure[2] += rp
        cost_vec[3] += rt
        # currently only place `reg_pressure[3]` is updated
        reg_pressure[3] += rp
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
    for ld ∈ loopdependencies(opp)
      (ld ∈ deps) || push!(deps, ld)
    end
    for ld ∈ reduceddependencies(opp)
      (ld ∈ deps) || push!(deps, ld)
    end
    if isload(opp)
      (opp.ref == compref) && return true
    else
      store_load_deps!(deps, opp, compref) && return true
    end
  end
  false
end
function store_load_deps(op::Operation)::Union{Nothing,Vector{Symbol}}
  isstore(op) || return nothing
  deps = copy(loopdependencies(op))
  store_load_deps!(deps, op) ? deps : nothing
end
function store_load_deps(ops::Vector{Operation})
  sld = Vector{Vector{Symbol}}(undef, length(ops))
  for i ∈ eachindex(sld)
    sldᵢ = store_load_deps(ops[i])
    sldᵢ ≢ nothing && (sld[i] = sldᵢ)
  end
  sld
end

function add_constant_offset_load_elmination_cost!(
    X, R, choose_to_inline, ls::LoopSet, op::Operation, iters, unrollsyms::UnrollSymbols, u₁reduces::Bool, u₂reduces::Bool, Wshift::Int, size_T::Int, opisininnerloop::Bool
)
    @unpack u₁loopsym, u₂loopsym, vloopsym = unrollsyms
    offset, uid = maxnegativeoffset(ls, op, unrollsyms)
    if -4 < offset < 0
        udependent_reduction = (-1 - offset) / 3
        uindependent_increase = (4 + offset) / 3
        rt, lat, rp = cost(ls, op, (u₁loopsym, u₂loopsym), vloopsym, Wshift, size_T)
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
        X[r             ] += rt * uindependent_increase
        R[r == 3 ? 4 : r] += rp * uindependent_increase
        X[i             ] += rt * udependent_reduction
        R[i == 3 ? 4 : i] += rp * udependent_reduction
        choose_to_inline[] = true
        return true
    else
        return false
    end
end

function update_cost_vec!(costs, cost, u₁reduces, u₂reduces)
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
function update_reg_pres!(rp, cost, u₁reduces, u₂reduces)
    if u₁reduces# & u₂reduces
        rp[4] -= cost
    elseif u₂reduces # cost decreased by unrolling u₂loop
        rp[2] += cost
    # elseif u₁reduces # cost decreased by unrolling u₁loop
        # rp[4] -= cost
    else # no cost decrease; cost must be repeated
        rp[1] += cost
    end
end
function child_dependent_u₁u₂(op::Operation)
  u₁ = u₂ = false
  for opc ∈ children(op)
    u₁ |= isu₁unrolled(opc)
    u₂ |= isu₂unrolled(opc)
  end
  u₁, u₂
end
function evaluate_cost_tile(ls::LoopSet, order::Vector{Symbol}, unrollsyms::UnrollSymbols, anyisbit::Bool = false)
  nops = length(operations(ls))
  iters = Vector{Float64}(undef, nops)
  reduced_by_unrolling = Array{Bool}(undef, 2, 2, nops)
  fill_children!(ls)
  evaluate_cost_tile!(iters, reduced_by_unrolling, ls, order, unrollsyms, anyisbit)
end
function evaluate_cost_tile!(
  iters::Vector{Float64}, reduced_by_unrolling::Array{Bool,3}, ls::LoopSet, order::Vector{Symbol}, unrollsyms::UnrollSymbols, anyisbit::Bool,
  sld::Vector{Vector{Symbol}} = store_load_deps(operations(ls)), holdopinreg::Vector{Bool} = holdopinregister(ls)
)
  N = length(order)
  @assert N ≥ 2 "Cannot tile merely $N loops!"
  @unpack u₁loopsym, u₂loopsym, vloopsym = unrollsyms
  cacheunrolled!(ls, u₁loopsym, u₂loopsym, vloopsym)
  # println("\n")
  # @show order unrollsyms
  # u₂loopsym = order[1]
  # u₁loopsym = order[2]
  ops = operations(ls)
  nops = length(ops)
  included_vars = resize!(ls.included_vars, nops)
  descendentsininnerloop = resize!(ls.place_after_loop, nops)
  @inbounds for i ∈ 1:nops
    included_vars[i] = false
    descendentsininnerloop[i] = false
    reduced_by_unrolling[1,1,i] = false
    reduced_by_unrolling[2,1,i] = false
    reduced_by_unrolling[1,2,i] = false
    reduced_by_unrolling[2,2,i] = false
    iters[i] = -99.9
  end
  innerloop = last(order)
  nested_loop_syms = Symbol[]# Set{Symbol}()
  # Need to check if fusion is possible
  size_T = biggest_type_size(ls)
  W, Wshift = lsvecwidthshift(ls, vloopsym, size_T)
  # costs =
  # cost_mat[1] / ( unrolled * u₂loopsym)
  # cost_mat[2] / ( u₂loopsym)
  # cost_mat[3] / ( unrolled)
  # cost_mat[4]
  
  cost_vec = cost_vec_buf(ls)
  reg_pressure = reg_pres_buf(ls)
  iter::Float64 = 1.0
  u₁reached = u₂reached = false
  choose_to_inline = Ref(false)
  copyto!(names(ls), order); reverse!(names(ls))
  prefetch_good_idea = false
  # goes from outer to inner
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
    iter *= itersym === vloopsym ? num_iterations(looplength, W) : looplength
    # check which vars we can define at this level of loop nest
    for (id, op) ∈ enumerate(ops)
      # isconstant(op) && continue
      # @assert id == identifier(op)+1 # testing, for now
      # won't define if already defined...
      included_vars[id] && continue
      # it must also be a subset of defined symbols
      _maybe_continue = true
      for ld ∈ loopdependencies(op)
        mbc = false
        for nls ∈ nested_loop_syms
          if ld === nls
            mbc = true
            break
          end
        end
        if !mbc
          _maybe_continue = false
          break
        end
      end
      # @assert _maybe_continue == all(ld -> ld ∈ nested_loop_syms, loopdependencies(op))
      _maybe_continue || continue
      # all(ld -> ld ∈ nested_loop_syms, loopdependencies(op)) || continue
      # rd = reduceddependencies(op)
      # if hasintersection(rd, @view(nested_loop_syms[1:end-length(rd)]))
      #     return 0,0,Inf,false
      # end
      (isassigned(sld, id) && any(s -> (s ∉ sld[id]), nested_loop_syms)) && return 0,0,Inf,false
      included_vars[id] = true
      if isconstant(op)
        depends_on_u₁, depends_on_u₂ = isunrolled_sym(op, u₁loopsym, u₂loopsym, vloopsym)
        reduced_by_unrolling[1,1,id] = !depends_on_u₁
        reduced_by_unrolling[2,1,id] = !depends_on_u₂
      else
        depends_on_u₁ = isu₁unrolled(op)
        depends_on_u₂ = isu₂unrolled(op)
        reduced_by_unrolling[1,1,id] = (u₁reached) & !depends_on_u₁
        reduced_by_unrolling[2,1,id] = (u₂reached) & !depends_on_u₂
      end
      # cost is reduced by unrolling u₁ if it is interior to u₁loop (true if either u₁reached, or if depends on u₂ [or u₁]) and doesn't depend on u₁
      inner₁ = u₁reached | depends_on_u₂
      inner₂ = u₂reached | depends_on_u₁
      # if isconstantop(op)
      # if iscompute(op)
      #   @show inner₁, depends_on_u₁, inner₂, depends_on_u₂, op
      # end
      reduced_by_unrolling[1,2,id] = inner₁ & !depends_on_u₁
      reduced_by_unrolling[2,2,id] = inner₂ & !depends_on_u₂
      # else
      #   reduced_by_unrolling[1,2,id] = inner₁ & !depends_on_u₁
      #   reduced_by_unrolling[2,2,id] = inner₂ & !depends_on_u₂
      # end
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
        # println("constoffelim")
        continue
      elseif load_elimination_cost_factor!(cost_vec, reg_pressure, choose_to_inline, ls, op, iters[id], unrollsyms, Wshift, size_T)
        # println("loadelim")
        # A[i,j-1], A[i,j]
        continue
      end
      #elseif isconstant(op)
    end
    rt, lat, rp = cost(ls, op, (u₁loopsym, u₂loopsym), vloopsym, Wshift, size_T)
    if isload(op) & (!prefetch_good_idea)
      prefetch_good_idea = prefetchisagoodidea(ls, op, UnrollArgs(ls, 4, unrollsyms, 4, 0)) ≠ 0
    end
    # rp = (opisininnerloop && !(loadintostore(ls, op))) ? rp : zero(rp) # we only care about register pressure within the inner most loop
    rp = opisininnerloop ? rp : zero(rp) # we only care about register pressure within the inner most loop
    rto = rt
    rt *= iters[id]
    # @show (u₁reducesrt, u₂reducesrt), (u₁reducesrp, u₂reducesrp), rto, rt, lat, rp, op
    if isstore(op) & (!u₁reducesrt) & (!u₂reducesrt)
      irreducible_storecosts += rt
    end
    update_cost_vec!(cost_vec, rt, u₁reducesrt, u₂reducesrt)
    # if child_dependent_u₁u₂(op)
    if iscompute(op) | isload(op)
      if !holdopinreg[id]
        rp = max(zero(rp), rp - one(rp))
      elseif innerloop ∈ loopdependencies(op)
        u₁c, u₂c = child_dependent_u₁u₂(op)
        # The idea here is that we check if the operation is executed in the innermost loop nest
        #   - If this is true, then the operation must be recalculated on every iteration of the inner most loop
        # 
        # Then we check if it's children don't depend on both unrolled loops. If they do not depend on both,
        # we assume that these children can safely overwrite the results of this operation.
        # (Because this op has to be recalculated on the next iteration, there's no point to holding it somewhere)
        if !(u₁c & u₂c)
        # if u₁reducesrp & u₂reducesrp & !(u₁c & u₂c)
          rp = max(zero(rp), rp - one(rp))
        end
      end
    end
    # @show u₁reducesrp, u₂reducesrp, rp, op
    update_reg_pres!(reg_pressure, rp, u₁reducesrp, u₂reducesrp)
    # end
    # update_costs!(reg_pressure, rp, u₁reducesrp, u₂reducesrp)
  end
  # reg_pressure[1] = max(reg_pressure[1], length(ls.outer_reductions))
  # @inbounds ((cost_vec[4] > 0) || ((cost_vec[2] > 0) & (cost_vec[3] > 0))) || return 0,0,Inf,false
  # reg_pres[4] == remaining_registers
  costpenalty = ((reg_pressure[1] + reg_pressure[2] + reg_pressure[3]) > reg_pressure[4]) ? 2 : 1
  u₁v = vloopsym === u₁loopsym; u₂v = vloopsym === u₂loopsym
  visbit = anyisbit && ls.loopindexesbit[getloopid(ls,vloopsym)]
  round_uᵢ = if visbit
    (u₁v ? -1 : (u₂v ? -2 : 0))
  elseif prefetch_good_idea
    (u₁v ? 1 : (u₂v ? 2 : 0))
  else
    0
  end
  if (irreducible_storecosts / sum(cost_vec) ≥ 0.5) && !any(op -> loadintostore(ls, op), operations(ls))
    u₁, u₂ = if visbit
      vecsforbyte = 8 ÷ ls.vector_width
      u₁v ? (vecsforbyte,1) : (1,vecsforbyte)
    else
      (1, 1)
    end
    ucost = unroll_cost(cost_vec, 1, 1, length(getloop(ls, u₁loopsym)), length(getloop(ls, u₂loopsym)))
  else
    u₁, u₂, ucost = solve_unroll(ls, u₁loopsym, u₂loopsym, cost_vec, reg_pressure, W, vloopsym, round_uᵢ)
  end
  outer_reduct_penalty = length(ls.outer_reductions) * (u₁ + isodd(u₁))
  favor_bigger_u₂ = u₁ - u₂
  # favor_smaller_vloopsym = (u₁v ? u₁ : -u₁) + (u₂v ?  u₂ : -u₂)
  favor_smaller_vectorized = (u₁v ⊻ u₂v) ? (u₁v ? u₁ - u₂ : u₂ - u₁) : 0
  favor_u₁_vectorized = -0.2u₁v
  favoring_heuristics = favor_bigger_u₂ + 0.5favor_smaller_vectorized + favor_u₁_vectorized
  costpenalty = costpenalty * ucost + stride_penalty(ls, order) + outer_reduct_penalty + favoring_heuristics
  u₁, u₂, costpenalty, choose_to_inline[]
end

function holdopinregister(ls::LoopSet)
  ops = operations(ls)
  childdeps = fill(false, length(ops))
  # childdeps = fill(false, length(ls.loops), length(ops))
  for i ∈ eachindex(ops)
    op = ops[i]
    ld = loopdependencies(op)
    found = false
    for opc ∈ children(op)
      for j ∈ loopdependencies(opc)
        if j ∉ ld
          found = true
          break
        end
      end
      if found
        childdeps[i] = true
        break
      end
    end
  end
  childdeps
end

# struct LoopOrders
#     syms::Vector{Int}
#     buff::Vector{Int}
# end
# function LoopOrders(ls::LoopSet)
#   syms = collect(Base.OneTo(num_loops(ls)))
#   LoopOrders(syms, similar(syms))
# end
struct LoopOrders
  syms_nr::Vector{Symbol}
  syms_r::Vector{Symbol}
  buff::Vector{Symbol}
end

function outer_reduct_loopordersplit(ls::LoopSet)
  ops = operations(ls)
  nonouterreducts = Int[]
  for i ∈ eachindex(ops)
    i ∈ ls.outer_reductions || push!(nonouterreducts, i)
  end
  reductsyms = Symbol[]
  nonreductsyms = Symbol[]
  for l ∈ ls.loopsymbols
    isreduct = false
    for opid ∈ nonouterreducts
      if l ∈ reduceddependencies(ops[opid])
        isreduct = true
        push!(reductsyms, l)
        break
      end
    end
    isreduct || push!(nonreductsyms, l)
  end
  reductsyms, nonreductsyms
end
function loopordersplit(ls::LoopSet)
  reductsyms = Symbol[]
  nonreductsyms = Symbol[]
  for l ∈ ls.loopsymbols
    isreduct = false
    for op ∈ operations(ls)
      if l ∈ reduceddependencies(op)
        isreduct = true
        push!(reductsyms, l)
        break
      end
    end
    isreduct || push!(nonreductsyms, l)
  end
  reductsyms, nonreductsyms
end
function LoopOrders(ls::LoopSet)
  if length(ls.outer_reductions) == 0
    reductsyms, nonreductsyms = loopordersplit(ls)
  else
    reductsyms, nonreductsyms = outer_reduct_loopordersplit(ls)
  end
  LoopOrders(nonreductsyms, reductsyms, Vector{Symbol}(undef, length(ls.loopsymbols)))
end

nonreductview(lo::LoopOrders) = view(lo.buff, 1:length(lo.syms_nr))
reductview(lo::LoopOrders) = view(lo.buff, 1+length(lo.syms_nr):length(lo.buff))
function Base.iterate(lo::LoopOrders)
  copyto!(nonreductview(lo), lo.syms_nr)
  copyto!(reductview(lo), lo.syms_r)
  nr = length(lo.syms_nr); r = length(lo.syms_r)
  state = zeros(Int, nr + r)
  lo.buff, (view(state, 1:nr), view(state, 1+nr:nr+r))
end

function advance_state!(state)
  N = length(state)
  N > 1 || return false
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
function advance_state!(state, Nr)
  state_nr = view(state, 1:Nr)
  advance_state!(state_nr) && return true
  fill!(state_nr, 0)
  advance_state!(view(state, 1+Nr:length(state)))
end
swap!(x::AbstractVector, i::Int, j::Int) = (x[j], x[i]) = (x[i], x[j])
function swap!(dest::AbstractVector{Symbol}, src::AbstractVector{Symbol}, offs::AbstractVector{Int})
  copyto!(dest, src)
  for i ∈ eachindex(offs)
    sᵢ = offs[i]
    sᵢ == 0 || swap!(dest, i, i + sᵢ)
  end
end
# This is not a good algorithm
function Base.iterate(lo::LoopOrders, (state_nr, state_r))
  if advance_state!(state_nr)
    swap!(nonreductview(lo), lo.syms_nr, state_nr)
  else
    advance_state!(state_r) || return nothing
    fill!(state_nr, 0)
    copyto!(nonreductview(lo), lo.syms_nr)
    swap!(reductview(lo), lo.syms_r, state_r)
  end
  lo.buff, (state_nr, state_r)
end

function reject_reorder(ls::LoopSet, loop::Int, isu₂::Bool)
  r = ls.validreorder[loop]
  r < Core.ifelse(isu₂, 0x01, 0x03)
end
function reject_reorder(ls::LoopSet, loop::Symbol, isu₂::Bool)
  for i ∈ eachindex(ls.loops)
    ls.loops[i].itersymbol === loop && return reject_reorder(ls, i, isu₂)
  end
  false
end

function choose_unroll_order(ls::LoopSet, lowest_cost::Float64 = Inf, sld::Vector{Vector{Symbol}} = store_load_deps(operations(ls)), v::Int = 0)
  iszero(length(offsetloadcollection(ls).opidcollectionmap)) && fill_offset_memop_collection!(ls)
  lo = LoopOrders(ls)
  new_order, state = iterate(lo) # right now, new_order === best_order
  vec_iter_prealloc = v == 0 ? new_order : [(ls.loops)[v].itersymbol]
  best_order = similar(new_order)
  best_vec = first(new_order)
  while true
    vec_iter = v == 0 ? new_order : vec_iter_prealloc
    for new_vec ∈ vec_iter
      reject_reorder(ls, new_vec, false) && continue
      cost_temp = evaluate_cost_unroll(ls, new_order, new_vec, lowest_cost, sld)
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

function choose_tile(ls::LoopSet, sld::Vector{Vector{Symbol}} = store_load_deps(operations(ls)), v::Int = 0)
  iszero(length(offsetloadcollection(ls).opidcollectionmap)) && fill_offset_memop_collection!(ls)
  lo = LoopOrders(ls)
  best_order = ls.loop_order.bestorder
  vec_iter_prealloc = v == 0 ? best_order : [(ls.loops)[v].itersymbol]
  bestu₁ = bestu₂ = best_vec = Symbol("")
  u₁ = u₂ = 0; lowest_cost = Inf; loadelim = false
  anyisbit = any(ls.loopindexesbit)

  nops = length(operations(ls))
  iters = Vector{Float64}(undef, nops)
  reduced_by_unrolling = Array{Bool}(undef, 2, 2, nops)
  holdopinreg = holdopinregister(ls)
  for newu₂ ∈ ls.loopsymbols
    reject_reorder(ls, newu₂, true) && continue
    for newu₁ ∈ ls.loopsymbols#@view(new_order[nt+1:end])
      (((newu₁ == newu₂) || reject_reorder(ls, newu₁, true)) || reject_candidate(ls, newu₁, newu₂)) && continue
      new_order, state = iterate(lo) # right now, new_order === best_order
      while true
        vec_iter = v == 0 ? new_order : vec_iter_prealloc
        for new_vec ∈ vec_iter
          reject_reorder(ls, new_vec, false) && continue
          if anyisbit && ls.loopindexesbit[getloopid(ls,new_vec)]
            # ((new_vec === newu₁) || (new_vec === newu₂)) || continue
            (new_vec === newu₁) || continue
          end
          u₁temp, u₂temp, cost_temp, loadelim_temp = evaluate_cost_tile!(iters, reduced_by_unrolling, ls, new_order, UnrollSymbols(newu₁, newu₂, new_vec), anyisbit, sld, holdopinreg)
          # if cost_temp < lowest_cost # leads to 4 vmovapds
          if cost_temp ≤ lowest_cost # lead to 2 vmovapds
            lowest_cost = cost_temp
            u₁, u₂ = u₁temp, u₂temp
            best_vec = new_vec
            bestu₂ = newu₂
            bestu₁ = newu₁
            loadelim = loadelim_temp
            copyto!(best_order, new_order)
            save_tilecost!(ls)
          end
        end
        iter = iterate(lo, state)
        iter === nothing && break
        new_order, state = iter
      end
    end
  end
  ls.loadelimination = loadelim
  # `any(...)` is a HACK: specialization can fail in Tullio's codegen, resulting in bad performance. See https://github.com/JuliaSIMD/LoopVectorization.jl/issues/245
  shouldinline = (looplengthprod(ls) < 4097.0) #|| any(op -> iscompute(op) && iszero(length(loopdependencies(op))), operations(ls))
  best_order, bestu₁, bestu₂, best_vec, u₁, u₂, lowest_cost, shouldinline
end
function mismatchedstorereductions(ls::LoopSet)
  reduceddeps = Vector{Symbol}[]
  nreduceddeps = 0
  for op ∈ operations(ls)
    isstore(op) || continue
    rd = reduceddependencies(first(parents(op)))
    if nreduceddeps ≠ 0
      length(rd) == nreduceddeps || return true
    else
      nreduceddeps = length(rd)
    end
    for rdo ∈ reduceddeps
      for s ∈ rdo
        s ∈ rd || return true
      end
    end
    push!(reduceddeps, rd)
  end
  false
end
# Last in order is the inner most loop
function choose_order_cost(ls::LoopSet, v::Int = 0)
  fill_children!(ls)
  resize!(ls.loop_order, length(ls.loopsymbols))
  sld = store_load_deps(operations(ls))
  if num_loops(ls) > 1
    torder, tunroll, ttile, tvec, tU, tT, tc, shouldinline = choose_tile(ls, sld, v)
  else
    torder = names(ls) # dummy
    tunroll = ttile = tvec = Symbol("##undefined##") # dummy
    tU = tT = 0 # dummy
    tc = Inf
  end
  uorder, uvec, uc = choose_unroll_order(ls, tc, sld, v)
  mismatched = mismatchedstorereductions(ls)
  if num_loops(ls) > 1 && tc ≤ uc
    @assert ls.loop_order.bestorder === torder
    # copyto!(ls.loop_order.bestorder, torder)
    return torder, tunroll, ttile, tvec, tU, tT, Core.ifelse(mismatched, Inf, tc), shouldinline
    # return torder, tvec, 4, 4#5, 5
  else
    copyto!(ls.loop_order.bestorder, uorder)
    UF, uunroll = determine_unroll_factor(ls, uorder, uvec)
    return uorder, uunroll, Symbol("##undefined##"), uvec, UF, -1, Core.ifelse(mismatched, Inf, uc), true
  end
end
function choose_order(ls::LoopSet)
    order, unroll, tile, vec, u₁, u₂, c = choose_order_cost(ls)
    order, unroll, tile, vec, u₁, u₂
end
