# using LightGraphs


isdense(::Type{<:DenseArray}) = true

"""
ShortVector{T} simply wraps a Vector{T}, but uses a different hash function that is faster for short vectors to support using it as the keys of a Dict.
This hash function scales O(N) with length of the vectors, so it is slow for long vectors.
"""
struct ShortVector{T} <: DenseVector{T}
    data::Vector{T}
end
Base.@propagate_inbounds Base.getindex(x::ShortVector, I...) = x.data[I...]
Base.@propagate_inbounds Base.setindex!(x::ShortVector, v, I...) = x.data[I...] = v
@inbounds Base.length(x::ShortVector) = length(x.data)
@inbounds Base.size(x::ShortVector) = size(x.data)
@inbounds Base.strides(x::ShortVector) = strides(x.data)
@inbounds Base.push!(x::ShortVector, v) = push!(x.data, v)
@inbounds Base.append!(x::ShortVector, v) = append!(x.data, v)
function Base.hash(x::ShortVector, h::UInt)
    @inbounds for n ∈ eachindex(x)
        h = hash(x[n], h)
    end
    h
end



@enum NodeType begin
    memload
    memstore
    compute
end


struct Operation
    elementbytes::Int
    instruction::Symbol
    node_type::NodeType
    # dependencies::ShortVector{Symbol}
    dependencies::Set{Symbol}
    # dependencies::Set{Symbol}
    parents::Vector{Operation}
    children::Vector{Operation}
    numerical_metadata::Vector{Float64}
    symbolic_metadata::Vector{Symbol}
    function Operation(elementbytes, instruction, node_type)
        new(
            elementbytes, instruction, node_type,
            Set{Symbol}(), Operation[], Operation[], Float64[], Symbol[]
        )
    end
end

function isreduction(op::Operation)
    (op.node_type == memstore) && (length(op.symbolic_metadata) < length(op.dependencies)) && issubset(op.symbolic_metadata, op.dependencies)
end
isload(op::Operation) = op.node_type == memload
isstore(op::Operation) = op.node_type == memstore
accesses_memory(op::Operation) = isload(op) | isstore(op)
elsize(op::Operation) = op.elementbytes
dependson(op::Operation, sym::Symbol) = sym ∈ op.dependencies

function stride(op::Operation, sym::Symbol)
    @assert accesses_memory(op) "This operation does not access memory!"
    # access stride info?
end
# function

struct Node
    type::DataType
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
function Loop(itersymbol::Symbol, rangesym::Symbol, rangehint::Int = 1_000_000)
    Loop( itersymbol, rangehint, rangesym, false )
end

# Must make it easy to iterate
struct LoopSet
    loops::Dict{Symbol,Loop} # sym === loops[sym].itersymbol
    operations::Vector{Operation}
    
end

function Base.length(ls::LoopSet, is::Symbol)
    ls.loops[is].rangehint
end
function variables(ls::LoopSet)

end
function loopdependencies(var::Operation)

end
function sym(var::Operation)
    
end
function instruction(var::Operation)

end
function accesses_memory(var::Operation)

end
function stride(var::Operation, sym::Symbol)

end
operations(ls::LoopSet) = ls.operations
function cost(var::Operation, unrolled::Symbol, dim::Int)
    c = cost(instruction(var), Wshift, T)::Int
    if accesses_memory(var)
        # either vbroadcast/reductionstore, vmov(a/u)pd, or gather/scatter
        if (unrolled ∈ loopdependencies(var))
            if (stride(var, unrolled) != 1) || !isdense(var) # need gather/scatter
                c *= W
            # else # vmov(a/u)pd
            end
        elseif sym(var) == :setindex! # broadcast or reductionstore; if store we want to penalize reduction
            c *= 2
        end
    end
    c
end

    # Base._return_type()

function biggest_type(ls::LoopSet)
    maximum(elsize, ls.operations)
end



# evaluates cost of evaluating loop in given order
function evaluate_cost_unroll(
    ls::LoopSet, order::ShortVector{Symbol}, unrolled::Symbol, max_cost = typemax(Float64)
)
    included_vars = Set{Symbol}()
    nested_loop_syms = Set{Symbol}()
    total_cost = 0.0
    iter = 1.0
    # Need to check if fusion is possible
    W, Wshift = VectorizationBase.pick_vector_width_shift(length(ls, unrolled), biggest_type(ls))::Tuple{Int,Int}
    for itersym ∈ order
        # Add to set of defined symbles
        push!(nested_loop_syms, itersym)
        liter = Float64(length(ls, itersym))
        if itersym == unrolled
            liter /= W
        end
        iter *= liter
        # check which vars we can define at this level of loop nest
        for var ∈ operations(ls)
            # won't define if already defined...
            sym(var) ∈ included_vars && continue
            # it must also be a subset of defined symbols
            loopdependencies(var) ⊆ nested_loop_syms || continue
            added_vars += 1
            push!(included_vars, sym(var))
            
            total_cost += iter * cost(var, W, Wshift, unrolled, liter)
            total_cost > max_cost && return total_cost # abort if more expensive; we only want to know the cheapest
        end
    end
    total_cost
end

# only covers unrolled ops; everything else considered lifted?
function depchain_cost!(
    skip::Set{Symbol}, ls::LoopSet, op::Operation, unrolled::Symbol, Wshift::Int, size_T::Int
)
    
end
   
function determine_unroll_factor(
    ls::LoopSet, order::ShortVector{Symbol}, unrolled::Symbol, Wshift::Int, size_T::Int
)
    # The strategy is to use an unroll factor of 1, unless there appears to be loop carried dependencies (ie, num_reductions > 0)
    # The assumption here is that unrolling provides no real benefit, unless it is needed to enable OOO execution by breaking up these dependency chains
    num_reductions = sum(isreduction, operations(ls))
    iszero(num_reductions) && return 1
    # So if num_reductions > 0, we set the unroll factor to be high enough so that the CPU can be kept busy
    # if there are, U = max(1, round(Int, max(latency) * throughput / num_reductions)) = max(1, round(Int, latency / (recip_througput * num_reductions)))
    latency = 0
    recip_throughput = 0.0
    visited_nodes = Set{Symbol}()
    for op ∈ operations(ls)
        if isreduction(op) && dependson(op, unrolled)
            l, rt = cost_of_chain()
            num_reductions += 1
            sl, rt = cost(instruction(op), Wshift, size_T)
            latency = max(sl, latency)
            recip_throughput += rt
        end
    end
    

    
end
function evaluate_cost_tile(
    ls::LoopSet, order::ShortVector{Symbol}, tiler, tilec, max_cost = typemax(Float64)
)
    
end

struct LoopOrders
    syms::Vector{Symbol}
end
function Base.iterate(lo::LoopOrders)
    ShortVector(lo.syms), zeros(Int, length(lo.syms))# - 1)
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
    syms = copy(lo.syms)
    for i ∈ eachindex(state)
        sᵢ = state[i]
        sᵢ == 0 || swap!(syms, i, i + sᵢ)
    end
    ShortVector(syms), state
end

function choose_order(ls::LoopSet)
    is = copy(itersyms(ls))
    best_cost = typemax(Int)
    for lo ∈ LoopOrders(ls)
        cost = evaluate_cost(ls, lo)
        
    end
end

# Here, we have to figure out how to convert the loopset into a vectorized expression.
# This must traverse in a parent -> child pattern
# but order is also dependent on which loop inds they depend on.
# Requires sorting 
function lower(ls::LoopSet)

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


