# using LightGraphs


isdense(::Type{<:DenseArray}) = true

@enum NodeType begin
    memload
    memstore
    reduction
    compute
end


struct Operation
    elementbytes::Int
    instruction::Symbol
    node_type::NodeType
    parents::Vector{Operation}
    children::Vector{Operation}
    metadata::Vector{Float64}
    function Operation(elementbytes, instruction, node_type)
        new(
            elementbytes, instruction, node_type,
            Operation[], Operation[], Float64[]
        )
    end
end

isreduction(op::Operation) = op.node_type == reduction
isload(op::Operation) = op.node_type == memload
isstore(op::Operation) = op.node_type == memstore
accesses_memory(op::Operation) = isload(op) | isstore(op)
Base.eltype(var::Operation) = op.outtype

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

function stride(op::Operation, sym::Symbol)
    @assert accesses_memory(op) "This operation does not access memory!"
    # access stride info?
end
function

struct Node
    type::DataType
end

# Must make it easy to iterate
struct LoopSet
    
end

function Base.length(ls::LoopSet, is::Symbol)

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

end



# evaluates cost of evaluating loop in given order
function evaluate_cost_unroll(
    ls::LoopSet, order::ShortVector{Symbol}, unrolled::Symbol, max_cost = typemax(Int)
)
    included_vars = Set{Symbol}()
    nested_loop_syms = Set{Symbol}()
    total_cost = 0.0
    iter = 1.0
    # Need to check if fusion is possible
    # W, Wshift = VectorizationBase.pick_vector_width_shift(length(ls, unrolled), biggest_type(ls))::Tuple{Int,Int}
    for itersym ∈ order
        # Add to set of defined symbles
        push!(nested_loop_syms, itersym)
        liter = length(ls, itersym)
        if itersym == unrolled
            liter /= W
        end
        iter *= liter
        # check which vars we can define at this level of loop nest
        for var ∈ variables(ls)
            # won't define if already defined...
            sym(var) ∈ included_vars && continue
            # it must also be a subset of defined symbols
            loopdependencies(var) ⊆ nested_loop_syms || continue
            added_vars += 1
            push!(included_vars, sym(var))
            
            total_cost += iter * cost(var, W, Wshift, unrolled, liter)
            total_cost > max_cost && return total_cost # abort
        end
    end
end
function evaluate_cost_tile(
    ls::LoopSet, order::ShortVector{Symbol}, tiler, tilec, max_cost = typemax(Int)
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


