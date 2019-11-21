# using LightGraphs

struct ShortIntVector
    data::Vector{Int}
end
function Base.hash(x::ShortIntVector, h::UInt)
    d = x.data
    @inbounds for n ∈ eachindex(d)
        h = hash(d[n], h)
    end
    h
end


@enum NodeType begin
    input
    store
    reduction
end

struct Node
    type::DataType
end

# Must make it easy to iterate
struct LoopSet
    
end

# evaluates cost of evaluating loop in given order
function evaluate_cost(
    ls::LoopSet, order::ShortIntVector
)
    included_vars = Set{Symbol}
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


