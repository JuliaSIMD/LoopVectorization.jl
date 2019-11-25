

# @enum CostScaling begin
    # NoCost
    # Linear
    # Unique
# end

struct InstructionCost
    scalar_latency::Int
    scalar_reciprical_throughput::Float64
    scaling::Float64 # sentinel values: -3 == no scaling; -2 == offset_scaling, -1 == linear scaling, >0 ->  == latency == reciprical throughput
    register_pressure::Int
end
InstructionCost(sl, srt, scaling = -3.0) = InstructionCost(sl, srt, scaling, 1)

function scalar_cost(instruction::InstructionCost)#, ::Type{T} = Float64) where {T}
    instruction.scalar_latency, instruction.scalar_reciprical_throughput
end
function vector_cost(instruction::InstructionCost, Wshift, ::Type{T} = Float64) where {T}
    sl, srt = scalar_cost(instruction)
    scaling = instruction.scaling
    if scaling == -3.0 || Wshift == 0
        return sl, srt
    elseif scaling == -2.0
        srt *= 1 << (Wshift + VectorizationBase.intlog2(sizeof(T)) - 4)
        if (sizeof(T) << Wshift) == VectorizationBase.REGISTER_SIZE # These instructions experience double latency with zmm
            sl += sl
        end
    elseif scaling == -1.0
        W = 1 << Wshift
        extra_latency = sl - srt
        srt *= W
        sl = srt + extra_latency
    else
        sl, srt = scaling, scaling
    end    
    sl, srt
end
function cost(instruction::InstructionCost, Wshift, ::Type{T}) where {T}
    Wshift == 0 ? scalar_cost(instruction) : vector_cost(instruction, Wshift, T)
end

# Just a semi-reasonable assumption; should not be that sensitive to anything other than loads
const OPAQUE_INSTRUCTION = InstructionSet(50.0, 50.0, -1.0, VectorizationBase.REGISTER_COUNT)

const COST = Dict{Symbol,InstructionCost}(
    :getindex => InstructionCost(3,0.5),
    :setindex! => InstructionCost(3,1.0), # but not a part of dependency chains, so not really twice as expensive?
    :+ => InstructionCost(4,0.5),
    :- => InstructionCost(4,0.5),
    :* => InstructionCost(4,0.5),
    :/ => InstructionCost(13,4.0,-2.0),
    :== => InstructionCost(1, 0.5),
    :isequal => InstructionCost(1, 0.5),
    :& => InstructionCost(1, 0.5),
    :| => InstructionCost(1, 0.5),
    :> => InstructionCost(1, 0.5),
    :< => InstructionCost(1, 0.5),
    :>= => InstructionCost(1, 0.5),
    :<= => InstructionCost(1, 0.5),
    :inv => InstructionCost(13,4.0,-2.0,2),
    :muladd => InstructionCost(4,0.5), # + and * will fuse into this, so much of the time they're not twice as expensive
    :fma => InstructionCost(4,0.5), # + and * will fuse into this, so much of the time they're not twice as expensive
    :vmuladd => InstructionCost(4,0.5), # + and * will fuse into this, so much of the time they're not twice as expensive
    :vfma => InstructionCost(4,0.5), # + and * will fuse into this, so much of the time they're not twice as expensive
    :vfmadd => InstructionCost(4,0.5), # + and * will fuse into this, so much of the time they're not twice as expensive
    :vfmsub => InstructionCost(4,0.5), # + and * will fuse into this, so much of the time they're not twice as expensive
    :vfnmadd => InstructionCost(4,0.5), # + and * will fuse into this, so much of the time they're not twice as expensive
    :vfnmsub => InstructionCost(4,0.5), # + and * will fuse into this, so much of the time they're not twice as expensive
    :sqrt => InstructionCost(15,4.0,-2.0),
    :log => InstructionCost(20,20.0,40.0,21),
    :exp => InstructionCost(20,20.0,20.0,19),
    :sin => InstructionCost(18,15.0,68.0,24),
    :cos => InstructionCost(18,15.0,68.0,27),
    :sincos => InstructionCost(25,22.0,70.0,27)
)

function sum_simd(x)
    s = zero(eltype(x))
    @simd for xᵢ ∈ x
        s += xᵢ
    end
    s
end
using LoopVectorization, BenchmarkTools
function sum_loopvec(x::AbstractVector{Float64})
    s = 0.0
    @vvectorize 4 for i ∈ eachindex(x)
        s += x[i]
    end
    s
end
x = rand(111);
@btime sum($x)
@btime sum_simd($x)
@btime sum_loopvec($x)


# const SIMDPIRATES_COST = Dict{Symbol,InstructionCost}()
# const SLEEFPIRATES_COST = Dict{Symbol,InstructionCost}()

# const MODULE_LOOKUP = Dict{Symbol,Dict{Symbol,InstructionCost}}(
    # :Base => BASE_COST,
    # :SIMDPirates => SIMDPIRATES_COST,
    # :SLEEFPirates => SLEEFPIRATES_COST
# )

