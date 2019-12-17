

# @enum CostScaling begin
    # NoCost
    # Linear
    # Unique
# end

struct InstructionCost
    scaling::Float64 # sentinel values: -3 == no scaling; -2 == offset_scaling, -1 == linear scaling, >0 ->  == latency == reciprical throughput
    scalar_reciprical_throughput::Float64
    scalar_latency::Int
    register_pressure::Int
end
InstructionCost(sl::Int, srt::Float64, scaling::Float64 = -3.0) = InstructionCost(scaling, srt, sl, 0)

function scalar_cost(instruction::InstructionCost)#, ::Type{T} = Float64) where {T}
    @unpack scalar_reciprical_throughput, scalar_latency, register_pressure = instruction
    scalar_reciprical_throughput, scalar_latency, register_pressure
end
function vector_cost(instruction::InstructionCost, Wshift, sizeof_T)
    srt, sl, srp = scalar_cost(instruction)
    scaling = instruction.scaling
    if scaling == -3.0 || Wshift == 0 # No scaling
        return srt, sl, srp
    elseif scaling == -2.0 # offset scaling
        srt *= 1 << (Wshift + VectorizationBase.intlog2(sizeof_T) - 4)
        if (sizeof_T << Wshift) == 64 # VectorizationBase.REGISTER_SIZE # These instructions experience double latency with zmm
            sl += sl
        end
    elseif scaling == -1.0 # linear scaling
        W = 1 << Wshift
        extra_latency = sl - srt
        srt *= W
        sl = srt + extra_latency
    else # we assume custom cost, and that latency == recip_throughput
        sl, srt = scaling, scaling
    end    
    srt, sl, srp
end
instruction_cost(instruction::Symbol) = get(COST, instruction, OPAQUE_INSTRUCTION)
scalar_cost(instr::Symbol) = scalar_cost(instruction_cost(instr))
vector_cost(instr::Symbol, Wshift, sizeof_T) = vector_cost(instruction_cost(instr), Wshift, sizeof_T)
function cost(instruction::InstructionCost, Wshift, sizeof_T)
    Wshift == 0 ? scalar_cost(instruction) : vector_cost(instruction, Wshift, sizeof_T)
end

function cost(instruction::Symbol, Wshift, sizeof_T)
    cost(
        instruction_cost(instruction),
        Wshift, sizeof_T
    )
end


# Just a semi-reasonable assumption; should not be that sensitive to anything other than loads
const OPAQUE_INSTRUCTION = InstructionCost(50, 50.0, -1.0, VectorizationBase.REGISTER_COUNT)

# Comments on setindex!
# 1. Not a part of dependency chains, so not really twice as expensive as getindex?
# 2. getindex loads a register, not setindex!, but we place cost on setindex!
#    as a heuristic means of approximating register pressure, since many loads can be
#    consolidated into a single register. The number of LICM-ed setindex!, on the other
#    hand, should indicate how many registers we're keeping live for the sake of eventually storing.
const COST = Dict{Symbol,InstructionCost}(
    :getindex => InstructionCost(3,0.5,-3.0,0),
    :setindex! => InstructionCost(3,1.0,-3.0,1),
    :zero => InstructionCost(1,0.5),
    :one => InstructionCost(3,0.5),
    :(+) => InstructionCost(4,0.5),
    :(-) => InstructionCost(4,0.5),
    :(*) => InstructionCost(4,0.5),
    :(/) => InstructionCost(13,4.0,-2.0),
    :(==) => InstructionCost(1, 0.5),
    :isequal => InstructionCost(1, 0.5),
    :(&) => InstructionCost(1, 0.5),
    :(|) => InstructionCost(1, 0.5),
    :(>) => InstructionCost(1, 0.5),
    :(<) => InstructionCost(1, 0.5),
    :(>=) => InstructionCost(1, 0.5),
    :(<=) => InstructionCost(1, 0.5),
    :inv => InstructionCost(13,4.0,-2.0,1),
    :muladd => InstructionCost(4,0.5), # + and * will fuse into this, so much of the time they're not twice as expensive
    :fma => InstructionCost(4,0.5), # + and * will fuse into this, so much of the time they're not twice as expensive
    :vmuladd => InstructionCost(4,0.5), # + and * will fuse into this, so much of the time they're not twice as expensive
    :vfma => InstructionCost(4,0.5), # + and * will fuse into this, so much of the time they're not twice as expensive
    :vfmadd => InstructionCost(4,0.5), # + and * will fuse into this, so much of the time they're not twice as expensive
    :vfmsub => InstructionCost(4,0.5), # - and * will fuse into this, so much of the time they're not twice as expensive
    :vfnmadd => InstructionCost(4,0.5), # + and -* will fuse into this, so much of the time they're not twice as expensive
    :vfnmsub => InstructionCost(4,0.5), # - and -* will fuse into this, so much of the time they're not twice as expensive
    :sqrt => InstructionCost(15,4.0,-2.0),
    :log => InstructionCost(20,20.0,40.0,20),
    :exp => InstructionCost(20,20.0,20.0,18),
    :sin => InstructionCost(18,15.0,68.0,23),
    :cos => InstructionCost(18,15.0,68.0,26),
    :sincos => InstructionCost(25,22.0,70.0,26)
)
for (k, v) ∈ COST # so we can look up Symbol(typeof(function))
    COST[Symbol("typeof(", k, ")")] = v
end

const CORRESPONDING_REDUCTION = Dict{Symbol,Symbol}(
    :(+) => :vsum,
    :(-) => :vsum,
    :(*) => :vprod,
    :(&) => :vall,
    :(|) => :vany,
    :muladd => :vsum,
    :fma => :vsum,
    :vmuladd => :vsum,
    :vfma => :vsum,
    :vfmadd => :vsum,
    :vfmsub => :vsum,
    :vfnmadd => :vsum,
    :vfnmsub => :vsum
)
const REDUCTION_TRANSLATION = Dict{Symbol,Symbol}(
    :(+) => :evadd,
    :vadd => :evadd,
    :(*) => :evmul,
    :vmul => :evmul,
    :(-) => :evadd,
    :vsub => :evadd,
    :(/) => :evmul,
    :vdiv => :evmul,
    :muladd => :evadd,
    :fma => :evadd,
    :vmuladd => :evadd,
    :vfma => :evadd,
    :vfmadd => :evadd,
    :vfmsub => :evadd,
    :vfnmadd => :evadd,
    :vfnmsub => :evadd
)
const REDUCTION_ZERO = Dict{Symbol,Symbol}(
    :(+) => :zero,
    :vadd => :zero,
    :(*) => :one,
    :vmul => :one,
    :(-) => :zero,
    :vsub => :zero,
    :(/) => :one,
    :vdiv => :one,
    :muladd => :zero,
    :fma => :zero,
    :vmuladd => :zero,
    :vfma => :zero,
    :vfmadd => :zero,
    :vfmsub => :zero,
    :vfnmadd => :zero,
    :vfnmsub => :zero    
)
# const SIMDPIRATES_COST = Dict{Symbol,InstructionCost}()
# const SLEEFPIRATES_COST = Dict{Symbol,InstructionCost}()

# const MODULE_LOOKUP = Dict{Symbol,Dict{Symbol,InstructionCost}}(
    # :Base => BASE_COST,
    # :SIMDPirates => SIMDPIRATES_COST,
    # :SLEEFPirates => SLEEFPIRATES_COST
# )

