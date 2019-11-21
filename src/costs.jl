

# @enum CostScaling begin
    # NoCost
    # Linear
    # Unique
# end

struct InstructionCost
    scalar_latency::Int
    scalar_reciprical_throughput::Float64
    scaling::Float64 # sentinel values: -2 == no scaling; -1 == scaling, >0 ->  == latency == reciprical throughput
    
end
InstructionCost(sl, srt) = InstructionCost(sl, srt, NoCost)

function scalar_cost(instruction::InstructionCost)#, ::Type{T} = Float64) where {T}
    instruction.scalar_latency, instruction.scalar_reciprical_throughput
end
function vector_cost(instruction::InstructionCost, Wshift, ::Type{T} = Float64) where {T}
    sl, srt = scalar_cost(instruction)
    scaling = instruction.scaling
    if scaling == NoCost || Wshift == 0
        returnsl, srt
    elseif scaling == Linear
        srt *= 1 << (Wshift + VectorizationBase.intlog2(sizeof(T)) - 4)
        if (sizeof(T) << Wshift) == VectorizationBase.REGISTER_SIZE # These instructions experience double latency with zmm
            sl += sl
        end
    end
    
    sl, srt
end
function cost(instruction::InstructionCost, Wshift, ::Type{T}) where {T}
    Wshift == 0 ? scalar_cost(instruction) : vector_cost(instruction, Wshift, T)
end

const COST = Dict{Symbol,InstructionCost}(
    :getindex => InstructionCost(3,0.5),
    :setindex! => InstructionCost(3,1.0), # but not a part of dependency chains, so not really twice as expensive?
    :+ => InstructionCost(4,0.5),
    :- => InstructionCost(4,0.5),
    :* => InstructionCost(4,0.5),
    :/ => InstructionCost(13,4.0,),
    :muladd => InstructionCost(0.5,4), # + and * will fuse into this, so much of the time they're not twice as expensive
    :sqrt => InstructionCost(),
    :log => InstructionCost(,,52.5),
    :exp => InstructionCost(,,30.0),
    :sin => InstructionCost(),
    :cos => InstructionCost(),
    :sincos => InstructionCost(),
    :
)

# const SIMDPIRATES_COST = Dict{Symbol,InstructionCost}()
# const SLEEFPIRATES_COST = Dict{Symbol,InstructionCost}()

# const MODULE_LOOKUP = Dict{Symbol,Dict{Symbol,InstructionCost}}(
    # :Base => BASE_COST,
    # :SIMDPirates => SIMDPIRATES_COST,
    # :SLEEFPirates => SLEEFPIRATES_COST
# )

