
struct Instruction
    mod::Symbol
    instr::Symbol
end
Instruction(instr::Symbol) = Instruction(:LoopVectorization, instr)
Base.convert(::Type{Instruction}, instr::Symbol) = Instruction(instr)
lower(instr::Instruction) = Expr(:(.), instr.mod, QuoteNode(instr.instr))
Base.Expr(instr::Instruction, args...) = Expr(:call, lower(instr), args...)::Expr
Base.hash(instr::Instruction, h::UInt64) = hash(instr.instr, hash(instr.mod, h))

const LOOPCONSTANT = Instruction(gensym())

struct InstructionCost
    scaling::Float64 # sentinel values: -3 == no scaling; -2 == offset_scaling, -1 == linear scaling, >0 ->  == latency == reciprical throughput
    scalar_reciprical_throughput::Float64
    scalar_latency::Int
    register_pressure::Int
end
InstructionCost(sl::Int, srt::Float64, scaling::Float64 = -3.0) = InstructionCost(scaling, srt, sl, 0)

nocost(c::InstructionCost) = c.scalar_reciprical_throughput == 0.0
flatcost(c::InstructionCost) = c.scaling == -3.0
offsetscaling(c::InstructionCost) = c.scaling == -2.0
linearscaling(c::InstructionCost) = c.scaling == -1.0

function scalar_cost(ic::InstructionCost)#, ::Type{T} = Float64) where {T}
    @unpack scalar_reciprical_throughput, scalar_latency, register_pressure = ic
    scalar_reciprical_throughput, scalar_latency, register_pressure
end
function vector_cost(ic::InstructionCost, Wshift, sizeof_T)
    srt, sl, srp = scalar_cost(ic)
    if flatcost(ic) || Wshift == 0 || nocost(ic) # No scaling
        return srt, sl, srp
    elseif offsetscaling(ic) # offset scaling
        srt *= 1 << (Wshift + VectorizationBase.intlog2(sizeof_T) - 4)
        if (sizeof_T << Wshift) == 64 # VectorizationBase.REGISTER_SIZE # These instructions experience double latency with zmm
            sl += sl
        end
    elseif linearscaling(ic) # linear scaling
        W = 1 << Wshift
        extra_latency = sl - srt
        srt *= W
        sl = round(Int, srt + extra_latency)
    else # we assume custom cost, and that latency == recip_throughput
        scaling = ic.scaling
        sl, srt = round(Int,scaling), scaling
    end    
    srt, sl, srp
end
instruction_cost(instruction::Instruction) = get(COST, instruction, OPAQUE_INSTRUCTION)
instruction_cost(instruction::Symbol) = instruction_cost(Instruction(instruction))
scalar_cost(instr::Instruction) = scalar_cost(instruction_cost(instr))
vector_cost(instr::Instruction, Wshift, sizeof_T) = vector_cost(instruction_cost(instr), Wshift, sizeof_T)
function cost(instruction::InstructionCost, Wshift, sizeof_T)
    Wshift == 0 ? scalar_cost(instruction) : vector_cost(instruction, Wshift, sizeof_T)
end

function cost(instruction::Instruction, Wshift, sizeof_T)
    cost(
        instruction_cost(instruction), Wshift, sizeof_T
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
const COST = Dict{Instruction,InstructionCost}(
    Instruction(:getindex) => InstructionCost(-3.0,0.5,3,1),
    Instruction(:setindex!) => InstructionCost(-3.0,1.0,3,0),
    Instruction(:zero) => InstructionCost(1,0.5),
    Instruction(:one) => InstructionCost(3,0.5),
    Instruction(:+) => InstructionCost(4,0.5),
    Instruction(:-) => InstructionCost(4,0.5),
    Instruction(:*) => InstructionCost(4,0.5),
    Instruction(:/) => InstructionCost(13,4.0,-2.0),
    Instruction(:vadd) => InstructionCost(4,0.5),
    Instruction(:vsub) => InstructionCost(4,0.5),
    Instruction(:vmul) => InstructionCost(4,0.5),
    Instruction(:vdiv) => InstructionCost(13,4.0,-2.0),
    Instruction(:abs2) => InstructionCost(4,0.5),
    Instruction(:vabs2) => InstructionCost(4,0.5),
    Instruction(:(==)) => InstructionCost(1, 0.5),
    Instruction(:isequal) => InstructionCost(1, 0.5),
    Instruction(:(&)) => InstructionCost(1, 0.5),
    Instruction(:(|)) => InstructionCost(1, 0.5),
    Instruction(:(>)) => InstructionCost(1, 0.5),
    Instruction(:(<)) => InstructionCost(1, 0.5),
    Instruction(:(>=)) => InstructionCost(1, 0.5),
    Instruction(:(<=)) => InstructionCost(1, 0.5),
    Instruction(:inv) => InstructionCost(13,4.0,-2.0,1),
    Instruction(:vinv) => InstructionCost(13,4.0,-2.0,1),
    Instruction(:muladd) => InstructionCost(4,0.5), # + and * will fuse into this, so much of the time they're not twice as expensive
    Instruction(:fma) => InstructionCost(4,0.5), # + and * will fuse into this, so much of the time they're not twice as expensive
    Instruction(:vmuladd) => InstructionCost(4,0.5), # + and * will fuse into this, so much of the time they're not twice as expensive
    Instruction(:vfma) => InstructionCost(4,0.5), # + and * will fuse into this, so much of the time they're not twice as expensive
    Instruction(:vfmadd) => InstructionCost(4,0.5), # + and * will fuse into this, so much of the time they're not twice as expensive
    Instruction(:vfmsub) => InstructionCost(4,0.5), # - and * will fuse into this, so much of the time they're not twice as expensive
    Instruction(:vfnmadd) => InstructionCost(4,0.5), # + and -* will fuse into this, so much of the time they're not twice as expensive
    Instruction(:vfnmsub) => InstructionCost(4,0.5), # - and -* will fuse into this, so much of the time they're not twice as expensive
    Instruction(:vfmadd_fast) => InstructionCost(4,0.5), # + and * will fuse into this, so much of the time they're not twice as expensive
    Instruction(:vfmsub_fast) => InstructionCost(4,0.5), # - and * will fuse into this, so much of the time they're not twice as expensive
    Instruction(:vfnmadd_fast) => InstructionCost(4,0.5), # + and -* will fuse into this, so much of the time they're not twice as expensive
    Instruction(:vfnmsub_fast) => InstructionCost(4,0.5), # - and -* will fuse into this, so much of the time they're not twice as expensive
    Instruction(:sqrt) => InstructionCost(15,4.0,-2.0),
    Instruction(:log) => InstructionCost(20,20.0,40.0,20),
    Instruction(:exp) => InstructionCost(20,20.0,20.0,18),
    Instruction(:sin) => InstructionCost(18,15.0,68.0,23),
    Instruction(:cos) => InstructionCost(18,15.0,68.0,26),
    Instruction(:sincos) => InstructionCost(25,22.0,70.0,26),
    Instruction(:identity) => InstructionCost(0,0.0,0.0,0),
    Instruction(:adjoint) => InstructionCost(0,0.0,0.0,0),
    Instruction(:transpose) => InstructionCost(0,0.0,0.0,0),
    # Symbol("##CONSTANT##") => InstructionCost(0,0.0)
)
# for (k, v) ∈ COST # so we can look up Symbol(typeof(function))
#     COST[Symbol("typeof(", lower(k), ")")] = v
# end

const CORRESPONDING_REDUCTION = Dict{Instruction,Instruction}(
    Instruction(:+) => Instruction(:vsum),
    Instruction(:-) => Instruction(:vsum),
    Instruction(:*) => Instruction(:vprod),
    Instruction(:vadd) => Instruction(:vsum),
    Instruction(:vsub) => Instruction(:vsum),
    Instruction(:vmul) => Instruction(:vprod),
    Instruction(:&) => Instruction(:vall),
    Instruction(:|) => Instruction(:vany),
    Instruction(:muladd) => Instruction(:vsum),
    Instruction(:fma) => Instruction(:vsum),
    Instruction(:vmuladd) => Instruction(:vsum),
    Instruction(:vfma) => Instruction(:vsum),
    Instruction(:vfmadd) => Instruction(:vsum),
    Instruction(:vfmsub) => Instruction(:vsum),
    Instruction(:vfnmadd) => Instruction(:vsum),
    Instruction(:vfnmsub) => Instruction(:vsum)
)
const REDUCTION_TRANSLATION = Dict{Instruction,Instruction}(
    Instruction(:+) => Instruction(:evadd),
    Instruction(:vadd) => Instruction(:evadd),
    Instruction(:*) => Instruction(:evmul),
    Instruction(:vmul) => Instruction(:evmul),
    Instruction(:-) => Instruction(:evadd),
    Instruction(:vsub) => Instruction(:evadd),
    Instruction(:/) => Instruction(:evmul),
    Instruction(:vfdiv) => Instruction(:evmul),
    Instruction(:muladd) => Instruction(:evadd),
    Instruction(:fma) => Instruction(:evadd),
    Instruction(:vmuladd) => Instruction(:evadd),
    Instruction(:vfma) => Instruction(:evadd),
    Instruction(:vfmadd) => Instruction(:evadd),
    Instruction(:vfmsub) => Instruction(:evadd),
    Instruction(:vfnmadd) => Instruction(:evadd),
    Instruction(:vfnmsub) => Instruction(:evadd)
)
const REDUCTION_ZERO = Dict{Instruction,Symbol}(
    Instruction(:+) => :zero,
    Instruction(:vadd) => :zero,
    Instruction(:*) => :one,
    Instruction(:vmul) => :one,
    Instruction(:-) => :zero,
    Instruction(:vsub) => :zero,
    Instruction(:/) => :one,
    Instruction(:vfdiv) => :one,
    Instruction(:muladd) => :zero,
    Instruction(:fma) => :zero,
    Instruction(:vmuladd) => :zero,
    Instruction(:vfma) => :zero,
    Instruction(:vfmadd) => :zero,
    Instruction(:vfmsub) => :zero,
    Instruction(:vfnmadd) => :zero,
    Instruction(:vfnmsub) => :zero    
)

lv(x) = GlobalRef(LoopVectorization, x)
# Fast functions, because common pattern is
const REDUCTION_SCALAR_COMBINE = Dict{Instruction,GlobalRef}(
    Instruction(:+) => lv(:reduced_add),
    Instruction(:vadd) => lv(:reduced_add),
    Instruction(:*) => lv(:reduced_prod),
    Instruction(:vmul) => lv(:reduced_prod),
    Instruction(:-) => lv(:reduced_add),
    Instruction(:vsub) => lv(:reduced_add),
    Instruction(:/) => lv(:reduced_prod),
    Instruction(:vfdiv) => lv(:reduced_prod),
    Instruction(:muladd) => lv(:reduced_add),
    Instruction(:fma) => lv(:reduced_add),
    Instruction(:vmuladd) => lv(:reduced_add),
    Instruction(:vfma) => lv(:reduced_add),
    Instruction(:vfmadd) => lv(:reduced_add),
    Instruction(:vfmsub) => lv(:reduced_add),
    Instruction(:vfnmadd) => lv(:reduced_add),
    Instruction(:vfnmsub) => lv(:reduced_add)
)

const FUNCTIONSYMBOLS = Dict{Type{<:Function},Instruction}(
    typeof(+) => :(+),
    typeof(SIMDPirates.vadd) => :(+),
    typeof(Base.FastMath.add_fast) => :(+),
    typeof(-) => :(-),
    typeof(SIMDPirates.vsub) => :(-),
    typeof(Base.FastMath.sub_fast) => :(-),
    typeof(*) => :(*),
    typeof(SIMDPirates.vmul) => :(*),
    typeof(Base.FastMath.mul_fast) => :(*),
    typeof(/) => :(/),
    typeof(SIMDPirates.vdiv) => :(/),
    typeof(Base.FastMath.div_fast) => :(/),
    typeof(==) => :(==),
    typeof(isequal) => :isequal,
    typeof(&) => :(&),
    typeof(|) => :(|),
    typeof(>) => :(>),
    typeof(<) => :(<),
    typeof(>=) => :(>=),
    typeof(<=) => :(<=),
    typeof(inv) => :inv,
    typeof(muladd) => :muladd,
    typeof(fma) => :fma,
    typeof(SIMDPirates.vmuladd) => :vmuladd,
    typeof(SIMDPirates.vfma) => :vfma,
    typeof(SIMDPirates.vfmadd) => :vfmadd,
    typeof(SIMDPirates.vfmsub) => :vfmsub,
    typeof(SIMDPirates.vfnmadd) => :vfnmadd,
    typeof(SIMDPirates.vfnmsub) => :vfnmsub,
    typeof(sqrt) => :sqrt,
    typeof(Base.FastMath.sqrt_fast) => :sqrt,
    typeof(SIMDPirates.vsqrt) => :sqrt,
    typeof(log) => :log,
    typeof(Base.FastMath.log_fast) => :log,
    typeof(SIMDPirates.vlog) => :log,
    typeof(SLEEFPirates.log) => :log,
    typeof(exp) => :exp,
    typeof(Base.FastMath.exp_fast) => :exp,
    typeof(SIMDPirates.vexp) => :exp,
    typeof(SLEEFPirates.exp) => :exp,
    typeof(sin) => :sin,
    typeof(Base.FastMath.sin_fast) => :sin,
    typeof(SLEEFPirates.sin) => :sin,
    typeof(cos) => :cos,
    typeof(Base.FastMath.cos_fast) => :cos,
    typeof(SLEEFPirates.cos) => :cos,
    typeof(sincos) => :sincos,
    typeof(Base.FastMath.sincos_fast) => :sincos,
    typeof(SLEEFPirates.sincos) => :sincos
)

