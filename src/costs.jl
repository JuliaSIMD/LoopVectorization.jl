
@static if VERSION < v"1.3"
    lv(x) = Expr(:(.), :LoopVectorization, QuoteNode(x))
else
    lv(x) = GlobalRef(LoopVectorization, x)
end



struct Instruction
    mod::Symbol
    instr::Symbol
end
Instruction(instr::Symbol) = Instruction(:LoopVectorization, instr)
Base.convert(::Type{Instruction}, instr::Symbol) = Instruction(instr)
lower(instr::Instruction) = Expr(:(.), instr.mod, QuoteNode(instr.instr))
Base.Expr(instr::Instruction, args...) = Expr(:call, lower(instr), args...)::Expr
Base.hash(instr::Instruction, h::UInt64) = hash(instr.instr, hash(instr.mod, h))
function Base.isless(instr1::Instruction, instr2::Instruction)
    if instr1.mod === instr2.mod
        isless(instr1.instr, instr2.instr)
    else
        isless(instr1.mod, instr2.mod)
    end
end
Base.isequal(ins1::Instruction, ins2::Instruction) = (ins1.instr === ins2.instr) && (ins1.mod === ins2.mod)

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
# instruction_cost(instruction::Symbol) = get(COST, instruction, OPAQUE_INSTRUCTION)
# instruction_cost(instruction::Instruction) = instruction_cost(instruction.instr)
instruction_cost(instruction::Instruction) = get(COST, instruction, OPAQUE_INSTRUCTION)
instruction_cost(instruction::Symbol) = instruction_cost(Instruction(instruction))
scalar_cost(instr::Instruction) = scalar_cost(instruction_cost(instr))
vector_cost(instr::Instruction, Wshift, sizeof_T) = vector_cost(instruction_cost(instr), Wshift, sizeof_T)
function cost(instruction::InstructionCost, Wshift, sizeof_T)
    Wshift == 0 ? scalar_cost(instruction) : vector_cost(instruction, Wshift, sizeof_T)
end

function cost(instruction::Instruction, Wshift, sizeof_T)
    cost( instruction_cost(instruction), Wshift, sizeof_T )
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
    Instruction(:conditionalstore!) => InstructionCost(-3.0,1.0,3,0),
    Instruction(:zero) => InstructionCost(1,0.5),
    Instruction(:one) => InstructionCost(3,0.5),
    Instruction(:(+)) => InstructionCost(4,0.5),
    Instruction(:(-)) => InstructionCost(4,0.5),
    Instruction(:(*)) => InstructionCost(4,0.5),
    Instruction(:(/)) => InstructionCost(13,4.0,-2.0),
    Instruction(:vadd) => InstructionCost(4,0.5),
    Instruction(:vsub) => InstructionCost(4,0.5),
    Instruction(:vmul) => InstructionCost(4,0.5),
    Instruction(:vfdiv) => InstructionCost(13,4.0,-2.0),
    Instruction(:evadd) => InstructionCost(4,0.5),
    Instruction(:evsub) => InstructionCost(4,0.5),
    Instruction(:evmul) => InstructionCost(4,0.5),
    Instruction(:evfdiv) => InstructionCost(13,4.0,-2.0),
    Instruction(:reduced_add) => InstructionCost(4,0.5),# ignoring reduction part of cost, might be nop
    Instruction(:reduced_prod) => InstructionCost(4,0.5),# ignoring reduction part of cost, might be nop
    Instruction(:reduce_to_add) => InstructionCost(0,0.0,0.0,0),
    Instruction(:reduce_to_prod) => InstructionCost(0,0.0,0.0,0),
    Instruction(:abs2) => InstructionCost(4,0.5),
    Instruction(:vabs2) => InstructionCost(4,0.5),
    Instruction(:(==)) => InstructionCost(1, 0.5),
    Instruction(:isequal) => InstructionCost(1, 0.5),
    Instruction(:(~)) => InstructionCost(1, 0.5),
    Instruction(:(&)) => InstructionCost(1, 0.5),
    Instruction(:(|)) => InstructionCost(1, 0.5),
    Instruction(:(>)) => InstructionCost(1, 0.5),
    Instruction(:(<)) => InstructionCost(1, 0.5),
    Instruction(:(>=)) => InstructionCost(1, 0.5),
    Instruction(:(<=)) => InstructionCost(1, 0.5),
    Instruction(:ifelse) => InstructionCost(1, 0.5),
    Instruction(:vifelse) => InstructionCost(1, 0.5),
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
    Instruction(:sqrt_fast) => InstructionCost(15,4.0,-2.0),
    Instruction(:log) => InstructionCost(20,20.0,40.0,20),
    Instruction(:exp) => InstructionCost(20,20.0,20.0,18),
    Instruction(:(^)) => InstructionCost(40,40.0,40.0,26), # FIXME
    Instruction(:sin) => InstructionCost(18,15.0,68.0,23),
    Instruction(:cos) => InstructionCost(18,15.0,68.0,26),
    Instruction(:sincos) => InstructionCost(25,22.0,70.0,26),
    Instruction(:sinpi) => InstructionCost(18,15.0,68.0,23),
    Instruction(:cospi) => InstructionCost(18,15.0,68.0,26),
    Instruction(:sincospi) => InstructionCost(25,22.0,70.0,26),
    Instruction(:log_fast) => InstructionCost(20,20.0,40.0,20),
    Instruction(:exp_fast) => InstructionCost(20,20.0,20.0,18),
    Instruction(:sin_fast) => InstructionCost(18,15.0,68.0,23),
    Instruction(:cos_fast) => InstructionCost(18,15.0,68.0,26),
    Instruction(:sincos_fast) => InstructionCost(25,22.0,70.0,26),
    Instruction(:sinpi_fast) => InstructionCost(18,15.0,68.0,23),
    Instruction(:cospi_fast) => InstructionCost(18,15.0,68.0,26),
    Instruction(:sincospi_fast) => InstructionCost(25,22.0,70.0,26),
    Instruction(:identity) => InstructionCost(0,0.0,0.0,0),
    Instruction(:adjoint) => InstructionCost(0,0.0,0.0,0),
    Instruction(:transpose) => InstructionCost(0,0.0,0.0,0),
    # Symbol("##CONSTANT##") => InstructionCost(0,0.0)
)

# const KNOWNINSTRUCTIONS = keys(COST)
# instruction(f, m) = f ∈ KNOWNINSTRUCTIONS ? Instruction(:LoopVectorization, f) : Instruction(m, f)
instruction(f::Symbol, m) = Instruction(f) ∈ keys(COST) ? Instruction(f) : Instruction(m, f)
# instruction(f, m) = get(COST, f, Instruction(m, f))

# for (k, v) ∈ COST # so we can look up Symbol(typeof(function))
#     COST[Symbol("typeof(", lower(k), ")")] = v
# end

const ADDITIVE_IN_REDUCTIONS = 1.0
const MULTIPLICATIVE_IN_REDUCTIONS = 2.0
const ANY = 3.0
const ALL = 4.0

const REDUCTION_CLASS = Dict{Symbol,Float64}(
    :+ => ADDITIVE_IN_REDUCTIONS,
    :- => ADDITIVE_IN_REDUCTIONS,
    :* => MULTIPLICATIVE_IN_REDUCTIONS,
    :vadd => ADDITIVE_IN_REDUCTIONS,
    :vsub => ADDITIVE_IN_REDUCTIONS,
    :vmul => MULTIPLICATIVE_IN_REDUCTIONS,
    :evadd => ADDITIVE_IN_REDUCTIONS,
    :evsub => ADDITIVE_IN_REDUCTIONS,
    :evmul => MULTIPLICATIVE_IN_REDUCTIONS,
    :& => ALL,
    :| => ANY,
    :muladd => ADDITIVE_IN_REDUCTIONS,
    :fma => ADDITIVE_IN_REDUCTIONS,
    :vmuladd => ADDITIVE_IN_REDUCTIONS,
    :vfma => ADDITIVE_IN_REDUCTIONS,
    :vfmadd => ADDITIVE_IN_REDUCTIONS,
    :vfmsub => ADDITIVE_IN_REDUCTIONS,
    :vfnmadd => ADDITIVE_IN_REDUCTIONS,
    :vfnmsub => ADDITIVE_IN_REDUCTIONS,
    :vfmadd_fast => ADDITIVE_IN_REDUCTIONS,
    :vfmsub_fast => ADDITIVE_IN_REDUCTIONS,
    :vfnmadd_fast => ADDITIVE_IN_REDUCTIONS,
    :vfnmsub_fast => ADDITIVE_IN_REDUCTIONS,
    :reduced_add => ADDITIVE_IN_REDUCTIONS,
    :reduced_prod => MULTIPLICATIVE_IN_REDUCTIONS,
    :reduced_all => ALL,
    :reduced_any => ANY
)
reduction_instruction_class(instr::Symbol) = get(REDUCTION_CLASS, instr, NaN)
reduction_instruction_class(instr::Instruction) = get(REDUCTION_CLASS, instr.instr, NaN)
function reduction_to_single_vector(x::Float64)
    x == 1.0 ? :evadd : x == 2.0 ? :evmul : x == 3.0 ? :vand : x == 4.0 ? :vor : throw("Reduction not found.")
end
reduction_to_single_vector(x) = reduction_to_single_vector(reduction_instruction_class(x))
function reduction_to_scalar(x::Float64)
    x == 1.0 ? :vsum : x == 2.0 ? :vprod : x == 3.0 ? :vany : x == 4.0 ? :vall : throw("Reduction not found.")
end
reduction_to_scalar(x) = reduction_to_scalar(reduction_instruction_class(x))
function reduction_scalar_combine(x::Float64)
    x == 1.0 ? :reduced_add : x == 2.0 ? :reduced_prod : x == 3.0 ? :reduced_any : x == 4.0 ? :reduced_all : throw("Reduction not found.")
end
reduction_scalar_combine(x) = reduction_scalar_combine(reduction_instruction_class(x))
function reduction_combine_to(x::Float64)
    x == 1.0 ? :reduce_to_add : x == 2.0 ? :reduce_to_prod : x == 3.0 ? :reduce_to_any : x == 4.0 ? :reduce_to_all : throw("Reduction not found.")
end
reduction_combine_to(x) = reduction_combine_to(reduction_instruction_class(x))
function reduction_zero(x::Float64) 
    x == 1.0 ? :zero : x == 2.0 ? :one : x == 3.0 ? :false : x == 4.0 ? :true : throw("Reduction not found.")
end
reduction_zero(x) = reduction_zero(reduction_instruction_class(x))

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
    typeof(SIMDPirates.vfmadd_fast) => :vfmadd_fast,
    typeof(SIMDPirates.vfmsub_fast) => :vfmsub_fast,
    typeof(SIMDPirates.vfnmadd_fast) => :vfnmadd_fast,
    typeof(SIMDPirates.vfnmsub_fast) => :vfnmsub_fast,
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

