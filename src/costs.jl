
lv(x) = GlobalRef(LoopVectorization, x)


"""
    Instruction

`Instruction` represents a function via its module and symbol. It is
similar to a `GlobalRef` and may someday be replaced by `GlobalRef`.
"""
struct Instruction
    mod::Symbol
    instr::Symbol
end
# lower(instr::Instruction) = Expr(:(.), instr.mod, QuoteNode(instr.instr))
# Base.convert(::Type{Expr}, instr::Instruction) = Expr(:(.), instr.mod, QuoteNode(instr.instr))
function callexpr(instr::Instruction)
    if instr.mod === :LoopVectorization
        Expr(:call, lv(instr.instr))
    else#if instr.mod === :Main
        Expr(:call, instr.instr)
    end
end
function callexpr(instr::Instruction, arg)
    ce = callexpr(instr)
    append!(ce.args, arg)
    ce
end
Base.hash(instr::Instruction, h::UInt64) = hash(instr.instr, hash(instr.mod, h))
# function Base.isless(instr1::Instruction, instr2::Instruction)
#     if instr1.mod === instr2.mod
#         isless(instr1.instr, instr2.instr)
#     else
#         isless(instr1.mod, instr2.mod)
#     end
# end
Base.isequal(ins1::Instruction, ins2::Instruction) = (ins1.instr === ins2.instr) && (ins1.mod === ins2.mod)

const LOOPCONSTANT = Instruction(:LoopVectorization, Symbol("LOOPCONSTANTINSTRUCTION"))

"""
    InstructionCost

Store parameters related to performance for individual CPU instructions.

$(TYPEDFIELDS)
"""
struct InstructionCost
    "A flag indicating how instruction cost scales with vector width (128, 256, or 512 bits)"
    scaling::Float64 # sentinel values: -3 == no scaling; -2 == offset_scaling, -1 == linear scaling, >0 ->  == latency == reciprocal throughput
    """The number of clock cycles per operation when many of the same operation are repeated in sequence.
    Think of it as the inverse of the flow rate at steady-state. It is typically ≤ the `scalar_latency`."""
    scalar_reciprocal_throughput::Float64
    """The minimum delay, in clock cycles, associated with the instruction.
    Think of it as the delay from turning on a faucet to when water starts coming out the end of the pipe.
    See also `scalar_reciprocal_throughput`."""
    scalar_latency::Int
    "Number of floating-point registered used"
    register_pressure::Int
end
InstructionCost(sl::Int, srt::Float64, scaling::Float64 = -3.0) = InstructionCost(scaling, srt, sl, 0)

nocost(c::InstructionCost) = c.scalar_reciprocal_throughput == 0.0
flatcost(c::InstructionCost) = c.scaling == -3.0
offsetscaling(c::InstructionCost) = c.scaling == -2.0
linearscaling(c::InstructionCost) = c.scaling == -1.0

function scalar_cost(ic::InstructionCost)#, ::Type{T} = Float64) where {T}
    @unpack scalar_reciprocal_throughput, scalar_latency, register_pressure = ic
    scalar_reciprocal_throughput, scalar_latency, register_pressure
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

const OPAQUE_INSTRUCTION = InstructionCost(-1.0, 40, 40.0, REGISTER_COUNT)

instruction_cost(instruction::Instruction) = instruction.mod === :LoopVectorization ? COST[instruction.instr] : OPAQUE_INSTRUCTION
instruction_cost(instruction::Symbol) = get(COST, instruction, OPAQUE_INSTRUCTION)
scalar_cost(instr::Instruction) = scalar_cost(instruction_cost(instr))
vector_cost(instr::Instruction, Wshift, sizeof_T) = vector_cost(instruction_cost(instr), Wshift, sizeof_T)
# function cost(instruction::InstructionCost, Wshift, sizeof_T)
#     Wshift == 0 ? scalar_cost(instruction) : vector_cost(instruction, Wshift, sizeof_T)
# end

# function cost(instruction::Instruction, Wshift, sizeof_T)
#     cost( instruction_cost(instruction), Wshift, sizeof_T )
# end


# Just a semi-reasonable assumption; should not be that sensitive to anything other than loads


# Comments on setindex!
# 1. Not a part of dependency chains, so not really twice as expensive as getindex?
# 2. getindex loads a register, not setindex!, but we place cost on setindex!
#    as a heuristic means of approximating register pressure, since many loads can be
#    consolidated into a single register. The number of LICM-ed setindex!, on the other
#    hand, should indicate how many registers we're keeping live for the sake of eventually storing.
const COST = Dict{Symbol,InstructionCost}(
    :getindex => InstructionCost(-3.0,0.5,3,1),
    :conditionalload => InstructionCost(-3.0,0.5,3,1),
    :setindex! => InstructionCost(-3.0,1.0,3,0),
    :conditionalstore! => InstructionCost(-3.0,1.0,3,0),
    :zero => InstructionCost(1,0.5),
    :one => InstructionCost(3,0.5),
    :(+) => InstructionCost(4,0.5),
    :(-) => InstructionCost(4,0.5),
    :(*) => InstructionCost(4,0.5),
    :(/) => InstructionCost(13,4.0,-2.0),
    :vadd => InstructionCost(4,0.5),
    :vadd1 => InstructionCost(4,0.5),
    :add_fast => InstructionCost(4,0.5),
    :vsub => InstructionCost(4,0.5),
    :sub_fast => InstructionCost(4,0.5),
    # :vadd! => InstructionCost(4,0.5),
    # :vsub! => InstructionCost(4,0.5),
    # :vmul! => InstructionCost(4,0.5),
    :vmul => InstructionCost(4,0.5),
    :mul_fast => InstructionCost(4,0.5),
    # :vfdiv => InstructionCost(13,4.0,-2.0),
    # :vfdiv! => InstructionCost(13,4.0,-2.0),
    :div_fast => InstructionCost(13,4.0,-2.0),
    # :evadd => InstructionCost(4,0.5),
    # :evsub => InstructionCost(4,0.5),
    # :evmul => InstructionCost(4,0.5),
    # :evfdiv => InstructionCost(13,4.0,-2.0),
    :vsum => InstructionCost(6,2.0),
    :vprod => InstructionCost(6,2.0),
    :reduced_add => InstructionCost(4,0.5),# ignoring reduction part of cost, might be nop
    :reduced_prod => InstructionCost(4,0.5),# ignoring reduction part of cost, might be nop
    :reduced_max => InstructionCost(4,0.5),# ignoring reduction part of cost, might be nop
    :reduced_min => InstructionCost(4,0.5),# ignoring reduction part of cost, might be nop
    :reduce_to_add => InstructionCost(0,0.0,0.0,0),
    :reduce_to_prod => InstructionCost(0,0.0,0.0,0),
    :abs => InstructionCost(1, 0.5),
    :abs2 => InstructionCost(4,0.5),
    # :vabs2 => InstructionCost(4,0.5),
    :(==) => InstructionCost(1, 0.5),
    :(!=) => InstructionCost(1, 0.5),
    :(isnan) => InstructionCost(1, 0.5),
    :(isfinite) => InstructionCost(2, 1.0),
    :(isinf) => InstructionCost(2, 1.0),
    :isequal => InstructionCost(1, 0.5),
    :(!) => InstructionCost(1, 0.5),
    :(~) => InstructionCost(1, 0.5),
    :(&) => InstructionCost(1, 0.5),
    :(|) => InstructionCost(1, 0.5),
    :(⊻) => InstructionCost(1, 0.5),
    :(%) => InstructionCost(13, 4.0, -2.0),
    :(rem) => InstructionCost(13, 4.0, -2.0),
    :(>) => InstructionCost(1, 0.5),
    :(<) => InstructionCost(1, 0.5),
    :(>=) => InstructionCost(1, 0.5),
    :(<=) => InstructionCost(1, 0.5),
    :(≥) => InstructionCost(1, 0.5),
    :(≤) => InstructionCost(1, 0.5),
    :(>>) => InstructionCost(1, 0.5),
    :(>>>) => InstructionCost(1, 0.5),
    :(<<) => InstructionCost(1, 0.5),
    :isodd => InstructionCost(1, 0.5),
    :iseven => InstructionCost(1, 0.5),
    :max => InstructionCost(4,0.5),
    :min => InstructionCost(4,0.5),
    :relu => InstructionCost(4,0.5),
    # Instruction(:ifelse) => InstructionCost(1, 0.5),
    :ifelse => InstructionCost(1, 0.5),
    :inv => InstructionCost(13,4.0,-2.0,1),
    # :vinv => InstructionCost(13,4.0,-2.0,1),
    :muladd => InstructionCost(4,0.5), # + and * will fuse into this, so much of the time they're not twice as expensive
    :fma => InstructionCost(4,0.5), # + and * will fuse into this, so much of the time they're not twice as expensive
    # :vmuladd => InstructionCost(4,0.5), # + and * will fuse into this, so much of the time they're not twice as expensive
    # :vfma => InstructionCost(4,0.5), # + and * will fuse into this, so much of the time they're not twice as expensive
    :vfmadd => InstructionCost(4,0.5), # + and * will fuse into this, so much of the time they're not twice as expensive
    :vfmsub => InstructionCost(4,0.5), # - and * will fuse into this, so much of the time they're not twice as expensive
    :vfnmadd => InstructionCost(4,0.5), # + and -* will fuse into this, so much of the time they're not twice as expensive
    :vfnmsub => InstructionCost(4,0.5), # - and -* will fuse into this, so much of the time they're not twice as expensive
    :vfmadd231 => InstructionCost(4,0.5), # + and * will fuse into this, so much of the time they're not twice as expensive
    :vfmsub231 => InstructionCost(4,0.5), # - and * will fuse into this, so much of the time they're not twice as expensive
    :vfnmadd231 => InstructionCost(4,0.5), # + and -* will fuse into this, so much of the time they're not twice as expensive
    :vfnmsub231 => InstructionCost(4,0.5), # - and -* will fuse into this, so much of the time they're not twice as expensive
    # :vfmadd! => InstructionCost(4,0.5), # + and * will fuse into this, so much of the time they're not twice as expensive
    # :vfnmadd! => InstructionCost(4,0.5), # + and -* will fuse into this, so much of the time they're not twice as expensive
    # :vfmsub! => InstructionCost(4,0.5), # + and * will fuse into this, so much of the time they're not twice as expensive
    # :vfnmsub! => InstructionCost(4,0.5), # + and -* will fuse into this, so much of the time they're not twice as expensive
    # :vfmadd_fast => InstructionCost(4,0.5), # + and * will fuse into this, so much of the time they're not twice as expensive
    # :vfmsub_fast => InstructionCost(4,0.5), # - and * will fuse into this, so much of the time they're not twice as expensive
    # :vfnmadd_fast => InstructionCost(4,0.5), # + and -* will fuse into this, so much of the time they're not twice as expensive
    # :vfnmsub_fast => InstructionCost(4,0.5), # - and -* will fuse into this, so much of the time they're not twice as expensive
    # :vfmaddaddone => InstructionCost(4,0.5), # - and -* will fuse into this, so much of the time they're not twice as expensive
    # :vmullog2 => InstructionCost(4,0.5),
    # :vmullog2add! => InstructionCost(4,0.5),
    # :vmullog10 => InstructionCost(4,0.5),
    # :vmullog10add! => InstructionCost(4,0.5),
    # :vdivlog2 => InstructionCost(13,4.0,-2.0),
    # :vdivlog2add! =>InstructionCost(13,4.0,-2.0),
    # :vdivlog10 => InstructionCost(13,4.0,-2.0),
    # :vdivlog10add! =>InstructionCost(13,4.0,-2.0),
    :sqrt => InstructionCost(15,4.0,-2.0),
    :sqrt_fast => InstructionCost(15,4.0,-2.0),
    :log => InstructionCost(20,20.0,20.0,20),
    :log1p => InstructionCost(20,25.0,25.0,20), # FIXME
    :exp => InstructionCost(20,20.0,20.0,18),
    :expm1 => InstructionCost(20,25.0,25.0,18), # FIXME
    :(^) => InstructionCost(40,40.0,40.0,26), # FIXME
    :sin => InstructionCost(18,15.0,68.0,23),
    :cos => InstructionCost(18,15.0,68.0,26),
    :sincos => InstructionCost(25,22.0,70.0,26),
    :sinpi => InstructionCost(18,15.0,68.0,23),
    :cospi => InstructionCost(18,15.0,68.0,26),
    :sincospi => InstructionCost(25,22.0,70.0,26),
    :log_fast => InstructionCost(20,20.0,40.0,20),
    :exp_fast => InstructionCost(20,20.0,20.0,18),
    :sin_fast => InstructionCost(18,15.0,68.0,23),
    :cos_fast => InstructionCost(18,15.0,68.0,26),
    :sincos_fast => InstructionCost(25,22.0,70.0,26),
    :sinpi_fast => InstructionCost(18,15.0,68.0,23),
    :cospi_fast => InstructionCost(18,15.0,68.0,26),
    :sincospi_fast => InstructionCost(25,22.0,70.0,26),
    :tanh => InstructionCost(40,40.0,40.0,26), # FIXME
    # :tanh_fast => InstructionCost(25,22.0,70.0,26), # FIXME
    :identity => InstructionCost(0,0.0,0.0,0),
    :adjoint => InstructionCost(0,0.0,0.0,0),
    :conj => InstructionCost(0,0.0,0.0,0),
    :transpose => InstructionCost(0,0.0,0.0,0),
    :first => InstructionCost(0,0.0,0.0,0),
    :second => InstructionCost(0,0.0,0.0,0),
    :third => InstructionCost(0,0.0,0.0,0),
    :fourth => InstructionCost(0,0.0,0.0,0),
    :fifth => InstructionCost(0,0.0,0.0,0),
    :sixth => InstructionCost(0,0.0,0.0,0),
    :seventh => InstructionCost(0,0.0,0.0,0),
    :eighth => InstructionCost(0,0.0,0.0,0),
    :ninth => InstructionCost(0,0.0,0.0,0),
    :prefetch => InstructionCost(0,0.0,0.0,0),
    :prefetch0 => InstructionCost(0,0.0,0.0,0),
    :prefetch1 => InstructionCost(0,0.0,0.0,0),
    :prefetch2 => InstructionCost(0,0.0,0.0,0)
)
@inline prefetch0(x, i) = VectorizationBase.prefetch(gep(stridedpointer(x), (data(i),)), Val{3}(), Val{0}())
@inline prefetch0(x, I::Tuple) = VectorizationBase.prefetch(gep(stridedpointer(x), data.(I)), Val{3}(), Val{0}())
@inline prefetch0(x, i, j) = VectorizationBase.prefetch(gep(stridedpointer(x), (data(i), data(j))), Val{3}(), Val{0}())
# @inline prefetch0(x, i, j, oi, oj) = VectorizationBase.prefetch(gep(stridedpointer(x), (data(i) + data(oi) - 1, data(j) + data(oj) - 1)), Val{3}(), Val{0}())
@inline prefetch1(x, i) = VectorizationBase.prefetch(gep(stridedpointer(x), (data(i),)), Val{2}(), Val{0}())
@inline prefetch1(x, i, j) = VectorizationBase.prefetch(gep(stridedpointer(x), (data(i), data(j))), Val{2}(), Val{0}())
# @inline prefetch1(x, i, j, oi, oj) = VectorizationBase.prefetch(gep(stridedpointer(x), (data(i) + data(oi) - 1, data(j) + data(oj) - 1)), Val{2}(), Val{0}())
@inline prefetch2(x, i) = VectorizationBase.prefetch(gep(stridedpointer(x), (data(i),)), Val{1}(), Val{0}())
@inline prefetch2(x, i, j) = VectorizationBase.prefetch(gep(stridedpointer(x), (data(i), data(j))), Val{1}(), Val{0}())
# @inline prefetch2(x, i, j, oi, oj) = VectorizationBase.prefetch(gep(stridedpointer(x), (data(i) + data(oi) - 1, data(j) + data(oj) - 1)), Val{1}(), Val{0}())

Base.convert(::Type{Instruction}, instr::Symbol) = Instruction(instr)
# instruction(f::Symbol, m) = f ∈ keys(COST) ? Instruction(f) : Instruction(m, f)
# instruction(f::Symbol) = f ∈ keys(COST) ? Instruction(:LoopVectorization, f) : Instruction(Symbol(""), f)
function instruction(f::Symbol)
    # f === :ifelse && return Instruction(:LoopVectorization, :ifelse)
    # @assert f ∈ keys(COST)
    f ∈ keys(COST) ? Instruction(:LoopVectorization, f) : Instruction(Symbol(""), f)
end
# instruction(f::Symbol, m::Symbol) = f ∈ keys(COST) ? Instruction(:LoopVectorization, f) : Instruction(m, f)
Instruction(instr::Symbol) = instruction(instr)

const ADDITIVE_IN_REDUCTIONS = 1.0
const MULTIPLICATIVE_IN_REDUCTIONS = 2.0
const ANY = 3.0
const ALL = 4.0
const MAX = 5.0
const MIN = 6.0

const REDUCTION_CLASS = Dict{Symbol,Float64}(
    :+ => ADDITIVE_IN_REDUCTIONS,
    :- => ADDITIVE_IN_REDUCTIONS,
    :* => MULTIPLICATIVE_IN_REDUCTIONS,
    :vadd => ADDITIVE_IN_REDUCTIONS,
    :vsub => ADDITIVE_IN_REDUCTIONS,
    # :vadd! => ADDITIVE_IN_REDUCTIONS,
    # :vsub! => ADDITIVE_IN_REDUCTIONS,
    :vmul => MULTIPLICATIVE_IN_REDUCTIONS,
    # :vmul! => MULTIPLICATIVE_IN_REDUCTIONS,
    # :evadd => ADDITIVE_IN_REDUCTIONS,
    # :evsub => ADDITIVE_IN_REDUCTIONS,
    # :evmul => MULTIPLICATIVE_IN_REDUCTIONS,
    :& => ALL,
    :| => ANY,
    :muladd => ADDITIVE_IN_REDUCTIONS,
    :fma => ADDITIVE_IN_REDUCTIONS,
    # :vmuladd => ADDITIVE_IN_REDUCTIONS,
    # :vfma => ADDITIVE_IN_REDUCTIONS,
    :vfmadd => ADDITIVE_IN_REDUCTIONS,
    :vfmsub => ADDITIVE_IN_REDUCTIONS,
    :vfnmadd => ADDITIVE_IN_REDUCTIONS,
    :vfnmsub => ADDITIVE_IN_REDUCTIONS,
    :vfmadd231 => ADDITIVE_IN_REDUCTIONS,
    :vfmsub231 => ADDITIVE_IN_REDUCTIONS,
    :vfnmadd231 => ADDITIVE_IN_REDUCTIONS,
    :vfnmsub231 => ADDITIVE_IN_REDUCTIONS,
    # :vfmadd! => ADDITIVE_IN_REDUCTIONS,
    # :vfnmadd! => ADDITIVE_IN_REDUCTIONS,
    # :vfmsub! => ADDITIVE_IN_REDUCTIONS,
    # :vfnmsub! => ADDITIVE_IN_REDUCTIONS,
    :vfmadd_fast => ADDITIVE_IN_REDUCTIONS,
    :vfmsub_fast => ADDITIVE_IN_REDUCTIONS,
    :vfnmadd_fast => ADDITIVE_IN_REDUCTIONS,
    :vfnmsub_fast => ADDITIVE_IN_REDUCTIONS,
    :reduced_add => ADDITIVE_IN_REDUCTIONS,
    :reduced_prod => MULTIPLICATIVE_IN_REDUCTIONS,
    :reduced_all => ALL,
    :reduced_any => ANY,
    :max => MAX,
    :min => MIN
)
reduction_instruction_class(instr::Symbol) = get(REDUCTION_CLASS, instr, NaN)
reduction_instruction_class(instr::Instruction) = reduction_instruction_class(instr.instr)
function reduction_to_single_vector(x::Float64)
    # x == 1.0 ? :evadd : x == 2.0 ? :evmul : x == 3.0 ? :vor : x == 4.0 ? :vand : x == 5.0 ? :max : x == 6.0 ? :min : throw("Reduction not found.")
    x == ADDITIVE_IN_REDUCTIONS ? :vadd : x == MULTIPLICATIVE_IN_REDUCTIONS ? :vmul : x == MAX ? :max : x == MIN ? :min : throw("Reduction not found.")
end
reduction_to_single_vector(x) = reduction_to_single_vector(reduction_instruction_class(x))
# function reduction_to_scalar(x::Float64)
#     # x == 1.0 ? :vsum : x == 2.0 ? :vprod : x == 3.0 ? :vany : x == 4.0 ? :vall : x == 5.0 ? :maximum : x == 6.0 ? :minimum : throw("Reduction not found.")
#     x == 1.0 ? :vsum : x == 2.0 ? :vprod : x == 5.0 ? :maximum : x == 6.0 ? :minimum : throw("Reduction not found.")
# end
# reduction_to_scalar(x) = reduction_to_scalar(reduction_instruction_class(x))
function reduction_to_scalar(x::Float64)
    x == ADDITIVE_IN_REDUCTIONS ? :vsum : x == MULTIPLICATIVE_IN_REDUCTIONS ? :vprod : x == MAX ? :vmaximum : x == MIN ? :vminimum : throw("Reduction not found.")
end
reduction_to_scalar(x) = reduction_to_scalar(reduction_instruction_class(x))
function reduction_scalar_combine(x::Float64)
    # x == 1.0 ? :reduced_add : x == 2.0 ? :reduced_prod : x == 3.0 ? :reduced_any : x == 4.0 ? :reduced_all : x == 5.0 ? :reduced_max : x == 6.0 ? :reduced_min : throw("Reduction not found.")
    x == ADDITIVE_IN_REDUCTIONS ? :reduced_add : x == MULTIPLICATIVE_IN_REDUCTIONS ? :reduced_prod : x == MAX ? :reduced_max : x == MIN ? :reduced_min : throw("Reduction not found.")
end
reduction_scalar_combine(x) = reduction_scalar_combine(reduction_instruction_class(x))
# function reduction_combine_to(x::Float64)
#     # x == 1.0 ? :reduce_to_add : x == 2.0 ? :reduce_to_prod : x == 3.0 ? :reduce_to_any : x == 4.0 ? :reduce_to_all : x == 5.0 ? :reduce_to_max : x == 6.0 ? :reduce_to_min : throw("Reduction not found.")
#     x == ADDITIVE_IN_REDUCTIONS ? :reduce_to_add : x == MULTIPLICATIVE_IN_REDUCTIONS ? :reduce_to_prod : x == MAX ? :reduce_to_max : x == MIN ? :reduce_to_min : throw("Reduction not found.")
# end
# reduction_combine_to(x) = reduction_combine_to(reduction_instruction_class(x))
function reduction_zero(x::Float64)
    # x == 1.0 ? :zero : x == 2.0 ? :one : x == 3.0 ? :false : x == 4.0 ? :true : x == 5.0 ? :typemin : x == 6.0 ? :typemax : throw("Reduction not found.")
    x == ADDITIVE_IN_REDUCTIONS ? :zero : x == MULTIPLICATIVE_IN_REDUCTIONS ? :one : x == MAX ? :typemin : x == MIN ? :typemax : throw("Reduction not found.")
end
reduction_zero(x) = reduction_zero(reduction_instruction_class(x))

function isreductcombineinstr(instr::Symbol)
    instr ∈ (:reduced_add, :reduced_prod, :reduce_to_add, :reduce_to_prod, :reduced_max, :reduced_min, :reduce_to_max, :reduce_to_min)
end
isreductcombineinstr(instr::Instruction) = isreductcombineinstr(instr.instr)

const FUNCTIONSYMBOLS = IdDict{Type{<:Function},Instruction}(
    typeof(+) => :(+),
    typeof(VectorizationBase.vadd) => :(+),
    # typeof(VectorizationBase.vadd!) => :(+),
    typeof(Base.FastMath.add_fast) => :(+),
    typeof(-) => :(-),
    typeof(VectorizationBase.vsub) => :(-),
    # typeof(VectorizationBase.vsub!) => :(-),
    typeof(Base.FastMath.sub_fast) => :(-),
    typeof(*) => :(*),
    typeof(VectorizationBase.vmul) => :(*),
    # typeof(VectorizationBase.vmul!) => :(*),
    typeof(Base.FastMath.mul_fast) => :(*),
    typeof(/) => :(/),
    typeof(^) => :(^),
    # typeof(VectorizationBase.vfdiv) => :(/),
    # typeof(VectorizationBase.vfdiv!) => :(/),
    typeof(VectorizationBase.vdiv) => :(/),
    typeof(Base.FastMath.div_fast) => :(/),
    typeof(==) => :(==),
    typeof(!=) => :(!=),
    typeof(isequal) => :isequal,
    typeof(isnan) => :isnan,
    typeof(isinf) => :isinf,
    typeof(isfinite) => :isfinite,
    typeof(abs) => :abs,
    typeof(abs2) => :abs2,
    typeof(~) => :(~),
    typeof(!) => :(!),
    typeof(&) => :(&),
    typeof(|) => :(|),
    typeof(⊻) => :(⊻),
    typeof(>) => :(>),
    typeof(<) => :(<),
    typeof(>=) => :(>=),
    typeof(<=) => :(<=),
    typeof(inv) => :inv,
    typeof(muladd) => :muladd,
    typeof(fma) => :fma,
    # typeof(VectorizationBase.vmuladd) => :vmuladd,
    # typeof(VectorizationBase.vfma) => :vfma,
    typeof(VectorizationBase.vfmadd) => :vfmadd,
    typeof(VectorizationBase.vfmsub) => :vfmsub,
    typeof(VectorizationBase.vfnmadd) => :vfnmadd,
    typeof(VectorizationBase.vfnmsub) => :vfnmsub,
    typeof(VectorizationBase.vfmadd231) => :vfmadd231,
    typeof(VectorizationBase.vfmsub231) => :vfmsub231,
    typeof(VectorizationBase.vfnmadd231) => :vfnmadd231,
    typeof(VectorizationBase.vfnmsub231) => :vfnmsub231,
    # typeof(VectorizationBase.vfmadd!) => :vfmadd!,
    # typeof(VectorizationBase.vfnmadd!) => :vfnmadd!,
    # typeof(VectorizationBase.vfmsub!) => :vfmsub!,
    # typeof(VectorizationBase.vfnmsub!) => :vfnmsub!,
    # typeof(VectorizationBase.vfmadd_fast) => :vfmadd_fast,
    # typeof(VectorizationBase.vfmsub_fast) => :vfmsub_fast,
    # typeof(VectorizationBase.vfnmadd_fast) => :vfnmadd_fast,
    # typeof(VectorizationBase.vfnmsub_fast) => :vfnmsub_fast,
    # typeof(vfmaddaddone) => :vfmaddaddone,
    # typeof(vmullog2) => :vmullog2,
    # typeof(vmullog2add!) => :vmullog2add!,
    # typeof(vmullog10) => :vmullog10,
    # typeof(vmullog10add!) => :vmullog10add!,
    # typeof(vdivlog2) => :vdivlog2,
    # typeof(vdivlog2add!) => :vdivlog2add!,
    # typeof(vdivlog10) => :vdivlog10,
    # typeof(vdivlog10add!) => :vdivlog10add!,
    typeof(sqrt) => :sqrt,
    typeof(Base.FastMath.sqrt_fast) => :sqrt,
    # typeof(VectorizationBase.vsqrt) => :sqrt,
    typeof(log) => :log,
    typeof(Base.FastMath.log_fast) => :log,
    typeof(log1p) => :log1p,
    # typeof(VectorizationBase.vlog) => :log,
    typeof(SLEEFPirates.log) => :log,
    typeof(exp) => :exp,
    typeof(Base.FastMath.exp_fast) => :exp,
    typeof(expm1) => :expm1,
    # typeof(VectorizationBase.vexp) => :exp,
    typeof(SLEEFPirates.exp) => :exp,
    typeof(sin) => :sin,
    typeof(Base.FastMath.sin_fast) => :sin,
    typeof(SLEEFPirates.sin) => :sin,
    typeof(cos) => :cos,
    typeof(Base.FastMath.cos_fast) => :cos,
    typeof(SLEEFPirates.cos) => :cos,
    typeof(sincos) => :sincos,
    typeof(Base.FastMath.sincos_fast) => :sincos,
    typeof(SLEEFPirates.sincos) => :sincos,
    typeof(Base.tanh) => :tanh,
    # typeof(SLEEFPirates.tanh_fast) => :tanh_fast,
    typeof(max) => :max,
    typeof(min) => :min,
    typeof(relu) => :relu,
    typeof(<<) => :<<,
    typeof(>>) => :>>,
    typeof(>>>) => :>>>,
    typeof(Base.ifelse) => :ifelse,
    typeof(ifelse) => :ifelse,
    typeof(identity) => :identity,
    typeof(conj) => :conj
)

# implement whitelist for avx_support that package authors may use to conservatively guard `@avx` application
for f ∈ keys(FUNCTIONSYMBOLS)
    @eval ArrayInterface.can_avx(::$(typeof(f))) = true
end
