
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
Base.isequal(ins1::Instruction, ins2::Instruction) =
  (ins1.instr === ins2.instr) && (ins1.mod === ins2.mod)

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
InstructionCost(sl::Int, srt::Float64, scaling::Float64 = -3.0) =
  InstructionCost(scaling, srt, sl, 0)

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
    if (sizeof_T << Wshift) == 64 # VectorizationBase.register_size() # These instructions experience double latency with zmm
      sl += sl
    end
  elseif linearscaling(ic) # linear scaling
    W = 1 << Wshift
    extra_latency = sl - srt
    srt *= W
    sl = round(Int, srt + extra_latency)
    # else # we assume custom cost, and that latency == recip_throughput
    #     scaling = ic.scaling
    #     sl, srt = round(Int,scaling), scaling
  end
  srt, sl, srp
end

const OPAQUE_INSTRUCTION = InstructionCost(-1.0, 20, 20.0, 16)

instruction_cost(instruction::Instruction) =
  instruction.mod === :LoopVectorization ? COST[instruction.instr] : OPAQUE_INSTRUCTION
instruction_cost(instruction::Symbol) = get(COST, instruction, OPAQUE_INSTRUCTION)
scalar_cost(instr::Instruction) = scalar_cost(instruction_cost(instr))
vector_cost(instr::Instruction, Wshift, sizeof_T) =
  vector_cost(instruction_cost(instr), Wshift, sizeof_T)
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
  :getindex => InstructionCost(-3.0, 0.5, 3, 0),
  :conditionalload => InstructionCost(-3.0, 0.5, 3, 0),
  :setindex! => InstructionCost(-3.0, 1.0, 3, 0),
  :conditionalstore! => InstructionCost(-3.0, 1.0, 3, 0),
  :zero => InstructionCost(1, 0.5),
  :one => InstructionCost(3, 0.5),
  :(+) => InstructionCost(4, 0.5),
  :(-) => InstructionCost(4, 0.5),
  :(*) => InstructionCost(4, 0.5),
  :(/) => InstructionCost(13, 4.0, -2.0),
  :vadd => InstructionCost(4, 0.5),
  :vadd1 => InstructionCost(4, 0.5),
  :add_fast => InstructionCost(4, 0.5),
  :vsub => InstructionCost(4, 0.5),
  :sub_fast => InstructionCost(4, 0.5),
  # :vadd! => InstructionCost(4,0.5),
  # :vsub! => InstructionCost(4,0.5),
  # :vmul! => InstructionCost(4,0.5),
  :vmul => InstructionCost(4, 0.5),
  :vmul_nsw => InstructionCost(4, 0.5),
  :vadd_nsw => InstructionCost(4, 0.5),
  :vsub_nsw => InstructionCost(4, 0.5),
  :mul_fast => InstructionCost(4, 0.5),
  # :vfdiv => InstructionCost(13,4.0,-2.0),
  # :vfdiv! => InstructionCost(13,4.0,-2.0),
  :rem_fast => InstructionCost(13, 4.0, -2.0), # FIXME
  :div_fast => InstructionCost(13, 4.0, -2.0),
  :vdiv_fast => InstructionCost(20, 4.0, -2.0), # FIXME
  :÷ => InstructionCost(13, 4.0, -2.0),
  # :evadd => InstructionCost(4,0.5),
  # :evsub => InstructionCost(4,0.5),
  # :evmul => InstructionCost(4,0.5),
  # :evfdiv => InstructionCost(13,4.0,-2.0),
  :vsum => InstructionCost(6, 2.0),
  :vprod => InstructionCost(6, 2.0),
  :reduced_add => InstructionCost(4, 0.5),# ignoring reduction part of cost, might be nop
  :reduced_prod => InstructionCost(4, 0.5),# ignoring reduction part of cost, might be nop
  :reduced_max => InstructionCost(4, 0.5),# ignoring reduction part of cost, might be nop
  :reduced_min => InstructionCost(4, 0.5),# ignoring reduction part of cost, might be nop
  :reduce_to_add => InstructionCost(0, 0.0, 0.0, 0),
  :reduce_to_prod => InstructionCost(0, 0.0, 0.0, 0),
  :abs => InstructionCost(1, 0.5),
  :abs2 => InstructionCost(4, 0.5),
  :abs2_fast => InstructionCost(4, 0.5),
  :round => InstructionCost(4, 0.5),
  :(==) => InstructionCost(1, 0.5),
  :(!=) => InstructionCost(1, 0.5),
  :(≠) => InstructionCost(1, 0.5),
  :(===) => InstructionCost(1, 0.5),
  :(!==) => InstructionCost(1, 0.5),
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
  :max => InstructionCost(4, 0.5),
  :min => InstructionCost(4, 0.5),
  :max_fast => InstructionCost(4, 0.5),
  :min_fast => InstructionCost(4, 0.5),
  :relu => InstructionCost(4, 0.5),
  # Instruction(:ifelse) => InstructionCost(1, 0.5),
  :ifelse => InstructionCost(1, 0.5),
  :inv => InstructionCost(13, 4.0, -2.0, 1),
  :inv_fast => InstructionCost(10, 4.0, -2.0, 1), # FIXME
  # :vinv => InstructionCost(13,4.0,-2.0,1),
  :muladd => InstructionCost(4, 0.5), # + and * will fuse into this, so much of the time they're not twice as expensive
  :fma => InstructionCost(4, 0.5), # + and * will fuse into this, so much of the time they're not twice as expensive
  :vmuladd_fast => InstructionCost(4, 0.5), # + and * will fuse into this, so much of the time they're not twice as expensive
  :vfma_fast => InstructionCost(4, 0.5), # + and * will fuse into this, so much of the time they're not twice as expensive
  :vfmadd => InstructionCost(4, 0.5), # + and * will fuse into this, so much of the time they're not twice as expensive
  :vfmsub => InstructionCost(4, 0.5), # - and * will fuse into this, so much of the time they're not twice as expensive
  :vfnmadd => InstructionCost(4, 0.5), # + and -* will fuse into this, so much of the time they're not twice as expensive
  :vfnmsub => InstructionCost(4, 0.5), # - and -* will fuse into this, so much of the time they're not twice as expensive
  :vfmadd_fast => InstructionCost(4, 0.5), # + and * will fuse into this, so much of the time they're not twice as expensive
  :vfmsub_fast => InstructionCost(4, 0.5), # - and * will fuse into this, so much of the time they're not twice as expensive
  :vfnmadd_fast => InstructionCost(4, 0.5), # + and -* will fuse into this, so much of the time they're not twice as expensive
  :vfnmsub_fast => InstructionCost(4, 0.5), # - and -* will fuse into this, so much of the time they're not twice as expensive
  :vfmadd231 => InstructionCost(4, 0.5), # + and * will fuse into this, so much of the time they're not twice as expensive
  :vfmsub231 => InstructionCost(4, 0.5), # - and * will fuse into this, so much of the time they're not twice as expensive
  :vfnmadd231 => InstructionCost(4, 0.5), # + and -* will fuse into this, so much of the time they're not twice as expensive
  :vfnmsub231 => InstructionCost(4, 0.5), # - and -* will fuse into this, so much of the time they're not twice as expensive
  :vfmaddsub => InstructionCost(4, 0.5),
  :vfmsubadd => InstructionCost(4, 0.5),
  :sqrt => InstructionCost(15, 4.0, -2.0),
  :sqrt_fast => InstructionCost(15, 4.0, -2.0),
  :log => InstructionCost(-3.0, 15, 30, 11),
  :log2 => InstructionCost(-3.0, 15, 30, 11),
  :log10 => InstructionCost(-3.0, 15, 30, 11),
  :log_fast => InstructionCost(-3.0, 15, 30, 11),
  :log2_fast => InstructionCost(-3.0, 15, 30, 11),
  :log10_fast => InstructionCost(-3.0, 15, 30, 11),
  :log1p => InstructionCost(-3.0, 15, 30, 11),
  :exp => InstructionCost(-3.0, 13.0, 26.0, 14),
  :exp2 => InstructionCost(-3.0, 10.0, 40.0, 14),
  :exp10 => InstructionCost(-3.0, 13.0, 26.0, 14),
  :expm1 => InstructionCost(-3.0, 30.0, 60.0, 19),
  :(^) => InstructionCost(-3.0, 200.0, 400.0, 26), # FIXME
  :pow_fast => InstructionCost(-3.0, 200.0, 400.0, 26), # FIXME
  :sin => InstructionCost(-3, 30.0, 60.0, 23),
  :cos => InstructionCost(-3, 30.0, 60.0, 26),
  :sincos => InstructionCost(-3, 37.0, 85.0, 26),
  :sincos_fast => InstructionCost(-3, 30.0, 60.0, 26),
  :sinpi => InstructionCost(18, 15.0, 68.0, 23),
  :cospi => InstructionCost(18, 15.0, 68.0, 26),
  :sincospi => InstructionCost(25, 37.0, 70.0, 26),
  :log_fast => InstructionCost(20, 20.0, 40.0, 20),
  :exp_fast => InstructionCost(20, 20.0, 20.0, 18),
  :sin_fast => InstructionCost(18, 25.0, 50.0, 23),
  :cos_fast => InstructionCost(18, 25.0, 50.0, 26),
  :sinpi_fast => InstructionCost(18, 25.0, 50.0, 23),
  :cospi_fast => InstructionCost(18, 25.0, 50.0, 26),
  :sincospi_fast => InstructionCost(25, 25.0, 50.0, 26),
  :tanh => InstructionCost(-3.0, 80.0, 160.0, 26), # FIXME
  :tanh_fast => InstructionCost(-3.0, 30.0, 60.0, 20), # FIXME
  :sigmoid_fast => InstructionCost(-3.0, 16.0, 66.0, 15), # FIXME
  :identity => InstructionCost(0, 0.0, 0.0, 0),
  :adjoint => InstructionCost(0, 0.0, 0.0, 0),
  :conj => InstructionCost(0, 0.0, 0.0, 0),
  :transpose => InstructionCost(0, 0.0, 0.0, 0),
  :first => InstructionCost(0, 0.0, 0.0, 0),
  :second => InstructionCost(0, 0.0, 0.0, 0),
  :third => InstructionCost(0, 0.0, 0.0, 0),
  :fourth => InstructionCost(0, 0.0, 0.0, 0),
  :fifth => InstructionCost(0, 0.0, 0.0, 0),
  :sixth => InstructionCost(0, 0.0, 0.0, 0),
  :seventh => InstructionCost(0, 0.0, 0.0, 0),
  :eighth => InstructionCost(0, 0.0, 0.0, 0),
  :ninth => InstructionCost(0, 0.0, 0.0, 0),
  :tenth => InstructionCost(0, 0.0, 0.0, 0),
  :eleventh => InstructionCost(0, 0.0, 0.0, 0),
  :twelfth => InstructionCost(0, 0.0, 0.0, 0),
  :thirteenth => InstructionCost(0, 0.0, 0.0, 0),
  :last => InstructionCost(0, 0.0, 0.0, 0),
  :prefetch => InstructionCost(0, 0.0, 0.0, 0),
  :prefetch0 => InstructionCost(0, 0.0, 0.0, 0),
  :prefetch1 => InstructionCost(0, 0.0, 0.0, 0),
  :prefetch2 => InstructionCost(0, 0.0, 0.0, 0),
  :convert => InstructionCost(4, 0.5),
  :oftype => InstructionCost(4, 0.5),
  :vpermilps177 => InstructionCost(1, 1.0),
  :vmovsldup => InstructionCost(1, 1.0),
  :vmovshdup => InstructionCost(1, 1.0),
  :exponent => InstructionCost(8, 1.0),
  :significand => InstructionCost(8, 1.0),
)

# # @inline prefetch0(x::Ptr, i) = VectorizationBase.prefetch(x, Val{3}(), Val{0}())
# @inline prefetch0(x, i) = VectorizationBase.prefetch(gep(stridedpointer(x), (data(i),)), Val{3}(), Val{0}())
# @inline prefetch0(x, I::Tuple) = VectorizationBase.prefetch(gep(x, I), Val{3}(), Val{0}())
# @inline prefetch0(x, i, j) = VectorizationBase.prefetch(gep(stridedpointer(x), (data(i), data(j))), Val{3}(), Val{0}())
# # @inline prefetch0(x, i, j, oi, oj) = VectorizationBase.prefetch(gep(stridedpointer(x), (data(i) + data(oi) - 1, data(j) + data(oj) - 1)), Val{3}(), Val{0}())
# @inline prefetch1(x, i) = VectorizationBase.prefetch(gep(stridedpointer(x), (data(i),)), Val{2}(), Val{0}())
# @inline prefetch1(x, i, j) = VectorizationBase.prefetch(gep(stridedpointer(x), (data(i), data(j))), Val{2}(), Val{0}())
# # @inline prefetch1(x, i, j, oi, oj) = VectorizationBase.prefetch(gep(stridedpointer(x), (data(i) + data(oi) - 1, data(j) + data(oj) - 1)), Val{2}(), Val{0}())
# @inline prefetch2(x, i) = VectorizationBase.prefetch(gep(stridedpointer(x), (data(i),)), Val{1}(), Val{0}())
# @inline prefetch2(x, i, j) = VectorizationBase.prefetch(gep(stridedpointer(x), (data(i), data(j))), Val{1}(), Val{0}())
# # @inline prefetch2(x, i, j, oi, oj) = VectorizationBase.prefetch(gep(stridedpointer(x), (data(i) + data(oi) - 1, data(j) + data(oj) - 1)), Val{1}(), Val{0}())

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

struct IfElseOp{F}
  f::F
end
struct IfElseReducer{F}
  f::F
end
struct IfElseReduced{F}
  f::F
end
struct IfElseReduceTo{F}
  f::F
end
struct IfElseCollapser{F}
  f::F
end
@inline (ieo::IfElseOp)(a, b) = ifelse(ieo.f(a, b), a, b)

@inline (ier::IfElseReducer)(a) = VectorizationBase.ifelse_reduce(ier.f, a)
@inline function (ier::IfElseReducer)(a, b)
  f = ier.f
  r = VectorizationBase.ifelse_reduce(f, b)
  ifelse(f(a, r), a, r)
end
@inline (ier::IfElseReducer)(a::VecUnroll) =
  VecUnroll(VectorizationBase.fmap(ier, VectorizationBase.data(a)))
@inline (ier::IfElseReducer)(a::VecUnroll, b::VecUnroll) = VecUnroll(
  VectorizationBase.fmap(ier, VectorizationBase.data(a), VectorizationBase.data(b)),
)


@inline (ier::IfElseReduced)(x::NativeTypes, y::NativeTypes) = ifelse(ier.f(x, y), x, y)
@inline (ier::IfElseReduced)(x::AbstractSIMD{W}, y::AbstractSIMD{W}) where {W} =
  ifelse(ier.f(x, y), x, y)
@inline function (ier::IfElseReduced)(x::AbstractSIMD, y::AbstractSIMD)
  r = IfElseReduceTo(ier.f)(x, y)
  ifelse(ier.f(r, y), r, y)
end
@inline (ier::IfElseReduced)(x::VecUnroll, y::VecUnroll) =
  VecUnroll(fmap(ier, getfield(x, :data), getfield(y, :data)))
@inline function (ier::IfElseReduced)(x::AbstractSIMD, y::NativeTypes)
  f = ier.f
  r = VectorizationBase.ifelse_reduce(f, x)
  ifelse(f(r, y), r, y)
end


@inline (ier::IfElseReduceTo)(a::NativeTypes, ::NativeTypes) = a
@inline (ier::IfElseReduceTo)(a::AbstractSIMD, ::NativeTypes) =
  VectorizationBase.ifelse_reduce(ier.f, a)
@inline (ier::IfElseReduceTo)(a::AbstractSIMD{W}, ::AbstractSIMD{W}) where {W} = a
@inline function (ier::IfElseReduceTo)(a::AbstractSIMD, b::AbstractSIMD)
  x, y = VectorizationBase.splitvector(a) # halve recursively
  ier(ifelse(ier.f(x, y), x, y), b)
end
@inline (ier::IfElseReduceTo)(a::VecUnroll, b::VecUnroll) = VecUnroll(
  VectorizationBase.fmap(ier, VectorizationBase.data(a), VectorizationBase.data(b)),
)

@inline (iec::IfElseCollapser)(a) = VectorizationBase.collapse(IfElseOp(iec.f), a)
@inline (iec::IfElseCollapser)(a, ::StaticInt{C}) where {C} =
  VectorizationBase.contract(IfElseOp(iec.f), a, StaticInt{C}())

struct IfElseOpMirror{F,A,B}
  f::F
  a::A
  b::B
end
struct IfElseReducerMirror{F,A,B}
  f::F
  a::A
  b::B
end
struct IfElseReducedMirror{F,A,B}
  f::F
  a::A
  b::B
end
struct IfElseReduceToMirror{F,A}
  f::F
  a::A
end
struct IfElseCollapserMirror{F,A}
  f::F
  a::A
end
@inline (ieo::IfElseOpMirror)(a, b) = ifelse(ieo.f(ieo.a, ieo.b), a, b)

@inline _first_ifelse_reduce_mirror(f::F, a, b) where {F} =
  getfield(VectorizationBase.ifelse_reduce_mirror(f, a, b), 1, false)
@inline (ier::IfElseReducerMirror)(a) = _ifelse_reduce_mirror(ier.f, a, ier.a)
@inline function _ifelse_reduce_mirror(f::F, a, b, c, d) where {F}
  r, rm = VectorizationBase.ifelse_reduce_mirror(f, b, d)
  ifelse(f(c, rm), a, r)
end
@inline (ier::IfElseReducerMirror)(a, b) = _ifelse_reduce_mirror(ier.f, a, b, ier.a, ier.b)
@inline (ier::IfElseReducerMirror)(a::VecUnroll) = VecUnroll(
  VectorizationBase.fmap(
    _first_ifelse_reduce_mirror,
    ier.f,
    VectorizationBase.data(a),
    VectorizationBase.data(ier.a),
  ),
)
@inline function (ier::IfElseReducerMirror)(a::VecUnroll, b::VecUnroll)
  VecUnroll(
    VectorizationBase.fmap(
      _ifelse_reduce_mirror,
      ier.f,
      VectorizationBase.data(a),
      VectorizationBase.data(b),
      VectorizationBase.data(ier.a),
      VectorizationBase.data(ier.b),
    ),
  )
end

@inline IfElseReducedMirror(f::F, a::A) where {F,A} =
  IfElseReducedMirror{F,A,Nothing}(f, a, nothing)
@inline (ier::IfElseReducedMirror)(x::NativeTypes, y::NativeTypes) =
  ifelse(ier.f(ier.a, ier.b), x, y)
@inline (ier::IfElseReducedMirror)(x::AbstractSIMD{W}, y::AbstractSIMD{W}) where {W} =
  ifelse(ier.f(ier.a, ier.b), x, y)
@inline function _reduce_mirror(f::F, x, y, a, b) where {F}
  r, rm = IfElseReduceToMirror(f, a, b)(x, y)
  ifelse(f(r, y), r, y)
end
@inline (ier::IfElseReducedMirror)(x::AbstractSIMD, y::AbstractSIMD) =
  _reduce_mirror(ier.f, x, y, ier.a, ier.b)
@inline (ier::IfElseReducedMirror)(x::VecUnroll, y::VecUnroll) = VecUnroll(
  fmap(
    _reduce_mirror,
    ier.f,
    getfield(x, :data),
    getfield(y, :data),
    getfield(ier.a, :data),
    getfield(ier.b, :data),
  ),
)
@inline function (ier::IfElseReducedMirror)(x::AbstractSIMD, y::NativeTypes)
  f = ier.f
  r, rm = VectorizationBase.ifelse_reduce_mirror(f, x, ier.a)
  # @show rm, ier.b, r, y
  ifelse(f(rm, ier.b), r, y)
end


@inline (ier::IfElseReduceToMirror)(a::NativeTypes, ::NativeTypes) = a
@inline (ier::IfElseReduceToMirror)(a::AbstractSIMD, ::NativeTypes) =
  VectorizationBase.ifelse_reduce_mirror(ier.f, a, ier.a)
@inline (ier::IfElseReduceToMirror)(a::AbstractSIMD{W}, ::AbstractSIMD{W}) where {W} = a
@inline function (ier::IfElseReduceToMirror)(a::AbstractSIMD, b::AbstractSIMD)
  x, y = VectorizationBase.splitvector(a) # halve recursively
  w, z = VectorizationBase.splitvector(ier.a) # halve recursively
  f = ier.f
  fwz = f(w, z)
  IfElseReduceToMirror(f, ifelse(fwz, w, z))(ifelse(fwz, x, y), b)
end
@inline (ier::IfElseReduceToMirror)(a::VecUnroll, b::VecUnroll) = VecUnroll(
  VectorizationBase.fmap(ier, VectorizationBase.data(a), VectorizationBase.data(b)),
)

# @inline (iec::IfElseCollapserMirror)(a) = getfield(VectorizationBase.ifelse_collapse_mirror(iec.f, a, iec.a), 1, false)
# @inline (iec::IfElseCollapserMirror)(a, ::StaticInt{N}) where {N} = getfield(VectorizationBase.ifelse_collapse_mirror(iec.f, a, iec.a, StaticInt{N}()), 1, false)

@inline (iec::IfElseCollapserMirror)(a) =
  VectorizationBase.ifelse_collapse_mirror(iec.f, a, iec.a)
@inline (iec::IfElseCollapserMirror)(a, ::StaticInt{N}) where {N} =
  VectorizationBase.ifelse_collapse_mirror(iec.f, a, iec.a, StaticInt{N}())

# @inline function (iec::IfElseCollapserMirror)(a, ::StaticInt{C}) where {C}
#   VectorizationBase.contract(IfElseOp(iec.f), a, StaticInt{C}())
# end

const ADDITIVE_IN_REDUCTIONS = 1.0
const MULTIPLICATIVE_IN_REDUCTIONS = 2.0
const ANY = 3.0
const ALL = 4.0
const MAX = 5.0
const MIN = 6.0
const IFELSE = 7.0

const REDUCTION_CLASS = Dict{Symbol,Float64}(
  :+ => ADDITIVE_IN_REDUCTIONS,
  :- => ADDITIVE_IN_REDUCTIONS,
  :* => MULTIPLICATIVE_IN_REDUCTIONS,
  :vadd => ADDITIVE_IN_REDUCTIONS,
  :vsub => ADDITIVE_IN_REDUCTIONS,
  :add_fast => ADDITIVE_IN_REDUCTIONS,
  :sub_fast => ADDITIVE_IN_REDUCTIONS,
  :vadd_fast => ADDITIVE_IN_REDUCTIONS,
  :vsub_fast => ADDITIVE_IN_REDUCTIONS,
  # :vadd! => ADDITIVE_IN_REDUCTIONS,
  # :vsub! => ADDITIVE_IN_REDUCTIONS,
  :vmul => MULTIPLICATIVE_IN_REDUCTIONS,
  :mul_fast => MULTIPLICATIVE_IN_REDUCTIONS,
  :vmul_fast => MULTIPLICATIVE_IN_REDUCTIONS,
  # :vmul! => MULTIPLICATIVE_IN_REDUCTIONS,
  # :evadd => ADDITIVE_IN_REDUCTIONS,
  # :evsub => ADDITIVE_IN_REDUCTIONS,
  # :evmul => MULTIPLICATIVE_IN_REDUCTIONS,
  :& => ALL,
  :| => ANY,
  :muladd => ADDITIVE_IN_REDUCTIONS,
  :fma => ADDITIVE_IN_REDUCTIONS,
  :vmuladd_fast => ADDITIVE_IN_REDUCTIONS,
  :vfma_fast => ADDITIVE_IN_REDUCTIONS,
  :vfmadd => ADDITIVE_IN_REDUCTIONS,
  :vfmsub => ADDITIVE_IN_REDUCTIONS,
  :vfnmadd => ADDITIVE_IN_REDUCTIONS,
  :vfnmsub => ADDITIVE_IN_REDUCTIONS,
  :vfmadd_fast => ADDITIVE_IN_REDUCTIONS,
  :vfmsub_fast => ADDITIVE_IN_REDUCTIONS,
  :vfnmadd_fast => ADDITIVE_IN_REDUCTIONS,
  :vfnmsub_fast => ADDITIVE_IN_REDUCTIONS,
  :vfmadd231 => ADDITIVE_IN_REDUCTIONS,
  :vfmsub231 => ADDITIVE_IN_REDUCTIONS,
  :vfnmadd231 => ADDITIVE_IN_REDUCTIONS,
  :vfnmsub231 => ADDITIVE_IN_REDUCTIONS,
  # :vfmadd! => ADDITIVE_IN_REDUCTIONS,
  # :vfnmadd! => ADDITIVE_IN_REDUCTIONS,
  # :vfmsub! => ADDITIVE_IN_REDUCTIONS,
  # :vfnmsub! => ADDITIVE_IN_REDUCTIONS,
  :reduced_add => ADDITIVE_IN_REDUCTIONS,
  :reduced_prod => MULTIPLICATIVE_IN_REDUCTIONS,
  :reduced_all => ALL,
  :reduced_any => ANY,
  :max => MAX,
  :min => MIN,
  :max_fast => MAX,
  :min_fast => MIN,
  :vfmaddsub => ADDITIVE_IN_REDUCTIONS,
  :vfmsubadd => ADDITIVE_IN_REDUCTIONS,
)
reduction_instruction_class(instr::Symbol) = get(REDUCTION_CLASS, instr, NaN)
reduction_instruction_class(instr::Instruction) = reduction_instruction_class(instr.instr)
function reduction_to_single_vector(x::Float64)
  if x == ADDITIVE_IN_REDUCTIONS
    :collapse_add
  elseif x == MULTIPLICATIVE_IN_REDUCTIONS
    :collapse_mul
  elseif x == MAX
    :collapse_max
  elseif x == MIN
    :collapse_min
  elseif x == ALL
    :collapse_and
  elseif x == ANY
    :collapse_or
  else
    throw("Reduction not found.")
  end
end
function reduce_to_onevecunroll(x::Float64)
  if x == ADDITIVE_IN_REDUCTIONS
    :+
  elseif x == MULTIPLICATIVE_IN_REDUCTIONS
    :*
  elseif x == MAX
    :max
  elseif x == MIN
    :min
  elseif x == ALL
    :&
  elseif x == ANY
    :|
  else
    throw("Reduction not found.")
  end
end
function reduce_number_of_vectors(x::Float64)
  if x == ADDITIVE_IN_REDUCTIONS
    :contract_add
  elseif x == MULTIPLICATIVE_IN_REDUCTIONS
    :contract_mul
  elseif x == MAX
    :contract_max
  elseif x == MIN
    :contract_min
  elseif x == ALL
    :contract_and
  elseif x == ANY
    :contract_or
  else
    throw("Reduction not found.")
  end
end
function reduction_to_scalar(x::Float64)
  if x == ADDITIVE_IN_REDUCTIONS
    :vsum
  elseif x == MULTIPLICATIVE_IN_REDUCTIONS
    :vprod
  elseif x == MAX
    :vmaximum
  elseif x == MIN
    :vminimum
  elseif x == ALL
    :vall
  elseif x == ANY
    :vany
  else
    throw("Reduction not found.")
  end
end
function reduction_scalar_combine(x::Float64)
  # x == 1.0 ? :reduced_add : x == 2.0 ? :reduced_prod : x == 3.0 ? :reduced_any : x == 4.0 ? :reduced_all : x == 5.0 ? :reduced_max : x == 6.0 ? :reduced_min : throw("Reduction not found.")
  if x == ADDITIVE_IN_REDUCTIONS
    :reduced_add
  elseif x == MULTIPLICATIVE_IN_REDUCTIONS
    :reduced_prod
  elseif x == MAX
    :reduced_max
  elseif x == MIN
    :reduced_min
  elseif x == ALL
    :reduced_all
  elseif x == ANY
    :reduced_any
  else
    throw("Reduction not found.")
  end
end

function reduction_zero(x::Float64)
  # x == 1.0 ? :zero : x == 2.0 ? :one : x == 3.0 ? :false : x == 4.0 ? :true : x == 5.0 ? :typemin : x == 6.0 ? :typemax : throw("Reduction not found.")
  if x == ADDITIVE_IN_REDUCTIONS
    :zero
  elseif x == MULTIPLICATIVE_IN_REDUCTIONS
    :one
  elseif x == MAX
    :typemin
  elseif x == MIN
    :typemax
  elseif x == ALL
    :max_mask
  elseif x == ANY
    :zero_mask
  else
    throw("Reduction not found.")
  end
end
function reduction_zero_class(x::Symbol)::Float64
  if x === :one
    MULTIPLICATIVE_IN_REDUCTIONS
  elseif x === :typemin
    MAX
  elseif x === :typemax
    MIN
  elseif x === :max_mask
    ALL
  elseif x === :zero_mask
    ANY
  elseif x === :zero#sorted last, as should go into preamble_zeros
    ADDITIVE_IN_REDUCTIONS
  else
    throw("Reduction not found.")
  end
end
reduction_zero(x) = reduction_zero(reduction_instruction_class(x))


function isreductcombineinstr(instr::Symbol)
  instr ∈ (
    :reduced_add,
    :reduced_prod,
    :reduce_to_add,
    :reduce_to_prod,
    :reduced_max,
    :reduced_min,
    :reduce_to_max,
    :reduce_to_min,
  )
end
isreductcombineinstr(instr::Instruction) = isreductcombineinstr(instr.instr)

const FUNCTIONSYMBOLS = IdDict{Type{<:Function},Instruction}(
  typeof(+) => :add_fast,
  typeof(VectorizationBase.vadd) => :add_fast,
  typeof(add_fast) => :add_fast,
  typeof(-) => :sub_fast,
  typeof(VectorizationBase.vsub) => :sub_fast,
  typeof(sub_fast) => :sub_fast,
  typeof(*) => :mul_fast,
  typeof(VectorizationBase.vmul) => :mul_fast,
  typeof(mul_fast) => :mul_fast,
  typeof(/) => :div_fast,
  typeof(^) => :pow_fast,
  typeof(VectorizationBase.vdiv) => :div_fast,
  typeof(÷) => :(÷),
  typeof(div_fast) => :div_fast,
  typeof(rem_fast) => :rem_fast,
  typeof(==) => :(==),
  typeof(!=) => :(!=),
  typeof(isequal) => :isequal,
  typeof(isnan) => :isnan,
  typeof(isinf) => :isinf,
  typeof(isfinite) => :isfinite,
  typeof(abs) => :abs,
  typeof(abs2) => :abs2_fast,
  typeof(abs2_fast) => :abs2_fast,
  typeof(~) => :(~),
  typeof(!) => :(!),
  typeof(&) => :(&),
  typeof(|) => :(|),
  typeof(⊻) => :(⊻),
  typeof(>) => :(>),
  typeof(<) => :(<),
  typeof(>=) => :(>=),
  typeof(<=) => :(<=),
  typeof(inv) => :inv_fast,
  typeof(inv_fast) => :inv_fast,
  typeof(muladd) => :vmuladd_fast,
  typeof(fma) => :vfma_fast,
  typeof(VectorizationBase.vfma) => :vfma_fast,
  typeof(VectorizationBase.vmuladd) => :vmuladd_fast,
  typeof(VectorizationBase.vfmsub) => :vfmsub_fast,
  typeof(VectorizationBase.vfnmadd) => :vfnmadd_fast,
  typeof(VectorizationBase.vfnmsub) => :vfnmsub_fast,
  typeof(VectorizationBase.vfma_fast) => :vfma_fast,
  typeof(VectorizationBase.vmuladd_fast) => :vmuladd_fast,
  typeof(VectorizationBase.vfmsub_fast) => :vfmsub_fast,
  typeof(VectorizationBase.vfnmadd_fast) => :vfnmadd_fast,
  typeof(VectorizationBase.vfnmsub_fast) => :vfnmsub_fast,
  typeof(VectorizationBase.vfmadd231) => :vfmadd231,
  typeof(VectorizationBase.vfmsub231) => :vfmsub231,
  typeof(VectorizationBase.vfnmadd231) => :vfnmadd231,
  typeof(VectorizationBase.vfnmsub231) => :vfnmsub231,
  typeof(sqrt) => :sqrt_fast,
  typeof(sqrt_fast) => :sqrt_fast,
  typeof(log) => :log_fast,
  typeof(log2) => :log2_fast,
  typeof(log10) => :log10_fast,
  typeof(log_fast) => :log_fast,
  typeof(log1p) => :log1p,
  # typeof(VectorizationBase.vlog) => :log,
  typeof(SLEEFPirates.log) => :log,
  typeof(exp) => :exp,
  typeof(exp2) => :exp2,
  typeof(exp10) => :exp10,
  typeof(Base.FastMath.exp_fast) => :exp,
  typeof(expm1) => :expm1,
  # typeof(VectorizationBase.vexp) => :exp,
  typeof(SLEEFPirates.exp) => :exp,
  typeof(sin) => :sin_fast,
  typeof(Base.FastMath.sin_fast) => :sin_fast,
  typeof(SLEEFPirates.sin) => :sin_fast,
  typeof(cos) => :cos_fast,
  typeof(Base.FastMath.cos_fast) => :cos_fast,
  typeof(SLEEFPirates.cos) => :cos_fast,
  typeof(sincos) => :sincos_fast,
  typeof(Base.FastMath.sincos_fast) => :sincos_fast,
  typeof(SLEEFPirates.sincos) => :sincos_fast,
  typeof(tan) => :tan_fast,
  typeof(Base.FastMath.sincos_fast) => :sincos_fast,
  typeof(SLEEFPirates.sincos) => :sincos_fast,
  typeof(Base.tanh) => :tanh,
  typeof(tanh_fast) => :tanh_fast,
  typeof(sigmoid_fast) => :sigmoid_fast,
  typeof(max) => :max_fast,
  typeof(min) => :min_fast,
  typeof(max_fast) => :max_fast,
  typeof(min_fast) => :min_fast,
  typeof(relu) => :relu,
  typeof(<<) => :<<,
  typeof(>>) => :>>,
  typeof(>>>) => :>>>,
  typeof(%) => :(%),
  typeof(÷) => :(÷),
  typeof(Base.ifelse) => :ifelse,
  typeof(ifelse) => :ifelse,
  typeof(identity) => :identity,
  typeof(conj) => :identity,#conj,
  typeof(÷) => :vdiv_fast,
  # typeof(zero) => :zero,
  # typeof(one) => :one,
  # typeof(axes) => :axes,
  # typeof(eltype) => :eltype
)

# implement whitelist for avx_support that package authors may use to conservatively guard `@turbo` application
for f ∈ keys(FUNCTIONSYMBOLS)
  @eval ArrayInterface.can_avx(::$f) = true
end
