module LoopVectorization

using VectorizationBase, SIMDPirates, SLEEFPirates, MacroTools, Parameters
using VectorizationBase: REGISTER_SIZE, REGISTER_COUNT, extract_data, num_vector_load_expr,
    mask, masktable, pick_vector_width_val, valmul, valrem, valmuladd, valadd, valsub
using SIMDPirates: VECTOR_SYMBOLS, evadd, evmul, vrange, reduced_add, reduced_prod
using Base.Broadcast: Broadcasted, DefaultArrayStyle
using LinearAlgebra: Adjoint, Transpose
using MacroTools: prewalk, postwalk


export LowDimArray, stridedpointer, vectorizable,
    @avx, ∗,
    vmap, vmap!

const SLEEFPiratesDict = Dict{Symbol,Tuple{Symbol,Symbol}}(
    :sin => (:SLEEFPirates, :sin_fast),
    :sinpi => (:SLEEFPirates, :sinpi),
    :cos => (:SLEEFPirates, :cos_fast),
    :cospi => (:SLEEFPirates, :cospi),
    :tan => (:SLEEFPirates, :tan_fast),
    # :log => (:SLEEFPirates, :log_fast),
    :log => (:SIMDPirates, :vlog),
    :log10 => (:SLEEFPirates, :log10),
    :log2 => (:SLEEFPirates, :log2),
    :log1p => (:SLEEFPirates, :log1p),
    # :exp => (:SLEEFPirates, :exp),
    :exp => (:SIMDPirates, :vexp),
    :exp2 => (:SLEEFPirates, :exp2),
    :exp10 => (:SLEEFPirates, :exp10),
    :expm1 => (:SLEEFPirates, :expm1),
    :inv => (:SIMDPirates, :vinv), # faster than sqrt_fast
    :sqrt => (:SIMDPirates, :sqrt), # faster than sqrt_fast
    :rsqrt => (:SIMDPirates, :rsqrt),
    :cbrt => (:SLEEFPirates, :cbrt_fast),
    :asin => (:SLEEFPirates, :asin_fast),
    :acos => (:SLEEFPirates, :acos_fast),
    :atan => (:SLEEFPirates, :atan_fast),
    :sinh => (:SLEEFPirates, :sinh),
    :cosh => (:SLEEFPirates, :cosh),
    :tanh => (:SLEEFPirates, :tanh),
    :asinh => (:SLEEFPirates, :asinh),
    :acosh => (:SLEEFPirates, :acosh),
    :atanh => (:SLEEFPirates, :atanh),
    # :erf => :(SLEEFPirates.erf),
    # :erfc => :(SLEEFPirates.erfc),
    # :gamma => :(SLEEFPirates.gamma),
    # :lgamma => :(SLEEFPirates.lgamma),
    :trunc => (:SLEEFPirates, :trunc),
    :floor => (:SLEEFPirates, :floor),
    :ceil => (:SIMDPirates, :ceil),
    :abs => (:SIMDPirates, :vabs),
    :sincos => (:SLEEFPirates, :sincos_fast),
    # :pow => (:SLEEFPirates, :pow_fast),
    :^ => (:SLEEFPirates, :pow_fast),
    # :sincospi => (:SLEEFPirates, :sincospi_fast),
    # :pow => (:SLEEFPirates, :pow),
    # :hypot => (:SLEEFPirates, :hypot_fast),
    :mod => (:SLEEFPirates, :mod),
    # :copysign => :copysign
    :one => (:SIMDPirates, :vone),
    :zero => (:SIMDPirates, :vzero),
    :erf => (:SIMDPirates, :verf)
)

include("costs.jl")
include("operations.jl")
include("graphs.jl")
include("broadcast.jl")
include("determinestrategy.jl")
include("lowering.jl")
include("constructors.jl")
include("map.jl")
# include("precompile.jl")
# _precompile_()

end # module
