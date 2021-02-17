module LoopVectorization

# if (!isnothing(get(ENV, "TRAVIS_BRANCH", nothing)) || !isnothing(get(ENV, "APPVEYOR", nothing))) && isdefined(Base, :Experimental) && isdefined(Base.Experimental, Symbol("@optlevel"))
    # @eval Base.Experimental.@optlevel 1
# end

using VectorizationBase, SLEEFPirates, UnPack, OffsetArrays
using VectorizationBase: register_size, register_count, cache_linesize, has_opmask_registers,
    mask, pick_vector_width, MM, data, grouped_strided_pointer,
    maybestaticlength, maybestaticsize, staticm1, staticp1, staticmul, vzero,
    maybestaticrange, offsetprecalc, lazymul,
    maybestaticfirst, maybestaticlast, scalar_less, scalar_greaterequal, gep, gesp, pointerforcomparison, NativeTypes,
    vfmadd, vfmsub, vfnmadd, vfnmsub, vfmadd_fast, vfmsub_fast, vfnmadd_fast, vfnmsub_fast, vfmadd231, vfmsub231, vfnmadd231, vfnmsub231,
    vfma_fast, vmuladd_fast, vdiv_fast, vadd_fast, vsub_fast, vmul_fast,
    relu, stridedpointer, StridedPointer, StridedBitPointer, AbstractStridedPointer,
    reduced_add, reduced_prod, reduce_to_add, reduce_to_prod, reduced_max, reduced_min, reduce_to_max, reduce_to_min,
    vsum, vprod, vmaximum, vminimum, unwrap, Unroll, VecUnroll,
    preserve_buffer, zero_vecunroll, vbroadcast_vecunroll, _vzero, _vbroadcast,
    contract_add, collapse_add,
    contract_mul, collapse_mul,
    contract_max, collapse_max,
    contract_min, collapse_min,
    contract_and, collapse_and,
    contract_or,  collapse_or


using IfElse: ifelse

using ThreadingUtilities
using SLEEFPirates: pow
using Base.Broadcast: Broadcasted, DefaultArrayStyle
using LinearAlgebra: Adjoint, Transpose
using Base.Meta: isexpr
using DocStringExtensions
import LinearAlgebra # for check_args

using Base.FastMath: add_fast, sub_fast, mul_fast, div_fast, inv_fast, abs2_fast, rem_fast, max_fast, min_fast


using ArrayInterface
using ArrayInterface: OptionallyStaticUnitRange, Zero, One, StaticBool, True, False, reduce_tup, indices
const Static = ArrayInterface.StaticInt

using Requires


export LowDimArray, stridedpointer, indices,
    @avx, @_avx, *ˡ, _avx_!,
    vmap, vmap!, vmapt, vmapt!, vmapnt, vmapnt!, vmapntt, vmapntt!,
    tanh_fast, sigmoid_fast,
    vfilter, vfilter!, vmapreduce, vreduce

const VECTORWIDTHSYMBOL, ELTYPESYMBOL = Symbol("##Wvecwidth##"), Symbol("##Tloopeltype##")

include("vectorizationbase_compat/contract_pass.jl")
include("vectorizationbase_compat/subsetview.jl")
include("closeopen.jl")
include("getconstindexes.jl")
include("predicates.jl")
include("map.jl")
include("filter.jl")
include("costs.jl")
include("operations.jl")
include("graphs.jl")
include("operation_evaluation_order.jl")
include("memory_ops_common.jl")
include("add_loads.jl")
include("add_stores.jl")
include("add_compute.jl")
include("add_constants.jl")
include("add_ifelse.jl")
include("determinestrategy.jl")
include("line_number_nodes.jl")
include("loopstartstopmanager.jl")
include("lower_compute.jl")
include("lower_constant.jl")
include("lower_memory_common.jl")
include("lower_load.jl")
include("lower_store.jl")
include("lowering.jl")
include("split_loops.jl")
include("condense_loopset.jl")
include("reconstruct_loopset.jl")
include("constructors.jl")
include("user_api_conveniences.jl")
include("mapreduce.jl")
include("broadcast.jl")

"""
LoopVectorization provides macros and functions that combine SIMD vectorization and
loop-reordering so as to improve performance:

- [`@avx`](@ref): transform `for`-loops and broadcasting
- [`@_avx`](@ref): similar to `@avx` but does not use type information
- [`vmap`](@ref) and `vmap!`: vectorized version of `map` and `map!`
- [`vmapnt`](@ref) and `vmapnt!`: non-temporal variants of `vmap` and `vmap!`
- [`vmapntt`](@ref) and `vmapntt!`: threaded variants of `vmapnt` and `vmapnt!`
- [`vfilter`](@ref) and `vfilter!`: vectorized versions of `filter` and `filter!`
"""
LoopVectorization

include("precompile.jl")
_precompile_()

# import ChainRulesCore, ForwardDiff
# include("vmap_grad.jl")
function __init__()
    @require ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4" begin
        @require ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210" include("vmap_grad.jl")
    end
end

end # module
