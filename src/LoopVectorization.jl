module LoopVectorization

using Static: StaticInt, gt, static
using VectorizationBase, SLEEFPirates, UnPack, OffsetArrays
using VectorizationBase:
  mask,
  MM,
  AbstractMask,
  data,
  grouped_strided_pointer,
  AbstractSIMD,
  vzero,
  offsetprecalc,
  lazymul,
  vadd_nw,
  vadd_nsw,
  vadd_nuw,
  vsub_nw,
  vsub_nsw,
  vsub_nuw,
  vmul_nw,
  vmul_nsw,
  vmul_nuw,
  vfmaddsub,
  vfmsubadd,
  vpermilps177,
  vmovsldup,
  vmovshdup,
  maybestaticfirst,
  maybestaticlast,
  gep,
  gesp,
  NativeTypes, #llvmptr,
  vfmadd,
  vfmsub,
  vfnmadd,
  vfnmsub,
  vfmadd_fast,
  vfmsub_fast,
  vfnmadd_fast,
  vfnmsub_fast,
  vfmadd231,
  vfmsub231,
  vfnmadd231,
  vfnmsub231,
  vfma_fast,
  vmuladd_fast,
  vdiv_fast,
  vadd_fast,
  vsub_fast,
  vmul_fast,
  relu,
  stridedpointer,
  StridedPointer,
  StridedBitPointer,
  AbstractStridedPointer,
  _vload,
  _vstore!,
  reduced_add,
  reduced_prod,
  reduce_to_add,
  reduce_to_prod,
  reduced_max,
  reduced_min,
  reduce_to_max,
  reduce_to_min,
  reduced_all,
  reduced_any,
  reduce_to_all,
  reduce_to_any,
  vsum,
  vprod,
  vmaximum,
  vminimum,
  vany,
  vall,
  unwrap,
  Unroll,
  VecUnroll,
  preserve_buffer,
  zero_vecunroll,
  vbroadcast_vecunroll,
  _vzero,
  _vbroadcast,
  contract_add,
  collapse_add,
  contract_mul,
  collapse_mul,
  contract_max,
  collapse_max,
  contract_min,
  collapse_min,
  contract_and,
  collapse_and,
  contract_or,
  collapse_or,
  max_mask,
  maybestaticsize#,zero_mask

using HostCPUFeatures:
  pick_vector_width, register_size, register_count, has_opmask_registers
using CPUSummary: num_threads, num_cores, cache_linesize, cache_size

using LayoutPointers: stridedpointer_preserve, GroupedStridedPointers

using IfElse: ifelse

using ThreadingUtilities, PolyesterWeave
using Base.Broadcast: Broadcasted, DefaultArrayStyle
using LinearAlgebra: Adjoint, Transpose, Diagonal
using Base.Meta: isexpr
using DocStringExtensions
import LinearAlgebra # for check_args

using Base.FastMath:
  add_fast,
  sub_fast,
  mul_fast,
  div_fast,
  inv_fast,
  abs2_fast,
  rem_fast,
  max_fast,
  min_fast,
  pow_fast,
  sqrt_fast
using SLEEFPirates:
  log_fast, log2_fast, log10_fast, pow, sin_fast, cos_fast, sincos_fast, tan_fast

using ArrayInterface
using ArrayInterface:
  OptionallyStaticUnitRange,
  OptionallyStaticRange,
  Zero,
  One,
  StaticBool,
  True,
  False,
  reduce_tup,
  indices,
  UpTri,
  LoTri,
  strides,
  offsets,
  size,
  axes,
  StrideIndex
using CloseOpenIntervals: AbstractCloseOpen, CloseOpen#, SafeCloseOpen
# @static if VERSION ≥ v"1.6.0-rc1" #TODO: delete `else` when dropping 1.5 support
# using ArrayInterface: static_step
# else # Julia 1.5 did not define `step` on CartesianIndices
@inline static_step(x) = ArrayInterface.static_step(x)
@inline static_step(x::CartesianIndices) =
  VectorizationBase.CartesianVIndex(map(static_step, x.indices))
# end
const Static = StaticInt

export LowDimArray,
  stridedpointer,
  indices,
  static,
  @avx,
  @avxt,
  @turbo,
  @tturbo,
  *ˡ,
  _turbo_!,
  vmap,
  vmap!,
  vmapt,
  vmapt!,
  vmapnt,
  vmapnt!,
  vmapntt,
  vmapntt!,
  tanh_fast,
  sigmoid_fast,
  vfilter,
  vfilter!,
  vmapreduce,
  vreduce

const VECTORWIDTHSYMBOL, ELTYPESYMBOL, MASKSYMBOL =
  Symbol("##Wvecwidth##"), Symbol("##Tloopeltype##"), Symbol("##mask##")


include("vectorizationbase_compat/contract_pass.jl")
include("vectorizationbase_compat/subsetview.jl")
include("getconstindexes.jl")
include("predicates.jl")
include("simdfunctionals/map.jl")
include("simdfunctionals/filter.jl")
include("modeling/costs.jl")
include("modeling/operations.jl")
include("modeling/graphs.jl")
include("codegen/operation_evaluation_order.jl")
include("parse/memory_ops_common.jl")
include("parse/add_loads.jl")
include("parse/add_stores.jl")
include("parse/add_compute.jl")
include("parse/add_constants.jl")
include("parse/add_ifelse.jl")
include("modeling/determinestrategy.jl")
include("codegen/line_number_nodes.jl")
include("codegen/loopstartstopmanager.jl")
include("codegen/lower_compute.jl")
include("codegen/lower_constant.jl")
include("codegen/lower_memory_common.jl")
include("codegen/lower_load.jl")
include("codegen/lower_store.jl")
include("codegen/lowering.jl")
include("codegen/split_loops.jl")
include("codegen/lower_threads.jl")
include("condense_loopset.jl")
include("transforms.jl")
include("reconstruct_loopset.jl")
include("constructors.jl")
include("user_api_conveniences.jl")
include("simdfunctionals/mapreduce.jl")
include("broadcast.jl")

"""
LoopVectorization provides macros and functions that combine SIMD vectorization and
loop-reordering so as to improve performance:

- [`@turbo`](@ref): transform `for`-loops and broadcasting
- [`vmapreduce`](@ref): vectorized version of `mapreduce`
- [`vreduce`](@ref): vectorized version of `reduce`
- [`vmap`](@ref) and `vmap!`: vectorized version of `map` and `map!`
- [`vmapnt`](@ref) and `vmapnt!`: non-temporal variants of `vmap` and `vmap!`
- [`vmapntt`](@ref) and `vmapntt!`: threaded variants of `vmapnt` and `vmapnt!`
- [`vfilter`](@ref) and `vfilter!`: vectorized versions of `filter` and `filter!`
"""
LoopVectorization

include("precompile.jl")
_precompile_()

_vreduce(+, Float64[1.0])
# matmul_params(64, 32, 64)

# import ChainRulesCore, ForwardDiff
# include("vmap_grad.jl")
using ChainRulesCore, ForwardDiff, SpecialFunctions
include("simdfunctionals/vmap_grad_rrule.jl")
include("simdfunctionals/vmap_grad_forwarddiff.jl")
@inline SpecialFunctions.erf(x::AbstractSIMD) = VectorizationBase.verf(float(x))

end # module
