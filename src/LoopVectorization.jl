module LoopVectorization

using VectorizationBase, SIMDPirates, SLEEFPirates, Parameters
using VectorizationBase: REGISTER_SIZE, REGISTER_COUNT, extract_data, num_vector_load_expr,
    mask, masktable, pick_vector_width_val, valmul, valrem, valmuladd, valadd, valsub, _MM,
    maybestaticlength, maybestaticsize, staticm1, subsetview, vzero, stridedpointer_for_broadcast,
    Static, StaticUnitRange, StaticLowerUnitRange, StaticUpperUnitRange,
    PackedStridedPointer, SparseStridedPointer, RowMajorStridedPointer, StaticStridedPointer, StaticStridedStruct
using SIMDPirates: VECTOR_SYMBOLS, evadd, evmul, vrange, reduced_add, reduced_prod, reduce_to_add, reduce_to_prod,
    sizeequivalentfloat, sizeequivalentint, vadd!, vsub!, vmul!, vfdiv!, vfmadd!, vfnmadd!, vfmsub!, vfnmsub!,
    vfmadd231, vfmsub231, vfnmadd231, vfnmsub231, #prefetch,
    vmullog2, vmullog10, vdivlog2, vdivlog10, vmullog2add!, vmullog10add!, vdivlog2add!, vdivlog10add!, vfmaddaddone
using Base.Broadcast: Broadcasted, DefaultArrayStyle
using LinearAlgebra: Adjoint, Transpose

const SUPPORTED_TYPES = Union{Float16,Float32,Float64,Integer}

export LowDimArray, stridedpointer, vectorizable,
    @avx, @_avx, *ˡ, _avx_!,
    vmap, vmap!, vmapnt, vmapnt!, vmapntt, vmapntt!,
    vfilter, vfilter!


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
include("broadcast.jl")
include("determinestrategy.jl")
include("lower_compute.jl")
include("lower_constant.jl")
include("lower_memory_common.jl")
include("lower_load.jl")
include("lower_store.jl")
include("lowering.jl")
include("condense_loopset.jl")
include("reconstruct_loopset.jl")
include("constructors.jl")


include("precompile.jl")
_precompile_()

end # module
