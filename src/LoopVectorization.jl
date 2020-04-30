module LoopVectorization

using VectorizationBase, SIMDPirates, SLEEFPirates, UnPack
using VectorizationBase: REGISTER_SIZE, REGISTER_COUNT, extract_data, num_vector_load_expr,
    mask, masktable, pick_vector_width_val, valmul, valrem, valmuladd, valadd, valsub, _MM,
    maybestaticlength, maybestaticsize, staticm1, subsetview, vzero, stridedpointer_for_broadcast,
    Static, StaticUnitRange, StaticLowerUnitRange, StaticUpperUnitRange, unwrap, maybestaticrange,
    AbstractColumnMajorStridedPointer, AbstractRowMajorStridedPointer, AbstractSparseStridedPointer, AbstractStaticStridedPointer,
    PackedStridedPointer, SparseStridedPointer, RowMajorStridedPointer, StaticStridedPointer, StaticStridedStruct,
    maybestaticfirst, maybestaticlast, scalar_less, scalar_greater
using SIMDPirates: VECTOR_SYMBOLS, evadd, evsub, evmul, evfdiv, vrange, reduced_add, reduced_prod, reduce_to_add, reduce_to_prod, vsum, vprod, vmaximum, vminimum,
    sizeequivalentfloat, sizeequivalentint, vadd!, vsub!, vmul!, vfdiv!, vfmadd!, vfnmadd!, vfmsub!, vfnmsub!,
    vfmadd231, vfmsub231, vfnmadd231, vfnmsub231, sizeequivalentfloat, sizeequivalentint, #prefetch,
    vmullog2, vmullog10, vdivlog2, vdivlog10, vmullog2add!, vmullog10add!, vdivlog2add!, vdivlog10add!, vfmaddaddone
using SLEEFPirates: pow
using Base.Broadcast: Broadcasted, DefaultArrayStyle
using LinearAlgebra: Adjoint, Transpose
using Base.Meta: isexpr
using DocStringExtensions


const NativeTypes = Union{Bool, Base.HWReal}

export LowDimArray, stridedpointer, vectorizable,
    @avx, @_avx, *ˡ, _avx_!,
    vmap, vmap!, vmapnt, vmapnt!, vmapntt, vmapntt!,
    vfilter, vfilter!

const VECTORWIDTHSYMBOL, ELTYPESYMBOL = Symbol("##Wvecwidth##"), Symbol("##Tloopeltype##")


include("vectorizationbase_extensions.jl")
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
include("broadcast.jl")
include("determinestrategy.jl")
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

end # module
