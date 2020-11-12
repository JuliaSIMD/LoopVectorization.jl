module LoopVectorization

if (!isnothing(get(ENV, "TRAVIS_BRANCH", nothing)) || !isnothing(get(ENV, "APPVEYOR", nothing))) && isdefined(Base, :Experimental) && isdefined(Base.Experimental, Symbol("@optlevel"))
    @eval Base.Experimental.@optlevel 1
end

using VectorizationBase, SLEEFPirates, UnPack, OffsetArrays, ArrayInterface
using VectorizationBase: NativeTypes, REGISTER_SIZE, MM, insertelement, extractelement
# using VectorizationBase: num_vector_load_expr,
#     mask, pick_vector_width_val, MM, vzero, stridedpointer_for_broadcast,
#     Zero, unwrap, maybestaticrange, REGISTER_COUNT,
#     AbstractColumnMajorStridedPointer, AbstractRowMajorStridedPointer, AbstractSparseStridedPointer, AbstractStaticStridedPointer,
#     PackedStridedPointer, SparseStridedPointer, RowMajorStridedPointer, StaticStridedPointer, StaticStridedStruct, offsetprecalc,
#     maybestaticfirst, maybestaticlast, noalias!, gesp, gepbyte, pointerforcomparison, NativeTypes,
#     reduced_add, reduced_prod, reduce_to_add, reduced_max, reduced_min, vsum, vprod, vmaximum, vminimum,
#     sizeequivalentfloat, sizeequivalentint, vfmadd231, vfmsub231, vfnmadd231, vfnmsub231, sizeequivalentfloat, sizeequivalentint, relu
    # vadd!, vsub!, vmul!, vfdiv!, vfmadd!, vfnmadd!, vfmsub!, vfnmsub!,
    # vmullog2, vmullog10, vdivlog2, vdivlog10, vmullog2add!, vmullog10add!, vdivlog2add!, vdivlog10add!, vfmaddaddone, vadd1
# using SLEEFPirates: pow # why?
using Base: OneTo, setindex
using Base.Broadcast: Broadcasted, DefaultArrayStyle
using LinearAlgebra: Adjoint, Transpose
using Base.Meta: isexpr
using DocStringExtensions
using ArrayInterface: StaticInt
import LinearAlgebra # for check_args

# using Base.FastMath: add_fast, sub_fast, mul_fast, div_fast

# export LowDimArray, stridedpointer,
#     @avx, @_avx, *ˡ, _avx_!,
#     vmap, vmap!, vmapt, vmapt!, vmapnt, vmapnt!, vmapntt, vmapntt!,
#     vfilter, vfilter!, vmapreduce, vreduce

const VECTORWIDTHSYMBOL, ELTYPESYMBOL = Symbol("##Wvecwidth##"), Symbol("##Tloopeltype##")

include("utilities/datastructures.jl")
include("loopset/arrayreferences.jl")
include("loopset/dependencies.jl")
include("loopset/loops.jl")
include("loopset/polyhedra.jl")
include("loopset/operations.jl")
include("loopset/loopset.jl")
include("optimizing/costs.jl")
include("optimizing/determinestrategy.jl")

# include("getconstindexes.jl")
# include("vectorizationbase_extensions.jl")
# include("predicates.jl")
include("map.jl")
include("filter.jl")
# include("costs.jl")
# include("operations.jl")
# include("graphs.jl")
# include("operation_evaluation_order.jl")
# include("memory_ops_common.jl")
# include("add_loads.jl")
# include("add_stores.jl")
# include("add_compute.jl")
# include("add_constants.jl")
# include("add_ifelse.jl")
# include("determinestrategy.jl")
# include("loopstartstopmanager.jl")
# include("lower_compute.jl")
# include("lower_constant.jl")
# include("lower_memory_common.jl")
# include("lower_load.jl")
# include("lower_store.jl")
# include("lowering.jl")
# include("split_loops.jl")
# include("condense_loopset.jl")
# include("reconstruct_loopset.jl")
# include("constructors.jl")
# include("user_api_conveniences.jl")
# include("mapreduce.jl")
# include("broadcast.jl")

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

# include("precompile.jl")
# _precompile_()

end # module
