module LoopVectorization

# if (!isnothing(get(ENV, "TRAVIS_BRANCH", nothing)) || !isnothing(get(ENV, "APPVEYOR", nothing))) && isdefined(Base, :Experimental) && isdefined(Base.Experimental, Symbol("@optlevel"))
    # @eval Base.Experimental.@optlevel 1
# end

using VectorizationBase, SLEEFPirates, UnPack, OffsetArrays
using VectorizationBase: REGISTER_SIZE, REGISTER_COUNT, data,
    mask, pick_vector_width_val, MM,
    maybestaticlength, maybestaticsize, staticm1, staticp1, staticmul, vzero,
    Zero, maybestaticrange, offsetprecalc, lazymul,
    maybestaticfirst, maybestaticlast, scalar_less, gep, gesp, pointerforcomparison, NativeTypes,
    vfmadd, vfmsub, vfnmadd, vfnmsub, vfmadd231, vfmsub231, vfnmadd231, vfnmsub231, vadd, vsub, vmul,
    relu, stridedpointer, StridedPointer, StridedBitPointer, AbstractStridedPointer,
    reduced_add, reduced_prod, reduce_to_add, reduce_to_prod, reduced_max, reduced_min, reduce_to_max, reduce_to_min,
    vsum, vprod, vmaximum, vminimum, vstorent!

using IfElse: ifelse

# missing: stridedpointer_for_broadcast, noalias!, gepbyte, 
# using SIMDPirates: VECTOR_SYMBOLS, evadd, evsub, evmul, evfdiv, vrange, 
#     reduced_add, reduced_prod, reduce_to_add, reduced_max, reduced_min, vsum, vprod, vmaximum, vminimum,
#     sizeequivalentfloat, sizeequivalentint, vadd!, vsub!, vmul!, vfdiv!, vfmadd!, vfnmadd!, vfmsub!, vfnmsub!,
#     vfmadd231, vfmsub231, vfnmadd231, vfnmsub231, sizeequivalentfloat, sizeequivalentint, #prefetch,
#     vmullog2, vmullog10, vdivlog2, vdivlog10, vmullog2add!, vmullog10add!, vdivlog2add!, vdivlog10add!, vfmaddaddone, vadd1, relu
using SLEEFPirates: pow
using Base.Broadcast: Broadcasted, DefaultArrayStyle
using LinearAlgebra: Adjoint, Transpose
using Base.Meta: isexpr
using DocStringExtensions
import LinearAlgebra # for check_args

using Base.FastMath: add_fast, sub_fast, mul_fast, div_fast

using ArrayInterface
using ArrayInterface: OptionallyStaticUnitRange, Zero, One
const Static = ArrayInterface.StaticInt



export LowDimArray, stridedpointer,
    @avx, @_avx, *ˡ, _avx_!,
    vmap, vmap!, vmapt, vmapt!, vmapnt, vmapnt!, vmapntt, vmapntt!,
    vfilter, vfilter!, vmapreduce, vreduce

@inline unwrap(::Val{N}) where {N} = N
@inline unwrap(::Static{N}) where {N} = N
@inline unwrap(x) = x

const VECTORWIDTHSYMBOL, ELTYPESYMBOL = Symbol("##Wvecwidth##"), Symbol("##Tloopeltype##")


include("vectorizationbase_compat/contract_pass.jl")
include("vectorizationbase_compat/subsetview.jl")
include("getconstindexes.jl")
# include("vectorizationbase_extensions.jl")
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
include("zygoterules.jl")

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
