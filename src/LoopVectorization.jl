module LoopVectorization

using VectorizationBase, SIMDPirates, SLEEFPirates, MacroTools, Parameters
using VectorizationBase: REGISTER_SIZE, REGISTER_COUNT, extract_data, num_vector_load_expr,
    mask, masktable, pick_vector_width_val, valmul, valrem, valmuladd, valadd, valsub, _MM,
    maybestaticlength, maybestaticsize, staticm1, subsetview,
    Static, StaticUnitRange, StaticLowerUnitRange, StaticUpperUnitRange,
    PackedStridedPointer, SparseStridedPointer, RowMajorStridedPointer, StaticStridedPointer, StaticStridedStruct
using SIMDPirates: VECTOR_SYMBOLS, evadd, evmul, vrange, reduced_add, reduced_prod, reduce_to_add, reduce_to_prod
using Base.Broadcast: Broadcasted, DefaultArrayStyle
using LinearAlgebra: Adjoint, Transpose
using MacroTools: prewalk, postwalk


export LowDimArray, stridedpointer, vectorizable,
    @avx, *ˡ, ∗,
    vmap, vmap!


include("map.jl")
include("costs.jl")
include("operations.jl")
include("graphs.jl")
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

export @_avx, _avx, @_avx_, avx_!

# include("precompile.jl")
# _precompile_()

end # module
