using Test
using LoopVectorization
using LinearAlgebra

# const START_TIME = time()
# exceeds_time_limit() = (time() - START_TIME) > 35 * 60

function clenshaw(x, coeff)
    len_c = length(coeff)
    tmp = zero(x)
    ret = zero(x)
    for i in len_c:-1:2
        ret     = muladd(x,2tmp,coeff[i]-ret)
        ret,tmp = tmp,ret
    end
    ret = muladd(x,tmp,coeff[1]-ret)
    return ret
end

"""
Causes `check_args` to fail.
"""
struct FallbackArrayWrapper{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
end
Base.size(A::FallbackArrayWrapper) = size(A.data)
Base.@propagate_inbounds Base.getindex(A::FallbackArrayWrapper, i::Vararg{Int, N}) where {N} = getindex(A.data, i...)
Base.@propagate_inbounds Base.setindex!(A::FallbackArrayWrapper, v, i::Vararg{Int, N}) where {N} = setindex!(A.data, v, i...)
Base.IndexStyle(::Type{<:FallbackArrayWrapper}) = IndexLinear()

@show LoopVectorization.REGISTER_COUNT

@time @testset "LoopVectorization.jl" begin
    
    @test isempty(detect_unbound_args(LoopVectorization))

    @time include("polyhedra.jl")

    # @time include("printmethods.jl")

    # @time include("fallback.jl")

    # @time include("utils.jl")

    # @time include("arraywrappers.jl")

    # @time include("check_empty.jl")

    # if isnothing(get(ENV, "TRAVIS_BRANCH", nothing)) || LoopVectorization.REGISTER_COUNT ≠ 32 || VERSION ≥ v"1.4"
    #     @time include("offsetarrays.jl")
    # end

    # @time include("tensors.jl")

    # @time include("map.jl")

    # @time include("filter.jl")
    
    # @time include("mapreduce.jl")

    # @time include("ifelsemasks.jl")

    # @time include("dot.jl")

    # @time include("special.jl")

    # @time include("gemv.jl")

    # @time include("miscellaneous.jl")

    # @time include("copy.jl")

    # @time include("broadcast.jl")

    # # I test  locally on master; times out on Travis.
    # if isnothing(get(ENV, "TRAVIS_BRANCH", nothing)) || LoopVectorization.REGISTER_COUNT ≠ 32 || VERSION ≥ v"1.4"
    #     @time include("gemm.jl")
    # end
end
