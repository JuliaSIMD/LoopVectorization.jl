using Test
using LoopVectorization
using LinearAlgebra
import InteractiveUtils

InteractiveUtils.versioninfo(stdout; verbose = true)

const START_TIME = time()

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

const RUN_SLOW_TESTS = LoopVectorization.REGISTER_COUNT ≤ 16 || !parse(Bool, get(ENV, "GITHUB_ACTIONS", "false"))
@show RUN_SLOW_TESTS

@time @testset "LoopVectorization.jl" begin
    
    @test isempty(detect_unbound_args(LoopVectorization))

    @time include("printmethods.jl")

    @time include("can_avx.jl")

    @time include("fallback.jl")

    @time include("utils.jl")

    @time include("arraywrappers.jl")

    @time include("check_empty.jl")

    @time include("offsetarrays.jl")

    @time include("tensors.jl")

    @time include("map.jl")

    @time include("filter.jl")
    
    @time include("mapreduce.jl")

    @time include("ifelsemasks.jl")

    @time include("dot.jl")

    @time include("special.jl")

    @time include("gemv.jl")

    @time include("miscellaneous.jl")

    @time include("copy.jl")

    @time include("broadcast.jl")

    @time include("gemm.jl")
end

const ELAPSED_MINUTES = (time() - START_TIME)/60
@test ELAPSED_MINUTES < 120
