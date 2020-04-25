using Test
using LoopVectorization
using LinearAlgebra

function clenshaw(x,coeff)
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

@show LoopVectorization.VectorizationBase.REGISTER_COUNT

@time @testset "LoopVectorization.jl" begin

    @time include("printmethods.jl")

    @time include("utils.jl")

    @time include("offsetarrays.jl")

    @time include("tensors.jl")

    @time include("map.jl")

    @time include("filter.jl")

    @time include("ifelsemasks.jl")

    @time include("dot.jl")

    @time include("special.jl")

    @time include("gemv.jl")

    @time include("miscellaneous.jl")

    @time include("copy.jl")

    @time include("broadcast.jl")

    @time include("gemm.jl")

end
