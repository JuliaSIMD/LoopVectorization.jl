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

@time @testset "LoopVectorization.jl" begin

    @time include("printmethods.jl")
    
    @time include("ifelsemasks.jl")

    @time include("dot.jl")

    @time include("special.jl")

    @time include("gemv.jl")

    @time include("miscellaneous.jl")

    @time include("copy.jl")

    @time include("broadcast.jl")

    @time include("map.jl")

    if Base.libllvm_version > v"7"
        @time include("filter.jl")
    end

    @time include("gemm.jl")

end
