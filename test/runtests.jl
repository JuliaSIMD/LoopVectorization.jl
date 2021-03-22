include("testsetup.jl")

import InteractiveUtils

InteractiveUtils.versioninfo(stdout; verbose = true)

const START_TIME = time()

@show LoopVectorization.register_count()

@show RUN_SLOW_TESTS

@time @testset "LoopVectorization.jl" begin

    # @time Aqua.test_all(LoopVectorization)
    # @test isempty(detect_unbound_args(LoopVectorization))

    @time include("printmethods.jl")

    @time include("can_avx.jl")

    @time include("fallback.jl")

    @time include("utils.jl")

    @time include("arraywrappers.jl")

    @time include("check_empty.jl")

    @time include("loopinductvars.jl")

    @time include("shuffleloadstores.jl")
    
    @time include("zygote.jl")

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

    @time include("threading.jl")

    @time include("tullio.jl")
end

const ELAPSED_MINUTES = (time() - START_TIME)/60
# @test ELAPSED_MINUTES < 180
@test ELAPSED_MINUTES < 240

