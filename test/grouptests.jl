const START_TIME = time()

@show LoopVectorization.register_count()

@show RUN_SLOW_TESTS

@time @testset "LoopVectorization.jl" begin

  @time if LOOPVECTORIZATION_TEST == "all" || LOOPVECTORIZATION_TEST == "part1"
    @time include("broadcast.jl")
    @time include("parsing_inputs.jl")
  end

  @time if LOOPVECTORIZATION_TEST == "all" || LOOPVECTORIZATION_TEST == "part2"
    if VERSION <= v"1.8" || isempty(VERSION.prerelease)
      using Aqua
      @time Aqua.test_all(LoopVectorization, ambiguities = false)
    end
    @test isempty(detect_unbound_args(LoopVectorization))

    @time include("printmethods.jl")

    @time include("can_avx.jl")

    @time include("fallback.jl")

    @time include("utils.jl")

    @time include("arraywrappers.jl")

    @time include("check_empty.jl")

    @time include("loopinductvars.jl")

    @time include("shuffleloadstores.jl")

    if VERSION < v"1.7-DEV"
      @time include("zygote.jl")
    else
      println("Skipping Zygote tests.")
    end

    @time include("map.jl")

    @time include("filter.jl")

    @time include("multiassignments.jl")

    @time include("reduction_untangling.jl")

    @time include("manyloopreductions.jl")

    @time include("simplemisc.jl")

    @time include("convolutions.jl")

    @time include("ifelsemasks.jl")

    @time include("gemv.jl")

    @time include("dot.jl")

    @time include("special.jl")

    @time include("mapreduce.jl")

    @time include("index_processing.jl")
  end

  @time if LOOPVECTORIZATION_TEST == "all" || LOOPVECTORIZATION_TEST == "part3"
    @time include("rejectunroll.jl")

    @time include("miscellaneous.jl")

    @time include("copy.jl")

    @time include("tensors.jl")

    @time include("staticsize.jl")
  end

  @time if LOOPVECTORIZATION_TEST == "all" || LOOPVECTORIZATION_TEST == "part4"
    @time include("threading.jl")

    @time include("tullio.jl")

    @time include("iteration_bound_tests.jl")

    @time include("outer_reductions.jl")

    @time include("upperboundedintegers.jl")

    if VERSION ≥ v"1.6"
      @time include("quantum.jl")
    end

    @time include("offsetarrays.jl")
  end

  @time if LOOPVECTORIZATION_TEST == "all" || LOOPVECTORIZATION_TEST == "part5"
    @time include("steprange.jl")

    @time include("gemm.jl")

    @time include("inner_reductions.jl")
  end

  @time if LOOPVECTORIZATION_TEST == "all" || LOOPVECTORIZATION_TEST == "part6"
    cproj = Base.active_project()
    precompiledir = joinpath(@__DIR__, "precompile")
    Pkg.activate(joinpath(precompiledir, "LVUser"))
    @time include(joinpath(precompiledir, "precompile.jl"))
    Pkg.activate(cproj)
  end

end

const ELAPSED_MINUTES = (time() - START_TIME) / 60
# @test ELAPSED_MINUTES < 180
@test ELAPSED_MINUTES < 300
