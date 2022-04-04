using LoopVectorization, ProgressMeter
const LOOPVECBENCHDIR = joinpath(pkgdir(LoopVectorization), "benchmark")
include(joinpath(LOOPVECBENCHDIR, "benchmarkflops.jl"))
