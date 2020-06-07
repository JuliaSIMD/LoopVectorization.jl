using LoopVectorization
const LOOPVECBENCHDIR = joinpath(pkgdir(LoopVectorization), "benchmark")
include(joinpath(LOOPVECBENCHDIR, "benchmarkflops.jl"))

