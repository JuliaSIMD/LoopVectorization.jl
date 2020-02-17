# pkgdir(pkg::String) = abspath(joinpath(dirname(Base.find_package(pkg)), ".."))
# const LOOPVECBENCHDIR = joinpath(pkgdir("LoopVectorization"), "benchmarks")
# includet(joinpath(LOOPVECBENCHDIR, "driver.jl"))

pkgdir(pkg::String) = abspath(joinpath(dirname(Base.find_package(pkg)), ".."))
const LOOPVECBENCHDIR = joinpath(pkgdir("LoopVectorization"), "benchmark")
include(joinpath(LOOPVECBENCHDIR, "benchmarkflops.jl"))
include(joinpath(LOOPVECBENCHDIR, "plotbenchmarks.jl"))


using Distributed

addprocs((Sys.CPU_THREADS >> 1)-1); nprocs()

@everywhere begin
    pkgdir(pkg::String) = abspath(joinpath(dirname(Base.find_package(pkg)), ".."))
    const LOOPVECBENCHDIR = joinpath(pkgdir("LoopVectorization"), "benchmark")
    include(joinpath(LOOPVECBENCHDIR, "benchmarkflops.jl"))
    # BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1
end

AmulB_bench = benchmark_AmulB(2:256)
AmulBt_bench = benchmark_AmulBt(2:256)
AtmulB_bench = benchmark_AtmulB(2:256)
AtmulBt_bench = benchmark_AtmulBt(2:256)
dot_bench = benchmark_dot(2:256)
selfdot_bench = benchmark_selfdot(2:256)
Amulvb_bench = benchmark_Amulvb(2:256)
Atmulvb_bench = benchmark_Atmulvb(2:256)
dot3_bench = benchmark_dot3(2:256)
sse_bench = benchmark_sse(2:256)
exp_bench = benchmark_exp(2:256)
aplusBc_bench = benchmark_aplusBc(2:256)
AplusAt_bench = benchmark_AplusAt(2:256)
randomaccess_bench = benchmark_random_access(2:256)
logdettriangle_bench = benchmark_logdettriangle(2:256)

v = 1
filetype = "svg"
const PICTURES = joinpath(pkgdir("LoopVectorization"), "docs", "src", "assets")
save(joinpath(PICTURES, "bench_exp_v$v.$filetype"), plot(exp_bench));
save(joinpath(PICTURES, "bench_logdettriangle_v$v.$filetype"), plot(logdettriangle_bench));
save(joinpath(PICTURES, "bench_AmulB_v$v.$filetype"), plot(AmulB_bench));
save(joinpath(PICTURES, "bench_AmulBt_v$v.$filetype"), plot(AmulBt_bench));
save(joinpath(PICTURES, "bench_AtmulB_v$v.$filetype"), plot(AtmulB_bench));
save(joinpath(PICTURES, "bench_AtmulBt_v$v.$filetype"), plot(AtmulBt_bench));
save(joinpath(PICTURES, "bench_dot_v$v.$filetype"), plot(dot_bench));
save(joinpath(PICTURES, "bench_selfdot_v$v.$filetype"), plot(selfdot_bench));
save(joinpath(PICTURES, "bench_dot3_v$v.$filetype"), plot(dot3_bench));
save(joinpath(PICTURES, "bench_sse_v$v.$filetype"), plot(sse_bench));
save(joinpath(PICTURES, "bench_aplusBc_v$v.$filetype"), plot(aplusBc_bench));
save(joinpath(PICTURES, "bench_AplusAt_v$v.$filetype"), plot(AplusAt_bench));
save(joinpath(PICTURES, "bench_random_access_v$v.$filetype"), plot(randomaccess_bench));
save(joinpath(PICTURES, "bench_Amulvb_v$v.$filetype"), plot(Amulvb_bench));
save(joinpath(PICTURES, "bench_Atmulvb_v$v.$filetype"), plot(Atmulvb_bench));




plot(gemm_bench)
plot(AtmulB_bench)
plot(dot_bench)
plot(selfdot_bench)
plot(gemv_bench)
plot(dot3_bench)
plot(sse_bench)
plot(exp_bench)
plot(aplusBc_bench)
plot(AplusAt_bench)



