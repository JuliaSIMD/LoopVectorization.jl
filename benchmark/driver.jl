# pkgdir(pkg::String) = abspath(joinpath(dirname(Base.find_package(pkg)), ".."))
# const LOOPVECBENCHDIR = joinpath(pkgdir("LoopVectorization"), "benchmarks")
# includet(joinpath(LOOPVECBENCHDIR, "driver.jl"))

using Distributed

pkgdir(pkg::String) = abspath(joinpath(dirname(Base.find_package(pkg)), ".."))
const LOOPVECBENCHDIR = joinpath(pkgdir("LoopVectorization"), "benchmark")
include(joinpath(LOOPVECBENCHDIR, "benchmarkflops.jl"))
include(joinpath(LOOPVECBENCHDIR, "plotbenchmarks.jl"))


addprocs((Sys.CPU_THREADS >> 1)-1); nprocs()

@everywhere begin
    pkgdir(pkg::String) = abspath(joinpath(dirname(Base.find_package(pkg)), ".."))
    const LOOPVECBENCHDIR = joinpath(pkgdir("LoopVectorization"), "benchmark")
    include(joinpath(LOOPVECBENCHDIR, "benchmarkflops.jl"))
    # BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1
end


# sizes = 23:23
sizes = 256:-1:2

filter2d_dynamic_bench = benchmark_filter2ddynamic(sizes)#512:-1:2)
filter2d_3x3_bench = benchmark_filter2d3x3(sizes)#512:-1:2)
filter2d_unrolled_bench = benchmark_filter2dunrolled(sizes)#512:-1:2)

AmulB_bench = benchmark_AmulB(sizes)
AmulBt_bench = benchmark_AmulBt(sizes)
AtmulB_bench = benchmark_AtmulB(sizes)
AtmulBt_bench = benchmark_AtmulBt(sizes)
dot_bench = benchmark_dot(sizes)
selfdot_bench = benchmark_selfdot(sizes)
Amulvb_bench = benchmark_Amulvb(sizes)
Atmulvb_bench = benchmark_Atmulvb(sizes)
dot3_bench = benchmark_dot3(sizes)
sse_bench = benchmark_sse(sizes)
aplusBc_bench = benchmark_aplusBc(sizes)
AplusAt_bench = benchmark_AplusAt(sizes)
exp_bench = benchmark_exp(sizes)
randomaccess_bench = benchmark_random_access(sizes)
logdettriangle_bench = benchmark_logdettriangle(sizes)

v = 1
filetype = "svg"
const PICTURES = joinpath(pkgdir("LoopVectorization"), "docs", "src", "assets")
save(joinpath(PICTURES, "bench_filter2d_dynamic_v$v.$filetype"), plot(filter2d_dynamic_bench));
save(joinpath(PICTURES, "bench_filter2d_3x3_v$v.$filetype"), plot(filter2d_3x3_bench));
save(joinpath(PICTURES, "bench_filter2d_unrolled_v$v.$filetype"), plot(filter2d_unrolled_bench));
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
save(joinpath(PICTURES, "bench_Amulvb_v$v.$filetype"), plot(Amulvb_bench));
save(joinpath(PICTURES, "bench_Atmulvb_v$v.$filetype"), plot(Atmulvb_bench));
save(joinpath(PICTURES, "bench_exp_v$v.$filetype"), plot(exp_bench));
save(joinpath(PICTURES, "bench_random_access_v$v.$filetype"), plot(randomaccess_bench));
save(joinpath(PICTURES, "bench_logdettriangle_v$v.$filetype"), plot(logdettriangle_bench));




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



