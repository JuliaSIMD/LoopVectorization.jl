# pkgdir(pkg::String) = abspath(joinpath(dirname(Base.find_package(pkg)), ".."))
# const LOOPVECBENCHDIR = joinpath(pkgdir("LoopVectorization"), "benchmarks")
# includet(joinpath(LOOPVECBENCHDIR, "driver.jl"))

using Distributed, LoopVectorization

const LOOPVECBENCHDIR = joinpath(pkgdir(LoopVectorization), "benchmark")
include(joinpath(LOOPVECBENCHDIR, "benchmarkflops.jl"))
include(joinpath(LOOPVECBENCHDIR, "plotbenchmarks.jl"))


addprocs((Sys.CPU_THREADS >> 1)-1); nworkers()

@everywhere begin
    using LoopVectorization
    const LOOPVECBENCHDIR = joinpath(pkgdir(LoopVectorization), "benchmark")
    include(joinpath(LOOPVECBENCHDIR, "benchmarkflops.jl"))
    # BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1
end


# sizes = 23:23
sizes = 256:-1:2
selfdot_bench = benchmark_selfdot(sizes)

AmulB_bench = benchmark_AmulB(sizes)
AmulBt_bench = benchmark_AmulBt(sizes)
AtmulB_bench = benchmark_AtmulB(sizes)
AtmulBt_bench = benchmark_AtmulBt(sizes)
Amulvb_bench = benchmark_Amulvb(sizes)
Atmulvb_bench = benchmark_Atmulvb(sizes)

filter2d_dynamic_bench = benchmark_filter2ddynamic(sizes)#512:-1:2)
filter2d_3x3_bench = benchmark_filter2d3x3(sizes)#512:-1:2)
filter2d_unrolled_bench = benchmark_filter2dunrolled(sizes)#512:-1:2)

dot3_bench = benchmark_dot3(sizes)
dot_bench = benchmark_dot(sizes)
sse_bench = benchmark_sse(sizes)
aplusBc_bench = benchmark_aplusBc(sizes)
AplusAt_bench = benchmark_AplusAt(sizes)
vexp_bench = benchmark_exp(sizes)
randomaccess_bench = benchmark_random_access(sizes)
logdettriangle_bench = benchmark_logdettriangle(sizes)

v = 1
using Cairo, Fontconfig
const PICTURES = joinpath(pkgdir(LoopVectorization), "docs", "src", "assets")
function saveplot(f, br)
    draw(PNG(joinpath(PICTURES, f * "$v.png"), 12inch, 8inch), plot(br))
end

saveplot("bench_filter2d_dynamic_v", filter2d_dynamic_bench);
saveplot("bench_filter2d_3x3_v", filter2d_3x3_bench);
saveplot("bench_filter2d_unrolled_v", filter2d_unrolled_bench);
saveplot("bench_dot_v", dot_bench);
saveplot("bench_selfdot_v", selfdot_bench);
saveplot("bench_dot3_v", dot3_bench);
saveplot("bench_sse_v", sse_bench);
saveplot("bench_aplusBc_v", aplusBc_bench);
saveplot("bench_AplusAt_v", AplusAt_bench);
saveplot("bench_exp_v", vexp_bench);
saveplot("bench_random_access_v", randomaccess_bench);
saveplot("bench_logdettriangle_v", logdettriangle_bench);
saveplot("bench_AmulB_v", AmulB_bench);
saveplot("bench_AmulBt_v", AmulBt_bench);
saveplot("bench_AtmulB_v", AtmulB_bench);
saveplot("bench_AtmulBt_v", AtmulBt_bench);
saveplot("bench_Amulvb_v", Amulvb_bench);
saveplot("bench_Atmulvb_v", Atmulvb_bench);




# plot(gemm_bench)
# plot(AtmulB_bench)
# plot(dot_bench)
# plot(selfdot_bench)
# plot(gemv_bench)
# plot(dot3_bench)
# plot(sse_bench)
# plot(vexp_bench)
# plot(aplusBc_bench)
# plot(AplusAt_bench)



