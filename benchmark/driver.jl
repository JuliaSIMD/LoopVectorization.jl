# pkgdir(pkg::String) = abspath(joinpath(dirname(Base.find_package(pkg)), ".."))
# const LOOPVECBENCHDIR = joinpath(pkgdir("LoopVectorization"), "benchmarks")
# includet(joinpath(LOOPVECBENCHDIR, "driver.jl"))

pkgdir(pkg::String) = abspath(joinpath(dirname(Base.find_package(pkg)), ".."))
const LOOPVECBENCHDIR = joinpath(pkgdir("LoopVectorization"), "benchmark")
include(joinpath(LOOPVECBENCHDIR, "benchmarkflops.jl"))
include(joinpath(LOOPVECBENCHDIR, "plotbenchmarks.jl"))


using Distributed

addprocs(13);

@everywhere begin
    pkgdir(pkg::String) = abspath(joinpath(dirname(Base.find_package(pkg)), ".."))
    const LOOPVECBENCHDIR = joinpath(pkgdir("LoopVectorization"), "benchmark")
    include(joinpath(LOOPVECBENCHDIR, "benchmarkflops.jl"))
    BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1
end

gemm_future = @spawnat 2 benchmark_gemm(2:256);
AtmulB_future = @spawnat 3 benchmark_AtmulB(2:256);
dot_future = @spawnat 4 benchmark_dot(2:256);
selfdot_future = @spawnat 5 benchmark_selfdot(2:256);
gemv_future = @spawnat 6 benchmark_gemv(2:256);
dot3_future = @spawnat 7 benchmark_dot3(2:256);
sse_future = @spawnat 8 benchmark_sse(2:256);
exp_future = @spawnat 9 benchmark_exp(2:256);
aplusBc_future = @spawnat 10 benchmark_aplusBc(2:256);
AplusAt_future = @spawnat 11 benchmark_AplusAt(2:256);
randomaccess_future = @spawnat 12 benchmark_random_access(2:256);
AmulBt_future = @spawnat 13 benchmark_AmulBt(2:256);
Atmulvb_future = @spawnat 14 benchmark_Atmulvb(2:256);

dot_bench = fetch(dot_future)
selfdot_bench = fetch(selfdot_future)
AplusAt_bench = fetch(AplusAt_future)
gemv_bench = fetch(gemv_future)
randomaccess_bench = fetch(randomaccess_future)
dot3_bench = fetch(dot3_future)
sse_bench = fetch(sse_future)
exp_bench = fetch(exp_future)
aplusBc_bench = fetch(aplusBc_future)
gemm_bench = fetch(gemm_future)
AtmulB_bench = fetch(AtmulB_future)
AmulBt_bench = fetch(AmulBt_future)
Atmulvb_bench = fetch(Atmulvb_future)

v = 2
const PICTURES = "/home/chriselrod/Pictures"
save(joinpath(PICTURES, "bench_gemm_v$v.png"), plot(gemm_bench));
save(joinpath(PICTURES, "bench_AtmulB_v$v.png"), plot(AtmulB_bench));
save(joinpath(PICTURES, "bench_dot_v$v.png"), plot(dot_bench));
save(joinpath(PICTURES, "bench_selfdot_v$v.png"), plot(selfdot_bench));
save(joinpath(PICTURES, "bench_gemv_v$v.png"), plot(gemv_bench));
save(joinpath(PICTURES, "bench_dot3_v$v.png"), plot(dot3_bench));
save(joinpath(PICTURES, "bench_sse_v$v.png"), plot(sse_bench));
save(joinpath(PICTURES, "bench_exp_v$v.png"), plot(exp_bench));
save(joinpath(PICTURES, "bench_aplusBc_v$v.png"), plot(aplusBc_bench));
save(joinpath(PICTURES, "bench_AplusAt_v$v.png"), plot(AplusAt_bench));
save(joinpath(PICTURES, "bench_random_access_v$v.png"), plot(randomaccess_bench));
save(joinpath(PICTURES, "bench_AmulBt_v$b.png"), plot(AmulBt_bench));
save(joinpath(PICTURES, "bench_Atmulvb_v$v.png"), plot(Atmulvb_bench));

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



