# pkgdir(pkg::String) = abspath(joinpath(dirname(Base.find_package(pkg)), ".."))
# const LOOPVECBENCHDIR = joinpath(pkgdir("LoopVectorization"), "benchmarks")
# includet(joinpath(LOOPVECBENCHDIR, "driver.jl"))

using Distributed

addprocs(9);

@everywhere begin
    pkgdir(pkg::String) = abspath(joinpath(dirname(Base.find_package(pkg)), ".."))
    const LOOPVECBENCHDIR = joinpath(pkgdir("LoopVectorization"), "benchmarks")
    include(joinpath(LOOPVECBENCHDIR, "benchmarkflops.jl"))
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

gemm_bench = fetch(gemm_future)
AtmulB_bench = fetch(AtmulB_future)
dot_bench = fetch(dot_future)
selfdot_bench = fetch(selfdot_future)
gemv_bench = fetch(gemv_future)
dot3_bench = fetch(dot3_future)
sse_bench = fetch(sse_future)
exp_bench = fetch(exp_future)
aplusBc_bench = fetch(aplusBc_future)


include(joinpath(LOOPVECBENCHDIR, "plotbenchmarks.jl"))

