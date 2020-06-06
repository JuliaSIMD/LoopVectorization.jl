# pkgdir(pkg::String) = abspath(joinpath(dirname(Base.find_package(pkg)), ".."))
# const LOOPVECBENCHDIR = joinpath(pkgdir("LoopVectorization"), "benchmarks")
# includet(joinpath(LOOPVECBENCHDIR, "driver.jl"))

using Distributed, LoopVectorization, JLD2

const LOOPVECBENCHDIR = joinpath(pkgdir(LoopVectorization), "benchmark")
include(joinpath(LOOPVECBENCHDIR, "benchmarkflops.jl"))
include(joinpath(LOOPVECBENCHDIR, "plotbenchmarks.jl"))


nprocs_to_add() = (Sys.CPU_THREADS >> 1) - 1
start_worker(wid) = remotecall(include, wid, joinpath(LOOPVECBENCHDIR, "setup_worker.jl"))
function start_workers()
    addprocs(nprocs_to_add())
    foreach(wait, map(start_worker, workers()))
end
stop_workers() = rmprocs(workers())
addprocs(); nworkers()

pmap_startstop(f, s) = (start_workers(); r = pmap(f, s); stop_workers(); r)

blastests() = [
    "LoopVectorization",
    "Julia", "Clang",
    "GFortran", "icc", "ifort",
    "g++ & Eigen-3", "clang++ & Eigen-3",
    "GFortran-builtin", "ifort-builtin",
    "OpenBLAS", "MKL"
]    
function benchmark_AmulB(sizes)
    br = BenchmarkResult(blastests(), sizes)
    sm = br.sizedresults.results
    pmap_startstop(is -> A_mul_B_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end
function benchmark_AmulBt(sizes)
    br = BenchmarkResult(blastests(), sizes)
    sm = br.sizedresults.results
    pmap_startstop(is -> A_mul_Bt_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end
function benchmark_AtmulB(sizes)
    br = BenchmarkResult(blastests(), sizes)
    sm = br.sizedresults.results
    pmap_startstop(is -> At_mul_B_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end
function benchmark_AtmulBt(sizes)
    br = BenchmarkResult(blastests(), sizes)
    sm = br.sizedresults.results
    pmap_startstop(is -> At_mul_Bt_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end
function benchmark_dot(sizes)
    tests = ["LoopVectorization", "Julia", "Clang", "GFortran", "icc", "ifort", "g++ & Eigen-3", "clang++ & Eigen-3"]#, "OpenBLAS"]
    br = BenchmarkResult(tests, sizes)
    sm = br.sizedresults.results
    pmap_startstop(is -> dot_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end
function benchmark_selfdot(sizes)
    tests = ["LoopVectorization", "Julia", "Clang", "GFortran", "icc", "ifort", "g++ & Eigen-3", "clang++ & Eigen-3"]#, "OpenBLAS"]
    br = BenchmarkResult(tests, sizes)
    sm = br.sizedresults.results
    pmap_startstop(is -> selfdot_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end
function benchmark_Amulvb(sizes)
    br = BenchmarkResult(blastests(), sizes)
    sm = br.sizedresults.results
    pmap_startstop(is -> A_mul_vb_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end
function benchmark_Atmulvb(sizes)
    br = BenchmarkResult(blastests(), sizes)
    sm = br.sizedresults.results
    pmap_startstop(is -> At_mul_vb_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end
function benchmark_dot3(sizes)
    tests = ["LoopVectorization", "Julia", "Clang", "GFortran", "icc", "ifort", "g++ & Eigen-3", "clang++ & Eigen-3", "LinearAlgebra" ]
    br = BenchmarkResult(tests, sizes)
    sm = br.sizedresults.results
    pmap_startstop(is -> dot3_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end
function benchmark_sse(sizes)
    tests = ["LoopVectorization", "Julia", "Clang", "GFortran", "icc", "ifort", "g++ & Eigen-3", "clang++ & Eigen-3", "MKL"]
    br = BenchmarkResult(tests, sizes)
    sm = br.sizedresults.results
    pmap_startstop(is -> sse_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end
function benchmark_exp(sizes)
    tests = ["LoopVectorization", "Julia", "Clang", "GFortran", "icc", "ifort"]
    br = BenchmarkResult(tests, sizes)
    sm = br.sizedresults.results
    pmap_startstop(is -> exp_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end
function benchmark_aplusBc(sizes)
    tests = ["LoopVectorization", "Julia", "Clang", "GFortran", "icc", "ifort", "g++ & Eigen-3", "clang++ & Eigen-3"]
    br = BenchmarkResult(tests, sizes)
    sm = br.sizedresults.results
    pmap_startstop(is -> aplusBc_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end
function benchmark_AplusAt(sizes)
    tests = ["LoopVectorization", "Julia", "Clang", "GFortran", "icc", "ifort", "g++ & Eigen-3", "clang++ & Eigen-3", "GFortran-builtin", "ifort-builtin"]
    br = BenchmarkResult(tests, sizes)
    sm = br.sizedresults.results
    pmap_startstop(is -> AplusAt_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end
function benchmark_random_access(sizes)
    tests = ["LoopVectorization", "Julia", "Clang", "GFortran", "icc", "ifort"]
    br = BenchmarkResult(tests, sizes)
    sm = br.sizedresults.results
    pmap_startstop(is -> randomaccess_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end
function benchmark_logdettriangle(sizes)
    # tests = ["LoopVectorization", "Julia", "Clang", "GFortran", "icc", "ifort", "g++ & Eigen-3", "clang++ & Eigen-3", "LinearAlgebra"]
    tests = ["LoopVectorization", "Julia", "Clang", "GFortran", "icc", "ifort", "LinearAlgebra"]
    br = BenchmarkResult(tests, sizes)
    sm = br.sizedresults.results
    pmap_startstop(is -> logdettriangle_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end
function benchmark_filter2d(sizes, K)
    tests = ["LoopVectorization", "Julia", "Clang", "GFortran", "icc", "ifort"]
    br = BenchmarkResult(tests, sizes)
    sm = br.sizedresults.results
    pmap_startstop(is -> filter2d_bench_run!(sm, is[2], is[1], K), enumerate(sizes))
    br
end
function benchmark_filter2ddynamic(sizes)
    K = OffsetArray(rand(Float64, 3, 3), -1:1, -1:1)
    benchmark_filter2d(sizes, K)
end
function benchmark_filter2d3x3(sizes)
    K = SizedOffsetMatrix{Float64,-1,1,-1,1}(rand(3,3))
    benchmark_filter2d(sizes, K)
end
function benchmark_filter2dunrolled(sizes)
    tests = ["LoopVectorization", "Julia", "Clang", "GFortran", "icc", "ifort"]
    br = BenchmarkResult(tests, sizes)
    sm = br.sizedresults.results
    K = SizedOffsetMatrix{Float64,-1,1,-1,1}(rand(3,3))
    pmap_startstop(is -> filter2dunrolled_bench_run!(sm, is[2], is[1], K), enumerate(sizes))
    br
end



# sizes = 23:23
sizes = 256:-1:2
longsizes = 1024:-1:2

logdettriangle_bench = benchmark_logdettriangle(sizes); println("logdet(LowerTriangular(A)) benchmark results:"); println(logdettriangle_bench)
dot3_bench = benchmark_dot3(sizes); println("x' * A * y benchmark results:"); println(dot3_bench)

AmulB_bench = benchmark_AmulB(sizes); println("A * B benchmark results:"); println(AmulB_bench)
AmulBt_bench = benchmark_AmulBt(sizes); println("A * B' benchmark results:"); println(AmulBt_bench)
AtmulBt_bench = benchmark_AtmulBt(sizes); println("A' * B' benchmark results:"); println(AtmulBt_bench)
AtmulB_bench = benchmark_AtmulB(sizes); println("A' * B benchmark results:"); println(AtmulB_bench)

Amulvb_bench = benchmark_Amulvb(sizes); println("A * b benchmark results:"); println(Amulvb_bench)
Atmulvb_bench = benchmark_Atmulvb(sizes); println("A' * b benchmark results:"); println(Atmulvb_bench)

dot_bench = benchmark_dot(longsizes); println("a' * b benchmark results:"); println(dot_bench)
selfdot_bench = benchmark_selfdot(longsizes); println("a' * a benchmark results:"); println(selfdot_bench)
sse_bench = benchmark_sse(sizes); println("Benchmark resutls of summing squared error:"); println(sse_bench)
aplusBc_bench = benchmark_aplusBc(sizes); println("Benchmark results of a .+ B .* c':"); println(aplusBc_bench)
AplusAt_bench = benchmark_AplusAt(sizes); println("Benchmark results of A * A':"); println(AplusAt_bench)

filter2d_dynamic_bench = benchmark_filter2ddynamic(sizes); println("Benchmark results for dynamically sized 3x3 convolution:"); println(filter2d_dynamic_bench)
filter2d_3x3_bench = benchmark_filter2d3x3(sizes); println("Benchmark results for statically sized 3x3 convolution:"); println(filter2d_3x3_bench)
filter2d_unrolled_bench = benchmark_filter2dunrolled(sizes); println("Benchmark results for unrolled 3x3 convolution:"); println(filter2d_unrolled_bench)

vexp_bench = benchmark_exp(sizes); println("Benchmark results of exponentiating a vector:"); println(vexp_bench)
randomaccess_bench = benchmark_random_access(sizes); println("Benchmark results from using a vector of indices:"); println(randomaccess_bench)

const v = 1
using Cairo, Fontconfig
const PICTURES = joinpath(pkgdir(LoopVectorization), "docs", "src", "assets")
saveplot(f, br) = draw(PNG(joinpath(PICTURES, f * "$v.png"), 12inch, 8inch), plot(br))

saveplot("bench_dot3_v", dot3_bench);
saveplot("bench_dot_v", dot_bench);
saveplot("bench_selfdot_v", selfdot_bench);
saveplot("bench_sse_v", sse_bench);
saveplot("bench_aplusBc_v", aplusBc_bench);
saveplot("bench_AplusAt_v", AplusAt_bench);
saveplot("bench_AmulB_v", AmulB_bench);
saveplot("bench_AmulBt_v", AmulBt_bench);
saveplot("bench_AtmulB_v", AtmulB_bench);
saveplot("bench_AtmulBt_v", AtmulBt_bench);
saveplot("bench_Amulvb_v", Amulvb_bench);
saveplot("bench_Atmulvb_v", Atmulvb_bench);

saveplot("bench_logdettriangle_v", logdettriangle_bench);
saveplot("bench_filter2d_dynamic_v", filter2d_dynamic_bench);
saveplot("bench_filter2d_3x3_v", filter2d_3x3_bench);
saveplot("bench_filter2d_unrolled_v", filter2d_unrolled_bench);
saveplot("bench_exp_v", vexp_bench);
saveplot("bench_random_access_v", randomaccess_bench);

# @load "benchmarkresults.jld2" logdettriangle_bench filter2d_dynamic_bench filter2d_3x3_bench filter2d_unrolled_bench vexp_bench randomaccess_bench

@save "benchmarkresults.jld2" logdettriangle_bench filter2d_dynamic_bench filter2d_3x3_bench filter2d_unrolled_bench dot_bench selfdot_bench dot3_bench sse_bench aplusBc_bench AplusAt_bench vexp_bench randomaccess_bench AmulB_bench AmulBt_bench AtmulB_bench AtmulBt_bench Amulvb_bench Atmulvb_bench

