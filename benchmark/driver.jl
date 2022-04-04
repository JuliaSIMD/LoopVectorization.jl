# pkgdir(pkg::String) = abspath(joinpath(dirname(Base.find_package(pkg)), ".."))
# const LOOPVECBENCHDIR = joinpath(pkgdir("LoopVectorization"), "benchmarks")
# includet(joinpath(LOOPVECBENCHDIR, "driver.jl"))

using Distributed, LoopVectorization, JLD2, ProgressMeter
const LOOPVECBENCHDIR = joinpath(pkgdir(LoopVectorization), "benchmark")
include(joinpath(LOOPVECBENCHDIR, "benchmarkflops.jl"))
include(joinpath(LOOPVECBENCHDIR, "plotbenchmarks.jl"))


nprocs_to_add() = ((Sys.CPU_THREADS)::Int >> 1)
# nprocs_to_add() = ((Sys.CPU_THREADS)::Int >> 1) - 1
start_worker(wid) = remotecall(include, wid, joinpath(LOOPVECBENCHDIR, "setup_worker.jl"))
function start_workers(nprocs = nprocs_to_add())
  addprocs(nprocs, exeflags = "--project=$(Base.active_project())")
  foreach(wait, map(start_worker, workers()))
end
stop_workers() = rmprocs(workers())


function blastests()
  tests = ["LoopVectorization", "Julia", "Clang", "GFortran"]
  INTEL_BENCH && push!(tests, "icc", "ifort")
  push!(tests, "g++ & Eigen-3", "clang++ & Eigen-3", "GFortran-builtin", "OpenBLAS")
  INTEL_BENCH && push!(tests, "ifort-builtin")
  MKL_BENCH && push!(tests, "MKL")
  tests
end
function benchmark_AmulB(sizes)
  tests = blastests()
  start_workers(nprocs_to_add() >> 1)
  sm = SharedMatrix(Matrix{Float64}(undef, length(tests), length(sizes)))
  @showprogress pmap(is -> A_mul_B_bench!(sm, is[2], is[1]), enumerate(sizes))
  br = BenchmarkResult(Matrix(sm), tests, sizes)
  stop_workers()
  br
end
function benchmark_AmulBt(sizes)
  tests = blastests()
  start_workers(nprocs_to_add() >> 1)
  sm = SharedMatrix(Matrix{Float64}(undef, length(tests), length(sizes)))
  @showprogress pmap(is -> A_mul_Bt_bench!(sm, is[2], is[1]), enumerate(sizes))
  br = BenchmarkResult(Matrix(sm), tests, sizes)
  stop_workers()
  br
end
function benchmark_AtmulB(sizes)
  tests = blastests()
  start_workers(nprocs_to_add() >> 1)
  sm = SharedMatrix(Matrix{Float64}(undef, length(tests), length(sizes)))
  @showprogress pmap(is -> At_mul_B_bench!(sm, is[2], is[1]), enumerate(sizes))
  br = BenchmarkResult(Matrix(sm), tests, sizes)
  stop_workers()
  br
end
function benchmark_AtmulBt(sizes)
  tests = blastests()
  start_workers(nprocs_to_add() >> 1)
  sm = SharedMatrix(Matrix{Float64}(undef, length(tests), length(sizes)))
  @showprogress pmap(is -> At_mul_Bt_bench!(sm, is[2], is[1]), enumerate(sizes))
  br = BenchmarkResult(Matrix(sm), tests, sizes)
  stop_workers()
  br
end
function dot_tests()
  tests = ["LoopVectorization", "Julia", "Clang", "GFortran"]
  INTEL_BENCH && push!(tests, "icc", "ifort")
  push!(tests, "g++ & Eigen-3", "clang++ & Eigen-3")
  tests
end
function benchmark_dot(sizes)
  tests = dot_tests()
  start_workers()
  sm = SharedMatrix(Matrix{Float64}(undef, length(tests), length(sizes)))
  @showprogress pmap(is -> dot_bench!(sm, is[2], is[1]), enumerate(sizes))
  br = BenchmarkResult(Matrix(sm), tests, sizes)
  stop_workers()
  br
end
function benchmark_selfdot(sizes)
  tests = dot_tests()
  start_workers()
  sm = SharedMatrix(Matrix{Float64}(undef, length(tests), length(sizes)))
  @showprogress pmap(is -> selfdot_bench!(sm, is[2], is[1]), enumerate(sizes))
  br = BenchmarkResult(Matrix(sm), tests, sizes)
  stop_workers()
  br
end
function benchmark_Amulvb(sizes)
  tests = blastests()
  start_workers()
  sm = SharedMatrix(Matrix{Float64}(undef, length(tests), length(sizes)))
  @showprogress pmap(is -> A_mul_vb_bench!(sm, is[2], is[1]), enumerate(sizes))
  br = BenchmarkResult(Matrix(sm), tests, sizes)
  stop_workers()
  br
end
function benchmark_Atmulvb(sizes)
  tests = blastests()
  start_workers()
  sm = SharedMatrix(Matrix{Float64}(undef, length(tests), length(sizes)))
  @showprogress pmap(is -> At_mul_vb_bench!(sm, is[2], is[1]), enumerate(sizes))
  br = BenchmarkResult(Matrix(sm), tests, sizes)
  stop_workers()
  br
end
function benchmark_dot3(sizes)
  tests = ["LoopVectorization", "Julia", "Clang", "GFortran"]
  INTEL_BENCH && push!(tests, "icc", "ifort")
  push!(tests, "g++ & Eigen-3", "clang++ & Eigen-3", "LinearAlgebra")
  start_workers()
  sm = SharedMatrix(Matrix{Float64}(undef, length(tests), length(sizes)))
  @showprogress pmap(is -> dot3_bench!(sm, is[2], is[1]), enumerate(sizes))
  br = BenchmarkResult(Matrix(sm), tests, sizes)
  stop_workers()
  br
end
function benchmark_sse(sizes)
  tests = ["LoopVectorization", "Julia", "Clang", "GFortran"]
  INTEL_BENCH && push!(tests, "icc", "ifort")
  push!(tests, "g++ & Eigen-3", "clang++ & Eigen-3")
  MKL_BENCH && push!(tests, "MKL")

  start_workers()
  sm = SharedMatrix(Matrix{Float64}(undef, length(tests), length(sizes)))
  @showprogress pmap(is -> sse_bench!(sm, is[2], is[1]), enumerate(sizes))
  br = BenchmarkResult(Matrix(sm), tests, sizes)
  stop_workers()
  br
end
function benchmark_exp(sizes)
  tests = ["LoopVectorization", "Julia", "Clang", "GFortran"]
  INTEL_BENCH && push!(tests, "icc", "ifort")
  start_workers()
  sm = SharedMatrix(Matrix{Float64}(undef, length(tests), length(sizes)))
  @showprogress pmap(is -> exp_bench!(sm, is[2], is[1]), enumerate(sizes))
  br = BenchmarkResult(Matrix(sm), tests, sizes)
  stop_workers()
  br
end
function benchmark_aplusBc(sizes)
  tests = ["LoopVectorization", "Julia", "Clang", "GFortran"]
  INTEL_BENCH && push!(tests, "icc", "ifort")
  push!(tests, "g++ & Eigen-3", "clang++ & Eigen-3")
  start_workers()
  sm = SharedMatrix(Matrix{Float64}(undef, length(tests), length(sizes)))
  @showprogress pmap(is -> aplusBc_bench!(sm, is[2], is[1]), enumerate(sizes))
  br = BenchmarkResult(Matrix(sm), tests, sizes)
  stop_workers()
  br
end
function benchmark_AplusAt(sizes)
  tests = ["LoopVectorization", "Julia", "Clang", "GFortran"]
  INTEL_BENCH && push!(tests, "icc", "ifort")
  push!(tests, "g++ & Eigen-3", "clang++ & Eigen-3", "GFortran-builtin")
  INTEL_BENCH && push!(tests, "ifort-builtin")
  start_workers()
  sm = SharedMatrix(Matrix{Float64}(undef, length(tests), length(sizes)))
  @showprogress pmap(is -> AplusAt_bench!(sm, is[2], is[1]), enumerate(sizes))
  br = BenchmarkResult(Matrix(sm), tests, sizes)
  stop_workers()
  br
end
function benchmark_random_access(sizes)
  tests = ["LoopVectorization", "Julia", "Clang", "GFortran"]
  INTEL_BENCH && push!(tests, "icc", "ifort")
  start_workers()
  sm = SharedMatrix(Matrix{Float64}(undef, length(tests), length(sizes)))
  @showprogress pmap(is -> randomaccess_bench!(sm, is[2], is[1]), enumerate(sizes))
  br = BenchmarkResult(Matrix(sm), tests, sizes)
  stop_workers()
  br
end
function benchmark_logdettriangle(sizes)
  # tests = ["LoopVectorization", "Julia", "Clang", "GFortran", "icc", "ifort", "g++ & Eigen-3", "clang++ & Eigen-3", "LinearAlgebra"]
  tests = ["LoopVectorization", "Julia", "Clang", "GFortran"]
  INTEL_BENCH && push!(tests, "icc", "ifort")
  push!(tests, "LinearAlgebra")
  start_workers()
  sm = SharedMatrix(Matrix{Float64}(undef, length(tests), length(sizes)))
  @showprogress pmap(is -> logdettriangle_bench!(sm, is[2], is[1]), enumerate(sizes))
  br = BenchmarkResult(Matrix(sm), tests, sizes)
  stop_workers()
  br
end
function benchmark_filter2d(sizes, K)
  tests = ["LoopVectorization", "Julia", "Clang", "GFortran"]
  INTEL_BENCH && push!(tests, "icc", "ifort")
  start_workers()
  sm = SharedMatrix(Matrix{Float64}(undef, length(tests), length(sizes)))
  @showprogress pmap(is -> filter2d_bench_run!(sm, is[2], is[1], K), enumerate(sizes))
  br = BenchmarkResult(Matrix(sm), tests, sizes)
  stop_workers()
  br
end
function benchmark_filter2ddynamic(sizes)
  K = OffsetArray(rand(Float64, 3, 3), -1:1, -1:1)
  benchmark_filter2d(sizes, K)
end
function benchmark_filter2d3x3(sizes)
  K = SizedOffsetMatrix{Float64,-1,1,-1,1}(rand(3, 3))
  benchmark_filter2d(sizes, K)
end
function benchmark_filter2dunrolled(sizes)
  tests = ["LoopVectorization", "Julia", "Clang", "GFortran"]
  INTEL_BENCH && push!(tests, "icc", "ifort")
  start_workers()
  sm = SharedMatrix(Matrix{Float64}(undef, length(tests), length(sizes)))
  K = SizedOffsetMatrix{Float64,-1,1,-1,1}(rand(3, 3))
  @showprogress pmap(
    is -> filter2dunrolled_bench_run!(sm, is[2], is[1], K),
    enumerate(sizes),
  )
  br = BenchmarkResult(Matrix(sm), tests, sizes)
  stop_workers()
  br
end



# sizes = 23:23
sizes = 256:-1:2
longsizes = 1024:-1:2

println("logdet(LowerTriangular(A)) benchmark results:");
logdettriangle_bench = benchmark_logdettriangle(sizes);
println(logdettriangle_bench);
println("x' * A * y benchmark results:");
dot3_bench = benchmark_dot3(sizes);
println(dot3_bench);

println("A * B benchmark results:");
AmulB_bench = benchmark_AmulB(sizes);
println(AmulB_bench);
println("A * B' benchmark results:");
AmulBt_bench = benchmark_AmulBt(sizes);
println(AmulBt_bench);
println("A' * B' benchmark results:");
AtmulBt_bench = benchmark_AtmulBt(sizes);
println(AtmulBt_bench);
println("A' * B benchmark results:");
AtmulB_bench = benchmark_AtmulB(sizes);
println(AtmulB_bench);

println("A * b benchmark results:");
Amulvb_bench = benchmark_Amulvb(sizes);
println(Amulvb_bench);
println("A' * b benchmark results:");
Atmulvb_bench = benchmark_Atmulvb(sizes);
println(Atmulvb_bench);

println("a' * b benchmark results:");
dot_bench = benchmark_dot(longsizes);
println(dot_bench);
println("a' * a benchmark results:");
selfdot_bench = benchmark_selfdot(longsizes);
println(selfdot_bench);

println("Benchmark results of a .+ B .* c':");
aplusBc_bench = benchmark_aplusBc(sizes);
println(aplusBc_bench);
println("Benchmark results of A .+ A':");
AplusAt_bench = benchmark_AplusAt(sizes);
println(AplusAt_bench);

println("Benchmark results for dynamically sized 3x3 convolution:");
filter2d_dynamic_bench = benchmark_filter2ddynamic(sizes);
println(filter2d_dynamic_bench);
println("Benchmark results for statically sized 3x3 convolution:");
filter2d_3x3_bench = benchmark_filter2d3x3(sizes);
println(filter2d_3x3_bench);
println("Benchmark results for unrolled 3x3 convolution:");
filter2d_unrolled_bench = benchmark_filter2dunrolled(sizes);
println(filter2d_unrolled_bench);

println("Benchmark resutls of summing squared error:");
sse_bench = benchmark_sse(sizes);
println(sse_bench);
println("Benchmark results of exponentiating a vector:");
vexp_bench = benchmark_exp(sizes);
println(vexp_bench);
println("Benchmark results from using a vector of indices:");
randomaccess_bench = benchmark_random_access(sizes);
println(randomaccess_bench);

const v = 2
# using Cairo, Fontconfig
const PICTURES = joinpath(pkgdir(LoopVectorization), "docs", "src", "assets")
# saveplot(f, br) = draw(PNG(joinpath(PICTURES, f * "$v.png"), 12inch, 8inch), plot(br))
saveplot(f, br) = draw(SVG(joinpath(PICTURES, f * "$v.svg"), 12inch, 8inch), plot(br))

# If only rerunning a few, remove them from load.
# @load "benchmarkresults.jld2" logdettriangle_bench filter2d_dynamic_bench filter2d_3x3_bench filter2d_unrolled_bench dot_bench selfdot_bench dot3_bench sse_bench aplusBc_bench AplusAt_bench vexp_bench randomaccess_bench AmulB_bench AmulBt_bench AtmulB_bench AtmulBt_bench Amulvb_bench Atmulvb_bench

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

@save "benchmarkresults.jld2" logdettriangle_bench filter2d_dynamic_bench filter2d_3x3_bench filter2d_unrolled_bench dot_bench selfdot_bench dot3_bench sse_bench aplusBc_bench AplusAt_bench vexp_bench randomaccess_bench AmulB_bench AmulBt_bench AtmulB_bench AtmulBt_bench Amulvb_bench Atmulvb_bench
