pkgdir(pkg::String) = abspath(joinpath(dirname(Base.find_package(pkg)), ".."))
const LOOPVECBENCHDIR = joinpath(pkgdir("LoopVectorization"), "benchmarks")
include(joinpath(LOOPVECBENCHDIR, "looptests.jl"))
include(joinpath(LOOPVECBENCHDIR, "loadsharedlibs.jl"))

using PrettyTables, BenchmarkTools
struct SizedResults{V <: AbstractVector} <: AbstractMatrix{String}
    results::Matrix{Float64}
    sizes::V
end
function Base.size(sr::SizedResults)
    M, N = size(sr.results)
    N, M+1
end
struct BenchmarkResult{V}
    tests::Vector{String}
    sizedresults::SizedResults{V}
end
function BenchmarkResult(tests, sizes)
    ntests = length(tests); nsizes = length(sizes)
    BenchmarkResult(
        append!(["Size"], tests),
        SizedResults(Matrix{Float64}(undef, ntests, nsizes), sizes)
    )
end
function Base.getindex(br::SizedResults, row, col)
    col == 1 ? string(br.sizes[row]) : string(br.results[col - 1, row])
end
Base.setindex!(br::BenchmarkResult, v, i...) = br.sizedresults.results[i...] = v
function Base.show(io::IO, br::BenchmarkResult)
    pretty_table(io, br.sizedresults, br.tests)
end
function alloc_matrices(s::NTuple{3,Int})
    M, K, N = s
    C = Matrix{Float64}(undef, M, N)
    A = rand(M, K)
    B = rand(K, N)
    C, A, B
end
alloc_matrices(s::Int) = alloc_matrices((s,s,s))
gflop(s::Int) = s^3 * 1e-9
gflop(s::NTuple{3,Int}) = prod(s) * 1e-9
function benchmark_gemm(sizes)
    tests = [BLAS.vendor() === :mkl ? "IntelMKL" : "OpenBLAS", "Julia", "Clang-Polly", "GFort-loops", "GFort-intrinsic", "LoopVectorization"]
    br = BenchmarkResult(tests, sizes)
    for (i,s) ∈ enumerate(sizes)
        C, A, B = alloc_matrices(s)
        n_gflop = gflop(s)
        br[1,i] = n_gflop / @belapsed mul!($C, $A, $B)
        Cblas = copy(C)
        br[2,i] = n_gflop / @belapsed jgemm_nkm!($C, $A, $B)
        @assert C ≈ Cblas "Julia gemm wrong?"
        br[3,i] = n_gflop / @belapsed cgemm_nkm!($C, $A, $B)
        @assert C ≈ Cblas "Polly gemm wrong?"
        br[4,i] = n_gflop / @belapsed fgemm_nkm!($C, $A, $B)
        @assert C ≈ Cblas "Fort gemm wrong?"
        br[5,i] = n_gflop / @belapsed fgemm_builtin!($C, $A, $B)
        @assert C ≈ Cblas "Fort intrinsic gemm wrong?"
        br[6,i] = n_gflop / @belapsed gemmavx!($C, $A, $B)
        @assert C ≈ Cblas "LoopVec gemm wrong?"
    end
    br
end

using VegaLite, IndexedTables
function plot(br::BenchmarkResult)
    res = vec(br.sizedresults.results)
    brsizes = br.sizedresults.sizes
    sizes = Vector{eltype(brsizes)}(undef, length(res))
    ntests = length(br.tests) - 1
    for i ∈ 0:length(brsizes)-1
        si = brsizes[i+1]
        for j ∈ 1:ntests
            sizes[j + i*ntests] = si
        end
    end
    tests = vcat((@view(br.tests[2:end]) for _ ∈ eachindex(brsizes))...)
    t = table((GFLOPS = res, Size = sizes, Method = tests))
    t |> @vlplot(
        :line,
        x = :Size,
        y = :GFLOPS,
        color = :Method
    )
end


