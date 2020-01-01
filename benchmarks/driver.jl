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

function alloc_matrices(s::NTuple{3,Int})
    M, K, N = s
    C = Matrix{Float64}(undef, M, N)
    A = rand(M, K)
    B = rand(K, N)
    C, A, B
end
alloc_matrices(s::Int) = alloc_matrices((s,s,s))
gflop(s::Int) = s^3 * 2e-9
gflop(s::NTuple{3,Int}) = prod(s) * 2e-9
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
function benchmark_dot(sizes)
    tests = [BLAS.vendor() === :mkl ? "IntelMKL" : "OpenBLAS", "Julia", "Clang-Polly", "GFort-loops", "LoopVectorization"]
    br = BenchmarkResult(tests, sizes)
    for (i,s) ∈ enumerate(sizes)
        a = rand(s); b = rand(s);
        n_gflop = s * 2e-9
        br[1,i] = n_gflop / @belapsed dot($a, $b)
        dotblas = dot(a, b)
        br[2,i] = n_gflop / @belapsed jdot($a, $b)
        @assert jdot(a,b) ≈ dotblas "Julia dot wrong?"
        br[3,i] = n_gflop / @belapsed cdot($a, $b)
        @assert cdot(a,b) ≈ dotblas "Polly dot wrong?"
        br[4,i] = n_gflop / @belapsed fdot($a, $b)
        @assert fdot(a,b) ≈ dotblas "Fort dot wrong?"
        br[5,i] = n_gflop / @belapsed jdotavx($a, $b)
        @assert jdotavx(a,b) ≈ dotblas "LoopVec dot wrong?"
    end
    br
end
function benchmark_selfdot(sizes)
    tests = [BLAS.vendor() === :mkl ? "IntelMKL" : "OpenBLAS", "Julia", "Clang-Polly", "GFort-loops", "LoopVectorization"]
    br = BenchmarkResult(tests, sizes)
    for (i,s) ∈ enumerate(sizes)
        a = rand(s);
        n_gflop = s * 2e-9
        br[1,i] = n_gflop / @belapsed dot($a, $a)
        dotblas = dot(a, a)
        br[2,i] = n_gflop / @belapsed jselfdot($a)
        @assert jselfdot(a) ≈ dotblas "Julia dot wrong?"
        br[3,i] = n_gflop / @belapsed cselfdot($a)
        @assert cselfdot(a) ≈ dotblas "Polly dot wrong?"
        br[4,i] = n_gflop / @belapsed fselfdot($a)
        @assert fselfdot(a) ≈ dotblas "Fort dot wrong?"
        br[5,i] = n_gflop / @belapsed jselfdotavx($a)
        @assert jselfdotavx(a) ≈ dotblas "LoopVec dot wrong?"
    end
    br
end
totwotuple(i::Int) = (i,i)
totwotuple(i::Tuple{Int,Int}) = i
function sse!(Xβ, y, X, β)
    mul!(copyto!(Xβ, y), X, β, 1.0, -1.0)
    dot(Xβ, Xβ)
end
function benchmark_sse(sizes)
    tests = [BLAS.vendor() === :mkl ? "IntelMKL" : "OpenBLAS", "Julia", "Clang-Polly", "GFort-loops", "GFort-intrinsic", "LoopVectorization"]
    br = BenchmarkResult(tests, sizes)
    for (i,s) ∈ enumerate(sizes)
        N, P = totwotuple(s)
        y = rand(N); β = rand(P)
        X = randn(N, P)
        Xβ = similar(y)
        n_gflop = 2e-9*(P*N + 2N)
        br[1,i] = n_gflop / @belapsed sse!($Xβ, $y, $X, $β)
        lpblas = sse!(Xβ, y, X, β)
        br[2,i] = n_gflop / @belapsed jOLSlp($y, $X, $β)
        @assert jOLSlp(y, X, β) ≈ lpblas "Julia wrong?"
        br[3,i] = n_gflop / @belapsed cOLSlp($y, $X, $β)
        @assert cOLSlp(y, X, β) ≈ lpblas "Polly wrong?"
        br[4,i] = n_gflop / @belapsed fOLSlp($y, $X, $β)
        @assert fOLSlp(y, X, β) ≈ lpblas "Fort wrong?"
        br[5,i] = n_gflop / @belapsed jOLSlp_avx($y, $X, $β)
        @assert jOLSlp_avx(y, X, β) ≈ lpblas "LoopVec wrong?"
    end
    br
end

function benchmark_exp(sizes)
    tests = ["Julia", "GFort-loops", "LoopVectorization"]
    br = BenchmarkResult(tests, sizes)
    for (i,s) ∈ enumerate(sizes)
        a = rand(s); b = similar(a)
        n_gflop = s # not really gflops
        br[1,i] = n_gflop / @belapsed @. $b = exp($a)
        baseb = copy(b)
        br[2,i] = n_gflop / @belapsed fvexp!($b, $a)
        @assert b ≈ baseb "Fort wrong?"
        br[3,i] = n_gflop / @belapsed @avx @. $b = exp($a)
        @assert b ≈ baseb "LoopVec wrong?"
    end
    br
end

function benchmark_aplusBc(sizes)
    tests = ["Julia", "Clang-Polly", "GFort-loops", "LoopVectorization"]
    br = BenchmarkResult(tests, sizes)
    for (i,s) ∈ enumerate(sizes)
        M, N = totwotuple(s)
        a = rand(M); B = rand(M,N); c = rand(N);
        c′ = c'; D = similar(B)
        n_gflop = 2e-9 * M*N
        br[1,i] = n_gflop / @belapsed @. $D = $a + $B * $c′
        Dcopy = copy(D)
        br[2,i] = n_gflop / @belapsed caplusBc!($D, $a, $B, $c)
        @assert D ≈ Dcopy "Polly wrong?"
        br[3,i] = n_gflop / @belapsed faplusBc!($D, $a, $B, $c)
        @assert D ≈ Dcopy "Fort wrong?"
        br[4,i] = n_gflop / @belapsed @avx @. $D = $a + $B * $c′
        @assert D ≈ Dcopy "LoopVec wrong?"
    end
    br
end



