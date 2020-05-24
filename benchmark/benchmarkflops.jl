include(joinpath(LOOPVECBENCHDIR, "looptests.jl"))
include(joinpath(LOOPVECBENCHDIR, "loadsharedlibs.jl"))

using BenchmarkTools, SharedArrays
struct SizedResults{V <: AbstractVector} <: AbstractMatrix{String}
    results::SharedMatrix{Float64}
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
        SizedResults(SharedMatrix{Float64}(ntests, nsizes), sizes)
    )
end
function Base.getindex(br::SizedResults, row, col)
    col == 1 ? string(br.sizes[row]) : string(br.results[col - 1, row])
end
Base.setindex!(br::BenchmarkResult, v, i...) = br.sizedresults.results[i...] = v
function Base.vcat(br1::BenchmarkResult, br2::BenchmarkResult)
    BenchmarkResult(
        br1.tests,
        SizedResults(
            SharedMatrix(hcat(br1.sizedresults.results, br2.sizedresults.results)),
            vcat(br1.sizedresults.sizes, br2.sizedresults.sizes)
        )
    )
end

tothreetuple(i::Int) = (i,i,i)
tothreetuple(i::NTuple{3,Int}) = i
function matmul_bench!(br, C, A, B, i)
    M, N = size(C); K = size(B,1);
    n_gflop = M*K*N*2e-9
    Cblas = A * B
    br[1,i] = n_gflop / @belapsed gemmavx!($C, $A, $B)
    @assert C ≈ Cblas "LoopVec gemm wrong?"; fill!(C, NaN)
    br[2,i] = n_gflop / @belapsed jgemm!($C, $A, $B)
    @assert C ≈ Cblas "Julia gemm wrong?"; fill!(C, NaN)
    br[3,i] = n_gflop / @belapsed cgemm!($C, $A, $B)
    @assert C ≈ Cblas "Clang gemm wrong?"; fill!(C, NaN)
    br[4,i] = n_gflop / @belapsed fgemm!($C, $A, $B)
    @assert C ≈ Cblas "Fort gemm wrong?"; fill!(C, NaN)
    br[5,i] = n_gflop / @belapsed icgemm!($C, $A, $B)
    @assert C ≈ Cblas "icc gemm wrong?"; fill!(C, NaN)
    br[6,i] = n_gflop / @belapsed ifgemm!($C, $A, $B)
    @assert C ≈ Cblas "ifort gemm wrong?"; fill!(C, NaN)
    br[7,i] = n_gflop / @belapsed egemm!($C, $A, $B)
    @assert C ≈ Cblas "eigen gemm wrong?"; fill!(C, NaN)
    br[8,i] = n_gflop / @belapsed iegemm!($C, $A, $B)
    @assert C ≈ Cblas "i-eigen gemm wrong?"; fill!(C, NaN)
    br[9,i] = n_gflop / @belapsed fgemm_builtin!($C, $A, $B)
    @assert C ≈ Cblas "Fort builtin gemm wrong?"; fill!(C, NaN)
    br[10,i] = n_gflop / @belapsed ifgemm_builtin!($C, $A, $B)
    @assert C ≈ Cblas "ifort builtin gemm wrong?"; fill!(C, NaN)
    br[11,i] = n_gflop / @belapsed mul!($C, $A, $B);
    fill!(C, NaN)
    br[12,i] = n_gflop / @belapsed dgemmmkl!($C, $A, $B)
    @assert C ≈ Cblas "MKL JIT gemm wrong?"
    # br[12,i] = n_gflop / @belapsed gemmavx!($C, $A, $B)
end
function A_mul_B_bench!(br, s, i)
    M, K, N = tothreetuple(s)
    C = Matrix{Float64}(undef, M, N)
    A = rand(M, K)
    B = rand(K, N)
    matmul_bench!(br, C, A, B, i)
end
function A_mul_B_bench!(br, s, i)
    M, K, N = tothreetuple(s)
    C = Matrix{Float64}(undef, M, N)
    A = rand(M, K)
    B = rand(K, N)
    matmul_bench!(br, C, A, B, i)
end
function A_mul_Bt_bench!(br, s, i)
    M, K, N = tothreetuple(s)
    C = Matrix{Float64}(undef, M, N)
    A = rand(M, K)
    B = rand(N, K)'
    matmul_bench!(br, C, A, B, i)
end
function At_mul_B_bench!(br, s, i)
    M, K, N = tothreetuple(s)
    C = Matrix{Float64}(undef, M, N)
    A = rand(K, M)'
    B = rand(K, N)
    matmul_bench!(br, C, A, B, i)
end
function At_mul_Bt_bench!(br, s, i)
    M, K, N = tothreetuple(s)
    C = Matrix{Float64}(undef, M, N)
    A = rand(K, M)'
    B = rand(N, K)'
    matmul_bench!(br, C, A, B, i)
end

blastests() = [
    "LoopVectorization",
    "Julia", "Clang",
    "GFortran", "icc", "ifort",
    "g++ & Eigen-3", "clang++ & Eigen-3",
    "GFortran-builtin", "ifort-builtin",
    BLAS.vendor() === :mkl ? "IntelMKL" : "OpenBLAS", "MKL"
]    

function benchmark_AmulB(sizes)
    br = BenchmarkResult(blastests(), sizes)
    sm = br.sizedresults.results
    pmap(is -> A_mul_B_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end
function benchmark_AmulBt(sizes)
    br = BenchmarkResult(blastests(), sizes)
    sm = br.sizedresults.results
    pmap(is -> A_mul_Bt_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end
function benchmark_AtmulB(sizes)
    br = BenchmarkResult(blastests(), sizes)
    sm = br.sizedresults.results
    pmap(is -> At_mul_B_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end
function benchmark_AtmulBt(sizes)
    br = BenchmarkResult(blastests(), sizes)
    sm = br.sizedresults.results
    pmap(is -> At_mul_Bt_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end

function dot_bench!(br, s, i)
    a = rand(s); b = rand(s);
    dotblas = dot(a, b)
    n_gflop = s * 2e-9
    br[1,i] = n_gflop / @belapsed jdotavx($a, $b)
    @assert jdotavx(a,b) ≈ dotblas "LoopVec dot wrong?"
    br[2,i] = n_gflop / @belapsed jdot($a, $b)
    @assert jdot(a,b) ≈ dotblas "Julia dot wrong?"
    br[3,i] = n_gflop / @belapsed cdot($a, $b)
    @assert cdot(a,b) ≈ dotblas "Clang dot wrong?"
    br[4,i] = n_gflop / @belapsed fdot($a, $b)
    @assert fdot(a,b) ≈ dotblas "Fort dot wrong?"
    br[5,i] = n_gflop / @belapsed icdot($a, $b)
    @assert icdot(a,b) ≈ dotblas "icc dot wrong?"
    br[6,i] = n_gflop / @belapsed ifdot($a, $b)
    @assert ifdot(a,b) ≈ dotblas "ifort dot wrong?"
    br[7,i] = n_gflop / @belapsed edot($a, $b)
    @assert edot(a,b) ≈ dotblas "eigen dot wrong?"
    br[8,i] = n_gflop / @belapsed iedot($a, $b)
    @assert iedot(a,b) ≈ dotblas "i-eigen dot wrong?"
    br[9,i] = n_gflop / @belapsed dot($a, $b)
end
function benchmark_dot(sizes)
    tests = ["LoopVectorization", "Julia", "Clang", "GFortran", "icc", "ifort", "g++ & Eigen-3", "clang++ & Eigen-3", BLAS.vendor() === :mkl ? "IntelMKL" : "OpenBLAS"]
    br = BenchmarkResult(tests, sizes)
    sm = br.sizedresults.results
    pmap(is -> dot_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end
function selfdot_bench!(br, s, i)
    a = rand(s); b = rand(s);
    dotblas = dot(a, a)
    n_gflop = s * 2e-9
    br[1,i] = n_gflop / @belapsed jselfdotavx($a)
    @assert jselfdotavx(a) ≈ dotblas "LoopVec dot wrong?"
    br[2,i] = n_gflop / @belapsed jselfdot($a)
    @assert jselfdot(a) ≈ dotblas "Julia dot wrong?"
    br[3,i] = n_gflop / @belapsed cselfdot($a)
    @assert cselfdot(a) ≈ dotblas "Clang dot wrong?"
    br[4,i] = n_gflop / @belapsed fselfdot($a)
    @assert fselfdot(a) ≈ dotblas "Fort dot wrong?"
    br[5,i] = n_gflop / @belapsed icselfdot($a)
    @assert cselfdot(a) ≈ dotblas "icc dot wrong?"
    br[6,i] = n_gflop / @belapsed ifselfdot($a)
    @assert fselfdot(a) ≈ dotblas "ifort dot wrong?"
    br[7,i] = n_gflop / @belapsed eselfdot($a)
    @assert eselfdot(a) ≈ dotblas "eigen dot wrong?"
    br[8,i] = n_gflop / @belapsed ieselfdot($a)
    @assert ieselfdot(a) ≈ dotblas "i-eigen dot wrong?"
    br[9,i] = n_gflop / @belapsed dot($a, $a)
end
function benchmark_selfdot(sizes)
    tests = ["LoopVectorization", "Julia", "Clang", "GFortran", "icc", "ifort", "g++ & Eigen-3", "clang++ & Eigen-3", BLAS.vendor() === :mkl ? "IntelMKL" : "OpenBLAS"]
    br = BenchmarkResult(tests, sizes)
    sm = br.sizedresults.results
    pmap(is -> selfdot_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end

totwotuple(i::Int) = (i,i)
totwotuple(i::Tuple{Int,Int}) = i
function gemv_bench!(br, x, A, y, i)
    M, N = size(A)
    n_gflop = M*N * 2e-9
    xblas = A * y
    br[1,i] = n_gflop / @belapsed jgemvavx!($x, $A, $y)
    @assert x ≈ xblas "LoopVec wrong?"; fill!(x, NaN);
    br[2,i] = n_gflop / @belapsed jgemv!($x, $A, $y)
    @assert x ≈ xblas "Julia wrong?"; fill!(x, NaN);
    br[3,i] = n_gflop / @belapsed cgemv!($x, $A, $y)
    @assert x ≈ xblas "Clang wrong?"; fill!(x, NaN);
    br[4,i] = n_gflop / @belapsed fgemv!($x, $A, $y)
    @assert x ≈ xblas "Fort wrong?"; fill!(x, NaN);
    br[5,i] = n_gflop / @belapsed icgemv!($x, $A, $y)
    @assert x ≈ xblas "icc wrong?"; fill!(x, NaN);
    br[6,i] = n_gflop / @belapsed ifgemv!($x, $A, $y)
    @assert x ≈ xblas "ifort wrong?"; fill!(x, NaN);
    br[7,i] = n_gflop / @belapsed egemv!($x, $A, $y)
    @assert x ≈ xblas "eigen wrong?"; fill!(x, NaN);
    br[8,i] = n_gflop / @belapsed iegemv!($x, $A, $y)
    @assert x ≈ xblas "i-eigen wrong?"; fill!(x, NaN);
    br[9,i] = n_gflop / @belapsed fgemv_builtin!($x, $A, $y)
    @assert x ≈ xblas "Fort wrong?"; fill!(x, NaN);
    br[10,i] = n_gflop / @belapsed ifgemv_builtin!($x, $A, $y)
    @assert x ≈ xblas "ifort wrong?"; fill!(x, NaN);
    br[11,i] = n_gflop / @belapsed mul!($x, $A, $y)
    br[12,i] = n_gflop / @belapsed dgemvmkl!($x, $A, $y)
    @assert x ≈ xblas "gemvmkl wrong?"; fill!(x, NaN);
end
function A_mul_vb_bench!(br, s, i)
    M, N = totwotuple(s)
    x = Vector{Float64}(undef, M);
    A = rand(M, N);
    y = rand(N);
    gemv_bench!(br, x, A, y, i)
end
function At_mul_vb_bench!(br, s, i)
    M, N = totwotuple(s)
    x = Vector{Float64}(undef, M);
    A = rand(N, M)';
    y = rand(N);
    gemv_bench!(br, x, A, y, i)
end
function benchmark_Amulvb(sizes)
    br = BenchmarkResult(blastests(), sizes)
    sm = br.sizedresults.results
    pmap(is -> A_mul_vb_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end
function benchmark_Atmulvb(sizes)
    br = BenchmarkResult(blastests(), sizes)
    sm = br.sizedresults.results
    pmap(is -> At_mul_vb_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end

function dot3_bench!(br, s, i)
    M, N = totwotuple(s)
    x = rand(M); A = rand(M, N); y = rand(N);
    dotblas = dot(x, A, y)
    n_gflop = M*N * 3e-9
    br[1,i] = n_gflop / @belapsed jdot3avx($x, $A, $y)
    @assert jdot3avx(x, A, y) ≈ dotblas "LoopVec dot wrong?"
    br[2,i] = n_gflop / @belapsed jdot3($x, $A, $y)
    @assert jdot3(x, A, y) ≈ dotblas "Julia dot wrong?"
    br[3,i] = n_gflop / @belapsed cdot3($x, $A, $y)
    @assert cdot3(x, A, y) ≈ dotblas "Clang dot wrong?"
    br[4,i] = n_gflop / @belapsed fdot3($x, $A, $y)
    @assert fdot3(x, A, y) ≈ dotblas "Fort dot wrong?"
    br[5,i] = n_gflop / @belapsed icdot3($x, $A, $y)
    @assert icdot3(x, A, y) ≈ dotblas "icc dot wrong?"
    br[6,i] = n_gflop / @belapsed ifdot3($x, $A, $y)
    @assert ifdot3(x, A, y) ≈ dotblas "ifort dot wrong?"
    br[7,i] = n_gflop / @belapsed edot3($x, $A, $y)
    @assert edot3(x, A, y) ≈ dotblas "eigen dot wrong?"
    br[8,i] = n_gflop / @belapsed iedot3($x, $A, $y)
    @assert iedot3(x, A, y) ≈ dotblas "c-eigen dot wrong?"
    br[9,i] = n_gflop / @belapsed dot($x, $A, $y)
end
function benchmark_dot3(sizes)
    tests = ["LoopVectorization", "Julia", "Clang", "GFortran", "icc", "ifort", "g++ & Eigen-3", "clang++ & Eigen-3", "LinearAlgebra" ]
    br = BenchmarkResult(tests, sizes)
    sm = br.sizedresults.results
    pmap(is -> dot3_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end
function sse!(Xβ, y, X, β)
    mul!(copyto!(Xβ, y), X, β, 1.0, -1.0)
    dot(Xβ, Xβ)
end
sse_totwotuple(s::NTuple{2}) = s
sse_totwotuple(s::Integer) = ((3s) >> 1, s >> 1)

function sse_bench!(br, s, i)
    N, P = sse_totwotuple(s)
    y = rand(N); β = rand(P)
    X = randn(N, P)
    Xβ = similar(y)
    lpblas = sse!(Xβ, y, X, β)
    n_gflop = 2e-9*(P*N + 2N)
    br[1,i] = n_gflop / @belapsed jOLSlp_avx($y, $X, $β)
    @assert jOLSlp_avx(y, X, β) ≈ lpblas "LoopVec wrong?"
    br[2,i] = n_gflop / @belapsed jOLSlp($y, $X, $β)
    @assert jOLSlp(y, X, β) ≈ lpblas "Julia wrong?"
    br[3,i] = n_gflop / @belapsed cOLSlp($y, $X, $β)
    @assert cOLSlp(y, X, β) ≈ lpblas "Clang wrong?"
    br[4,i] = n_gflop / @belapsed fOLSlp($y, $X, $β)
    @assert fOLSlp(y, X, β) ≈ lpblas "Fort wrong?"
    br[5,i] = n_gflop / @belapsed icOLSlp($y, $X, $β)
    @assert icOLSlp(y, X, β) ≈ lpblas "icc wrong?"
    br[6,i] = n_gflop / @belapsed ifOLSlp($y, $X, $β)
    @assert ifOLSlp(y, X, β) ≈ lpblas "ifort wrong?"
    br[7,i] = n_gflop / @belapsed eOLSlp($y, $X, $β)
    @assert eOLSlp(y, X, β) ≈ lpblas "eigen wrong?"
    br[8,i] = n_gflop / @belapsed ieOLSlp($y, $X, $β)
    @assert ieOLSlp(y, X, β) ≈ lpblas "i-eigen wrong?"
    br[9,i] = n_gflop / @belapsed sse!($Xβ, $y, $X, $β)
end
function benchmark_sse(sizes)
    tests = ["LoopVectorization", "Julia", "Clang", "GFortran", "icc", "ifort", "g++ & Eigen-3", "clang++ & Eigen-3", BLAS.vendor() === :mkl ? "IntelMKL" : "OpenBLAS"]
    br = BenchmarkResult(tests, sizes)
    sm = br.sizedresults.results
    pmap(is -> sse_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end

function exp_bench!(br, s, i)
    a = rand(s); b = similar(a)
    n_gflop = 1e-9*s # not really gflops
    br[1,i] = n_gflop / @belapsed @avx @. $b = exp($a)
    baseb = copy(b)
    br[2,i] = n_gflop / @belapsed @. $b = exp($a)
    @assert b ≈ baseb "LoopVec wrong?"
    br[3,i] = n_gflop / @belapsed cvexp!($b, $a)
    @assert b ≈ baseb "Clang wrong?"
    br[4,i] = n_gflop / @belapsed fvexp!($b, $a)
    @assert b ≈ baseb "Fort wrong?"
    br[5,i] = n_gflop / @belapsed icvexp!($b, $a)
    @assert b ≈ baseb "icc wrong?"
    br[6,i] = n_gflop / @belapsed ifvexp!($b, $a)
    @assert b ≈ baseb "ifort wrong?"
end
function benchmark_exp(sizes)
    tests = ["LoopVectorization", "Julia", "Clang", "GFortran", "icc", "ifort"]
    br = BenchmarkResult(tests, sizes)
    sm = br.sizedresults.results
    pmap(is -> exp_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end

function aplusBc_bench!(br, s, i)
    M, N = totwotuple(s)
    a = rand(M); B = rand(M,N); c = rand(N);
    c′ = c'; D = similar(B)
    n_gflop = 2e-9 * M*N
    br[1,i] = n_gflop / @belapsed @avx @. $D = $a + $B * $c′
    Dcopy = copy(D); fill!(D, NaN);
    br[2,i] = n_gflop / @belapsed @. $D = $a + $B * $c′
    @assert D ≈ Dcopy "LoopVec wrong?"
    br[3,i] = n_gflop / @belapsed caplusBc!($D, $a, $B, $c)
    @assert D ≈ Dcopy "Clang wrong?"; fill!(D, NaN);
    br[4,i] = n_gflop / @belapsed faplusBc!($D, $a, $B, $c)
    @assert D ≈ Dcopy "Fort wrong?"; fill!(D, NaN);
    br[5,i] = n_gflop / @belapsed icaplusBc!($D, $a, $B, $c)
    @assert D ≈ Dcopy "icc wrong?"; fill!(D, NaN);
    br[6,i] = n_gflop / @belapsed ifaplusBc!($D, $a, $B, $c)
    @assert D ≈ Dcopy "ifort wrong?"; fill!(D, NaN);
    br[7,i] = n_gflop / @belapsed eaplusBc!($D, $a, $B, $c)
    @assert D ≈ Dcopy "eigen wrong?"; fill!(D, NaN);
    br[8,i] = n_gflop / @belapsed ieaplusBc!($D, $a, $B, $c)
    @assert D ≈ Dcopy "i-eigen wrong?"; fill!(D, NaN);
end
function benchmark_aplusBc(sizes)
    tests = ["LoopVectorization", "Julia", "Clang", "GFortran", "icc", "ifort", "g++ & Eigen-3", "clang++ & Eigen-3"]
    br = BenchmarkResult(tests, sizes)
    sm = br.sizedresults.results
    pmap(is -> aplusBc_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end

function AplusAt_bench!(br, s, i)
    A = rand(s,s); B = similar(A)
    n_gflop = 1e-9*s^2
    br[1,i] = n_gflop / @belapsed @avx @. $B = $A + $A'
    baseB = copy(B); fill!(B, NaN);
    br[2,i] = n_gflop / @belapsed @. $B = $A + $A'
    @assert B ≈ baseB "LoopVec wrong?"
    br[3,i] = n_gflop / @belapsed cAplusAt!($B, $A)
    @assert B ≈ baseB "Clang wrong?"; fill!(B, NaN);
    br[4,i] = n_gflop / @belapsed fAplusAt!($B, $A)
    @assert B ≈ baseB "Fort wrong?"; fill!(B, NaN);
    br[5,i] = n_gflop / @belapsed icAplusAt!($B, $A)
    @assert B ≈ baseB "icc wrong?"; fill!(B, NaN);
    br[6,i] = n_gflop / @belapsed ifAplusAt!($B, $A)
    @assert B ≈ baseB "ifort wrong?"; fill!(B, NaN);
    br[7,i] = n_gflop / @belapsed eAplusAt!($B, $A)
    @assert B ≈ baseB "eigen wrong?"; fill!(B, NaN);
    br[8,i] = n_gflop / @belapsed ieAplusAt!($B, $A)
    @assert B ≈ baseB "i-eigen wrong?"; fill!(B, NaN);
    br[9,i] = n_gflop / @belapsed fAplusAt_builtin!($B, $A)
    @assert B ≈ baseB "Fort-builtin wrong?"; fill!(B, NaN);
    br[10,i] = n_gflop / @belapsed ifAplusAt_builtin!($B, $A)
    @assert B ≈ baseB "ifort-builtin wrong?"; fill!(B, NaN);
end
function benchmark_AplusAt(sizes)
    tests = ["LoopVectorization", "Julia", "Clang", "GFortran", "icc", "ifort", "g++ & Eigen-3", "clang++ & Eigen-3", "GFortran-builtin", "ifort-builtin"]
    br = BenchmarkResult(tests, sizes)
    sm = br.sizedresults.results
    pmap(is -> AplusAt_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end

function randomaccess_bench!(br, s, i)
    A, C = totwotuple(s)
    P = rand(A, C) .+= 0.5;
    basis = rand(1:C, A, C);
    coefs = randn(C);
    n_gflop = 1e-9*(A*C + C)
    p = randomaccess(P, basis, coefs);
    br[1,i] = n_gflop / @belapsed randomaccessavx($P, $basis, $coefs)
    @assert p ≈ randomaccessavx(P, basis, coefs) "LoopVec wrong?"
    br[2,i] = n_gflop / @belapsed  randomaccess($P, $basis, $coefs)
    br[3,i] = n_gflop / @belapsed crandomaccess($P, $basis, $coefs)
    @assert p ≈ crandomaccess(P, basis, coefs) "Clang wrong?"
    br[4,i] = n_gflop / @belapsed frandomaccess($P, $basis, $coefs)
    @assert p ≈ frandomaccess(P, basis, coefs) "Fort wrong?"
    br[5,i] = n_gflop / @belapsed icrandomaccess($P, $basis, $coefs)
    @assert p ≈ icrandomaccess(P, basis, coefs) "icc wrong?"
    br[6,i] = n_gflop / @belapsed ifrandomaccess($P, $basis, $coefs)
    @assert p ≈ ifrandomaccess(P, basis, coefs) "ifort wrong?"
end
function benchmark_random_access(sizes)
    tests = ["LoopVectorization", "Julia", "Clang", "GFortran", "icc", "ifort"]
    br = BenchmarkResult(tests, sizes)
    sm = br.sizedresults.results
    pmap(is -> randomaccess_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end

function logdettriangle_bench!(br, s, i)
    S = randn(s, 2s)
    U = cholesky(Symmetric(S * S')).U
    n_gflop = 1e-9*s
    ld = logdet(U)
    br[1,i] = n_gflop / @belapsed jlogdettriangleavx($U)
    @assert ld ≈ jlogdettriangleavx(U) "LoopVec wrong?"
    br[2,i] = n_gflop / @belapsed jlogdettriangle($U)
    @assert ld ≈ jlogdettriangle(U) "Julia wrong?"
    br[3,i] = n_gflop / @belapsed clogdettriangle($U)
    @assert ld ≈ clogdettriangle(U) "Clang wrong?"
    br[4,i] = n_gflop / @belapsed flogdettriangle($U)
    @assert ld ≈ flogdettriangle(U) "Fort wrong?"
    br[5,i] = n_gflop / @belapsed iclogdettriangle($U)
    @assert ld ≈ iclogdettriangle(U) "icc wrong?"
    br[6,i] = n_gflop / @belapsed iflogdettriangle($U)
    @assert ld ≈ iflogdettriangle(U) "ifort wrong?"
    # br[7,i] = n_gflop / @belapsed elogdettriangle($U)
    # @assert ld ≈ elogdettriangle(U) "eigen wrong?"; fill!(B, NaN);
    # br[8,i] = n_gflop / @belapsed ielogdettriangle($U)
    # @assert ld ≈ ielogdettriangle(U) "i-eigen wrong?"; fill!(B, NaN);
    br[7,i] = n_gflop / @belapsed logdet($U)
end
function benchmark_logdettriangle(sizes)
    # tests = ["LoopVectorization", "Julia", "Clang", "GFortran", "icc", "ifort", "g++ & Eigen-3", "clang++ & Eigen-3", "LinearAlgebra"]
    tests = ["LoopVectorization", "Julia", "Clang", "GFortran", "icc", "ifort", "LinearAlgebra"]
    br = BenchmarkResult(tests, sizes)
    sm = br.sizedresults.results
    pmap(is -> logdettriangle_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end


function filter2d_bench_run!(br, s, i, K)
    A = rand(s + 2, s + 2)
    B = OffsetArray(similar(A, (s,s)), 1, 1)
    Mk, Nk = size(K)
    n_gflop = 1e-9 * (2Mk * Nk - 1) * s^2
    br[1,i] = n_gflop / @belapsed filter2davx!($B, $A, $K)
    Bcopy = copy(B); fill!(B, NaN);
    br[2,i] = n_gflop / @belapsed filter2d!($B, $A, $K)
    @assert B ≈ Bcopy "LoopVec wrong?"
    br[3,i] = n_gflop / @belapsed cfilter2d!($B, $A, $K)
    @assert B ≈ Bcopy "Clang wrong?"
    br[4,i] = n_gflop / @belapsed ffilter2d!($B, $A, $K)
    @assert B ≈ Bcopy "Fort wrong?"
    br[5,i] = n_gflop / @belapsed icfilter2d!($B, $A, $K)
    @assert B ≈ Bcopy "icc wrong?"
    br[6,i] = n_gflop / @belapsed iffilter2d!($B, $A, $K)
    @assert B ≈ Bcopy "ifort wrong?"
end
function benchmark_filter2d(sizes, K)
    tests = ["LoopVectorization", "Julia", "Clang", "GFortran", "icc", "ifort"]
    br = BenchmarkResult(tests, sizes)
    sm = br.sizedresults.results
    pmap(is -> filter2d_bench_run!(sm, is[2], is[1], K), enumerate(sizes))
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

function filter2dunrolled_bench_run!(br, s, i, K)
    A = rand(s + 2, s + 2)
    B = OffsetArray(similar(A, (s,s)), 1, 1)
    Mk, Nk = size(K)
    n_gflop = 1e-9 * (2Mk * Nk - 1) * s^2
    br[1,i] = n_gflop / @belapsed filter2dunrolledavx!($B, $A, $K)
    Bcopy = copy(B); fill!(B, NaN);
    br[2,i] = n_gflop / @belapsed filter2dunrolled!($B, $A, $K)
    @assert B ≈ Bcopy "LoopVec wrong?"
    br[3,i] = n_gflop / @belapsed cfilter2dunrolled!($B, $A, $K)
    @assert B ≈ Bcopy "Clang wrong?"
    br[4,i] = n_gflop / @belapsed ffilter2dunrolled!($B, $A, $K)
    @assert B ≈ Bcopy "Fort wrong?"
    br[5,i] = n_gflop / @belapsed icfilter2dunrolled!($B, $A, $K)
    @assert B ≈ Bcopy "icc wrong?"
    br[6,i] = n_gflop / @belapsed iffilter2dunrolled!($B, $A, $K)
    @assert B ≈ Bcopy "ifort wrong?"
end
function benchmark_filter2dunrolled(sizes)
    tests = ["LoopVectorization", "Julia", "Clang", "GFortran", "icc", "ifort"]
    br = BenchmarkResult(tests, sizes)
    sm = br.sizedresults.results
    K = SizedOffsetMatrix{Float64,-1,1,-1,1}(rand(3,3))
    pmap(is -> filter2dunrolled_bench_run!(sm, is[2], is[1], K), enumerate(sizes))
    br
end
