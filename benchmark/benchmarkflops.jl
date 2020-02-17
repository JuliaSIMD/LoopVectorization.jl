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


tothreetuple(i::Int) = (i,i,i)
tothreetuple(i::NTuple{3,Int}) = i
function matmul_bench!(br, C, A, B, i)
    M, N = size(C); K = size(B,1);
    n_gflop = M*K*N*2e-9
    br[1,i] = n_gflop / @belapsed mul!($C, $A, $B)
    Cblas = copy(C)
    br[2,i] = n_gflop / @belapsed jgemm!($C, $A, $B)
    @assert C ≈ Cblas "Julia gemm wrong?"
    br[3,i] = n_gflop / @belapsed cgemm!($C, $A, $B)
    @assert C ≈ Cblas "Polly gemm wrong?"
    br[4,i] = n_gflop / @belapsed fgemm!($C, $A, $B)
    @assert C ≈ Cblas "Fort gemm wrong?"
    br[5,i] = n_gflop / @belapsed fgemm_builtin!($C, $A, $B)
    @assert C ≈ Cblas "Fort intrinsic gemm wrong?"
    br[6,i] = n_gflop / @belapsed icgemm!($C, $A, $B)
    @assert C ≈ Cblas "icc gemm wrong?"
    br[7,i] = n_gflop / @belapsed ifgemm!($C, $A, $B)
    @assert C ≈ Cblas "ifort gemm wrong?"
    br[8,i] = n_gflop / @belapsed ifgemm_builtin!($C, $A, $B)
    @assert C ≈ Cblas "ifort intrinsic gemm wrong?"
    br[9,i] = n_gflop / @belapsed gemmavx!($C, $A, $B)
    @assert C ≈ Cblas "LoopVec gemm wrong?"
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

function benchmark_AmulB(sizes)
    tests = [BLAS.vendor() === :mkl ? "IntelMKL" : "OpenBLAS", "Julia", "Clang-Polly", "GFortran", "GFort-intrinsic", "icc", "ifort", "ifort-intrinsic", "LoopVectorization"]
    br = BenchmarkResult(tests, sizes)
    sm = br.sizedresults.results
    pmap(is -> A_mul_B_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end
function benchmark_AmulBt(sizes)
    tests = [BLAS.vendor() === :mkl ? "IntelMKL" : "OpenBLAS", "Julia", "Clang-Polly", "GFortran", "GFort-intrinsic", "icc", "ifort", "ifort-intrinsic", "LoopVectorization"]
    br = BenchmarkResult(tests, sizes)
    sm = br.sizedresults.results
    pmap(is -> A_mul_Bt_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end
function benchmark_AtmulB(sizes)
    tests = [BLAS.vendor() === :mkl ? "IntelMKL" : "OpenBLAS", "Julia", "Clang-Polly", "GFortran", "GFort-intrinsic", "icc", "ifort", "ifort-intrinsic", "LoopVectorization"]
    br = BenchmarkResult(tests, sizes)
    sm = br.sizedresults.results
    pmap(is -> At_mul_B_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end
function benchmark_AtmulBt(sizes)
    tests = [BLAS.vendor() === :mkl ? "IntelMKL" : "OpenBLAS", "Julia", "Clang-Polly", "GFortran", "GFort-intrinsic", "icc", "ifort", "ifort-intrinsic", "LoopVectorization"]
    br = BenchmarkResult(tests, sizes)
    sm = br.sizedresults.results
    pmap(is -> At_mul_Bt_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end

function dot_bench!(br, s, i)
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
    br[5,i] = n_gflop / @belapsed icdot($a, $b)
    @assert cdot(a,b) ≈ dotblas "icc dot wrong?"
    br[6,i] = n_gflop / @belapsed ifdot($a, $b)
    @assert fdot(a,b) ≈ dotblas "ifort dot wrong?"
    br[7,i] = n_gflop / @belapsed jdotavx($a, $b)
    @assert jdotavx(a,b) ≈ dotblas "LoopVec dot wrong?"
end
function benchmark_dot(sizes)
    tests = [BLAS.vendor() === :mkl ? "IntelMKL" : "OpenBLAS", "Julia", "Clang-Polly", "GFortran", "icc", "ifort", "LoopVectorization"]
    br = BenchmarkResult(tests, sizes)
    sm = br.sizedresults.results
    pmap(is -> dot_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end
function selfdot_bench!(br, s, i)
    a = rand(s); b = rand(s);
    n_gflop = s * 2e-9
    br[1,i] = n_gflop / @belapsed dot($a, $a)
    dotblas = dot(a, a)
    br[2,i] = n_gflop / @belapsed jselfdot($a)
    @assert jselfdot(a) ≈ dotblas "Julia dot wrong?"
    br[3,i] = n_gflop / @belapsed cselfdot($a)
    @assert cselfdot(a) ≈ dotblas "Polly dot wrong?"
    br[4,i] = n_gflop / @belapsed fselfdot($a)
    @assert fselfdot(a) ≈ dotblas "Fort dot wrong?"
    br[5,i] = n_gflop / @belapsed icselfdot($a)
    @assert cselfdot(a) ≈ dotblas "icc dot wrong?"
    br[6,i] = n_gflop / @belapsed ifselfdot($a)
    @assert fselfdot(a) ≈ dotblas "ifort dot wrong?"
    br[7,i] = n_gflop / @belapsed jselfdotavx($a)
    @assert jselfdotavx(a) ≈ dotblas "LoopVec dot wrong?"
end
function benchmark_selfdot(sizes)
    tests = [BLAS.vendor() === :mkl ? "IntelMKL" : "OpenBLAS", "Julia", "Clang-Polly", "GFortran", "icc", "ifort", "LoopVectorization"]
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
    br[1,i] = n_gflop / @belapsed mul!($x, $A, $y)
    xblas = copy(x)
    br[2,i] = n_gflop / @belapsed jgemv!($x, $A, $y)
    @assert x ≈ xblas "Julia wrong?"
    br[3,i] = n_gflop / @belapsed cgemv!($x, $A, $y)
    @assert x ≈ xblas "Polly wrong?"
    br[4,i] = n_gflop / @belapsed fgemv!($x, $A, $y)
    @assert x ≈ xblas "Fort wrong?"
    br[5,i] = n_gflop / @belapsed fgemv_builtin!($x, $A, $y)
    @assert x ≈ xblas "Fort wrong?"
    br[6,i] = n_gflop / @belapsed icgemv!($x, $A, $y)
    @assert x ≈ xblas "icc wrong?"
    br[7,i] = n_gflop / @belapsed ifgemv!($x, $A, $y)
    @assert x ≈ xblas "ifort wrong?"
    br[8,i] = n_gflop / @belapsed ifgemv_builtin!($x, $A, $y)
    @assert x ≈ xblas "ifort wrong?"
    br[9,i] = n_gflop / @belapsed jgemvavx!($x, $A, $y)
    @assert x ≈ xblas "LoopVec wrong?"
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
    tests = [BLAS.vendor() === :mkl ? "IntelMKL" : "OpenBLAS", "Julia", "Clang-Polly", "GFortran", "GFort-intrinsic", "icc", "ifort", "ifort-intrinsic", "LoopVectorization"]
    br = BenchmarkResult(tests, sizes)
    sm = br.sizedresults.results
    pmap(is -> A_mul_vb_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end
function benchmark_Atmulvb(sizes)
    tests = [BLAS.vendor() === :mkl ? "IntelMKL" : "OpenBLAS", "Julia", "Clang-Polly", "GFortran", "GFort-intrinsic", "icc", "ifort", "ifort-intrinsic", "LoopVectorization"]
    br = BenchmarkResult(tests, sizes)
    sm = br.sizedresults.results
    pmap(is -> At_mul_vb_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end

function dot3_bench!(br, s, i)
    M, N = totwotuple(s)
    x = rand(M); A = rand(M, N); y = rand(N);
    n_gflop = M*N * 3e-9
    br[1,i] = n_gflop / @belapsed dot($x, $A, $y)
    dotblas = dot(x, A, y)
    br[2,i] = n_gflop / @belapsed jdot3($x, $A, $y)
    @assert jdot3(x, A, y) ≈ dotblas "Julia dot wrong?"
    br[3,i] = n_gflop / @belapsed cdot3($x, $A, $y)
    @assert cdot3(x, A, y) ≈ dotblas "Polly dot wrong?"
    br[4,i] = n_gflop / @belapsed fdot3($x, $A, $y)
    @assert fdot3(x, A, y) ≈ dotblas "Fort dot wrong?"
    br[5,i] = n_gflop / @belapsed icdot3($x, $A, $y)
    @assert cdot3(x, A, y) ≈ dotblas "icc dot wrong?"
    br[6,i] = n_gflop / @belapsed ifdot3($x, $A, $y)
    @assert fdot3(x, A, y) ≈ dotblas "ifort dot wrong?"
    br[7,i] = n_gflop / @belapsed jdot3avx($x, $A, $y)
    @assert jdot3avx(x, A, y) ≈ dotblas "LoopVec dot wrong?"
end
function benchmark_dot3(sizes)
    tests = [BLAS.vendor() === :mkl ? "IntelMKL" : "OpenBLAS", "Julia", "Clang-Polly", "GFortran", "icc", "ifort", "LoopVectorization"]
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
    n_gflop = 2e-9*(P*N + 2N)
    br[1,i] = n_gflop / @belapsed sse!($Xβ, $y, $X, $β)
    lpblas = sse!(Xβ, y, X, β)
    br[2,i] = n_gflop / @belapsed jOLSlp($y, $X, $β)
    @assert jOLSlp(y, X, β) ≈ lpblas "Julia wrong?"
    br[3,i] = n_gflop / @belapsed cOLSlp($y, $X, $β)
    @assert cOLSlp(y, X, β) ≈ lpblas "Polly wrong?"
    br[4,i] = n_gflop / @belapsed fOLSlp($y, $X, $β)
    @assert fOLSlp(y, X, β) ≈ lpblas "Fort wrong?"
    br[5,i] = n_gflop / @belapsed icOLSlp($y, $X, $β)
    @assert cOLSlp(y, X, β) ≈ lpblas "icc wrong?"
    br[6,i] = n_gflop / @belapsed ifOLSlp($y, $X, $β)
    @assert fOLSlp(y, X, β) ≈ lpblas "ifort wrong?"
    br[7,i] = n_gflop / @belapsed jOLSlp_avx($y, $X, $β)
    @assert jOLSlp_avx(y, X, β) ≈ lpblas "LoopVec wrong?"
end
function benchmark_sse(sizes)
    tests = [BLAS.vendor() === :mkl ? "IntelMKL" : "OpenBLAS", "Julia", "Clang-Polly", "GFortran", "icc", "ifort", "LoopVectorization"]
    br = BenchmarkResult(tests, sizes)
    sm = br.sizedresults.results
    pmap(is -> sse_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end

function exp_bench!(br, s, i)
    a = rand(s); b = similar(a)
    n_gflop = 1e-9*s # not really gflops
    br[1,i] = n_gflop / @belapsed @. $b = exp($a)
    baseb = copy(b)
    br[2,i] = n_gflop / @belapsed cvexp!($b, $a)
    @assert b ≈ baseb "Clang wrong?"
    br[3,i] = n_gflop / @belapsed fvexp!($b, $a)
    @assert b ≈ baseb "Fort wrong?"
    br[4,i] = n_gflop / @belapsed icvexp!($b, $a)
    @assert b ≈ baseb "icc wrong?"
    br[5,i] = n_gflop / @belapsed ifvexp!($b, $a)
    @assert b ≈ baseb "ifort wrong?"
    br[6,i] = n_gflop / @belapsed @avx @. $b = exp($a)
    @assert b ≈ baseb "LoopVec wrong?"
end
function benchmark_exp(sizes)
    tests = ["Julia", "Clang-Polly", "GFortran", "icc", "ifort", "LoopVectorization"]
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
    br[1,i] = n_gflop / @belapsed @. $D = $a + $B * $c′
    Dcopy = copy(D)
    br[2,i] = n_gflop / @belapsed caplusBc!($D, $a, $B, $c)
    @assert D ≈ Dcopy "Polly wrong?"
    br[3,i] = n_gflop / @belapsed faplusBc!($D, $a, $B, $c)
    @assert D ≈ Dcopy "Fort wrong?"
    br[4,i] = n_gflop / @belapsed icaplusBc!($D, $a, $B, $c)
    @assert D ≈ Dcopy "icc wrong?"
    br[5,i] = n_gflop / @belapsed ifaplusBc!($D, $a, $B, $c)
    @assert D ≈ Dcopy "ifort wrong?"
    br[6,i] = n_gflop / @belapsed @avx @. $D = $a + $B * $c′
    @assert D ≈ Dcopy "LoopVec wrong?"
end
function benchmark_aplusBc(sizes)
    tests = ["Julia", "Clang-Polly", "GFortran", "icc", "ifort", "LoopVectorization"]
    br = BenchmarkResult(tests, sizes)
    sm = br.sizedresults.results
    pmap(is -> aplusBc_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end

function AplusAt_bench!(br, s, i)
    A = rand(s,s); B = similar(A)
    n_gflop = 1e-9*s^2
    br[1,i] = n_gflop / @belapsed @. $B = $A + $A'
    baseB = copy(B)
    br[2,i] = n_gflop / @belapsed cAplusAt!($B, $A)
    @assert B ≈ baseB "Clang wrong?"
    br[3,i] = n_gflop / @belapsed fAplusAt!($B, $A)
    @assert B ≈ baseB "Fort wrong?"
    br[4,i] = n_gflop / @belapsed fAplusAt_builtin!($B, $A)
    @assert B ≈ baseB "Fort-builtin wrong?"
    br[5,i] = n_gflop / @belapsed icAplusAt!($B, $A)
    @assert B ≈ baseB "icc wrong?"
    br[6,i] = n_gflop / @belapsed ifAplusAt!($B, $A)
    @assert B ≈ baseB "ifort wrong?"
    br[7,i] = n_gflop / @belapsed ifAplusAt_builtin!($B, $A)
    @assert B ≈ baseB "ifort-builtin wrong?"
    br[8,i] = n_gflop / @belapsed @avx @. $B = $A + $A'
    @assert B ≈ baseB "LoopVec wrong?"
end
function benchmark_AplusAt(sizes)
    tests = ["Julia", "Clang-Polly", "GFortran", "GFortran-builtin", "icc", "ifort", "ifort-builtin", "LoopVectorization"]
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
    br[1,i] = n_gflop / @belapsed  randomaccess($P, $basis, $coefs)
    br[2,i] = n_gflop / @belapsed crandomaccess($P, $basis, $coefs)
    @assert p ≈ crandomaccess(P, basis, coefs) "Clang wrong?"
    br[3,i] = n_gflop / @belapsed frandomaccess($P, $basis, $coefs)
    @assert p ≈ frandomaccess(P, basis, coefs) "Fort wrong?"
    br[4,i] = n_gflop / @belapsed icrandomaccess($P, $basis, $coefs)
    @assert p ≈ icrandomaccess(P, basis, coefs) "icc wrong?"
    br[5,i] = n_gflop / @belapsed ifrandomaccess($P, $basis, $coefs)
    @assert p ≈ ifrandomaccess(P, basis, coefs) "ifort wrong?"
    br[6,i] = n_gflop / @belapsed randomaccessavx($P, $basis, $coefs)
    @assert p ≈ randomaccessavx(P, basis, coefs) "LoopVec wrong?"
end
function benchmark_random_access(sizes)
    tests = ["Julia", "Clang-Polly", "GFortran", "icc", "ifort", "LoopVectorization"]
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
    br[1,i] = n_gflop / @belapsed logdet($U)
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
    br[7,i] = n_gflop / @belapsed jlogdettriangleavx($U)
    @assert ld ≈ jlogdettriangleavx(U) "LoopVec wrong?"
end
function benchmark_logdettriangle(sizes)
    tests = ["Julia-builtin", "Julia", "Clang-Polly", "GFortran", "icc", "ifort", "LoopVectorization"]
    br = BenchmarkResult(tests, sizes)
    sm = br.sizedresults.results
    pmap(is -> logdettriangle_bench!(sm, is[2], is[1]), enumerate(sizes))
    br
end

