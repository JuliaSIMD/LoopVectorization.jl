
using LoopVectorization.VectorizationBase: REGISTER_SIZE

pkgdir(pkg::String) = abspath(joinpath(dirname(Base.find_package(pkg)), ".."))
const LOOPVECBENCHDIR = joinpath(pkgdir("LoopVectorization"), "benchmarks")
include(joinpath(LOOPVECBENCHDIR, "looptests.jl"))

const LIBCTEST = joinpath(LOOPVECBENCHDIR, "libctests.so")
const LIBFTEST = joinpath(LOOPVECBENCHDIR, "libftests.so")

# requires Clang with polly to build
if !isfile(LIBCTEST)
    cfile = joinpath(LOOPVECBENCHDIR, "looptests.c")
    run(`clang -Ofast -march=native -mprefer-vector-width=$(8REGISTER_SIZE) -mllvm -polly -mllvm -polly-vectorizer=stripmine -shared -fPIC $cfile -o $LIBCTEST`)
end
if !isfile(LIBFTEST)
    ffile = joinpath(LOOPVECBENCHDIR, "looptests.f90") # --param max-unroll-times defaults to ≥8, which is generally excessive
    run(`gfortran -Ofast -march=native -funroll-loops --param max-unroll-times=4 -floop-nest-optimize -mprefer-vector-width=$(8REGISTER_SIZE) -shared -fPIC $ffile -o $LIBFTEST`)
end

for order ∈ (:kmn, :knm, :mkn, :mnk, :nkm, :nmk)
    gemm = Symbol(:gemm_, order)
    @eval function $(Symbol(:c, gemm, :!))(C, A, B)
        M, N = size(C); K = size(B, 1)
        ccall(
            ($(QuoteNode(gemm)), LIBCTEST), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Clong, Clong, Clong),
            C, A, B, M, K, N
        )
    end
    @eval function $(Symbol(:f, gemm, :!))(C, A, B)
        M, N = size(C); K = size(B, 1)
        ccall(
            ($(QuoteNode(gemm)), LIBFTEST), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Clong}, Ref{Clong}, Ref{Clong}),
            C, A, B, Ref(M), Ref(K), Ref(N)
        )
    end
end

function fgemm_builtin!(C, A, B)
    M, N = size(C); K = size(B, 1)
    ccall(
        (:gemmbuiltin, LIBFTEST), Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Clong}, Ref{Clong}, Ref{Clong}),
        C, A, B, Ref(M), Ref(K), Ref(N)
    )
end

function cdot(a, b)
    N = length(a)
    ccall(
        (:dot, LIBCTEST), Float64,
        (Ptr{Float64}, Ptr{Float64}, Clong),
        a, b, N
    )
end
function fdot(a, b)
    N = length(a)
    d = Ref{Float64}()
    ccall(
        (:dot, LIBFTEST), Cvoid,
        (Ref{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Clong}),
        d, a, b, Ref(N)
    )
    d[]
end
function cselfdot(a)
    N = length(a)
    ccall(
        (:selfdot, LIBCTEST), Float64,
        (Ptr{Float64}, Clong),
        a, N
    )
end
function fselfdot(a)
    N = length(a)
    d = Ref{Float64}()
    ccall(
        (:selfdot, LIBFTEST), Cvoid,
        (Ref{Float64}, Ptr{Float64}, Ref{Clong}),
        d, a, Ref(N)
    )
    d[]
end

function cgemv!(y, A, x)
    M, K = size(A)
    ccall(
        (:gemv, LIBCTEST), Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Clong, Clong),
        y, A, x, M, K
    )
end
function fgemv!(y, A, x)
    M, K = size(A)
    ccall(
        (:gemv, LIBFTEST), Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Clong}, Ref{Clong}),
        y, A, x, Ref(M), Ref(K)
    )
end
function fgemv_builtin!(y, A, x)
    M, K = size(A)
    ccall(
        (:gemvbuiltin, LIBFTEST), Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Clong}, Ref{Clong}),
        y, A, x, Ref(M), Ref(K)
    )
end

function caplusBc!(D, a, B, c)
    M, K = size(B)
    ccall(
        (:aplusBc, LIBCTEST), Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Clong, Clong),
        D, a, B, c, M, K
    )
end
function faplusBc!(D, a, B, c)
    M, K = size(B)
    ccall(
        (:aplusBc, LIBFTEST), Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Clong}, Ref{Clong}),
        D, a, B, c, Ref(M), Ref(K)
    )
end
function cOLSlp(y, X, β)
    N, P = size(X)
    ccall(
        (:OLSlp, LIBCTEST), Float64,
        (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Clong, Clong),
        y, X, β, N, P
    )
end
function fOLSlp(y, X, β)
    N, P = size(X)
    lp = Ref{Float64}()
    ccall(
        (:OLSlp, LIBFTEST), Cvoid,
        (Ref{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Clong}, Ref{Clong}),
        lp, y, X, β, Ref(N), Ref(P)
    )
    lp[]
end
function fvexp!(b, a)
    N = length(b)
    ccall(
        (:vexp, LIBFTEST), Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Ref{Clong}),
        b, a, Ref(N)
    )
end
function fvexpsum(a)
    N = length(a)
    s = Ref{Float64}()
    ccall(
        (:svexp, LIBFTEST), Cvoid,
        (Ref{Float64}, Ptr{Float64}, Ref{Clong}),
        s, a, Ref(N)
    )
    s[]
end

