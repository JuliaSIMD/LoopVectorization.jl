
using LoopVectorization.VectorizationBase: REGISTER_SIZE

pkgdir(pkg::String) = abspath(joinpath(dirname(Base.find_package(pkg)), ".."))
const LOOPVECBENCHDIR = joinpath(pkgdir("LoopVectorization"), "benchmarks")
include(joinpath(LOOPVECBENCHDIR, "looptests.jl"))

const LIBCTEST = joinpath(LOOPVECBENCHDIR, "libctests.so")
const LIBFTEST = joinpath(LOOPVECBENCHDIR, "libftests.so")

# requires Clang with polly to build
cfile = joinpath(LOOPVECBENCHDIR, "looptests.c")
if !isfile(LIBCTEST) || mtime(cfile) > mtime(LIBCTEST)    
    run(`clang -Ofast -march=native -mprefer-vector-width=$(8REGISTER_SIZE) -lm -mllvm -polly -mllvm -polly-vectorizer=stripmine -shared -fPIC $cfile -o $LIBCTEST`)
end
ffile = joinpath(LOOPVECBENCHDIR, "looptests.f90")
if !isfile(LIBFTEST) || mtime(ffile) > mtime(LIBFTEST)
    # --param max-unroll-times defaults to ≥8, which is generally excessive
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
function cAtmulB!(C, A, B)
    M, N = size(C); K = size(B, 1)
    ccall(
        (:AtmulB, LIBCTEST), Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Clong, Clong, Clong),
        C, A, B, M, K, N
    )
end
function fAtmulB!(C, A, B)
    M, N = size(C); K = size(B, 1)
    ccall(
        (:AtmulB, LIBFTEST), Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Clong}, Ref{Clong}, Ref{Clong}),
        C, A, B, Ref(M), Ref(K), Ref(N)
    )
end
function fAtmulB_builtin!(C, A, B)
    M, N = size(C); K = size(B, 1)
    ccall(
        (:AtmulBbuiltin, LIBFTEST), Cvoid,
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
function cdot3(x, A, y)
    M, N = size(A)
    ccall(
        (:dot3, LIBCTEST), Float64,
        (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Clong, Clong),
        x, A, y, M, N
    )
end
function fdot3(x, A, y)
    M, N = size(A)
    d = Ref{Float64}()
    ccall(
        (:dot3, LIBFTEST), Cvoid,
        (Ref{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Clong}, Ref{Clong}),
        d, x, A, y, Ref(M), Ref(N)
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
function cvexp!(b, a)
    N = length(b)
    ccall(
        (:vexp, LIBCTEST), Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Clong),
        b, a, N
    )
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

