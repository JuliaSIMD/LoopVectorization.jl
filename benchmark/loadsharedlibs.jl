
using LoopVectorization.VectorizationBase: REGISTER_SIZE

pkgdir(pkg::String) = abspath(joinpath(dirname(Base.find_package(pkg)), ".."))
const LOOPVECBENCHDIR = joinpath(pkgdir("LoopVectorization"), "benchmark")
include(joinpath(LOOPVECBENCHDIR, "looptests.jl"))

const LIBCTEST = joinpath(LOOPVECBENCHDIR, "libctests.so")
const LIBFTEST = joinpath(LOOPVECBENCHDIR, "libftests.so")
const LIBICTEST = joinpath(LOOPVECBENCHDIR, "libictests.so")
const LIBIFTEST = joinpath(LOOPVECBENCHDIR, "libiftests.so")

# requires Clang with polly to build
cfile = joinpath(LOOPVECBENCHDIR, "looptests.c")
if !isfile(LIBCTEST) || mtime(cfile) > mtime(LIBCTEST)    
    run(`clang -Ofast -march=native -mprefer-vector-width=$(8REGISTER_SIZE) -lm -mllvm -polly -mllvm -polly-vectorizer=stripmine -shared -fPIC $cfile -o $LIBCTEST`)
end
if !isfile(LIBICTEST) || mtime(cfile) > mtime(LIBICTEST)
    run(`icc -fast -qopt-zmm-usage=high -shared -fPIC $cfile -o $LIBICTEST`)
end
ffile = joinpath(LOOPVECBENCHDIR, "looptests.f90")
if !isfile(LIBFTEST) || mtime(ffile) > mtime(LIBFTEST)
    # --param max-unroll-times defaults to ≥8, which is generally excessive
    run(`gfortran -Ofast -march=native -funroll-loops --param max-unroll-times=4 -floop-nest-optimize -mprefer-vector-width=$(8REGISTER_SIZE) -shared -fPIC $ffile -o $LIBFTEST`)
end
if !isfile(LIBIFTEST) || mtime(ffile) > mtime(LIBIFTEST)
    run(`ifort -fast -qopt-zmm-usage=high -shared -fPIC $ffile -o $LIBIFTEST`)
end

for (prefix,Cshared,Fshared) ∈ ((Symbol(""),LIBCTEST,LIBFTEST), (:i,LIBICTEST,LIBIFTEST))
    for order ∈ (:kmn, :knm, :mkn, :mnk, :nkm, :nmk)
        gemm = Symbol(:gemm_, order)
        @eval function $(Symbol(prefix, :c, gemm, :!))(C, A, B)
            M, N = size(C); K = size(B, 1)
            ccall(
                ($(QuoteNode(gemm)), $Cshared), Cvoid,
                (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Clong, Clong, Clong),
                C, A, B, M, K, N
            )
        end
        @eval function $(Symbol(prefix, :f, gemm, :!))(C, A, B)
            M, N = size(C); K = size(B, 1)
            ccall(
                ($(QuoteNode(gemm)), $Fshared), Cvoid,
                (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Clong}, Ref{Clong}, Ref{Clong}),
                C, A, B, Ref(M), Ref(K), Ref(N)
            )
        end
    end
    @eval @inline $(Symbol(prefix,:cgemm!))(C, A, B) = $(Symbol(prefix, :cgemm_nkm!))(C, A, B)
    @eval @inline $(Symbol(prefix,:fgemm!))(C, A, B) = $(Symbol(prefix, :fgemm_nkm!))(C, A, B)
    @eval function $(Symbol(prefix,:fgemm_builtin!))(C, A, B)
        M, N = size(C); K = size(B, 1)
        ccall(
            (:gemmbuiltin, $Fshared), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Clong}, Ref{Clong}, Ref{Clong}),
            C, A, B, Ref(M), Ref(K), Ref(N)
        )
    end
    @eval @inline function $(Symbol(prefix,:cgemm!))(C, A::Adjoint, B)
        M, N = size(C); K = size(B, 1)
        ccall(
            (:AtmulB, $Cshared), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Clong, Clong, Clong),
            C, parent(A), B, M, K, N
        )
    end
    @eval @inline function $(Symbol(prefix,:fgemm!))(C, A::Adjoint, B)
        M, N = size(C); K = size(B, 1)
        ccall(
            (:AtmulB, $Fshared), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Clong}, Ref{Clong}, Ref{Clong}),
            C, parent(A), B, Ref(M), Ref(K), Ref(N)
        )
    end
    @eval @inline function $(Symbol(prefix,:fgemm_builtin!))(C, A::Adjoint, B)
        M, N = size(C); K = size(B, 1)
        ccall(
            (:AtmulBbuiltin, $Fshared), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Clong}, Ref{Clong}, Ref{Clong}),
            C, parent(A), B, Ref(M), Ref(K), Ref(N)
        )
    end

    @eval function $(Symbol(prefix,:cdot))(a, b)
        N = length(a)
        ccall(
            (:dot, $Cshared), Float64,
            (Ptr{Float64}, Ptr{Float64}, Clong),
            a, b, N
        )
    end
    @eval function $(Symbol(prefix,:fdot))(a, b)
        N = length(a)
        d = Ref{Float64}()
        ccall(
            (:dot, $Fshared), Cvoid,
            (Ref{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Clong}),
            d, a, b, Ref(N)
        )
        d[]
    end
    @eval function $(Symbol(prefix,:cselfdot))(a)
        N = length(a)
        ccall(
            (:selfdot, $Cshared), Float64,
            (Ptr{Float64}, Clong),
            a, N
        )
    end
    @eval function $(Symbol(prefix,:fselfdot))(a)
        N = length(a)
        d = Ref{Float64}()
        ccall(
            (:selfdot, $Fshared), Cvoid,
            (Ref{Float64}, Ptr{Float64}, Ref{Clong}),
            d, a, Ref(N)
        )
        d[]
    end
    @eval function $(Symbol(prefix,:cdot3))(x, A, y)
        M, N = size(A)
        ccall(
            (:dot3, $Cshared), Float64,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Clong, Clong),
            x, A, y, M, N
        )
    end
    @eval function $(Symbol(prefix,:fdot3))(x, A, y)
        M, N = size(A)
        d = Ref{Float64}()
        ccall(
            (:dot3, $Fshared), Cvoid,
            (Ref{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Clong}, Ref{Clong}),
            d, x, A, y, Ref(M), Ref(N)
        )
        d[]
    end

    @eval function $(Symbol(prefix,:cgemv!))(y, A, x)
        M, K = size(A)
        ccall(
            (:gemv, $Cshared), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Clong, Clong),
            y, A, x, M, K
        )
    end
    @eval function $(Symbol(prefix,:fgemv!))(y, A, x)
        M, K = size(A)
        ccall(
            (:gemv, $Fshared), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Clong}, Ref{Clong}),
            y, A, x, Ref(M), Ref(K)
        )
    end
    @eval function $(Symbol(prefix,:fgemv_builtin!))(y, A, x)
        M, K = size(A)
        ccall(
            (:gemvbuiltin, $Fshared), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Clong}, Ref{Clong}),
            y, A, x, Ref(M), Ref(K)
        )
    end

    @eval function $(Symbol(prefix,:caplusBc!))(D, a, B, c)
        M, K = size(B)
        ccall(
            (:aplusBc, $Cshared), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Clong, Clong),
            D, a, B, c, M, K
        )
    end
    @eval function $(Symbol(prefix,:faplusBc!))(D, a, B, c)
        M, K = size(B)
        ccall(
            (:aplusBc, $Fshared), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Clong}, Ref{Clong}),
            D, a, B, c, Ref(M), Ref(K)
        )
    end
    @eval function $(Symbol(prefix,:cOLSlp))(y, X, β)
        N, P = size(X)
        ccall(
            (:OLSlp, $Cshared), Float64,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Clong, Clong),
            y, X, β, N, P
        )
    end
    @eval function $(Symbol(prefix,:fOLSlp))(y, X, β)
        N, P = size(X)
        lp = Ref{Float64}()
        ccall(
            (:OLSlp, $Fshared), Cvoid,
            (Ref{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Clong}, Ref{Clong}),
            lp, y, X, β, Ref(N), Ref(P)
        )
        lp[]
    end
    @eval function $(Symbol(prefix,:cvexp!))(b, a)
        N = length(b)
        ccall(
            (:vexp, $Cshared), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Clong),
            b, a, N
        )
    end
    @eval function $(Symbol(prefix,:fvexp!))(b, a)
        N = length(b)
        ccall(
            (:vexp, $Fshared), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ref{Clong}),
            b, a, Ref(N)
        )
    end
    @eval function $(Symbol(prefix,:fvexpsum))(a)
        N = length(a)
        s = Ref{Float64}()
        ccall(
            (:svexp, $Fshared), Cvoid,
            (Ref{Float64}, Ptr{Float64}, Ref{Clong}),
            s, a, Ref(N)
        )
        s[]
    end

    @eval function $(Symbol(prefix,:fAplusAt!))(B, A)
        N = size(B,1)
        ccall(
            (:AplusAt, $Fshared), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ref{Clong}),
            B, A, Ref(N)
        )
    end
    @eval function $(Symbol(prefix,:fAplusAtbuiltin!))(B, A)
        N = size(B,1)
        ccall(
            (:AplusAtbuiltin, $Fshared), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ref{Clong}),
            B, A, Ref(N)
        )
    end
    @eval function $(Symbol(prefix,:cAplusAt!))(B, A)
        N = size(B,1)
        ccall(
            (:AplusAt, $Cshared), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Clong),
            B, A, N
        )
    end
    @eval function $(Symbol(prefix,:crandomaccess))(P, basis, coefs)
        A, C = size(P)
        ccall(
            (:randomaccess, $Cshared), Float64,
            (Ptr{Float64}, Ptr{Clong}, Ptr{Float64}, Clong, Clong),
            P, basis, coefs, A, C
        )
    end
    @eval function $(Symbol(prefix,:frandomaccess))(P, basis, coefs)
        A, C = size(P)
        p = Ref{Float64}()
        ccall(
            (:randomaccess, $Cshared), Cvoid,
            (Ref{Float64}, Ptr{Float64}, Ptr{Clong}, Ptr{Float64}, Ref{Clong}, Ref{Clong}),
            p, P, basis, coefs, Ref(A), Ref(C)
        )
        p[]
    end
    
end
