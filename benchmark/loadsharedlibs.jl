using LinearAlgebra, LoopVectorization
using LoopVectorization.VectorizationBase: REGISTER_SIZE

# const LOOPVECBENCHDIR = joinpath(pkgdir(LoopVectorization), "benchmark")
include(joinpath(LOOPVECBENCHDIR, "looptests.jl"))

const LIBCTEST = joinpath(LOOPVECBENCHDIR, "libctests.so")
const LIBFTEST = joinpath(LOOPVECBENCHDIR, "libftests.so")
const LIBICTEST = joinpath(LOOPVECBENCHDIR, "libictests.so")
const LIBIFTEST = joinpath(LOOPVECBENCHDIR, "libiftests.so")
const LIBEIGENTEST = joinpath(LOOPVECBENCHDIR, "libetest.so")
const LIBIEIGENTEST = joinpath(LOOPVECBENCHDIR, "libietest.so")
const LIBDIRECTCALLJIT = joinpath(LOOPVECBENCHDIR, "libdcjtest.so")

# requires Clang with polly to build
cfile = joinpath(LOOPVECBENCHDIR, "looptests.c")
if !isfile(LIBCTEST) || mtime(cfile) > mtime(LIBCTEST)    
    run(`/usr/bin/clang -Ofast -march=native -mprefer-vector-width=$(8REGISTER_SIZE) -lm -shared -fPIC $cfile -o $LIBCTEST`)
    # run(`/usr/local/bin/clang -Ofast -march=native -mprefer-vector-width=$(8REGISTER_SIZE) -lm -mllvm -polly -mllvm -polly-vectorizer=stripmine -shared -fPIC $cfile -o $LIBCTEST`)
end
if !isfile(LIBICTEST) || mtime(cfile) > mtime(LIBICTEST)
    run(`icc -fast -qopt-zmm-usage=high -fargument-noalias-global -qoverride-limits -shared -fPIC $cfile -o $LIBICTEST`)
end
ffile = joinpath(LOOPVECBENCHDIR, "looptests.f90")
if !isfile(LIBFTEST) || mtime(ffile) > mtime(LIBFTEST)
    # --param max-unroll-times defaults to ≥8, which is generally excessive
    run(`gfortran -Ofast -march=native -funroll-loops -mprefer-vector-width=$(8REGISTER_SIZE) -shared -fPIC $ffile -o $LIBFTEST`)
    # run(`gfortran -Ofast -march=native -funroll-loops -floop-nest-optimize -mprefer-vector-width=$(8REGISTER_SIZE) -shared -fPIC $ffile -o $LIBFTEST`)
end
if !isfile(LIBIFTEST) || mtime(ffile) > mtime(LIBIFTEST)
    run(`ifort -fast -qopt-zmm-usage=high -qoverride-limits -shared -fPIC $ffile -o $LIBIFTEST`)
end

# g++ -> ICE, so Clang and ICPC
eigenfile = joinpath(LOOPVECBENCHDIR, "looptestseigen.cpp")
if !isfile(LIBEIGENTEST) || mtime(eigenfile) > mtime(LIBEIGENTEST)
    # Clang seems to have trouble finding includes
    run(`g++ -O3 -march=native -mprefer-vector-width=$(8REGISTER_SIZE) -DEIGEN_VECTORIZE_AVX512 -I/usr/include/eigen3 -shared -fPIC $eigenfile -o $LIBEIGENTEST`)
    
end
if !isfile(LIBIEIGENTEST) || mtime(eigenfile) > mtime(LIBIEIGENTEST)
    # run(`/usr/bin/clang++ -Ofast -march=native -mprefer-vector-width=$(8REGISTER_SIZE) -DEIGEN_VECTORIZE_AVX512 -I/usr/include/c++/9 -I/usr/include/c++/9/x86_64-generic-linux -I/usr/include/eigen3 -shared -fPIC $eigenfile -o $LIBEIGENTEST`)
    run(`/usr/bin/clang++ -Ofast -march=native -mprefer-vector-width=$(8REGISTER_SIZE) -DEIGEN_VECTORIZE_AVX512 -I/usr/include/eigen3 -shared -fPIC $eigenfile -o $LIBEIGENTEST`)
    # run(`icpc -fast -qopt-zmm-usage=high -fargument-noalias-global -qoverride-limits -I/usr/include/eigen3 -shared -fPIC $eigenfile -o $LIBIEIGENTEST`)
end

MKL_ROOT = "/home/chriselrod/intel"
directcalljitfile = joinpath(LOOPVECBENCHDIR, "directcalljit.f90")
if !isfile(LIBDIRECTCALLJIT) || mtime(directcalljitfile) > mtime(LIBDIRECTCALLJIT)
    run(`ifort -fast -DMKL_DIRECT_CALL_SEQ_JIT -fpp -qopt-zmm-usage=high -Wl,--start-group $(joinpath(MKL_ROOT,"mkl/lib/intel64/libmkl_intel_lp64.a")) $(joinpath(MKL_ROOT,"mkl/lib/intel64/libmkl_sequential.a")) $(joinpath(MKL_ROOT,"mkl/lib/intel64/libmkl_core.a")) -Wl,--end-group -I$(joinpath(MKL_ROOT, "mkl/include")) -I$(joinpath(MKL_ROOT, "compilers_and_libraries_2020.1.217/linux/mkl/include/intel64/lp64")) -shared -fPIC $directcalljitfile -o $LIBDIRECTCALLJIT`)
    # run(`gfortran -Ofast -march=native -DMKL_DIRECT_CALL_SEQ_JIT -cpp -mprefer-vector-width=$(8REGISTER_SIZE) -Wl,--start-group $(joinpath(MKL_ROOT,"mkl/lib/intel64/libmkl_intel_lp64.a")) $(joinpath(MKL_ROOT,"mkl/lib/intel64/libmkl_sequential.a")) $(joinpath(MKL_ROOT,"mkl/lib/intel64/libmkl_core.a")) -Wl,--end-group -I$(joinpath(MKL_ROOT, "mkl/include")) -I$(joinpath(MKL_ROOT, "compilers_and_libraries_2020.1.217/linux/mkl/include/intel64/lp64")) -shared -fPIC $directcalljitfile -o $LIBDIRECTCALLJIT`)
    
    # run(`gfortran -Ofast -march=native -DMKL_DIRECT_CALL_SEQ_JIT -cpp -mprefer-vector-width=$(8REGISTER_SIZE) -shared -fPIC $directcalljitfile -o $LIBDIRECTCALLJIT`)
end

istransposed(x) = false
istransposed(x::Adjoint) = true
istransposed(x::Transpose) = true
"""
If transposed, requires them to be square
"""
function dgemmmkl!(C::AbstractMatrix{Float64}, A::AbstractMatrix{Float64}, B::AbstractMatrix{Float64})
    M, N = size(C); K = size(B, 1)
    ccall(
        (:dgemmjit, LIBDIRECTCALLJIT), Cvoid,
        (Ptr{Float64},Ptr{Float64},Ptr{Float64},Ref{Int},Ref{Int},Ref{Int},Ref{Bool},Ref{Bool}),
        parent(C), parent(A), parent(B),
        Ref(M), Ref(K), Ref(N),
        Ref(istransposed(A)), Ref(istransposed(B))
    )
end
mkl_set_num_threads(N::Integer) = ccall((:set_num_threads, LIBDIRECTCALLJIT), Cvoid, (Ref{UInt32},), Ref(N % UInt32))
mkl_set_num_threads(1)
"""
If transposed, requires them to be square
"""
function dgemvmkl!(y::AbstractVector{Float64}, A::AbstractMatrix{Float64}, x::AbstractVector{Float64})
    M, N = size(A);
    ccall(
        (:dgemvjit, LIBDIRECTCALLJIT), Cvoid,
        (Ptr{Float64},Ptr{Float64},Ptr{Float64},Ref{Int},Ref{Int},Ref{Bool}),
        parent(y), parent(A), parent(x),
        Ref(M), Ref(N), Ref(istransposed(A))
    )
end

for (prefix,Cshared,Fshared,Eshared) ∈ ((Symbol(""),LIBCTEST,LIBFTEST,LIBEIGENTEST), (:i,LIBICTEST,LIBIFTEST,LIBIEIGENTEST))
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
    @eval $(Symbol(prefix,:cgemm!))(C, A, B) = $(Symbol(prefix, :cgemm_nkm!))(C, A, B)
    @eval $(Symbol(prefix,:fgemm!))(C, A, B) = $(Symbol(prefix, :fgemm_nkm!))(C, A, B)
    @eval function $(Symbol(prefix,:egemm!))(C, A, B)
        M, N = size(C); K = size(B, 1)
        ccall(
            (:AmulB, $Eshared), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Clong, Clong, Clong),
            C, A, B, M, K, N
        )
    end
    @eval function $(Symbol(prefix,:fgemm_builtin!))(C, A, B)
        M, N = size(C); K = size(B, 1)
        ccall(
            (:gemmbuiltin, $Fshared), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Clong}, Ref{Clong}, Ref{Clong}),
            C, A, B, Ref(M), Ref(K), Ref(N)
        )
    end
for (p,s) ∈ [(:c,Cshared) (:e,Eshared)]
    @eval function $(Symbol(prefix,p,:gemm!))(C, A::Adjoint, B)
        M, N = size(C); K = size(B, 1)
        ccall(
            (:AtmulB, $s), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Clong, Clong, Clong),
            C, parent(A), B, M, K, N
        )
    end
end
    @eval function $(Symbol(prefix,:fgemm!))(C, A::Adjoint, B)
        M, N = size(C); K = size(B, 1)
        ccall(
            (:AtmulB, $Fshared), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Clong}, Ref{Clong}, Ref{Clong}),
            C, parent(A), B, Ref(M), Ref(K), Ref(N)
        )
    end
    @eval function $(Symbol(prefix,:fgemm_builtin!))(C, A::Adjoint, B)
        M, N = size(C); K = size(B, 1)
        ccall(
            (:AtmulBbuiltin, $Fshared), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Clong}, Ref{Clong}, Ref{Clong}),
            C, parent(A), B, Ref(M), Ref(K), Ref(N)
        )
    end
for (p,s) ∈ [(:c,Cshared) (:e,Eshared)]
    @eval function $(Symbol(prefix,p,:gemm!))(C, A, B::Adjoint)
        M, N = size(C); K = size(B, 1)
        ccall(
            (:AmulBt, $s), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Clong, Clong, Clong),
            C, A, parent(B), M, K, N
        )
    end
end
    @eval function $(Symbol(prefix,:fgemm!))(C, A, B::Adjoint)
        M, N = size(C); K = size(B, 1)
        ccall(
            (:AmulBt, $Fshared), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Clong}, Ref{Clong}, Ref{Clong}),
            C, A, parent(B), Ref(M), Ref(K), Ref(N)
        )
    end
    @eval function $(Symbol(prefix,:fgemm_builtin!))(C, A, B::Adjoint)
        M, N = size(C); K = size(B, 1)
        ccall(
            (:AmulBtbuiltin, $Fshared), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Clong}, Ref{Clong}, Ref{Clong}),
            C, A, parent(B), Ref(M), Ref(K), Ref(N)
        )
    end
for (p,s) ∈ [(:c,Cshared) (:e,Eshared)]
    @eval function $(Symbol(prefix,p,:gemm!))(C, A::Adjoint, B::Adjoint)
        M, N = size(C); K = size(B, 1)
        ccall(
            (:AtmulBt, $s), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Clong, Clong, Clong),
            C, parent(A), parent(B), M, K, N
        )
    end
end
    @eval function $(Symbol(prefix,:fgemm!))(C, A::Adjoint, B::Adjoint)
        M, N = size(C); K = size(B, 1)
        ccall(
            (:AtmulBt, $Fshared), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Clong}, Ref{Clong}, Ref{Clong}),
            C, parent(A), parent(B), Ref(M), Ref(K), Ref(N)
        )
    end
    @eval function $(Symbol(prefix,:fgemm_builtin!))(C, A::Adjoint, B::Adjoint)
        M, N = size(C); K = size(B, 1)
        ccall(
            (:AtmulBtbuiltin, $Fshared), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Clong}, Ref{Clong}, Ref{Clong}),
            C, parent(A), parent(B), Ref(M), Ref(K), Ref(N)
        )
    end
for (p,s) ∈ [(:c,Cshared) (:e,Eshared)]
    @eval function $(Symbol(prefix,p,:dot))(a, b)
        N = length(a)
        ccall(
            (:dot, $s), Float64,
            (Ptr{Float64}, Ptr{Float64}, Clong),
            a, b, N
        )
    end
end
    @eval function $(Symbol(prefix,:fdot))(a, b)
        N = length(a)
        ccall(
            (:dot, $Fshared), Float64,
            (Ptr{Float64}, Ptr{Float64}, Ref{Clong}),
            a, b, Ref(N)
        )
    end
for (p,s) ∈ [(:c,Cshared) (:e,Eshared)]
    @eval function $(Symbol(prefix,p,:selfdot))(a)
        N = length(a)
        ccall(
            (:selfdot, $s), Float64,
            (Ptr{Float64}, Clong),
            a, N
        )
    end
end
@eval function $(Symbol(prefix,:fselfdot))(a)
        N = length(a)
        ccall(
            (:selfdot, $Fshared), Float64,
            (Ptr{Float64}, Ref{Clong}),
            a, Ref(N)
        )
end
for (p,s) ∈ [(:c,Cshared) (:e,Eshared)]
    @eval function $(Symbol(prefix,p,:dot3))(x, A, y)
        M, N = size(A)
        ccall(
            (:dot3, $s), Float64,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Clong, Clong),
            x, A, y, M, N
        )
    end
end
    @eval function $(Symbol(prefix,:fdot3))(x, A, y)
        M, N = size(A)
        ccall(
            (:dot3, $Fshared), Float64,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Clong}, Ref{Clong}),
            x, A, y, Ref(M), Ref(N)
        )
    end
for (p,s) ∈ [(:c,Cshared) (:e,Eshared)]
    @eval function $(Symbol(prefix,p,:gemv!))(y, A, x)
        M, K = size(A)
        ccall(
            ($(QuoteNode(p === :c ? :gemv : :Amulvb)), $s), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Clong, Clong),
            y, A, x, M, K
        )
    end
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
for (p,s) ∈ [(:c,Cshared) (:e,Eshared)]
    @eval function $(Symbol(prefix,p,:gemv!))(y, A::Adjoint, x)
        M, K = size(A)
        ccall(
            (:Atmulvb, $s), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Clong, Clong),
            y, parent(A), x, M, K
        )
    end
end
@eval function $(Symbol(prefix,:fgemv!))(y, A::Adjoint, x)
        M, K = size(A)
        ccall(
            (:Atmulvb, $Fshared), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Clong}, Ref{Clong}),
            y, parent(A), x, Ref(M), Ref(K)
        )
    end
    @eval function $(Symbol(prefix,:fgemv_builtin!))(y, A::Adjoint, x)
        M, K = size(A)
        ccall(
            (:Atmulvbbuiltin, $Fshared), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Clong}, Ref{Clong}),
            y, parent(A), x, Ref(M), Ref(K)
        )
    end
for (p,s) ∈ [(:c,Cshared) (:e,Eshared)]
    @eval function $(Symbol(prefix,p,:aplusBc!))(D, a, B, c)
        M, K = size(B)
        ccall(
            (:aplusBc, $s), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Clong, Clong),
            D, a, B, c, M, K
        )
    end
end
    @eval function $(Symbol(prefix,:faplusBc!))(D, a, B, c)
        M, K = size(B)
        ccall(
            (:aplusBc, $Fshared), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Clong}, Ref{Clong}),
            D, a, B, c, Ref(M), Ref(K)
        )
    end
for (p,s) ∈ [(:c,Cshared) (:e,Eshared)]
    @eval function $(Symbol(prefix,p,:OLSlp))(y, X, β)
        N, P = size(X)
        ccall(
            (:OLSlp, $s), Float64,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Clong, Clong),
            y, X, β, N, P
        )
    end
end
    @eval function $(Symbol(prefix,:fOLSlp))(y, X, β)
        N, P = size(X)
        ccall(
            (:OLSlp, $Fshared), Float64,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Clong}, Ref{Clong}),
            y, X, β, Ref(N), Ref(P)
        )
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
    @eval function $(Symbol(prefix,:fAplusAt_builtin!))(B, A)
        N = size(B,1)
        ccall(
            (:AplusAtbuiltin, $Fshared), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ref{Clong}),
            B, A, Ref(N)
        )
    end
    for (p,s) ∈ [(:c,Cshared) (:e,Eshared)]
        @eval function $(Symbol(prefix,p,:AplusAt!))(B, A)
            N = size(B,1)
            ccall(
                (:AplusAt, $s), Cvoid,
                (Ptr{Float64}, Ptr{Float64}, Clong),
                B, A, N
            )
        end
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
        ccall(
            (:randomaccess, $Fshared), Float64,
            (Ptr{Float64}, Ptr{Clong}, Ptr{Float64}, Ref{Clong}, Ref{Clong}),
            P, basis, coefs, Ref(A), Ref(C)
        )
    end
    for (p,s) ∈ [(:c,Cshared) (:e,Eshared)]
        @eval function $(Symbol(prefix,p,:logdettriangle))(T::Union{LowerTriangular,UpperTriangular})
            N = size(T,1)
            Tp = parent(T)
            ccall(
                (:logdettriangle, $s), Float64,
                (Ptr{Float64}, Clong),
                Tp, N
            )
        end
    end
    @eval function $(Symbol(prefix,:flogdettriangle))(T::Union{LowerTriangular,UpperTriangular})
        N = size(T,1)
        Tp = parent(T)
        ccall(
            (:logdettriangle, $Fshared), Float64,
            (Ptr{Float64}, Ref{Clong}),
            Tp, Ref(N)
        )
    end
    @eval function $(Symbol(prefix,:cfilter2d!))(B::OffsetArray, A::AbstractArray, K::OffsetArray)
        Ma, Na = size(A)
        offset = first(B.offsets)
        ccall(
            (:filter2d, $Cshared), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Clong, Clong, Clong),
            parent(B), A, parent(K), Ma, Na, offset
        )
    end
    @eval function $(Symbol(prefix,:ffilter2d!))(B::OffsetArray, A::AbstractArray, K::OffsetArray)
        Ma, Na = size(A)
        offset = first(B.offsets)
        ccall(
            (:filter2d, $Fshared), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Clong}, Ref{Clong}, Ref{Clong}),
            parent(B), A, parent(K), Ref(Ma), Ref(Na), Ref(offset)
        )
    end
    @eval function $(Symbol(prefix,:cfilter2d!))(B::OffsetArray, A::AbstractArray, K::SizedOffsetMatrix{Float64,-1,1,-1,1})
        Ma, Na = size(A)
        ccall(
            (:filter2d3x3, $Cshared), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Clong, Clong),
            parent(B), A, K, Ma, Na
        )
    end
    @eval function $(Symbol(prefix,:ffilter2d!))(B::OffsetArray, A::AbstractArray, K::SizedOffsetMatrix{Float64,-1,1,-1,1})
        Ma, Na = size(A)
        ccall(
            (:filter2d3x3, $Fshared), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Clong}, Ref{Clong}),
            parent(B), A, K, Ref(Ma), Ref(Na)
        )
    end
    @eval function $(Symbol(prefix,:cfilter2dunrolled!))(B::OffsetArray, A::AbstractArray, K::SizedOffsetMatrix{Float64,-1,1,-1,1})
        Ma, Na = size(A)
        ccall(
            (:filter2d3x3unrolled, $Cshared), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Clong, Clong),
            parent(B), A, K, Ma, Na
        )
    end
    @eval function $(Symbol(prefix,:ffilter2dunrolled!))(B::OffsetArray, A::AbstractArray, K::SizedOffsetMatrix{Float64,-1,1,-1,1})
        Ma, Na = size(A)
        ccall(
            (:filter2d3x3unrolled, $Fshared), Cvoid,
            (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Clong}, Ref{Clong}),
            parent(B), A, K, Ref(Ma), Ref(Na)
        )
    end
    
end
