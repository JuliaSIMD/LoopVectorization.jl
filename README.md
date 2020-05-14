# LoopVectorization

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://chriselrod.github.io/LoopVectorization.jl/stable)
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://chriselrod.github.io/LoopVectorization.jl/latest)
[![Build Status](https://travis-ci.com/chriselrod/LoopVectorization.jl.svg?branch=master)](https://travis-ci.com/chriselrod/LoopVectorization.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/chriselrod/LoopVectorization.jl?svg=true)](https://ci.appveyor.com/project/chriselrod/LoopVectorization-jl)
[![Codecov](https://codecov.io/gh/chriselrod/LoopVectorization.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/chriselrod/LoopVectorization.jl)

## Installation

```julia
using Pkg
Pkg.add("LoopVectorization")
```
LoopVectorization is supported on Julia 1.1 and later. It is tested on Julia 1.1, 1.3, and nightly.

## Warning

Misusing LoopVectorization can have [serious consequences](http://catb.org/jargon/html/N/nasal-demons.html). Like `@inbounds`, misusing it can lead to segfaults and memory corruption.
We expect that any time you use the `@avx` macro with a given block of code that you:
1. Are not indexing an array out of bounds. `@avx` does not perform any bounds checking.
2. Are not iterationg over an empty collection. Iterating over an empty loop such as `for i ∈ eachindex(Float64[])` is undefined behavior, and will likely result in the out of bounds memory accesses. Ensure that loops behave correctly.
3. Are not relying on a specific execution order. `@avx` can and will re-order operations and loops inside its scope, so the correctness cannot depend on a particular order. You cannot implement `cumsum` with `@avx`.

## Usage

This library provides the `@avx` macro, which may be used to prefix a `for` loop or broadcast statement.
It then tries to vectorize the loop to improve runtime performance.

The macro assumes that loop iterations can be reordered. It also currently supports simple nested loops, where loop bounds of inner loops are constant across iterations of the outer loop, and only a single loop at each level of noop lest. These limitations should be removed in a future version.

## Benchmarks

Please see the documentation for benchmarks versus base Julia, Clang-Polly, icc, ifort, gfortran, and Eigen. If you would believe any code or compiler flags can be improved, would like to submit your own benchmarks, or have Julia code using LoopVectorization that you would like to be tested for performance regressions on a semi-regular basis, please feel file an issue or PR with the code sample.

## Examples
### Dot Product
<details>
 <summaryClick me! ></summary>
<p>

A simple example with a single loop is the dot product:
```julia
julia> using LoopVectorization, BenchmarkTools

julia> function mydot(a, b)
           s = 0.0
           @inbounds @simd for i ∈ eachindex(a,b)
               s += a[i]*b[i]
           end
           s
       end
mydot (generic function with 1 method)

julia> function mydotavx(a, b)
           s = 0.0
           @avx for i ∈ eachindex(a,b)
               s += a[i]*b[i]
           end
           s
       end
mydotavx (generic function with 1 method)

julia> a = rand(256); b = rand(256);

julia> @btime mydot($a, $b)
  12.273 ns (0 allocations: 0 bytes)
62.61049816874535

julia> @btime mydotavx($a, $b)
  11.618 ns (0 allocations: 0 bytes)
62.61049816874536

julia> a = rand(255); b = rand(255);

julia> @btime mydot($a, $b)
  36.539 ns (0 allocations: 0 bytes)
62.29537331565549

julia> @btime mydotavx($a, $b)
  11.739 ns (0 allocations: 0 bytes)
62.29537331565549
```

On most recent CPUs, the performance of the dot product is bounded by
the speed at which it can load data; most recent x86_64 CPUs can perform
two aligned loads and two fused multiply adds (`fma`) per clock cycle.
However, the dot product requires two loads per `fma`.

A self-dot function, on the otherhand, requires one load per fma:
```julia
julia> function myselfdot(a)
           s = 0.0
           @inbounds @simd for i ∈ eachindex(a)
               s += a[i]*a[i]
           end
           s
       end
myselfdot (generic function with 1 method)

julia> function myselfdotavx(a)
           s = 0.0
           @avx for i ∈ eachindex(a)
               s += a[i]*a[i]
           end
           s
       end
myselfdotavx (generic function with 1 method)

julia> a = rand(256);

julia> @btime myselfdot($a)
  8.578 ns (0 allocations: 0 bytes)
90.16636687132868

julia> @btime myselfdotavx($a)
  9.560 ns (0 allocations: 0 bytes)
90.16636687132868

julia> @btime myselfdot($b)
  28.923 ns (0 allocations: 0 bytes)
83.20114563267853

julia> @btime myselfdotavx($b)
  9.174 ns (0 allocations: 0 bytes)
83.20114563267856
```
For this reason, the `@avx` version is roughly twice as fast. The `@inbounds @simd` version, however, is not, because it runs into the problem of loop carried dependencies: to add `a[i]*b[i]` to `s_new = s_old + a[i-j]*b[i-j]`, we must have first finished calculating `s_new`, but -- while two `fma` instructions can be initiated per cycle -- they each take several clock cycles to complete.
For this reason, we need to unroll the operation to run several independent instances concurrently. The `@avx` macro models this cost to try and pick an optimal unroll factor.

</p>
</details>

### Matrix Multiply
<details>
 <summaryClick me! ></summary>
<p>

We can also vectorize fancier loops. A likely familiar example to dive into:
```julia
julia> function mygemm!(C, A, B)
           @inbounds @fastmath for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
               Cmn = zero(eltype(C))
               for k ∈ 1:size(A,2)
                   Cmn += A[m,k] * B[k,n]
               end
               C[m,n] = Cmn
           end
       end
mygemm! (generic function with 1 method)

julia> function mygemmavx!(C, A, B)
           @avx for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
               Cmn = zero(eltype(C))
               for k ∈ 1:size(A,2)
                   Cmn += A[m,k] * B[k,n]
               end
               C[m,n] = Cmn
           end
       end
mygemmavx! (generic function with 1 method)

julia> M, K, N = 72, 75, 71;

julia> C1 = Matrix{Float64}(undef, M, N); A = randn(M, K); B = randn(K, N);

julia> C2 = similar(C1); C3 = similar(C1);

julia> @benchmark mygemmavx!($C1, $A, $B)
BenchmarkTools.Trial: 
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     7.381 μs (0.00% GC)
  median time:      7.415 μs (0.00% GC)
  mean time:        7.432 μs (0.00% GC)
  maximum time:     15.444 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     4

julia> @benchmark mygemm!($C2, $A, $B)
BenchmarkTools.Trial: 
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     230.790 μs (0.00% GC)
  median time:      231.288 μs (0.00% GC)
  mean time:        231.882 μs (0.00% GC)
  maximum time:     275.460 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     1

julia> using LinearAlgebra, Test

julia> @test all(C1 .≈ C2)
Test Passed

julia> BLAS.set_num_threads(1); BLAS.vendor()
:mkl

julia> @benchmark mul!($C3, $A, $B)
BenchmarkTools.Trial: 
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     6.830 μs (0.00% GC)
  median time:      6.861 μs (0.00% GC)
  mean time:        6.869 μs (0.00% GC)
  maximum time:     15.125 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     5

julia> @test all(C1 .≈ C3)
Test Passed
```
It can produce a decent macro kernel.
In the future, I would like it to also model the cost of memory movement in the L1 and L2 cache, and use these to generate loops around the macro kernel following the work of [Low, et al. (2016)](http://www.cs.utexas.edu/users/flame/pubs/TOMS-BLIS-Analytical.pdf).

Until then, performance will degrade rapidly compared to BLAS as the size of the matrices increase. The advantage of the `@avx` macro, however, is that it is general. Not every operation is supported by BLAS.

For example, what if `A` were the outer product of two vectors?
<!-- ```julia -->


<!-- ``` -->

</p>
</details>

### Broadcasting
<details>
 <summaryClick me! ></summary>
<p>

Another example, a straightforward operation expressed well via broadcasting and `*ˡ` (which is typed `*\^l`), the lazy matrix multiplication operator:
```julia
julia> using LoopVectorization, LinearAlgebra, BenchmarkTools, Test; BLAS.set_num_threads(1)

julia> a = rand(48); B = rand(48, 51); c = rand(51); d = rand(49);

julia> X1 =        a .+ B * (c .+ d');

julia> X2 = @avx @. a + B *ˡ (c + d');

julia> @test X1 ≈ X2
Test Passed

julia> buf1 = Matrix{Float64}(undef, length(c), length(d));

julia> buf2 = similar(X1);

julia> @btime $X1 .= $a .+ mul!($buf2, $B, ($buf1 .= $c .+ $d'));
  3.499 μs (0 allocations: 0 bytes)

julia> @btime @avx @. $X2 = $a + $B *ˡ ($c + $d');
  3.289 μs (0 allocations: 0 bytes)

julia> @test X1 ≈ X2
Test Passed
```
The lazy matrix multiplication operator `*ˡ` escapes broadcasts and fuses, making it easy to write code that avoids intermediates. However, I would recomend always checking if splitting the operation into pieces, or at least isolating the matrix multiplication, increases performance. That will often be the case, especially if the matrices are large, where a separate multiplication can leverage BLAS (and perhaps take advantage of threads).
This may improve as the optimizations within LoopVectorization improve.

</p>
</details>


### Dealing with structs
<details>
 <summaryClick me! ></summary>
<p>

The key to the `@avx` macro's performance gains is leveraging knowledge of exactly how data like `Float64`s and `Int`s are handled by a CPU. As such, it is not strightforward to generalize the `@avx` macro to work on arrays containing structs such as `Matrix{Complex{Float64}}`. Instead, it is currently recommended that users wishing to apply `@avx` to arrays of structs use packages such as [StructArrays.jl](https://github.com/JuliaArrays/StructArrays.jl) which transform an array where each element is a struct into a struct where each element is an array. Using StructArrays.jl, we can write a matrix multiply (gemm) kernel that works on matrices of `Complex{Float64}`s and `Complex{Int}`s:
```julia 
using LoopVectorization, LinearAlgebra, StructArrays, BenchmarkTools, Test

BLAS.set_num_threads(1); @show BLAS.vendor()

const MatrixFInt64 = Union{Matrix{Float64}, Matrix{Int}}

function mul_avx!(C::MatrixFInt64, A::MatrixFInt64, B::MatrixFInt64)
    @avx for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
        Cmn = zero(eltype(C))
        for k ∈ 1:size(A,2)
            Cmn += A[m,k] * B[k,n]
        end
        C[m,n] = Cmn
    end
end

function mul_add_avx!(C::MatrixFInt64, A::MatrixFInt64, B::MatrixFInt64, factor=1)
    @avx for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
        ΔCmn = zero(eltype(C))
        for k ∈ 1:size(A,2)
            ΔCmn += A[m,k] * B[k,n]
        end
        C[m,n] += factor * ΔCmn
    end
end

const StructMatrixComplexFInt64 = Union{StructArray{ComplexF64,2}, StructArray{Complex{Int},2}}

function mul_avx!(C:: StructMatrixComplexFInt64, A::StructMatrixComplexFInt64, B::StructMatrixComplexFInt64)
    mul_avx!(    C.re, A.re, B.re)     # C.re = A.re * B.re
    mul_add_avx!(C.re, A.im, B.im, -1) # C.re = C.re - A.im * B.im
    mul_avx!(    C.im, A.re, B.im)     # C.im = A.re * B.im
    mul_add_avx!(C.im, A.im, B.re)     # C.im = C.im + A.im * B.re
end
```
this `mul_avx!` kernel can now accept `StructArray` matrices of complex numbers and multiply them efficiently:
```julia
julia> M, K, N = 56, 57, 58
(56, 57, 58)

julia> A  = StructArray(randn(ComplexF64, M, K));

julia> B  = StructArray(randn(ComplexF64, K, N));

julia> C1 = StructArray(Matrix{ComplexF64}(undef, M, N));

julia> C2 = collect(similar(C1));

julia> @btime mul_avx!($C1, $A, $B)
  13.634 μs (0 allocations: 0 bytes)

julia> @btime mul!(    $C2, $(collect(A)), $(collect(B))); # collect turns the StructArray into a regular Array
  14.007 μs (0 allocations: 0 bytes)

julia> @test C1 ≈ C2
Test Passed
```

Similar approaches can be taken to make kernels working with a variety of numeric struct types such as [dual numbers](https://github.com/JuliaDiff/DualNumbers.jl), [DoubleFloats](https://github.com/JuliaMath/DoubleFloats.jl), etc. 

</p>
</details>

## Packages using LoopVectorization

* [Gaius.jl](https://github.com/MasonProtter/Gaius.jl)
* [MaBLAS.jl](https://github.com/YingboMa/MaBLAS.jl)
* [PaddedMatrices.jl](https://github.com/chriselrod/PaddedMatrices.jl)
* [RecursiveFactorization.jl](https://github.com/YingboMa/RecursiveFactorization.jl)
* [Tullio.jl](https://github.com/mcabbott/Tullio.jl)

If you're using LoopVectorization, please feel free to file a PR adding yours to the list!

