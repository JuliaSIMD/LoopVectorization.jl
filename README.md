# LoopVectorization

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://chriselrod.github.io/LoopVectorization.jl/stable)
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://chriselrod.github.io/LoopVectorization.jl/latest)
[![Build Status](https://github.com/chriselrod/LoopVectorization.jl/workflows/CI/badge.svg)](https://github.com/chriselrod/LoopVectorization.jl/actions?query=workflow%3ACI)
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
4. Loops increment by 1 on each iteration, e.g. `1:2:N` is not supported at the moment. (This requirement will eventually be lifted.)

## Usage

This library provides the `@avx` macro, which may be used to prefix a `for` loop or broadcast statement.
It then tries to vectorize the loop to improve runtime performance.

The macro assumes that loop iterations can be reordered. It also currently supports simple nested loops, where loop bounds of inner loops are constant across iterations of the outer loop, and only a single loop at each level of noop lest. These limitations should be removed in a future version.

## Benchmarks

Please see the documentation for benchmarks versus base Julia, Clang, icc, ifort, gfortran, and Eigen. If you would believe any code or compiler flags can be improved, would like to submit your own benchmarks, or have Julia code using LoopVectorization that you would like to be tested for performance regressions on a semi-regular basis, please feel file an issue or PR with the code sample.

## Examples
### Dot Product

LLVM/Julia by default generate essentially optimal code for a primary vectorized part of this loop. In many cases -- such as the dot product -- this vectorized part of the loop computes 4*SIMD-vector-width iterations at a time.
On the CPU I'm running these benchmarks on with `Float64` data, the SIMD-vector-width is 8, meaning it will compute 32 iterations at a time.
However, LLVM is very slow at handling the tails, `length(iterations) % 32`. For this reason, [in benchmark plots](https://chriselrod.github.io/LoopVectorization.jl/latest/examples/dot_product/) you can see performance drop as the size of the remainder increases.

For simple loops like a dot product, LoopVectorization.jl's most important optimization is to handle these tails more efficiently:
<details>
 <summaryClick me! ></summary>
<p>
 
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
  12.220 ns (0 allocations: 0 bytes)
62.67140864639772

julia> @btime mydotavx($a, $b) # performance is similar
  12.104 ns (0 allocations: 0 bytes)
62.67140864639772

julia> a = rand(255); b = rand(255);

julia> @btime mydot($a, $b) # with loops shorter by 1, the remainder is now 32, and it is slow
  36.530 ns (0 allocations: 0 bytes)
61.25056244423578

julia> @btime mydotavx($a, $b) # performance remains mostly unchanged.
  12.226 ns (0 allocations: 0 bytes)
61.250562444235776
```
</p>
</details>



### Matrix Multiply
<details>
 <summaryClick me! ></summary>
<p>

We can also vectorize fancier loops. A likely familiar example to dive into:
```julia
julia> function mygemm!(C, A, B)
           @inbounds @fastmath for m ∈ axes(A,1), n ∈ axes(B,2)
               Cmn = zero(eltype(C))
               for k ∈ axes(A,2)
                   Cmn += A[m,k] * B[k,n]
               end
               C[m,n] = Cmn
           end
       end
mygemm! (generic function with 1 method)

julia> function mygemmavx!(C, A, B)
           @avx for m ∈ axes(A,1), n ∈ axes(B,2)
               Cmn = zero(eltype(C))
               for k ∈ axes(A,2)
                   Cmn += A[m,k] * B[k,n]
               end
               C[m,n] = Cmn
           end
       end
mygemmavx! (generic function with 1 method)

julia> M, K, N = 191, 189, 171;

julia> C1 = Matrix{Float64}(undef, M, N); A = randn(M, K); B = randn(K, N);

julia> C2 = similar(C1); C3 = similar(C1);

julia> @benchmark mygemmavx!($C1, $A, $B)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     111.722 μs (0.00% GC)
  median time:      112.528 μs (0.00% GC)
  mean time:        112.673 μs (0.00% GC)
  maximum time:     189.400 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     1

julia> @benchmark mygemm!($C2, $A, $B)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     4.891 ms (0.00% GC)
  median time:      4.899 ms (0.00% GC)
  mean time:        4.899 ms (0.00% GC)
  maximum time:     5.049 ms (0.00% GC)
  --------------
  samples:          1021
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
  minimum time:     117.221 μs (0.00% GC)
  median time:      118.745 μs (0.00% GC)
  mean time:        118.892 μs (0.00% GC)
  maximum time:     193.826 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     1

julia> @test all(C1 .≈ C3)
Test Passed

julia> 2e-9M*K*N ./ (111.722e-6, 4.891e-3, 117.221e-6)
(110.50516460500171, 2.524199141279902, 105.32121377568868)
```
It can produce a good macro kernel. An implementation of matrix multiplication able to handle large matrices would need to be perform blocking and packing of arrays to prevent the operations from being memory bottle-necked.
Some day, LoopVectorization may itself may try to model the costs of memory movement in the L1 and L2 cache, and use these to generate loops around the macro kernel following the work of [Low, et al. (2016)](http://www.cs.utexas.edu/users/flame/pubs/TOMS-BLIS-Analytical.pdf).

But for now, you should view it as a tool for generating efficient computational kernels, leaving tasks of parallelization and cache efficiency to you.


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

julia> A = rand(5,77); B = rand(77, 51); C = rand(51,49); D = rand(49,51);

julia> X1 =      view(A,1,:) .+ B  *  (C .+ D');

julia> X2 = @avx view(A,1,:) .+ B .*ˡ (C .+ D');

julia> @test X1 ≈ X2
Test Passed

julia> buf1 = Matrix{Float64}(undef, size(C,1), size(C,2));

julia> buf2 = similar(X1);

julia> @btime $X1 .= view($A,1,:) .+ mul!($buf2, $B, ($buf1 .= $C .+ $D'));
  9.188 μs (0 allocations: 0 bytes)

julia> @btime @avx $X2 .= view($A,1,:) .+ $B .*ˡ ($C .+ $D');
  6.751 μs (0 allocations: 0 bytes)

julia> @test X1 ≈ X2
Test Passed

julia> AmulBtest!(X1, B, C, D, view(A,1,:))

julia> AmulBtest2!(X2, B, C, D, view(A,1,:))

julia> @test X1 ≈ X2
Test Passed
```
The lazy matrix multiplication operator `*ˡ` escapes broadcasts and fuses, making it easy to write code that avoids intermediates. However, I would recomend always checking if splitting the operation into pieces, or at least isolating the matrix multiplication, increases performance. That will often be the case, especially if the matrices are large, where a separate multiplication can leverage BLAS (and perhaps take advantage of threads).
This may improve as the optimizations within LoopVectorization improve.

Note that loops will be faster than broadcasting in general. This is because the behavior of broadcasts is determined by runtime information (i.e., dimensions other than the leading dimension of size `1` will be broadcasted; it is not known which these will be at compile time).
```julia
julia> function AmulBtest!(C,A,Bk,Bn,d)
          @avx for m ∈ axes(A,1), n ∈ axes(Bk,2)
             ΔCₘₙ = zero(eltype(C))
             for k ∈ axes(A,2)
                ΔCₘₙ += A[m,k] * (Bk[k,n] + Bn[n,k])
             end
             C[m,n] = ΔCₘₙ + d[m]
          end
       end
AmulBtest! (generic function with 1 method)

julia> AmulBtest!(X2, B, C, D, view(A,1,:))

julia> @test X1 ≈ X2
Test Passed

julia> @benchmark AmulBtest!($X2, $B, $C, $D, view($A,1,:))
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     5.793 μs (0.00% GC)
  median time:      5.816 μs (0.00% GC)
  mean time:        5.824 μs (0.00% GC)
  maximum time:     14.234 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     6
```

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
  13.525 μs (0 allocations: 0 bytes)

julia> @btime mul!(    $C2, $(collect(A)), $(collect(B))); # collect turns the StructArray into a regular Array
  14.003 μs (0 allocations: 0 bytes)

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
* [SnpArrays.jl](https://github.com/OpenMendel/SnpArrays.jl)
* [Tullio.jl](https://github.com/mcabbott/Tullio.jl)

If you're using LoopVectorization, please feel free to file a PR adding yours to the list!

