# LoopVectorization

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://chriselrod.github.io/LoopVectorization.jl/stable)
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://chriselrod.github.io/LoopVectorization.jl/latest)
[![Build Status](https://travis-ci.com/chriselrod/LoopVectorization.jl.svg?branch=master)](https://travis-ci.com/chriselrod/LoopVectorization.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/chriselrod/LoopVectorization.jl?svg=true)](https://ci.appveyor.com/project/chriselrod/LoopVectorization-jl)
[![Codecov](https://codecov.io/gh/chriselrod/LoopVectorization.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/chriselrod/LoopVectorization.jl)

## Installation
```
using Pkg
Pkg.add("LoopVectorization")
```


## Usage

This library provides the `@avx` macro, which may be used to prefix a `for` loop or broadcast statement.
It then tries to vectorize the loop to improve runtime performance.

The macro assumes that loop iterations can be reordered. It also currently supports simple nested loops, where loop bounds of inner loops are constant across iterations of the outer loop, and only a single loop at each level of noop lest. These limitations should be removed in a future version.

A simple example with a single loop is the dot product:
```julia
using LoopVectorization, BenchmarkTools
function mydot(a, b)
    s = 0.0
    @inbounds @simd for i ∈ eachindex(a,b)
        s += a[i]*b[i]
    end
    s
end
function mydotavx(a, b)
    s = 0.0
    @avx for i ∈ eachindex(a,b)
        s += a[i]*b[i]
    end
    s
end
a = rand(256); b = rand(256);
@btime mydot($a, $b)
@btime mydotavx($a, $b)
a = rand(43); b = rand(43);
@btime mydot($a, $b)
@btime mydotavx($a, $b)
```

On most recent CPUs, the performance of the dot product is bounded by
the speed at which it can load data; most recent x86_64 CPUs can perform
two aligned loads and two fused multiply adds (`fma`) per clock cycle.
However, the dot product requires two loads per `fma`.

A self-dot function, on the otherhand, requires one load per fma:
```julia
function myselfdot(a)
    s = 0.0
    @inbounds @simd for i ∈ eachindex(a)
        s += a[i]*a[i]
    end
    s
end
function myselfdotavx(a)
    s = 0.0
    @avx for i ∈ eachindex(a)
        s += a[i]*a[i]
    end
    s
end
a = rand(256);
@btime myselfdotavx($a)
@btime myselfdot($a)
@btime myselfdotavx($b)
@btime myselfdot($b)
```
For this reason, the `@avx` version is roughly twice as fast. The `@inbounds @simd` version, however, is not, because it runs into the problem of loop carried dependencies: to add `a[i]*b[i]` to `s_new = s_old + a[i-j]*b[i-j]`, we must have first finished calculating `s_new`, but -- while two `fma` instructions can be initiated per cycle -- they each take several clock cycles to complete.
For this reason, we need to unroll the operation to run several independent instances concurrently. The `@avx` macro models this cost to try and pick an optimal unroll factor.

Note that 14 and 12 nm Ryzen chips can only do 1 full width `fma` per clock cycle (and 2 loads), so they should see similar performance with the dot and selfdot. I haven't verified this, but would like to hear from anyone who can.


We can also vectorize fancier loops. A likely familiar example to dive into:
```julia
function mygemm!(C, A, B)
    @inbounds for i ∈ 1:size(A,1), j ∈ 1:size(B,2)
        Cᵢⱼ = 0.0
        @fastmath for k ∈ 1:size(A,2)
            Cᵢⱼ += A[i,k] * B[k,j]
        end
        C[i,j] = Cᵢⱼ
    end
end
function mygemmavx!(C, A, B)
    @avx for i ∈ 1:size(A,1), j ∈ 1:size(B,2)
        Cᵢⱼ = 0.0
        for k ∈ 1:size(A,2)
            Cᵢⱼ += A[i,k] * B[k,j]
        end
        C[i,j] = Cᵢⱼ
    end
end
M, K, N = 72, 75, 71;
C1 = Matrix{Float64}(undef, M, N); A = randn(M, K); B = randn(K, N);
C2 = similar(C1); C3 = similar(C1); 
@btime mygemmavx!($C1, $A, $B)
@btime mygemm!($C2, $A, $B)
using LinearAlgebra, Test
@test all(C1 .≈ C2)
BLAS.set_num_threads(1); BLAS.vendor()
@btime mul!($C3, $A, $B)
@test all(C1 .≈ C3)
```
It can produce a decent macro kernel.
In the future, I would like it to also model the cost of memory movement in the L1 and L2 cache, and use these to generate loops around the macro kernel following the work of [Low, et al. (2016)](http://www.cs.utexas.edu/users/flame/pubs/TOMS-BLIS-Analytical.pdf).

Until then, performance will degrade rapidly compared to BLAS as the size of the matrices increase. The advantage of the `@avx` macro, however, is that it is general. Not every operation is supported by BLAS.

For example, what if `A` were the outter product of two vectors?
<!-- ```julia -->


<!-- ``` -->


Another example, a straightforward operation expressed well via broadcasting:
```julia
a = rand(37); B = rand(37, 47); c = rand(47); c′ = c';

d1 =      @. a + B * c′;
d2 = @avx @. a + B * c′;

@test all(d1 .≈ d2)

@time @.      $d1 = $a + $B * $c′;
@time @avx @. $d2 = $a + $B * $c′;
@test all(d1 .≈ d2)
```
can be optimized in a similar manner to BLAS, albeit to a much smaller degree because the naive version already benefits from vectorization (unlike the naive BLAS).


You can also use `\ast` for lazy matrix multiplication that can fuse with broadcasts. `.\ast` behaves similarly, espcaping the broadcast (it is not applied elementwise). This allows you to use `@.` and fuse all the loops, even if the arguments to `\ast` are themselves broadcasted objects. However, it will often be the case that creating an intermediary is faster. I would recomend always checking if splitting the operation into pieces, or at least isolating the matrix multiplication, increases performance. That will often be the case, especially if the matrices are large, where a separate multiplication can leverage BLAS (and perhaps take advantage of threads).

At small sizes, this can be fast.
```julia

```




