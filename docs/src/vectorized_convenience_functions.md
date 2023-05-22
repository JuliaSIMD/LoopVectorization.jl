# Convenient Vectorized Functions

## vmap

This is simply a vectorized `map` function.

## vmapnt and vmapntt

These are like `vmap`, but use non-temporal (streaming) stores into the destination, to avoid polluting the cache. Likely to yield a performance increase if you wont be reading the values soon.
```julia
julia> using LoopVectorization, BenchmarkTools

julia> f(x,y) = exp(-0.5abs2(x - y))
f (generic function with 1 method)

julia> x = rand(10^8); y = rand(10^8); z = similar(x);

julia> @benchmark map!(f, $z, $x, $y)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     442.614 ms (0.00% GC)
  median time:      443.750 ms (0.00% GC)
  mean time:        443.664 ms (0.00% GC)
  maximum time:     444.730 ms (0.00% GC)
  --------------
  samples:          12
  evals/sample:     1

julia> @benchmark vmap!(f, $z, $x, $y)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     177.257 ms (0.00% GC)
  median time:      177.380 ms (0.00% GC)
  mean time:        177.423 ms (0.00% GC)
  maximum time:     177.956 ms (0.00% GC)
  --------------
  samples:          29
  evals/sample:     1

julia> @benchmark vmapnt!(f, $z, $x, $y)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     143.521 ms (0.00% GC)
  median time:      143.639 ms (0.00% GC)
  mean time:        143.645 ms (0.00% GC)
  maximum time:     143.821 ms (0.00% GC)
  --------------
  samples:          35
  evals/sample:     1

julia> Threads.nthreads()
36

julia> @benchmark vmapntt!(f, $z, $x, $y)
BenchmarkTools.Trial:
  memory estimate:  25.69 KiB
  allocs estimate:  183
  --------------
  minimum time:     30.065 ms (0.00% GC)
  median time:      30.130 ms (0.00% GC)
  mean time:        30.146 ms (0.00% GC)
  maximum time:     31.277 ms (0.00% GC)
  --------------
  samples:          166
  evals/sample:     1
```

## vfilter

This function requires LLVM 7 or greater, and is only likely to give better performance if your CPU has AVX512. This is because it uses the compressed store intrinsic, which was added in LLVM 7. AVX512 provides a corresponding instruction, making the operation fast, while other instruction sets must emulate it, and thus are likely to get similar performance with `LoopVectorization.vfilter` as they do from `Base.filter`.

```julia
julia> using LoopVectorization, BenchmarkTools

julia> x = rand(997);

julia> y1 = filter(a -> a > 0.7, x);

julia> y2 = vfilter(a -> a > 0.7, x);

julia> y1 == y2
true

julia> @benchmark filter(a -> a > 0.7, $x)
BenchmarkTools.Trial:
  memory estimate:  7.94 KiB
  allocs estimate:  1
  --------------
  minimum time:     955.389 ns (0.00% GC)
  median time:      1.050 μs (0.00% GC)
  mean time:        1.191 μs (9.72% GC)
  maximum time:     82.799 μs (94.92% GC)
  --------------
  samples:          10000
  evals/sample:     18

julia> @benchmark vfilter(a -> a > 0.7, $x)
BenchmarkTools.Trial:
  memory estimate:  7.94 KiB
  allocs estimate:  1
  --------------
  minimum time:     477.487 ns (0.00% GC)
  median time:      575.166 ns (0.00% GC)
  mean time:        711.526 ns (17.87% GC)
  maximum time:     9.257 μs (79.17% GC)
  --------------
  samples:          10000
  evals/sample:     193
```

## vmapreduce

Vectorized version of `mapreduce`. `vmapreduce(f, op, a, b, c)` applies `f(a[i], b[i], c[i])` for `i in eachindex(a,b,c)`, reducing the results to a scalar with `op`.

```julia
julia> using LoopVectorization, BenchmarkTools

julia> x = rand(127); y = rand(127);

julia> @btime vmapreduce(hypot, +, $x, $y)
  191.420 ns (0 allocations: 0 bytes)
96.75538300513509

julia> @btime mapreduce(hypot, +, $x, $y)
  1.777 μs (5 allocations: 1.25 KiB)
96.75538300513509
```

## vsum

Vectorized version of `sum`. `vsum(f, a)` applies `f(a[i])` for `i in eachindex(a)`, then sums the results.

```julia
julia> using LoopVectorization, BenchmarkTools

julia> x = rand(127);

julia> @btime vsum(hypot, $x)
  12.095 ns (0 allocations: 0 bytes)
66.65246070098374

julia> @btime sum(hypot, $x)
  16.992 ns (0 allocations: 0 bytes)
66.65246070098372
```


