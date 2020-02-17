# Convenient Vectorized Functions

## vmap

This is simply a vectorized `map` function.


## vfilter

This function requires LLVM 7 or greater, and is only likly to give better performance if your CPU has AVX512. This is because it uses the compressed store intrinsic, which was added in LLVM 7. AVX512 provides a corresponding instruction, making the operation fast, while other instruction sets must emulate it, and thus are likely to get similar performance with `LoopVectorization.vfilter` as they do from `Base.filter`.

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




