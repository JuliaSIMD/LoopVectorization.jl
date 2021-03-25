## Getting Started

To install `LoopVectorization.jl`, simply use the package and `] add LoopVectorization`, or
```julia
using Pkg
Pkg.add("LoopVectorization")
```

Currently LoopVectorization only supports rectangular iteration spaces, although I plan on extending it to triangular and ragged iteration spaces in the future.
This means that if you nest multiple loops, the number of iterations of the inner loops shouldn't be a function of the outer loops. For example,
```julia
using LoopVectorization 

function mvp(P, basis, coeffs::Vector{T}) where {T}
    C = length(coeffs)
    A = size(P, 1)
    p = zero(T)
    @avx for c ∈ 1:C
        pc = coeffs[c]
        for a = 1:A
            pc *= P[a, basis[a, c]]
        end
        p += pc
    end
    p
end

maxdeg = 20; nbasis = 1_000; dim = 15;
r = 1:maxdeg+1
basis = rand(r, (dim, nbasis));
coeffs = rand(T, nbasis);
P = rand(T, dim, maxdeg+1);

mvp(P, basis, coeffs)
```

Aside from loops, `LoopVectorization.jl` also supports broadcasting.
!!! danger
    Broadcasting an `Array` `A` when `size(A,1) == 1` is NOT SUPPORTED, unless this is known at compile time (e.g., broadcasting a transposed vector is fine). Otherwise, you will probably crash Julia.

```julia
julia> using LoopVectorization, BenchmarkTools

julia> M, K, N = 47, 73, 7;

julia> A = rand(M, K);

julia> b = rand(K);

julia> c = rand(M);

julia> d = rand(1,K,N);

julia> #You can use a LowDimArray when you have a leading dimension of size 1.
       ldad = LowDimArray{(false,true,true)}(d);

julia> E1 = Array{Float64}(undef, M, K, N);

julia> E2 = similar(E1);

julia> @benchmark @. $E1 = exp($A - $b' +    $d) * $c
BenchmarkTools.Trial: 
  memory estimate:  112 bytes
  allocs estimate:  5
  --------------
  minimum time:     224.142 μs (0.00% GC)
  median time:      225.773 μs (0.00% GC)
  mean time:        229.146 μs (0.00% GC)
  maximum time:     289.601 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     1

julia> @benchmark @avx @. $E2 = exp($A - $b' + $ldad) * $c
BenchmarkTools.Trial: 
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     19.666 μs (0.00% GC)
  median time:      19.737 μs (0.00% GC)
  mean time:        19.759 μs (0.00% GC)
  maximum time:     29.906 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     1

julia> E1 ≈ E2
true
```





