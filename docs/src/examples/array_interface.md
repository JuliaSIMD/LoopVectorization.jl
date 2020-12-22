

# Array Interface

LoopVectorization uses [ArrayInterface.jl](https://github.com/SciML/ArrayInterface.jl) to describe the memory layout of arrays. By supporting the interface, `LoopVectorization` will be able to support compatible `AbstractArray` types.
[StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) and [HybridArrays.jl](https://github.com/mateuszbaran/HybridArrays.jl) are two example libraries providing array types supporting the interface.

`StaticArrays.SArray` itself is not compatible, because `LoopVectorization` needs access to a pointer. However,`StaticArrays.MArray`s are compatible. Loops featuring `StaticArrays.StaticArray` will result in a fall-back loop being executed
that wasn't optimized by `LoopVectorization`, but instead simply had `@inbounds @fastmath` applied to the loop. This can often still yield reasonable to good performance, saving you from having to write more than one version of the loop
to get good performance and correct behavior just because the array types happen to be different.

By supporting the interface, using `LoopVectorization` can simplify implementing many operations like matrix multiply while still getting good performance. For example, instead of [a few hundred lines of code](https://github.com/JuliaArrays/StaticArrays.jl/blob/0e431022954f0207eeb2c4f661b9f76936105c8a/src/matrix_multiply.jl#L4) to define matix multiplication in `StaticArrays`, one could simply write:
```julia
using StaticArrays, LoopVectorization

@inline function AmulB!(C, A, B)
    @avx for n ∈ axes(C,2), m ∈ axes(C,1)
        C_m_n = zero(eltype(C))
        for k ∈ axes(B,1)
            C_m_n += A[m,k] * B[k,n]
        end
        C[m,n] = C_m_n
    end
    C
end
@inline AmulB(A::MMatrix{M,K,T}, B::MMatrix{K,N,T}) where {M,K,N,T} = AmulB!(MMatrix{M,N,T}(undef), A, B)
@inline AmulB(A::SMatrix, B::SMatrix) = SMatrix(AmulB(MMatrix(A), MMatrix(B)))
```
Through converting back and fourth between `SMatrix` and `MMatrix`, we can still use `LoopVectorization` to implement `SMatrix` multiplication, and in most cases get better performance than the unrolled methods from the library. Unfortunately, it is still suboptimal because the compiler isn't able to elide the copying, but the temporaries are all stack-allocated, making the code
allocateion free. We can benchmark our simple implementation vs the `StaticArrays.SMatrix` and `StaticArrays.MMatrix` methods:
```julia
using BenchmarkTools, LinearAlgebra, DataFrames, VegaLite
BLAS.set_num_threads(1);

matdims(x::Integer) = (x, x, x)
matdims(x::NTuple{3}) = x
matflop(x::Integer) = 2x^3
matflop(x::NTuple{3}) = 2prod(x)

function runbenches(sr, ::Type{T}, fa = identity, fb = identity) where {T}
    bench_results = Matrix{Float64}(undef, length(sr), 4);
    for (i,s) ∈ enumerate(sr)
        M, K, N = matdims(s)
        A_m = @MMatrix rand(T, M, K)
        B_m = @MMatrix rand(T, K, N)
        A_s = Ref(SMatrix(A_m));
        B_s = Ref(SMatrix(B_m));
        C_s_s = fa(A_s[]) * fb(B_s[]);
        C_s_l = AmulB(fa(A_s[]), fb(B_s[]))
        C_m_s = similar(C_s_s); mul!(C_m_s, fa(A_m), fb(B_m));
        C_m_l = similar(C_s_s); AmulB!(C_m_l, fa(A_m), fb(B_m));
        @assert Array(C_s_s) ≈ Array(C_s_l) ≈ Array(C_m_s) ≈ Array(C_m_l) # Once upon a time Julia crashed on ≈ for large static arrays
        bench_results[i,1] = @belapsed $fa($A_s[]) * $fb($B_s[])
        bench_results[i,2] = @belapsed AmulB($fa($A_s[]), $fb($B_s[]))
        bench_results[i,3] = @belapsed mul!($C_m_s, $fa($A_m), $fb($B_m))
        bench_results[i,4] = @belapsed AmulB!($C_m_l, $fa($A_m), $fb($B_m))
        @show s, bench_results[i,:]
    end
    gflops = @. 1e-9 * matflop(sr) / bench_results
    array_type = append!(fill("Static", 2length(sr)), fill("Mutable", 2length(sr)))
    sa = fill("StaticArrays", length(sr)); lv = fill("LoopVectorization", length(sr));
    matmul_lib = vcat(sa, lv, sa, lv);
    sizes = reduce(vcat, (sr for _ ∈ 1:4))
	DataFrame(
	    Size = sizes, Time = vec(bench_results), GFLOPS = vec(gflops),
		ArrayType = array_type, MatmulLib = matmul_lib, MulType = array_type .* ' ' .* matmul_lib
	)
end

df = runbenches(1:24, Float64);
df |> @vlplot(:line, x = :Size, y = :GFLOPS, color = :MulType, height=640,width=960) |> save("sarraymatmul.svg")
```
This yields:
![sarray_benchmarks](../assets/sarraymatmul.png)
Our `AmulB!` for `MMatrix`es was the fastest at all sizes except `2`x`2`, where it lost out to `AmulB` for `SMatrix`, which in turn was faster than the hundreds of lines of
`StaticArray`s code at all sizes except `3`x`3`,  `5`x`5`, and  `6`x`6`.



Additionally, `HybridArrays.jl` can be used when we have a mix of dynamic and statically sized arrays. Maybe we want to multiply two matrices, where each element is a `3`x`3` matrix:
```julia
using HybridArrays, StaticArrays, LoopVectorization, BenchmarkTools

A_static = [@SMatrix(rand(3,3)) for i in 1:32, j in 1:32];
B_static = [@SMatrix(rand(3,3)) for i in 1:32, j in 1:32];
C_static = similar(A_static);

A_hybrid = HybridArray{Tuple{StaticArrays.Dynamic(),StaticArrays.Dynamic(),3,3}}(permutedims(reshape(reinterpret(Float64, A_static), (3,3,size(A_static)...)), (3,4,1,2)));
B_hybrid = HybridArray{Tuple{StaticArrays.Dynamic(),StaticArrays.Dynamic(),3,3}}(permutedims(reshape(reinterpret(Float64, B_static), (3,3,size(B_static)...)), (3,4,1,2)));
C_hybrid = HybridArray{Tuple{StaticArrays.Dynamic(),StaticArrays.Dynamic(),3,3}}(permutedims(reshape(reinterpret(Float64, C_static), (3,3,size(C_static)...)), (3,4,1,2)));

# C is M x N x I x J
# A is M x K x I x L
# B is K x N x L x J
function bmul!(C, A, B)
    @avx for n in axes(C,2), m in axes(C,1), j in axes(C,4), i in axes(C,3)
        C_m_n_j_i = zero(eltype(C))
        for k in axes(B,1), l in axes(B,3)
            C_m_n_j_i += A[m,k,i,l] * B[k,n,l,j]
        end
        C[m,n,i,j] = C_m_n_j_i
    end
end
```
This yields
```julia
julia> @benchmark bmul!($C_hybrid, $A_hybrid, $B_hybrid)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     15.550 μs (0.00% GC)
  median time:      15.663 μs (0.00% GC)
  mean time:        15.685 μs (0.00% GC)
  maximum time:     50.286 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     1
  
julia> @benchmark mul!($C_static, $A_static, $B_static)
BenchmarkTools.Trial:
  memory estimate:  336 bytes
  allocs estimate:  6
  --------------
  minimum time:     277.736 μs (0.00% GC)
  median time:      278.035 μs (0.00% GC)
  mean time:        278.310 μs (0.00% GC)
  maximum time:     299.259 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     1

julia> all(I -> C_hybrid[Tuple(I)[1],Tuple(I)[2],:,:] ≈ C_static[I], CartesianIndices(C_static))
true

julia> length(C_hybrid) * size(B_hybrid,1) * size(B_hybrid,3) * 2e-9 / 15.55e-6 # GFLOPS loops + hybrid arrays
113.79241157556271

julia> length(C_hybrid) * size(B_hybrid,1) * size(B_hybrid,3) * 2e-9 / 277.736e-6 # GFLOPS LinearAlgebra.mul! + StaticArrays
6.371057407034018
```
When using `LoopVectorization` + `HybridArrays`, you may often find that you often get the best performance when the leading dimensions are either an even multiple of 8, or relatively large.
This will often mean not leading with a small static dimension, which is commonly best practice when not using `LoopVectorization`.

If you happen to like tensor operations such as from this last example, you're also strongly encouraged to check out [Tullio.jl](https://github.com/mcabbott/Tullio.jl) which provides index-notation that is both much more convenient and much less error-prone than writing out loops, and uses both `LoopVectorization` (if you `using LoopVectorization` before `@tullio`) as well as multiple threads to maximize performance.



