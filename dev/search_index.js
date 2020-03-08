var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#LoopVectorization.jl-1",
    "page": "Home",
    "title": "LoopVectorization.jl",
    "category": "section",
    "text": "This documentation is for LoopVectorization.jl. Please file an issue if you run into any problems."
},

{
    "location": "#Manual-Outline-1",
    "page": "Home",
    "title": "Manual Outline",
    "category": "section",
    "text": "Pages = [\n    \"getting_started.md\",\n    \"examples/matrix_multiplication.md\",\n    \"examples/matrix_vector_ops.md\",\n    \"examples/dot_product.md\",\n    \"examples/sum_of_squared_error.md\",\n    \"vectorized_convenience_functions.md\",\n    \"future_work.md\",\n	\"devdocs/overview.md\",\n	\"devdocs/loopset_structure.md\",\n	\"devdocs/constructing_loopsets.md\",\n	\"devdocs/evaluating_loops.md\",\n	\"devdocs/lowering.md\"\n]\nDepth = 1"
},

{
    "location": "getting_started/#",
    "page": "Getting Started",
    "title": "Getting Started",
    "category": "page",
    "text": ""
},

{
    "location": "getting_started/#Getting-Started-1",
    "page": "Getting Started",
    "title": "Getting Started",
    "category": "section",
    "text": "To install LoopVectorization.jl, simply use the package and ] add LoopVectorization, orusing Pkg\nPkg.add(\"LoopVectorization\")Currently LoopVectorization only supports rectangular iteration spaces, although I plan on extending it to triangular and ragged iteration spaces in the future. This means that if you nest multiple loops, the number of iterations of the inner loops shouldn\'t be a function of the outer loops. For example,using LoopVectorization \n\nfunction mvp(P, basis, coeffs::Vector{T}) where {T}\n    C = length(coeffs)\n    A = size(P, 1)\n    p = zero(T)\n    @avx for c ∈ 1:C\n        pc = coeffs[c]\n        for a = 1:A\n            pc *= P[a, basis[a, c]]\n        end\n        p += pc\n    end\n	p\nend\n\nmaxdeg = 20; nbasis = 1_000; dim = 15;\nr = 1:maxdeg+1\nbasis = rand(r, (dim, nbasis));\ncoeffs = rand(T, nbasis);\nP = rand(T, dim, maxdeg+1);\n\nmvp(P, basis, coeffs)Aside from loops, LoopVectorization.jl also supports broadcasting.danger: Danger\nBroadcasting an Array A when size(A,1) == 1 is NOT SUPPORTED, unless this is known at compile time (e.g., broadcasting a transposed vector is fine). Otherwise, you will probably crash Julia.julia> using LoopVectorization, BenchmarkTools\n\njulia> M, K, N = 47, 73, 7;\n\njulia> A = rand(M, K);\n\njulia> b = rand(K);\n\njulia> c = rand(M);\n\njulia> d = rand(1,K,N);\n\njulia> #You can use a LowDimArray when you have a leading dimension of size 1.\n       ldad = LowDimArray{(false,true,true)}(d);\n\njulia> E1 = Array{Float64}(undef, M, K, N);\n\njulia> E2 = similar(E1);\n\njulia> @benchmark @. $E1 = exp($A - $b\' +    $d) * $c\nBenchmarkTools.Trial: \n  memory estimate:  112 bytes\n  allocs estimate:  5\n  --------------\n  minimum time:     224.142 μs (0.00% GC)\n  median time:      225.773 μs (0.00% GC)\n  mean time:        229.146 μs (0.00% GC)\n  maximum time:     289.601 μs (0.00% GC)\n  --------------\n  samples:          10000\n  evals/sample:     1\n\njulia> @benchmark @avx @. $E2 = exp($A - $b\' + $ldad) * $c\nBenchmarkTools.Trial: \n  memory estimate:  0 bytes\n  allocs estimate:  0\n  --------------\n  minimum time:     19.666 μs (0.00% GC)\n  median time:      19.737 μs (0.00% GC)\n  mean time:        19.759 μs (0.00% GC)\n  maximum time:     29.906 μs (0.00% GC)\n  --------------\n  samples:          10000\n  evals/sample:     1\n\njulia> E1 ≈ E2\ntrue"
},

{
    "location": "examples/matrix_multiplication/#",
    "page": "Matrix Multiplication",
    "title": "Matrix Multiplication",
    "category": "page",
    "text": ""
},

{
    "location": "examples/matrix_multiplication/#Matrix-Multiplication-1",
    "page": "Matrix Multiplication",
    "title": "Matrix Multiplication",
    "category": "section",
    "text": "One of the friendliest problems for vectorization is matrix multiplication. Given M × K matrix 𝐀, and K × N matrix 𝐁, multiplying them is like performing M * N dot products of length K. We need M*K + K*N + M*N total memory, but M*K*N multiplications and additions, so there\'s a lot more arithmetic we can do relative to the memory needed.LoopVectorization currently doesn\'t do any memory-modeling or memory-based optimizations, so it will still run into problems as the size of matrices increases. But at smaller sizes, it\'s capable of achieving a healthy percent of potential GFLOPS. We can write a single function:@inline function A_mul_B!(𝐂, 𝐀, 𝐁)\n    @avx for m ∈ 1:size(𝐀,1), n ∈ 1:size(𝐁,2)\n        𝐂ₘₙ = zero(eltype(𝐂))\n        for k ∈ 1:size(𝐀,2)\n            𝐂ₘₙ += 𝐀[m,k] * 𝐁[k,n]\n        end\n        𝐂[m,n] = 𝐂ₘₙ\n    end\nendand this can handle all transposed/not-tranposed permutations. LoopVectorization will change loop orders and strategy as appropriate based on the types of the input matrices. For each of the others, I wrote separate functions to handle each case.  Letting all three matrices be square and Size x Size, we attain the following benchmark results:(Image: AmulB) This is classic GEMM, 𝐂 = 𝐀 * 𝐁. GFortran\'s intrinsic matmul function does fairly well, as does Clang-Polly, because Polly is designed to specfically recognize GEMM-like loops and optimize them. But all the compilers are well behind LoopVectorization here, which falls behind MKL\'s gemm beyond 56 × 56. The problem imposed by alignment is also striking: performance is much higher when the sizes are integer multiplies of 8. Padding arrays so that each column is aligned regardless of the number of rows can thus be very profitable. PaddedMatrices.jl offers just such arrays in Julia. I believe that is also what the -pad compiler flag does when using Intel\'s compilers.(Image: AmulBt) The optimal pattern for 𝐂 = 𝐀 * 𝐁ᵀ is almost identical to that for 𝐂 = 𝐀 * 𝐁. Yet, as soon as we deviate slightly from the gemm-loops, Clang-Polly\'s pattern matching doesn\'t identify the loops, and it fails to optimize at all. LoopVectorization and the three Intel-compiled versions all do well. Similarly, it seems that gfortran\'s matmul instrinsic function has only been optimized for the non-transposed case, so that the simple loops actually performed better here.ifort did equally well whethor or not 𝐁 was transposed, while LoopVectorization\'s performance degraded slightly faster as a function of size in the transposed case, because strides between memory accesses are larger when 𝐁 is transposed. But it still performed best of all the compiled loops over this size range, only losing to MKL. icc interestingly does better when it is transposed.GEMM is easiest when the matrix 𝐀 is not tranposed (assuming column-major memory layouts), because then you can sum up columns of 𝐀 to store into 𝐂. If 𝐀 were transposed, then we cannot efficiently load contiguous elements from 𝐀 that can best stored directly in 𝐂. So for 𝐂 = 𝐀ᵀ * 𝐁, contiguous vectors along the k-loop have to be reduced, adding some overhead. (Image: AtmulB) I am not sure what exactly MKL is doing, but it is able to maintain its performance. I suspect it may be able to efficiently transpose and pack the arrays.LoopVectorization and both ifort versions have similar performance, while icc isn\'t too far behind. Clang-Polly is far in last.When both 𝐀 and 𝐁 are transposed, the loops become rather awkward to vectorize. (Image: AtmulBt)LoopVectorization and MKL managed to do about as well as normal. The ifort and gfortran intrinsics also do fairly well here, perhaps because it can be expressed as:C = transpose(matmul(B, A))The ifort-loop version also did fairly well. The other loop versions did poorly."
},

{
    "location": "examples/matrix_vector_ops/#",
    "page": "Matrix-Vector Operations",
    "title": "Matrix-Vector Operations",
    "category": "page",
    "text": ""
},

{
    "location": "examples/matrix_vector_ops/#Matrix-Vector-Operations-1",
    "page": "Matrix-Vector Operations",
    "title": "Matrix-Vector Operations",
    "category": "section",
    "text": "Here I\'ll discuss a variety of Matrix-vector operations, naturally starting with matrix-vector multiplication.@inline function jgemvavx!(𝐲, 𝐀, 𝐱)\n    @avx for i ∈ eachindex(𝐲)\n        𝐲ᵢ = zero(eltype(𝐲))\n        for j ∈ eachindex(𝐱)\n            𝐲ᵢ += 𝐀[i,j] * 𝐱[j]\n        end\n        𝐲[i] = 𝐲ᵢ\n    end\nendUsing a square Size x Size matrix 𝐀, we find the following results. (Image: Amulvb)If 𝐀 is transposed, or equivalently, if we\'re instead computing x * 𝐀: (Image: Atmulvb)Finally, the three-argument dot product y\' * 𝐀 * x: (Image: dot3)The performance impact of alignment is dramatic here."
},

{
    "location": "examples/dot_product/#",
    "page": "Dot Products",
    "title": "Dot Products",
    "category": "page",
    "text": ""
},

{
    "location": "examples/dot_product/#Dot-Products-1",
    "page": "Dot Products",
    "title": "Dot Products",
    "category": "section",
    "text": "Dot products are simple the sum of the elementwise products of two vectors. They can be interpreted geometrically as (after normalizing by dividing by the norms of both vectors) yielding the cosine of the angle between them. This makes them useful for, for example, the No-U-Turn sampler to check for u-turns (i.e., to check if the current momentum is no longer in the same direction as the change in position).function jdotavx(a, b)\n    s = zero(eltype(a))\n    @avx for i ∈ eachindex(a, b)\n        s += a[i] * b[i]\n    end\n    s\nendTo execute the loop using SIMD (Single Instruction Multiple Data) instructions, you have to unroll the loop. Rather than evaluating the loop as written – adding element-wise products to a single accumulator one after the other – you can multiply short vectors loaded from a and b and add their results to a vector of accumulators. Most modern CPUs found in laptops or desktops have the AVX instruction set, which allows them to operate on 256 bit vectors – meaning the vectors can hold 4 double precision (64 bit) floats. Some have the AVX512 instruction set, which increases the vector size to 512 bits, and also adds many new instructions that make vectorizing easier. To be gemeral across CPUs and data types, I\'ll refer to the number of elements in the vectors with W. I\'ll also refer to unrolling a loop by a factor of W and loading vectors from it as \"vectorizing\" that loop.In addition to vectorizing the loop, we\'ll want to unroll it by an additional factor. Given that we have single or double precision floating point elements, most recent CPU cores have a potential throughput of two fused multiply-add (fma) instructions per clock cycle. However, it actually takes about four clock cycles for any of these instructions to execute; a single core is able to work on several in parallel.This means that if we used a single vector to accumulate a product, we\'d only get to perform one fused multiply add every four clock cycles: we\'d have to wait for one instruction to complete before starting the next. By using extra accumulation vectors, we can break up this dependency chain. If we had 8 accumulators, then theoretically we could perform two per clock cycle, and after the 4th cycle, our first operations are done so that we can reuse them.However, there is another bottle neck: we can only perform 2 aligned loads per clock cycle (or 1 unaligned load). [Alignment here means with respect to a memory address boundary, if your vectors are 256 bits, then a load/store is aligned if it is with respect to a memory address that is an integer multiple of 32 bytes (256 bits = 32 bytes).] Thus, in 4 clock cycles, we can do up to 8 loads. But each fma requires 2 loads, meaning we are limited to 4 of them per 4 clock cyles, and any unrolling beyond 4 gives us no benefit.Double precision benchmarks pitting Julia\'s builtin dot product (named MKL here), and code compiled with a variety of compilers: (Image: dot) What we just described is the core of the approach used by all these compilers. The variation in results is explained mostly by how they handle vectors with lengths that are not an integer multiple of W. I ran these on a computer with AVX512 so that W = 8. LLVM, the backend compiler of both Julia and Clang, shows rapid performance degredation as N % 4W increases, where N is the length of the vectors. This is because, to handle the remainder, it uses a scalar loop that runs as written: multiply and add single elements, one after the other. GCC (gfortran) stumbles in throughput, because it does not use separate accumulation vectors.The Intel compilers have a secondary vectorized loop without any additional unrolling that masks off excess lanes beyond N (for when N isn\'t an integer multiple of W). LoopVectorization uses if/ifelse checks to determine how many extra vectors are needed, the last of which is masked.Neither GCC nor LLVM use masks (without LoopVectorization\'s assitance).I am not certain, but I believe Intel and GCC check for the vector\'s alignment, and align them if neccessary. Julia guarantees that the start of arrays beyond a certain size are aligned, so this is not an optimization I have implemented. But it may be worthwhile for handling large matrices with a number of rows that isn\'t an integer multiple of W. For such matrices, the first column may be aligned, but the next will not be."
},

{
    "location": "examples/dot_product/#Dot-Self-1",
    "page": "Dot Products",
    "title": "Dot-Self",
    "category": "section",
    "text": "A related problem is taking the dot product of a vector with itself; taking the sum of squares is a common operation, for example when calculating the (log)density of independent normal variates:function jselfdotavx(a)\n    s = zero(eltype(a))\n    @avx for i ∈ eachindex(a)\n        s += a[i] * a[i]\n    end\n    s\nendBecause we only need a single load per fma-instruction, we can now benefit from having 8 separate accumulators. For this reason, LoopVectorization now unrolls by 8 – it decides how much to unroll by comparing the bottlenecks on throughput with latency. The other compilers do not change their behavior, so now LoopVectorization has the advantage: (Image: selfdot) This algorithm may need refinement, because Julia (without LoopVectorization) only unrolls by 4, yet achieves roughly the same performance as LoopVectorization at multiples of 4W = 32, although performance declines rapidly from there due to the slow scalar loop. Performance for most is much higher – more GFLOPS – than the normal dot product, but still under half of the CPU\'s potential 131.2 GFLOPS, suggesting that some other bottlenecks are preventing the core from attaining 2 fmas per clock cycle. Note also that 8W = 64, so we don\'t really have enough iterations of the loop to amortize the overhead of performing the reductions of all these vectors into a single scalar. By the time the vectors are long enough to do this, we\'ll start running into memory bandwidth bottlenecks."
},

{
    "location": "examples/sum_of_squared_error/#",
    "page": "Sum of squared error",
    "title": "Sum of squared error",
    "category": "page",
    "text": ""
},

{
    "location": "examples/sum_of_squared_error/#Sum-of-squared-error-1",
    "page": "Sum of squared error",
    "title": "Sum of squared error",
    "category": "section",
    "text": "To calculate (y - X * β)\'(y - X * β), we can use the following loop.function sse_avx(y, X, β)\n    lp = zero(eltype(y))\n    @avx for i ∈ eachindex(y)\n        δ = y[i]\n        for j ∈ eachindex(β)\n            δ -= X[i,j] * β[j]\n        end\n        lp += δ * δ\n    end\n    lp\nendThis example demonstrates the importance of (not) modeling memory bandwidth and cache, as the performance quickly drops dramatically. However, it still does much better than all the compiled loops, with only the BLAS gemv-based approach matching (and ultimately beating) it in performance, while the other compilers lagged well behind.Performance starts to degrade for sizes larger than 60. Letting N be the size, X was a 3N/2x N/2 matrix. Therefore, performance started to suffer when X had more than about 30 columns (performance is much less sensitive to the number of rows).(Image: sse)"
},

{
    "location": "vectorized_convenience_functions/#",
    "page": "Vectorized Convenience Functions",
    "title": "Vectorized Convenience Functions",
    "category": "page",
    "text": ""
},

{
    "location": "vectorized_convenience_functions/#Convenient-Vectorized-Functions-1",
    "page": "Vectorized Convenience Functions",
    "title": "Convenient Vectorized Functions",
    "category": "section",
    "text": ""
},

{
    "location": "vectorized_convenience_functions/#vmap-1",
    "page": "Vectorized Convenience Functions",
    "title": "vmap",
    "category": "section",
    "text": "This is simply a vectorized map function."
},

{
    "location": "vectorized_convenience_functions/#vmapnt-and-vmapntt-1",
    "page": "Vectorized Convenience Functions",
    "title": "vmapnt and vmapntt",
    "category": "section",
    "text": "These are like vmap, but use non-temporal (streaming) stores into the destination, to avoid polluting the cache. Likely to yield a performance increase if you wont be reading the values soon.julia> using LoopVectorization, BenchmarkTools\n\njulia> f(x,y) = exp(-0.5abs2(x - y))\nf (generic function with 1 method)\n\njulia> x = rand(10^8); y = rand(10^8); z = similar(x);\n\njulia> @benchmark map!(f, $z, $x, $y)\nBenchmarkTools.Trial:\n  memory estimate:  0 bytes\n  allocs estimate:  0\n  --------------\n  minimum time:     442.614 ms (0.00% GC)\n  median time:      443.750 ms (0.00% GC)\n  mean time:        443.664 ms (0.00% GC)\n  maximum time:     444.730 ms (0.00% GC)\n  --------------\n  samples:          12\n  evals/sample:     1\n\njulia> @benchmark vmap!(f, $z, $x, $y)\nBenchmarkTools.Trial:\n  memory estimate:  0 bytes\n  allocs estimate:  0\n  --------------\n  minimum time:     177.257 ms (0.00% GC)\n  median time:      177.380 ms (0.00% GC)\n  mean time:        177.423 ms (0.00% GC)\n  maximum time:     177.956 ms (0.00% GC)\n  --------------\n  samples:          29\n  evals/sample:     1\n\njulia> @benchmark vmapnt!(f, $z, $x, $y)\nBenchmarkTools.Trial:\n  memory estimate:  0 bytes\n  allocs estimate:  0\n  --------------\n  minimum time:     143.521 ms (0.00% GC)\n  median time:      143.639 ms (0.00% GC)\n  mean time:        143.645 ms (0.00% GC)\n  maximum time:     143.821 ms (0.00% GC)\n  --------------\n  samples:          35\n  evals/sample:     1\n\njulia> Threads.nthreads()\n36\n\njulia> @benchmark vmapntt!(f, $z, $x, $y)\nBenchmarkTools.Trial:\n  memory estimate:  25.69 KiB\n  allocs estimate:  183\n  --------------\n  minimum time:     30.065 ms (0.00% GC)\n  median time:      30.130 ms (0.00% GC)\n  mean time:        30.146 ms (0.00% GC)\n  maximum time:     31.277 ms (0.00% GC)\n  --------------\n  samples:          166\n  evals/sample:     1"
},

{
    "location": "vectorized_convenience_functions/#vfilter-1",
    "page": "Vectorized Convenience Functions",
    "title": "vfilter",
    "category": "section",
    "text": "This function requires LLVM 7 or greater, and is only likly to give better performance if your CPU has AVX512. This is because it uses the compressed store intrinsic, which was added in LLVM 7. AVX512 provides a corresponding instruction, making the operation fast, while other instruction sets must emulate it, and thus are likely to get similar performance with LoopVectorization.vfilter as they do from Base.filter.julia> using LoopVectorization, BenchmarkTools\n\njulia> x = rand(997);\n\njulia> y1 = filter(a -> a > 0.7, x);\n\njulia> y2 = vfilter(a -> a > 0.7, x);\n\njulia> y1 == y2\ntrue\n\njulia> @benchmark filter(a -> a > 0.7, $x)\nBenchmarkTools.Trial:\n  memory estimate:  7.94 KiB\n  allocs estimate:  1\n  --------------\n  minimum time:     955.389 ns (0.00% GC)\n  median time:      1.050 μs (0.00% GC)\n  mean time:        1.191 μs (9.72% GC)\n  maximum time:     82.799 μs (94.92% GC)\n  --------------\n  samples:          10000\n  evals/sample:     18\n\njulia> @benchmark vfilter(a -> a > 0.7, $x)\nBenchmarkTools.Trial:\n  memory estimate:  7.94 KiB\n  allocs estimate:  1\n  --------------\n  minimum time:     477.487 ns (0.00% GC)\n  median time:      575.166 ns (0.00% GC)\n  mean time:        711.526 ns (17.87% GC)\n  maximum time:     9.257 μs (79.17% GC)\n  --------------\n  samples:          10000\n  evals/sample:     193"
},

{
    "location": "future_work/#",
    "page": "Future Work",
    "title": "Future Work",
    "category": "page",
    "text": ""
},

{
    "location": "future_work/#Future-Plans-1",
    "page": "Future Work",
    "title": "Future Plans",
    "category": "section",
    "text": "Future plans for LoopVectorization:Support triangular iteration spaces.\nIdentify obvious loop-carried dependencies like A[j] and A[j-1].\nBe able to generate optimized kernels from simple loop-based implementations of operations like Cholesky decompositions or solving triangular systems of equations.\nModel memory and CPU-cache to possibly insert extra loops and packing of data when deemed profitable.\nTrack types of individual operations in the loops. Currently, multiple types in loops aren\'t really handled, so this is a bit brittle at the moment.\nHandle loops where arrays contain non-primitive types (e.g., Complex numbers) well.Contributions are more than welcome, and I would be happy to assist if anyone would like to take a stab at any of these. Otherwise, while LoopVectorization is a core component to much of my work, so that I will continue developing it, I have many other projects that require active development, so it will be a long time before I am able to address these myself."
},

{
    "location": "api/#",
    "page": "API reference",
    "title": "API reference",
    "category": "page",
    "text": ""
},

{
    "location": "api/#API-reference-1",
    "page": "API reference",
    "title": "API reference",
    "category": "section",
    "text": ""
},

{
    "location": "api/#LoopVectorization.@avx",
    "page": "API reference",
    "title": "LoopVectorization.@avx",
    "category": "macro",
    "text": "@avx\n\nAnnotate a for loop, or a set of nested for loops whose bounds are constant across iterations, to optimize the computation. For example:\n\nfunction AmulBavx!(C, A, B)\n    @avx for m ∈ 1:size(A,1), n ∈ 1:size(B,2)\n        Cₘₙ = zero(eltype(C))\n        for k ∈ 1:size(A,2)\n            Cₘₙ += A[m,k] * B[k,n]\n        end\n        C[m,n] = Cₘₙ\n    end\nend\n\nThe macro models the set of nested loops, and chooses an ordering of the three loops to minimize predicted computation time.\n\nIt may also apply to broadcasts:\n\njulia> using LoopVectorization\n\njulia> a = rand(100);\n\njulia> b = @avx exp.(2 .* a);\n\njulia> c = similar(b);\n\njulia> @avx @. c = exp(2a);\n\njulia> b ≈ c\ntrue\n\n\n\n\n\n"
},

{
    "location": "api/#LoopVectorization.@_avx",
    "page": "API reference",
    "title": "LoopVectorization.@_avx",
    "category": "macro",
    "text": "@_avx\n\nThis macro transforms loops similarly to @avx. While @avx punts to a generated function to enable type-based analysis, _@avx works on just the expressions. This requires that it makes a number of default assumptions.\n\n\n\n\n\n"
},

{
    "location": "api/#Macros-1",
    "page": "API reference",
    "title": "Macros",
    "category": "section",
    "text": "@avx\n@_avx"
},

{
    "location": "api/#LoopVectorization.vmap",
    "page": "API reference",
    "title": "LoopVectorization.vmap",
    "category": "function",
    "text": "vmap(f, a::AbstractArray)\nvmap(f, a::AbstractArray, b::AbstractArray, ...)\n\nSIMD-vectorized map, applying f to each element of a (or paired elements of a, b, ...) and returning a new array.\n\n\n\n\n\n"
},

{
    "location": "api/#LoopVectorization.vmap!",
    "page": "API reference",
    "title": "LoopVectorization.vmap!",
    "category": "function",
    "text": "vmap!(f, destination, a::AbstractArray)\nvmap!(f, destination, a::AbstractArray, b::AbstractArray, ...)\n\nVectorized-map!, applying f to each element of a (or paired elements of a, b, ...) and storing the result in destination.\n\n\n\n\n\n"
},

{
    "location": "api/#LoopVectorization.vmapnt",
    "page": "API reference",
    "title": "LoopVectorization.vmapnt",
    "category": "function",
    "text": "vmapnt(f, a::AbstractArray)\nvmapnt(f, a::AbstractArray, b::AbstractArray, ...)\n\nA \"non-temporal\" variant of vmap. This can improve performance in cases where destination will not be needed soon.\n\n\n\n\n\n"
},

{
    "location": "api/#LoopVectorization.vmapnt!",
    "page": "API reference",
    "title": "LoopVectorization.vmapnt!",
    "category": "function",
    "text": "vmapnt!(::Function, dest, args...)\n\nThis is a vectorized map implementation using nontemporal store operations. This means that the write operations to the destination will not go to the CPU\'s cache. If you will not immediately be reading from these values, this can improve performance because the writes won\'t pollute your cache. This can especially be the case if your arguments are very long.\n\njulia> using LoopVectorization, BenchmarkTools\n\njulia> x = rand(10^8); y = rand(10^8); z = similar(x);\n\njulia> f(x,y) = exp(-0.5abs2(x - y))\nf (generic function with 1 method)\n\njulia> @benchmark map!(f, $z, $x, $y)\nBenchmarkTools.Trial:\n  memory estimate:  0 bytes\n  allocs estimate:  0\n  --------------\n  minimum time:     439.613 ms (0.00% GC)\n  median time:      440.729 ms (0.00% GC)\n  mean time:        440.695 ms (0.00% GC)\n  maximum time:     441.665 ms (0.00% GC)\n  --------------\n  samples:          12\n  evals/sample:     1\n\njulia> @benchmark vmap!(f, $z, $x, $y)\nBenchmarkTools.Trial:\n  memory estimate:  0 bytes\n  allocs estimate:  0\n  --------------\n  minimum time:     178.147 ms (0.00% GC)\n  median time:      178.381 ms (0.00% GC)\n  mean time:        178.430 ms (0.00% GC)\n  maximum time:     179.054 ms (0.00% GC)\n  --------------\n  samples:          29\n  evals/sample:     1\n\njulia> @benchmark vmapnt!(f, $z, $x, $y)\nBenchmarkTools.Trial:\n  memory estimate:  0 bytes\n  allocs estimate:  0\n  --------------\n  minimum time:     144.183 ms (0.00% GC)\n  median time:      144.338 ms (0.00% GC)\n  mean time:        144.349 ms (0.00% GC)\n  maximum time:     144.641 ms (0.00% GC)\n  --------------\n  samples:          35\n  evals/sample:     1\n\n\n\n\n\n"
},

{
    "location": "api/#LoopVectorization.vmapntt",
    "page": "API reference",
    "title": "LoopVectorization.vmapntt",
    "category": "function",
    "text": "vmapntt(f, a::AbstractArray)\nvmapntt(f, a::AbstractArray, b::AbstractArray, ...)\n\nA threaded variant of vmapnt.\n\n\n\n\n\n"
},

{
    "location": "api/#LoopVectorization.vmapntt!",
    "page": "API reference",
    "title": "LoopVectorization.vmapntt!",
    "category": "function",
    "text": "vmapntt!(::Function, dest, args...)\n\nLike vmapnt! (see vmapnt!), but uses Threads.@threads for parallel execution.\n\n\n\n\n\n"
},

{
    "location": "api/#map-like-constructs-1",
    "page": "API reference",
    "title": "map-like constructs",
    "category": "section",
    "text": "vmap\nvmap!\nvmapnt\nvmapnt!\nvmapntt\nvmapntt!"
},

{
    "location": "api/#filter-like-constructs-1",
    "page": "API reference",
    "title": "filter-like constructs",
    "category": "section",
    "text": "vfilter\nvfilter!"
},

{
    "location": "devdocs/overview/#",
    "page": "Developer Overview",
    "title": "Developer Overview",
    "category": "page",
    "text": ""
},

{
    "location": "devdocs/overview/#Developer-Overview-1",
    "page": "Developer Overview",
    "title": "Developer Overview",
    "category": "section",
    "text": "Here I will try to explain how the library works for the curious or any would-be contributors.The library uses a LoopSet object to model loops. The key components of the library can be divided into:Defining the LoopSet objects.\nConstructing the LoopSet objects.\nDetermining the strategy of how to evaluate loops.\nLowering the loopset object into a Julia Expr following a strategy."
},

{
    "location": "devdocs/loopset_structure/#",
    "page": "LoopSet Structure",
    "title": "LoopSet Structure",
    "category": "page",
    "text": ""
},

{
    "location": "devdocs/loopset_structure/#LoopSet-Structure-1",
    "page": "LoopSet Structure",
    "title": "LoopSet Structure",
    "category": "section",
    "text": "The loopsets define loops as a set of operations that depend on one another, and also on loops. Cycles are not allowed, making it a directed acyclic graph. Currently, only single return values are supported. Lets use a set of nested loops performing matrix multiplication as an example. We can create a naive LoopSet from an expression (naive due to being created without access to any type information):julia> using LoopVectorization\n\njulia> AmulBq = :(for m ∈ 1:M, n ∈ 1:N\n           C[m,n] = zero(eltype(B))\n           for k ∈ 1:K\n               C[m,n] += A[m,k] * B[k,n]\n           end\n       end);\n\njulia> lsAmulB = LoopVectorization.LoopSet(AmulBq);This LoopSet consists of seven operations that define the relationships within the loop:julia> LoopVectorization.operations(lsAmulB)\n7-element Array{LoopVectorization.Operation,1}:\n var\"##RHS#256\" = var\"##zero#257\"\n C[m, n] = var\"##RHS#256\"\n var\"##tempload#258\" = A[m, k]\n var\"##tempload#259\" = B[k, n]\n var\"##reduction#260\" = var\"##reductzero#261\"\n var\"##reduction#260\" = LoopVectorization.vfmadd_fast(var\"##tempload#258\", var\"##tempload#259\", var\"##reduction#260\")\n var\"##RHS#256\" = LoopVectorization.reduce_to_add(var\"##reduction#260\", var\"##RHS#256\")The act of performing a \"reduction\" across a loop introduces a few extra operations that manage creating a \"zero\" with respect to the reduction, and then combining with the specified value using reduce_to_add, which performs any necessary type conversions, such as from an SVec vector-type to a scalar, if necessary. This simplifies code generation, by making the functions agnostic with respect to the actual vectorization decisions the library makes.Each operation is listed as depending on a set of loop iteration symbols:julia> LoopVectorization.loopdependencies.(LoopVectorization.operations(lsAmulB))\n7-element Array{Array{Symbol,1},1}:\n [:m, :n]\n [:m, :n]\n [:m, :k]\n [:k, :n]\n [:m, :n]\n [:m, :k, :n]\n [:m, :n]We can also see which of the operations each of these operations depend on:julia> LoopVectorization.operations(lsAmulB)[6]\nvar\"##reduction#260\" = LoopVectorization.vfmadd_fast(var\"##tempload#258\", var\"##tempload#259\", var\"##reduction#260\")\n\njulia> LoopVectorization.parents(ans)\n3-element Array{LoopVectorization.Operation,1}:\n var\"##tempload#258\" = A[m, k]\n var\"##tempload#259\" = B[k, n]\n var\"##reduction#260\" = var\"##reductzero#261\"\n ```\nReferences to arrays are represtened with an `ArrayReferenceMeta` data structure:julia julia> LoopVectorization.operations(lsAmulB)[3].ref LoopVectorization.ArrayReferenceMeta(LoopVectorization.ArrayReference(:A, [:m, :k], Int8[0, 0]), Bool[1, 1], Symbol(\"##vptr##_A\")) ``It contains the name of the parent array (:A), the indicies[:m,:k], and a boolean vector (Bool[1, 1]) indicating whether these indices are loop iterables. Note that the optimizer assumes arrays are column-major, and thus that it is efficient to read contiguous elements from the first index. In lower level terms, it means that [high-throughput vmov](https://www.felixcloutier.com/x86/movupd) instructions can be used rather than [low-throughput](https://www.felixcloutier.com/x86/vgatherdpd:vgatherqpd) [gathers](https://www.felixcloutier.com/x86/vgatherqps:vgatherqpd). Similar story for storing elements. When no axis has unit stride, the first given index will be the dummySymbol(\"##DISCONTIGUOUSSUBARRAY##\")`."
},

{
    "location": "devdocs/constructing_loopsets/#",
    "page": "Constructing LoopSets",
    "title": "Constructing LoopSets",
    "category": "page",
    "text": ""
},

{
    "location": "devdocs/constructing_loopsets/#Constructing-LoopSets-1",
    "page": "Constructing LoopSets",
    "title": "Constructing LoopSets",
    "category": "section",
    "text": "When applying the @avx macro to a broadcast expression, the LoopSet object is constructed by recursively evaluating add_broadcast! on all the fields. The function and involved operations are their relationships are straightforward to infer from the structure of nested broadcasts.julia> Meta.@lower @. f(g(a,b) + c) / d\n:($(Expr(:thunk, CodeInfo(\n    @ none within `top-level scope\'\n1 ─ %1 = Base.broadcasted(g, a, b)\n│   %2 = Base.broadcasted(+, %1, c)\n│   %3 = Base.broadcasted(f, %2)\n│   %4 = Base.broadcasted(/, %3, d)\n│   %5 = Base.materialize(%4)\n└──      return %5\n))))\n\njulia> @macroexpand @avx @. f(g(a,b) + c) / d\nquote\n    var\"##262\" = Base.broadcasted(g, a, b)\n    var\"##263\" = Base.broadcasted(+, var\"##262\", c)\n    var\"##264\" = Base.broadcasted(f, var\"##263\")\n    var\"##265\" = Base.broadcasted(/, var\"##264\", d)\n    var\"##266\" = LoopVectorization.vmaterialize(var\"##265\", Val{:Main}())\nendThese nested broadcasted objects already express information very similar to what the LoopSet objects hold. The dimensionality of the objects provides the information on the associated loop dependencies.When applying @avx to a loop expression, it creates a LoopSet without awareness to type information, and then condenses the information into a summary which is passed as type information to a generated function.julia> @macroexpand @avx for m ∈ 1:M, n ∈ 1:N\n           C[m,n] = zero(eltype(B))\n           for k ∈ 1:K\n               C[m,n] += A[m,k] * B[k,n]\n           end\n       end\nquote\n    var\"##vptr##_C\" = LoopVectorization.stridedpointer(C)\n    var\"##vptr##_A\" = LoopVectorization.stridedpointer(A)\n    var\"##vptr##_B\" = LoopVectorization.stridedpointer(B)\n    begin\n        $(Expr(:gc_preserve, :(LoopVectorization._avx_!(Val{(0, 0)}(), Tuple{:numericconstant, Symbol(\"##zero#270\"), LoopVectorization.OperationStruct(0x0000000000000012, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, LoopVectorization.constant, 0x00, 0x01), :LoopVectorization, :setindex!, LoopVectorization.OperationStruct(0x0000000000000012, 0x0000000000000000, 0x0000000000000000, 0x0000000000000007, LoopVectorization.memstore, 0x01, 0x02), :LoopVectorization, :getindex, LoopVectorization.OperationStruct(0x0000000000000013, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, LoopVectorization.memload, 0x02, 0x03), :LoopVectorization, :getindex, LoopVectorization.OperationStruct(0x0000000000000032, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, LoopVectorization.memload, 0x03, 0x04), :numericconstant, Symbol(\"##reductzero#274\"), LoopVectorization.OperationStruct(0x0000000000000012, 0x0000000000000000, 0x0000000000000003, 0x0000000000000000, LoopVectorization.constant, 0x00, 0x05), :LoopVectorization, :vfmadd_fast, LoopVectorization.OperationStruct(0x0000000000000132, 0x0000000000000003, 0x0000000000000000, 0x0000000000030405, LoopVectorization.compute, 0x00, 0x05), :LoopVectorization, :reduce_to_add, LoopVectorization.OperationStruct(0x0000000000000012, 0x0000000000000003, 0x0000000000000000, 0x0000000000000601, LoopVectorization.compute, 0x00, 0x01)}, Tuple{LoopVectorization.ArrayRefStruct(0x0000000000000101, 0x0000000000000102, 0xffffffffffffe03b), LoopVectorization.ArrayRefStruct(0x0000000000000101, 0x0000000000000103, 0xffffffffffffffd6), LoopVectorization.ArrayRefStruct(0x0000000000000101, 0x0000000000000302, 0xffffffffffffe056), LoopVectorization.ArrayRefStruct(0x0000000000000101, 0x0000000000000102, 0xffffffffffffffd6)}, Tuple{0, Tuple{}, Tuple{}, Tuple{}, Tuple{}, Tuple{(1, LoopVectorization.IntOrFloat), (5, LoopVectorization.IntOrFloat)}, Tuple{}}, (LoopVectorization.StaticLowerUnitRange{0}(M), LoopVectorization.StaticLowerUnitRange{0}(N), LoopVectorization.StaticLowerUnitRange{0}(K)), var\"##vptr##_C\", var\"##vptr##_A\", var\"##vptr##_B\", var\"##vptr##_C\")), :C, :A, :B))\n    end\nendThis summary is then reconstruced using the available type information. This type information can be used, for example, to realize an array has been tranposed, and thus correctly identify which axis contains contiguous elements that are efficient to load from. This is why  The three chief components of the summaries are the definitions of operations, e.g.::LoopVectorization, :getindex, LoopVectorization.OperationStruct(0x0000000000000013, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, LoopVectorization.memload, 0x02, 0x03)the referenced array objects:LoopVectorization.ArrayRefStruct(0x0000000000000101, 0x0000000000000102, 0xffffffffffffe03b)and the set of loop bounds:(LoopVectorization.StaticLowerUnitRange{0}(M), LoopVectorization.StaticLowerUnitRange{0}(N), LoopVectorization.StaticLowerUnitRange{0}(K))"
},

{
    "location": "devdocs/evaluating_loops/#",
    "page": "Determining the strategy for evaluating loops",
    "title": "Determining the strategy for evaluating loops",
    "category": "page",
    "text": ""
},

{
    "location": "devdocs/evaluating_loops/#Determining-the-strategy-for-evaluating-loops-1",
    "page": "Determining the strategy for evaluating loops",
    "title": "Determining the strategy for evaluating loops",
    "category": "section",
    "text": "The heart of the optimizatizations performed by LoopVectorization are given in the determinestrategy.jl file utilizing instruction costs specified in costs.jl. Essentially, it estimates the cost of different means of evaluating the loops. It iterates through the different possible loop orders, as well as considering which loops to unroll, and which to vectorize. It will consider unrolling 1 or 2 loops (but it could settle on unrolling by a factor of 1, i.e. not unrolling), and vectorizing 1."
},

{
    "location": "devdocs/lowering/#",
    "page": "Lowering",
    "title": "Lowering",
    "category": "page",
    "text": ""
},

{
    "location": "devdocs/lowering/#Lowering-1",
    "page": "Lowering",
    "title": "Lowering",
    "category": "section",
    "text": "The first step to lowering is picking a strategy for lowering the loops. Then a Julia expression is created following that strategy, converting each of the operations into Julia expressions. This task is made simpler via multiple dispatch making the lowering of the components independent of the larger picture. For example, a load will look likevload(vptr_A, (i,j,k))with the behavior of this load determined by the types of the arguments. Vectorization is expressed by making an index a _MM{W} type, rather than an integer, and operations with it will either produce another _MM{W} when it will still correspond to contiguous loads, or an SVec{W,<:Integer} if the resulting loads will be discontiguous, so that a gather or scatter! will be used. If all indexes are simply integers, then this produces a scalar load or store."
},

]}
