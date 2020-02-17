
Future plans for LoopVectorization:
1. Support triangular iteration spaces.
2. Identify obvious loop-carried dependencies like `A[j]` and `A[j-1]`.
3. Be able to generate optimized kernels from simple loop-based implementations of operations like Cholesky decompositions or solving triangular systems of equations.
4. Model memory and CPU-cache to possibly insert extra loops and packing of data when deemed profitable.
5. Track types of individual operations in the loops. Currently, multiple types in loops aren't really handled, so this is a bit brittle at the moment.
6. Handle loops where arrays contain non-primitive types (e.g., Complex numbers) well.

Contributions are more than welcome, and I would be happy to assist if anyone would like to take a stab at any of these.
Otherwise, while LoopVectorization is a core component to much of my work, so that I will continue developing it, I have many other projects that require active development, so it will be a long time before I am able to address these myself.

