# Determining the strategy for evaluating loops

The heart of the optimizatizations performed by LoopVectorization are given in the [determinestrategy.jl](https://github.com/chriselrod/LoopVectorization.jl/blob/master/src/determinestrategy.jl) file utilizing instruction costs specified in [costs.jl](https://github.com/chriselrod/LoopVectorization.jl/blob/master/src/costs.jl).
Essentially, it estimates the cost of different means of evaluating the loops. It iterates through the different possible loop orders, as well as considering which loops to unroll, and which to vectorize. It will consider unrolling 1 or 2 loops (but it could settle on unrolling by a factor of 1, i.e. not unrolling), and vectorizing 1.

The cost estimate is based on the costs of individual instructions and the number of times each one needs to be executed for the given strategy. The instruction cost can be broken into several components:

- The `scalar latency` is the minimum delay, in clock cycles, associated with the instruction. Think of it as the delay from turning on the water to when water starts coming out the hose.
- The `reciprocal throughput` is similar to the latency, but it measures the number of cycles per operation when many of the same operation are repeated in sequence.  Continuing our hose analogy, think of it as the inverse of the flow rate at steady-state. It is typically ≤ the `scalar latency`.
- The `register pressure` measures the register consumption by the operation

Data on individual instructions for specific architectures can be found on [Agner Fog's website](https://agner.org/optimize/instruction_tables.pdf). Most of the costs used were those for the Skylake-X architecture.

Examples of how these come into play:
- Vectorizing a loop will result in each instruction evaluating multiple iterations, but the costs of loads and stores will change based on the memory layouts of the accessed arrays.
- Unrolling can help reduce the number of times an operation must be performed, for example if it can allow us to reuse memory multiple times rather than reloading it every time it is needed.
- When there is a reduction, such as performing a sum, there is a dependency chain. Each `+` has to wait for the previous `+` to finish executing before it can begin, thus execution time is bounded by latency rather than minimum of the throughput of the `+` and load operations. By unrolling the loop, we can create multiple independent dependency chains.



