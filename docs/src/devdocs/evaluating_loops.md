
## Determining the strategy for evaluating loops

The heart of the optimizatizations performed by LoopVectorization are given in the [determinestrategy.jl](https://github.com/chriselrod/LoopVectorization.jl/blob/master/src/determinestrategy.jl) file utilizing instruction costs specified in [costs.jl](https://github.com/chriselrod/LoopVectorization.jl/blob/master/src/costs.jl).
Essentially, it estimates the cost of different means of evaluating the loops. It iterates through the different possible loop orders, as well as considering which loops to unroll, and which to vectorize. It will consider unrolling 1 or 2 loops (but it could settle on unrolling by a factor of 1, i.e. not unrolling), and vectorizing 1.

