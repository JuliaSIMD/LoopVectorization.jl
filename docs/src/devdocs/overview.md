# Developer Overview

Here I will try to explain how the library works for the curious or any would-be contributors.

The library uses a [LoopSet](https://github.com/chriselrod/LoopVectorization.jl/blob/master/src/graphs.jl#L146) object to model loops. The key components of the library can be divided into:
1. Defining the LoopSet objects.
2. Constructing the LoopSet objects.
3. Determining the strategy of how to evaluate loops.
4. Lowering the loopset object into a Julia `Expr` following a strategy.



