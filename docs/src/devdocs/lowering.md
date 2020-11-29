# Lowering

The first step to lowering is picking a strategy for lowering the loops. Then a Julia expression is created following that strategy, converting each of the operations into Julia expressions.
This task is made simpler via multiple dispatch making the lowering of the components independent of the larger picture. For example, a load will look like
```julia
vload(vptr_A, (i,j,k))
```
with the behavior of this load determined by the types of the arguments. Vectorization is expressed by making an index a `_MM{W}` type, rather than an integer, and operations with it will either produce another `_MM{W}` when it will still correspond to contiguous loads, or an `Vec{W,<:Integer}` if the resulting loads will be discontiguous, so that a `gather` or `scatter!` will be used. If all indexes are simply integers, then this produces a scalar load or store.


