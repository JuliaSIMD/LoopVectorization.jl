# LoopSet Structure

The loopsets define loops as a set of operations that depend on one another, and also on loops. Cycles are not allowed, making it a directed acyclic graph.
Let's use a set of nested loops performing matrix multiplication as an example. We can create a naive `LoopSet` from an expression (naive due to being created without access to any type information):
```julia
julia> using LoopVectorization

julia> AmulBq = :(for m ∈ 1:M, n ∈ 1:N
           C[m,n] = zero(eltype(B))
           for k ∈ 1:K
               C[m,n] += A[m,k] * B[k,n]
           end
       end);

julia> lsAmulB = LoopVectorization.LoopSet(AmulBq);
```
This LoopSet consists of seven operations that define the relationships within the loop:
```julia
julia> LoopVectorization.operations(lsAmulB)
7-element Array{LoopVectorization.Operation,1}:
 var"##RHS#256" = var"##zero#257"
 C[m, n] = var"##RHS#256"
 var"##tempload#258" = A[m, k]
 var"##tempload#259" = B[k, n]
 var"##reduction#260" = var"##reductzero#261"
 var"##reduction#260" = LoopVectorization.vfmadd_fast(var"##tempload#258", var"##tempload#259", var"##reduction#260")
 var"##RHS#256" = LoopVectorization.reduce_to_add(var"##reduction#260", var"##RHS#256")
```
The act of performing a "reduction" across a loop introduces a few extra operations that manage creating a "zero" with respect to the reduction, and then combining with the specified value using `reduce_to_add`, which performs any necessary type conversions, such as from an `Vec` vector-type to a scalar, if necessary. This simplifies code generation, by making the functions agnostic with respect to the actual vectorization decisions the library makes.

Each operation is listed as depending on a set of loop iteration symbols:
```julia
julia> LoopVectorization.loopdependencies.(LoopVectorization.operations(lsAmulB))
7-element Array{Array{Symbol,1},1}:
 [:m, :n]
 [:m, :n]
 [:m, :k]
 [:k, :n]
 [:m, :n]
 [:m, :k, :n]
 [:m, :n]
```
We can also see which of the operations each of these operations depend on:
```julia
julia> LoopVectorization.operations(lsAmulB)[6]
var"##reduction#260" = LoopVectorization.vfmadd_fast(var"##tempload#258", var"##tempload#259", var"##reduction#260")

julia> LoopVectorization.parents(ans)
3-element Array{LoopVectorization.Operation,1}:
 var"##tempload#258" = A[m, k]
 var"##tempload#259" = B[k, n]
 var"##reduction#260" = var"##reductzero#261"
```
References to arrays are represented with an `ArrayReferenceMeta` data structure:
```julia
julia> LoopVectorization.operations(lsAmulB)[3].ref
LoopVectorization.ArrayReferenceMeta(LoopVectorization.ArrayReference(:A, [:m, :k], Int8[0, 0]), Bool[1, 1], Symbol("##vptr##_A"))
```
It contains the name of the parent array (`:A`), the indicies `[:m,:k]`, and a boolean vector (`Bool[1, 1]`) indicating whether these indices are loop iterables. Note that the optimizer assumes arrays are column-major, and thus that it is efficient to read contiguous elements from the first index. In lower level terms, it means that [high-throughput vmov](https://www.felixcloutier.com/x86/movupd) instructions can be used rather than [low-throughput](https://www.felixcloutier.com/x86/vgatherdpd:vgatherqpd) [gathers](https://www.felixcloutier.com/x86/vgatherqps:vgatherqpd). Similar story for storing elements.
When no axis has unit stride, the first given index will be the dummy `Symbol("##DISCONTIGUOUSSUBARRAY##")`.

!!! warning
    Currently, only single return values are supported (tuple destructuring is not supported in assignments).
