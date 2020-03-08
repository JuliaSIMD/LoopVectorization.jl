# Constructing LoopSets

When applying the `@avx` macro to a broadcast expression, the LoopSet object is constructed by recursively evaluating [add_broadcast!](https://github.com/chriselrod/LoopVectorization.jl/blob/master/src/broadcast.jl#L166) on all the fields. The function and involved operations are their relationships are straightforward to infer from the structure of nested broadcasts.
```julia
julia> Meta.@lower @. f(g(a,b) + c) / d
:($(Expr(:thunk, CodeInfo(
    @ none within `top-level scope'
1 ─ %1 = Base.broadcasted(g, a, b)
│   %2 = Base.broadcasted(+, %1, c)
│   %3 = Base.broadcasted(f, %2)
│   %4 = Base.broadcasted(/, %3, d)
│   %5 = Base.materialize(%4)
└──      return %5
))))

julia> @macroexpand @avx @. f(g(a,b) + c) / d
quote
    var"##262" = Base.broadcasted(g, a, b)
    var"##263" = Base.broadcasted(+, var"##262", c)
    var"##264" = Base.broadcasted(f, var"##263")
    var"##265" = Base.broadcasted(/, var"##264", d)
    var"##266" = LoopVectorization.vmaterialize(var"##265", Val{:Main}())
end
```
These nested broadcasted objects already express information very similar to what the `LoopSet` objects hold. The dimensionality of the objects provides the information on the associated loop dependencies.

When applying `@avx` to a loop expression, it creates a `LoopSet` without awareness to type information, and then [condenses the information](https://github.com/chriselrod/LoopVectorization.jl/blob/master/src/condense_loopset.jl) into a summary which is passed as type information to a generated function.
```julia
julia> @macroexpand @avx for m ∈ 1:M, n ∈ 1:N
           C[m,n] = zero(eltype(B))
           for k ∈ 1:K
               C[m,n] += A[m,k] * B[k,n]
           end
       end
quote
    var"##vptr##_C" = LoopVectorization.stridedpointer(C)
    var"##vptr##_A" = LoopVectorization.stridedpointer(A)
    var"##vptr##_B" = LoopVectorization.stridedpointer(B)
    begin
        $(Expr(:gc_preserve, :(LoopVectorization._avx_!(Val{(0, 0)}(), Tuple{:numericconstant, Symbol("##zero#270"), LoopVectorization.OperationStruct(0x0000000000000012, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, LoopVectorization.constant, 0x00, 0x01), :LoopVectorization, :setindex!, LoopVectorization.OperationStruct(0x0000000000000012, 0x0000000000000000, 0x0000000000000000, 0x0000000000000007, LoopVectorization.memstore, 0x01, 0x02), :LoopVectorization, :getindex, LoopVectorization.OperationStruct(0x0000000000000013, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, LoopVectorization.memload, 0x02, 0x03), :LoopVectorization, :getindex, LoopVectorization.OperationStruct(0x0000000000000032, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, LoopVectorization.memload, 0x03, 0x04), :numericconstant, Symbol("##reductzero#274"), LoopVectorization.OperationStruct(0x0000000000000012, 0x0000000000000000, 0x0000000000000003, 0x0000000000000000, LoopVectorization.constant, 0x00, 0x05), :LoopVectorization, :vfmadd_fast, LoopVectorization.OperationStruct(0x0000000000000132, 0x0000000000000003, 0x0000000000000000, 0x0000000000030405, LoopVectorization.compute, 0x00, 0x05), :LoopVectorization, :reduce_to_add, LoopVectorization.OperationStruct(0x0000000000000012, 0x0000000000000003, 0x0000000000000000, 0x0000000000000601, LoopVectorization.compute, 0x00, 0x01)}, Tuple{LoopVectorization.ArrayRefStruct(0x0000000000000101, 0x0000000000000102, 0xffffffffffffe03b), LoopVectorization.ArrayRefStruct(0x0000000000000101, 0x0000000000000103, 0xffffffffffffffd6), LoopVectorization.ArrayRefStruct(0x0000000000000101, 0x0000000000000302, 0xffffffffffffe056), LoopVectorization.ArrayRefStruct(0x0000000000000101, 0x0000000000000102, 0xffffffffffffffd6)}, Tuple{0, Tuple{}, Tuple{}, Tuple{}, Tuple{}, Tuple{(1, LoopVectorization.IntOrFloat), (5, LoopVectorization.IntOrFloat)}, Tuple{}}, (LoopVectorization.StaticLowerUnitRange{0}(M), LoopVectorization.StaticLowerUnitRange{0}(N), LoopVectorization.StaticLowerUnitRange{0}(K)), var"##vptr##_C", var"##vptr##_A", var"##vptr##_B", var"##vptr##_C")), :C, :A, :B))
    end
end
```
This summary is then [reconstruced](https://github.com/chriselrod/LoopVectorization.jl/blob/master/src/reconstruct_loopset.jl) using the available type information. This type information can be used, for example, to realize an array has been tranposed, and thus correctly identify which axis contains contiguous elements that are efficient to load from. This is why 
The three chief components of the summaries are the definitions of operations, e.g.:
```julia
:LoopVectorization, :getindex, LoopVectorization.OperationStruct(0x0000000000000013, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, LoopVectorization.memload, 0x02, 0x03)
```
the referenced array objects:
```julia
LoopVectorization.ArrayRefStruct(0x0000000000000101, 0x0000000000000102, 0xffffffffffffe03b)
```
and the set of loop bounds:
```julia
(LoopVectorization.StaticLowerUnitRange{0}(M), LoopVectorization.StaticLowerUnitRange{0}(N), LoopVectorization.StaticLowerUnitRange{0}(K))
```




