# Internals reference

## Operation types

```@docs
LoopVectorization.OperationType
LoopVectorization.constant
LoopVectorization.memload
LoopVectorization.compute
LoopVectorization.memstore
LoopVectorization.loopvalue
```

## Operation

```@docs
LoopVectorization.Operation
```

## Instructions and costs

```@docs
LoopVectorization.Instruction
LoopVectorization.InstructionCost
```

## Array references

```@docs
LoopVectorization.ArrayReference
LoopVectorization.ArrayReferenceMeta
```

## Condensed types

These are used when encoding the `@turbo` block as a type parameter for passing through
to the `@generated` function.

```@docs
LoopVectorization.ArrayRefStruct
LoopVectorization.OperationStruct
```
