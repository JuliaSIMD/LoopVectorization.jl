
@enum IndexType::UInt8 NotAnIndex=0 LoopIndex=1 ComputedIndex=2 SymbolicIndex=3 LiteralIndex=4

struct ArrayRefStruct
    index_types::UInt64
    indices::UInt64
end
tup_to_vec(t::NTuple{W,T}) where {W,T} = ntuple(Val(W)) do w @inbounds Core.VecElement(t[w]) end
vec_to_tup(v::Vec{W,T}) where {W,T} = ntuple(Val(W)) do w @inbounds (v[w]).value end
vec_to_tup(v::SVec{W,T}) where {W,T} = ntuple(Val(W)) do w @inbounds (v[w]) end
function ArrayRefStruct(ls::LoopSet, mref::ArrayReferenceMeta)
    index_types = zero(UInt64)
    indices = vbroadcast(SVec{8,UInt64}, zero(UInt64))
    indv = mref.ref.indices
    start = 1 + (first(indv) === Symbol("##DISCONTIGUOUSSUBARRAY##"))
    for (n,ind) ∈ enumerate(@view(indv[start:end]))
        index_types <<= 8
        indices <<= 16
        if ind isa Int
            
        elseif mref.loopindex[n]
        else
        end
    end
    ArrayRefStruct( index_types, vec_to_tup(indices) )
end

struct OperationStruct
    instruction::Instruction
    loopdeps::UInt64
    reduceddeps::UInt64
    parents::UInt64
    array::UInt64
end
function findmatchingarray(ls::LoopSet, array::Symbol)
    id = zero(UInt64)
    for (as,_) ∈ ls.includedarrays
        id += one(UInt64)
        if as === arraysym
            return id
        end
    end
    zero(UInt64)
end
filled_4byte_chunks(u::UInt64) = leading_zeros(u) >> 2
num_loop_deps(os::OperationStruct) = filled_4byte_chunks(os.loopdeps)
num_reduced_deps(os::OperationStruct) = filled_4byte_chunks(os.reduced_deps)
num_parents(os::OperationStruct) = filled_4byte_chunks(os.parents)

function loodeps_uint(ls::LoopSet, op::Operation)
    ld = zero(UInt64) # leading_zeros(ld) >> 2 yields the number of loopdeps
    for d ∈ loopdependencies(op)
        ld <<= 4
        ld |= getloopid(ls, d)
    end
    ld
end
function OperationStruct(ls::LoopSet, op::Operation)
    instr = instruction(op)
    array = accesses_memory(op) ? findmatchingarray(ls, name(op.ref)) : zero(UInt64)
    
end
## turn a LoopSet into a type object which can be used to reconstruct the LoopSet.


# Try to condense in type stable manner
function condense_operations(ls::LoopSet)
    
end

