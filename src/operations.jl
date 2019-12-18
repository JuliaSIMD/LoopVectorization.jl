
@enum OperationType begin
    constant
    memload
    compute
    memstore
end

# const ID = Threads.Atomic{UInt}(0)


# TODO: can some computations be cached in the operations?
"""
if ooperation_type == memstore || operation_type == memstore# || operation_type == compute_new || operation_type == compute_update
symbolic metadata contains info on direct dependencies / placement within loop.

if isload(op) -> Symbol(:vptr_, first(op.reduced_deps))
if istore(op) -> Symbol(:vptr_, op.variable)
is how we access the memory.
If numerical_metadata[i] == -1
Symbol(:stride_, op.variable, :_, op.symbolic_metadata[i])
is the stride for loop index
symbolic_metadata[i]
"""
struct Operation
    identifier::Int
    variable::Symbol
    elementbytes::Int
    instruction::Symbol
    node_type::OperationType
    dependencies::Vector{Symbol}
    reduced_deps::Vector{Symbol}
    parents::Vector{Operation}
    # children::Vector{Operation}
    # numerical_metadata::Vector{Int} # stride of -1 indicates dynamic
    # symbolic_metadata::Vector{Symbol}
    function Operation(
        identifier,
        variable,
        elementbytes,
        instruction,
        node_type,
        dependencies = Symbol[],
        reduced_deps = Symbol[],
        parents = Operation[]
    )
        new(
            identifier, variable, elementbytes, instruction, node_type,
            convert(Vector{Symbol},dependencies),
            convert(Vector{Symbol},reduced_deps),
            convert(Vector{Operation},parents)
        )
    end
end

 # negligible save on allocations for operations that don't need these (eg, constants).
const NODEPENDENCY = Symbol[]
const NOPARENTS = Operation[]


function isreduction(op::Operation)
    ((op.node_type == compute) || (op.node_type == memstore)) && length(op.reduced_deps) > 0
    # (op.node_type == memstore) && (length(op.symbolic_metadata) < length(op.dependencies))# && issubset(op.symbolic_metadata, op.dependencies)
end
isload(op::Operation) = op.node_type == memload
iscompute(op::Operation) = op.node_type == compute
isstore(op::Operation) = op.node_type == memstore
isconstant(op::Operation) = op.node_type == constant
accesses_memory(op::Operation) = isload(op) | isstore(op)
elsize(op::Operation) = op.elementbytes
dependson(op::Operation, sym::Symbol) = sym ∈ op.dependencies
parents(op::Operation) = op.parents
# children(op::Operation) = op.children
loopdependencies(op::Operation) = op.dependencies
reduceddependencies(op::Operation) = op.reduced_deps
identifier(op::Operation) = op.identifier + 1
name(op::Operation) = op.variable
instruction(op::Operation) = op.instruction

function isouterreduction(op::Operation)
    if isconstant(op)
        op.instruction === Symbol("##CONSTANT##")
    elseif iscompute(op)
        var = op.variable
        for opp ∈ parents(op)
            opp.variable === var && opp.instruction === Symbol("##CONSTANT##") && return true
        end
        false
    else
        false
    end
end

# function hasintersection(s1::Set{T}, s2::Set{T}) where {T}
    # for x ∈ s1
        # x ∈ s2 && return true
    # end
    # false
# end

# function symposition(op::Operation, sym::Symbol)
    # findfirst(s -> s === sym, op.symbolic_metadata)
# end
# function stride(op::Operation, sym::Symbol)
    # @assert accesses_memory(op) "This operation does not access memory!"
    # # access stride info?
    # op.numerical_metadata[symposition(op,sym)]
# end



