
struct ArrayReference
    array::Symbol
    indices::Vector{Symbol}
    offsets::Vector{Int8}
end
ArrayReference(array, indices) = ArrayReference(array, indices, similar(indices, Int8))
function Base.isequal(x::ArrayReference, y::ArrayReference)
    x.array === y.array || return false
    xinds = x.indices
    yinds = y.indices
    nrefs = length(xinds)
    nrefs == length(yinds) || return false
    for n ∈ 1:nrefs
        xinds[n] === yinds[n] || return false
    end
    true
end
struct ArrayReferenceMeta
    ref::ArrayReference
    loopedindex::Vector{Bool}
    ptr::Symbol
end
function ArrayReferenceMeta(ref::ArrayReference, loopedindex, ptr = vptr(ref))
    ArrayReferenceMeta(
        ref, loopedindex, ptr
    )
end
# function Base.hash(x::ArrayReference, h::UInt)
    # @inbounds for n ∈ eachindex(x)
        # h = hash(x.ref[n], h)
    # end
    # hash(x.array, h)
# end
loopdependencies(ref::ArrayReferenceMeta) = ref.ref.indices
Base.convert(::Type{ArrayReference}, ref::ArrayReferenceMeta) = ref.ref
Base.:(==)(x::ArrayReference, y::ArrayReference) = isequal(x, y)
Base.:(==)(x::ArrayReferenceMeta, y::ArrayReferenceMeta) = isequal(x.ref, y.ref) && x.ptr === y.ptr

# Errors preferable than silently working?
Base.:(==)(x::ArrayReference, y::ArrayReferenceMeta) = x == y.ref
Base.:(==)(x::ArrayReferenceMeta, y::ArrayReference) = x.ref == y
Base.:(==)(x::ArrayReference, y) = false
Base.:(==)(x::ArrayReferenceMeta, y) = false

abstract type AbstractLoopOperation end

@enum OperationType begin
    constant
    memload
    compute
    memstore
    loopvalue
end

# TODO: can some computations be cached in the operations?
"""
"""
mutable struct Operation <: AbstractLoopOperation
    identifier::Int
    variable::Symbol
    elementbytes::Int
    instruction::Instruction
    node_type::OperationType
    dependencies::Vector{Symbol}
    reduced_deps::Vector{Symbol}
    parents::Vector{Operation}
    ref::ArrayReferenceMeta
    mangledvariable::Symbol
    reduced_children::Vector{Symbol}
    function Operation(
        identifier::Int,
        variable,
        elementbytes,
        instruction,
        node_type,
        dependencies = Symbol[],
        reduced_deps = Symbol[],
        parents = Operation[],
        ref::ArrayReferenceMeta = NOTAREFERENCE,
        reduced_children = Symbol[]
    )
        new(
            identifier, variable, elementbytes, instruction, node_type,
            convert(Vector{Symbol},dependencies),
            convert(Vector{Symbol},reduced_deps),
            convert(Vector{Operation},parents),
            ref, Symbol("##", variable, :_),
            reduced_children
        )
    end
end

function matches(op1::Operation, op2::Operation)
    op1 === op2 && return true
    op1.instruction === op2.instruction || return false
    op1.node_type == op2.node_type || return false
    if isconstant(op1)
        return false
    end
    op1.dependencies == op2.dependencies || return false
    op2.reduced_deps == op2.reduced_deps || return false
    if isload(op1)
        op1.ref == op2.ref || return false
    end
    nparents = length(parents(op1))
    nparents == length(parents(op2)) || return false
    for p ∈ 1:nparents
        matches(op1.parents[p], op2.parents[p]) || return false
    end
    true
end

 # negligible save on allocations for operations that don't need these (eg, constants).
const NODEPENDENCY = Symbol[]
const NOPARENTS = Operation[]

function Base.show(io::IO, op::Operation)
    if isconstant(op)
        if op.instruction === LOOPCONSTANT
            
            print(io, Expr(:(=), op.variable, 0))
        else
            print(io, Expr(:(=), op.variable, op.instruction.instr))
        end
    elseif isload(op)
        print(io, Expr(:(=), op.variable, Expr(:ref, name(op.ref), getindices(op)...)))
    elseif iscompute(op)
        print(io, Expr(:(=), op.variable, Expr(op.instruction, name.(parents(op))...)))
    elseif isstore(op)
        print(io, Expr(:(=), Expr(:ref, name(op.ref), getindices(op)...), name(first(parents(op)))))
    elseif isloopvalue(op)
        print(io, Expr(:(=), op.variable, op.variable))
    end
end

function isreduction(op::Operation)
    ((op.node_type == compute) || (op.node_type == memstore)) && length(op.reduced_deps) > 0
    # (op.node_type == memstore) && (length(op.symbolic_metadata) < length(op.dependencies))# && issubset(op.symbolic_metadata, op.dependencies)
end
optype(op::Operation) = op.node_type
isload(op::AbstractLoopOperation) = optype(op) == memload
iscompute(op::AbstractLoopOperation) = optype(op) == compute
isstore(op::AbstractLoopOperation) = optype(op) == memstore
isconstant(op::AbstractLoopOperation) = optype(op) == constant
isloopvalue(op::AbstractLoopOperation) = optype(op) == loopvalue
accesses_memory(op::AbstractLoopOperation) = isload(op) | isstore(op)
elsize(op::Operation) = op.elementbytes
dependson(op::Operation, sym::Symbol) = sym ∈ op.dependencies
parents(op::Operation) = op.parents
# children(op::Operation) = op.children
loopdependencies(op::Operation) = op.dependencies
reduceddependencies(op::Operation) = op.reduced_deps
reducedchildren(op::Operation) = op.reduced_children
identifier(op::Operation) = op.identifier + 1
vptr(x::Symbol) = Symbol("##vptr##_", x)
vptr(x::ArrayReference) = vptr(x.array)
vptr(x::ArrayReferenceMeta) = x.ptr
vptr(x::Operation) = x.ref.ptr
name(x::ArrayReference) = x.array
name(x::ArrayReferenceMeta) = x.ref.array
name(op::Operation) = op.variable
instruction(op::Operation) = op.instruction
isreductionzero(op::Operation, instr::Symbol) = op.instruction.mod === REDUCTION_ZERO[instr]
refname(op::Operation) = op.ref.ptr
isreductcombineinstr(op::Operation) = iscompute(op) && isreductcombineinstr(instruction(op))
"""
    mvar = mangledvar(op)

Returns the mangled variable name, for use in the produced expressions.
These names will be further processed if op is tiled and/or unrolled.

```julia
    if tiled ∈ loopdependencies(op) # `suffix` is tilenumber
        mvar = Symbol(op, suffix, :_)
    end
    if unrolled ∈ loopdependencies(op) # `u` is unroll number
        mvar = Symbol(op, u)
    end
```
"""
mangledvar(op::Operation) = op.mangledvariable

"""
Returns `0` if the op is the declaration of the constant outerreduction variable.
Returns `n`, where `n` is the constant declarations's index among parents(op), if op is an outter reduction.
Returns `-1` if not an outerreduction.
"""
function isouterreduction(op::Operation)
    if isconstant(op) # equivalent to checking if length(loopdependencies(op)) == 0
        op.instruction === LOOPCONSTANT ? 0 : -1
    elseif iscompute(op)
        var = op.variable
        for (n,opp) ∈ enumerate(parents(op))
            opp.variable === var && opp.instruction === LOOPCONSTANT && return n
        end
        -1
    else
        -1
    end
end

mutable struct ArrayReferenceMetaPosition
    mref::ArrayReferenceMeta
    parents::Vector{Operation}
    loopdependencies::Vector{Symbol}
    reduceddeps::Vector{Symbol}
    varname::Symbol
end
function ArrayReferenceMetaPosition(parents::Vector{Operation}, ldref::Vector{Symbol}, reduceddeps::Vector{Symbol}, varname::Symbol)
    ArrayReferenceMetaPosition( NOTAREFERENCE, parents, ldref, reduceddeps, varname )
end
function Operation(id::Int, var::Symbol, elementbytes::Int, instr, optype::OperationType, mpref::ArrayReferenceMetaPosition)
    Operation( id, var, elementbytes, instr, optype, mpref.loopdependencies, mpref.reduceddeps, mpref.parents, mpref.mref )
end
Base.:(==)(x::ArrayReferenceMetaPosition, y::ArrayReferenceMetaPosition) = x.mref.ref == y.mref.ref
# Avoid memory allocations by using this for ops that aren't references
const NOTAREFERENCE = ArrayReferenceMeta(ArrayReference(Symbol(""), Union{Symbol,Int}[]),Bool[],Symbol(""))
const NOTAREFERENCEMP = ArrayReferenceMetaPosition(NOTAREFERENCE, NOPARENTS, Symbol[], Symbol[],Symbol(""))
varname(::Nothing) = nothing
varname(mpref::ArrayReferenceMetaPosition) = mpref.varname
name(mpref::ArrayReferenceMetaPosition) = name(mpref.mref.ref)
loopdependencies(ref::ArrayReferenceMetaPosition) = ref.loopdependencies
reduceddependencies(ref::ArrayReferenceMetaPosition) = ref.reduceddeps
getindices(ref::ArrayReference) = ref.indices
getindices(mref::ArrayReferenceMeta) = mref.ref.indices
getindices(mpref::ArrayReferenceMetaPosition) = mpref.ref.ref.indices
getindices(op::Operation) = op.ref.ref.indices


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
