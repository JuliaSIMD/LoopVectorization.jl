"""
    ArrayReference

A type for encoding an array reference `A[i,j]` occurring inside an `@avx` block.

# Fields

$(TYPEDFIELDS)
"""
struct ArrayReference
    "The array variable"
    array::Symbol
    "The list of indices (e.g., `[:i, :j]`), or `name(op)` for computed indices."
    indices::Vector{Symbol}
    """Index offset, e.g., `a[i+7]` would store the `7`. `offsets` is also used
    to help identify opportunities for avoiding reloads, for example in `y[i] = x[i] - x[i-1]`,
    the previous load `x[i-1]` can be "carried over" to the next iteration.
    Only used for small (`Int8`) offsets."""    
    offsets::Vector{Int8}
end
ArrayReference(array, indices) = ArrayReference(array, indices, zeros(Int8, length(indices)))
function sameref(x::ArrayReference, y::ArrayReference)
    (x.array === y.array) && (x.indices == y.indices)
end
function Base.isequal(x::ArrayReference, y::ArrayReference)
    sameref(x, y) || return false
    xoffs = x.offsets; yoffs = y.offsets
    length(xoffs) == length(yoffs) || return false
    for n ∈ eachindex(xoffs)
        xoffs[n] == yoffs[n] || return false
    end
    true
end

"""
    ArrayReferenceMeta

A type similar to [`ArrayReference`](@ref) but holding additional information.

# Fields

$(TYPEDFIELDS)
"""
struct ArrayReferenceMeta
    "The `ArrayReference`"
    ref::ArrayReference
    "A vector of Bools indicating whether each index is a loop variable (`false` for operation-computed indices)"
    loopedindex::Vector{Bool}
    "Variable holding the pointer to the array's underlying storage"
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
"""
This definition is used to find the matching arrays for the LoopStartStopManager.
It checks the underly ArrayReferences, but also the vptr names. This is because different
slices of the same array will have the same ArrayReference, but different vptr names.
"""
sameref(x::ArrayReferenceMeta, y::ArrayReferenceMeta) = (vptr(x) === vptr(y)) && sameref(x.ref, y.ref)
Base.convert(::Type{ArrayReference}, ref::ArrayReferenceMeta) = ref.ref
Base.:(==)(x::ArrayReference, y::ArrayReference) = isequal(x, y)
Base.:(==)(x::ArrayReferenceMeta, y::ArrayReferenceMeta) = (x.ptr === y.ptr) && isequal(x.ref, y.ref)

# Errors preferable than silently working?
Base.:(==)(x::ArrayReference, y::ArrayReferenceMeta) = x == y.ref
Base.:(==)(x::ArrayReferenceMeta, y::ArrayReference) = x.ref == y
# Base.:(==)(x::ArrayReference, y) = false
# Base.:(==)(x::ArrayReferenceMeta, y) = false

abstract type AbstractLoopOperation end

"""
`OperationType` is an `@enum` for classifying supported operations that can appear in
`@avx` blocks. Type `LoopVectorization.OperationType` to see the different types.
"""
@enum OperationType begin
    constant
    memload
    compute
    memstore
    loopvalue
end
"An operation setting a variable to a constant value (e.g., `a = 0.0`)" constant
"An operation setting a variable from a memory location (e.g., `a = A[i,j]`)" memload
"An operation computing a new value from one or more variables (e.g., `a = b + c`)" compute
"An operation storing a value to a memory location (e.g., `A[i,j] = a`)" memstore
"""
`loopvalue` indicates an loop variable (`i` in `for i in ...`). These are the "parents" of `compute`
operations that involve the loop variables.
"""
loopvalue

# TODO: can some computations be cached in the operations?
"""
    Operation

A structure to encode a particular action occuring inside an `@avx` block.

# Fields

$(TYPEDFIELDS)

# Example

```jldoctest Operation; filter = r"\\"##.*\\""
julia> using LoopVectorization

julia> AmulBq = :(for m ∈ 1:M, n ∈ 1:N
           C[m,n] = zero(eltype(B))
           for k ∈ 1:K
               C[m,n] += A[m,k] * B[k,n]
           end
       end);

julia> lsAmulB = LoopVectorization.LoopSet(AmulBq);

julia> LoopVectorization.operations(lsAmulB)
6-element Vector{LoopVectorization.Operation}:
 var"##RHS#245" = var"##zero#246"
 C[m, n] = var"##RHS#245"
 var"##tempload#248" = A[m, k]
 var"##tempload#249" = B[k, n]
 var"##RHS#245" = LoopVectorization.vfmadd(var"##tempload#248", var"##tempload#249", var"##RHS#245")
 var"##RHS#245" = LoopVectorization.identity(var"##RHS#245")
```
Each one of these lines is a pretty-printed `Operation`.
"""
mutable struct Operation <: AbstractLoopOperation
    """A unique identifier for this operation.
    `identifer(op::Operation)` returns the index of this operation within `operations(ls::LoopSet)`."""
    identifier::Int
    """The name of the variable storing the result of this operation.
    For `a = val` this would be `:a`. For array assignments `A[i,j] = val` this would be `:A`."""
    variable::Symbol
    "Intended to be the size of the result, in bytes. Often inaccurate, not to be relied on."
    elementbytes::Int
    "The specific operator, e.g., `identity` or `+`"
    instruction::Instruction
    "The [`OperationType`](@ref) associated with this operation"
    node_type::OperationType
    "The loop variables this operation depends on"
    dependencies::Vector{Symbol}
    "Additional loop dependencies that must execute before this operation can be performed successfully (often needed in reductions)"
    reduced_deps::Vector{Symbol}
    "Operations whose result this operation depends on"
    parents::Vector{Operation}
    "Operations who depend on this result"
    children::Vector{Operation}
    "For `memload` or `memstore`, encodes the array location"
    ref::ArrayReferenceMeta
    "`gensymmed` name of result."
    mangledvariable::Symbol
    """Loop variables that *consumers* of this operation depend on.
    Often used in reductions to replicate assignment of initializers when unrolling."""
    reduced_children::Vector{Symbol}
    "Cached value for whether u₁loopsym ∈ loopdependencies(op)"
    u₁unrolled::Bool
    "Cached value for whether u₂loopsym ∈ loopdependencies(op)"
    u₂unrolled::Bool
    "Cached value for whether vectorized ∈ loopdependencies(op)"
    vectorized::Bool
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
            convert(Vector{Operation},parents), Operation[],
            ref, Symbol("##", variable, :_),
            reduced_children
        )
    end
end

isu₁unrolled(op::Operation) = op.u₁unrolled
isu₂unrolled(op::Operation) = op.u₂unrolled
isvectorized(op::Operation) = op.vectorized
function setunrolled!(op::Operation, u₁loopsym, u₂loopsym, vectorized)
    op.u₁unrolled = u₁loopsym ∈ loopdependencies(op)
    op.u₂unrolled = u₂loopsym ∈ loopdependencies(op)
    op.vectorized = vectorized ∈ loopdependencies(op)
    nothing
end

function matches(op1::Operation, op2::Operation)
    op1 === op2 && return true
    op1.instruction === op2.instruction || return false
    op1.node_type == op2.node_type || return false
    if isconstant(op1)
        return iszero(length(loopdependencies(op1))) && iszero(length(loopdependencies(op2))) && (mangledvar(op1) === mangledvar(op2))
    end
    op1.dependencies == op2.dependencies || return false
    op2.reduced_deps == op2.reduced_deps || return false
    if accesses_memory(op1)
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
        ref = Expr(:ref, name(op.ref)); append!(ref.args, getindices(op))
        print(io, Expr(:(=), op.variable, ref))
    elseif iscompute(op)
        print(io, Expr(:(=), op.variable, callexpr(op.instruction, map(name, parents(op)))))
    elseif isstore(op)
        ref = Expr(:ref, name(op.ref)); append!(ref.args, getindices(op))
        print(io, Expr(:(=), ref, name(first(parents(op)))))
    elseif isloopvalue(op)
        print(io, Expr(:(=), op.variable, first(loopdependencies(op))))
    end
end

function isreduction(op::Operation)
    ((op.node_type == compute) || (op.node_type == memstore)) && length(reduceddependencies(op)) > 0
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
children(op::Operation) = op.children
loopdependencies(op::Operation) = op.dependencies
reduceddependencies(op::Operation) = op.reduced_deps
reducedchildren(op::Operation) = op.reduced_children
identifier(op::Operation) = op.identifier + 1
vptr(x::Symbol) = Symbol("##vptr##_", x)
vptr(x::ArrayReference) = vptr(x.array)
vptr(x::ArrayReferenceMeta) = x.ptr
vptr(x::Operation) = x.ref.ptr
# vptrbase(x) = Symbol(vptr(x), "##BASE##")
name(x::ArrayReference) = x.array
name(x::ArrayReferenceMeta) = x.ref.array
name(op::Operation) = op.variable
instruction(op::Operation) = op.instruction
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
# function ArrayReferenceMetaPosition(parents::Vector{Operation}, ldref::Vector{Symbol}, reduceddeps::Vector{Symbol}, varname::Symbol)
#     ArrayReferenceMetaPosition( NOTAREFERENCE, parents, ldref, reduceddeps, varname )
# end
function Operation(id::Int, var::Symbol, elementbytes::Int, instr, optype::OperationType, mpref::ArrayReferenceMetaPosition)
    Operation( id, var, elementbytes, instr, optype, mpref.loopdependencies, mpref.reduceddeps, mpref.parents, mpref.mref )
end
Base.:(==)(x::ArrayReferenceMetaPosition, y::ArrayReferenceMetaPosition) = x.mref == y.mref
# Avoid memory allocations by using this for ops that aren't references
const NOTAREFERENCE = ArrayReferenceMeta(ArrayReference(Symbol(""), Union{Symbol,Int}[]),Bool[],Symbol(""))
const NOTAREFERENCEMP = ArrayReferenceMetaPosition(NOTAREFERENCE, NOPARENTS, Symbol[], Symbol[],Symbol(""))
varname(::Nothing) = nothing
varname(mpref::ArrayReferenceMetaPosition) = mpref.varname
name(mpref::ArrayReferenceMetaPosition) = name(mpref.mref.ref)
loopdependencies(ref::ArrayReferenceMetaPosition) = ref.loopdependencies
reduceddependencies(ref::ArrayReferenceMetaPosition) = ref.reduceddeps
arrayref(ref::ArrayReference) = ref
arrayref(ref::ArrayReferenceMeta) = ref.ref
arrayref(ref::ArrayReferenceMetaPosition) = ref.ref.ref
arrayref(op::Operation) = op.ref.ref
getindices(ref) = arrayref(ref).indices
getoffsets(ref) = arrayref(ref).offsets
const DISCONTIGUOUS = Symbol("##DISCONTIGUOUSSUBARRAY##")
function makediscontiguous!(inds)
    if iszero(length(inds)) || first(inds) !== DISCONTIGUOUS
        pushfirst!(inds, DISCONTIGUOUS)
    end
    nothing
end
isdiscontiguous(ref) = first(getindices(ref)) === DISCONTIGUOUS

function getindicesonly(ref)
    indices = getindices(ref)
    @view(indices[isdiscontiguous(ref) + 1:end])
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
