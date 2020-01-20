
struct ArrayReference
    array::Symbol
    indices::Vector{Symbol}
end
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


function ref_from_expr(ex, offset1::Int, offset2::Int)
    (ex.args[1 + offset1])::Symbol, @view(ex.args[2 + offset2:end])
end
ref_from_ref(ex::Expr) = ref_from_expr(ex, 0, 0)
ref_from_getindex(ex::Expr) = ref_from_expr(ex, 1, 1)
ref_from_setindex(ex::Expr) = ref_from_expr(ex, 1, 2)
function ref_from_expr(ex::Expr)
    if ex.head === :ref
        ref_from_ref(ex)
    else#if ex.head === :call
        f = first(ex.args)::Symbol
        f === :getindex ? ref_from_getindex(ex) : ref_from_setindex(ex)
    end
end

function Base.:(==)(x::ArrayReference, y::Expr)::Bool
    ya, yinds = if y.head === :ref
        ref_from_ref(y)
    elseif y.head === :call
        f = first(y.args)
        if f === :getindex
            ya, yinds = ref_from_getindex(y)
        elseif f === :setindex!
            ya, yinds = ref_from_setindex(y)
        else
            return false
        end
    else
        return false
    end
    x.array == ya || return false
end
Base.:(==)(x::ArrayReference, y::ArrayReferenceMeta) = x == y.ref
Base.:(==)(x::ArrayReferenceMeta, y::ArrayReference) = x.ref == y
Base.:(==)(x::ArrayReference, y) = false
Base.:(==)(x::ArrayReferenceMeta, y) = false

@enum OperationType begin
    constant
    memload
    compute
    memstore
end

# TODO: can some computations be cached in the operations?
"""
"""
struct Operation
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
    function Operation(
        identifier::Int,
        variable,
        elementbytes,
        instruction,
        node_type,
        dependencies = Symbol[],
        reduced_deps = Symbol[],
        parents = Operation[],
        ref::ArrayReferenceMeta = NOTAREFERENCE
    )
        new(
            identifier, variable, elementbytes, instruction, node_type,
            convert(Vector{Symbol},dependencies),
            convert(Vector{Symbol},reduced_deps),
            convert(Vector{Operation},parents),
            ref,
            Symbol("##", variable, :_)
        )
    end
end

function matches(op1::Operation, op2::Operation)
    op1.instruction === op2.instruction || return false
    op1.node_type == op2.node_type || return false
    isconstant(op1) && return false
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
    end
end

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

struct ArrayReferenceMetaPosition
    mref::ArrayReferenceMeta
    parents::Vector{Operation}
    loopdependencies::Vector{Symbol}
    reduceddeps::Vector{Symbol}
end
function ArrayReferenceMetaPosition(parents::Vector{Operation}, ldref::Vector{Symbol}, reduceddeps::Vector{Symbol})
    ArrayReferenceMetaPosition( NOTAREFERENCE, parents, ldref, reduceddeps )
end
function Operation(id::Int, var::Symbol, elementbytes::Int, instr, optype::OperationType, mpref::ArrayReferenceMetaPosition)
    Operation( id, var, elementbytes, instr, optype, mpref.loopdependencies, mpref.reduceddeps, mpref.parents, mpref.mref )
end
Base.:(==)(x::ArrayReferenceMetaPosition, y::ArrayReferenceMetaPosition) = x.mref.ref == y.mref.ref
# Avoid memory allocations by using this for ops that aren't references
const NOTAREFERENCE = ArrayReferenceMeta(ArrayReference(Symbol(""), Union{Symbol,Int}[]),Bool[],Symbol(""))
const NOTAREFERENCEMP = ArrayReferenceMetaPosition(NOTAREFERENCE, NOPARENTS, Symbol[], Symbol[])
name(mpref::ArrayReferenceMetaPosition) = name(mpref.mref.ref)

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



