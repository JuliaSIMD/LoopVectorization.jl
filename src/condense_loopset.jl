
@enum IndexType::UInt8 NotAnIndex=0 LoopIndex=1 ComputedIndex=2 SymbolicIndex=3

Base.:|(u::Unsigned, it::IndexType) = u | UInt8(it)
Base.:(==)(u::Unsigned, it::IndexType) = (u % UInt8) == UInt8(it)

struct ArrayRefStruct
    index_types::UInt64
    indices::UInt64
end

function findindoradd!(v::Vector{T}, s::T) where {T}
    ind = findfirst(sᵢ -> sᵢ == s, v)
    ind === nothing || return ind
    push!(v, s)
    length(v)
end
function ArrayRefStruct(ls::LoopSet, mref::ArrayReferenceMeta, arraysymbolinds::Vector{Symbol})
    index_types = zero(UInt64)
    indices = zero(UInt64)
    indv = mref.ref.indices
    # we can discard that the array was considered discontiguous, as it should be recovered from type information
    start = 1 + (first(indv) === Symbol("##DISCONTIGUOUSSUBARRAY##"))
    for (n,ind) ∈ enumerate(@view(indv[start:end]))
        index_types <<= 8
        indices <<= 8
        if mref.loopindex[n]
            index_types |= LoopIndex
        else
            parent = getop(opdict, ind, nothing)
            if parent === nothing
                index_types |= SymbolicIndex
                indices |= findindoradd!(arraysymbolinds, ind)
            else
                index_types |= ComputedIndex
                indices |= identifier(parent)
            end
        end
    end
    ArrayRefStruct( index_types, indices )
end

struct OperationStruct
    instruction::Instruction
    loopdeps::UInt64
    reduceddeps::UInt64
    parents::UInt64
    array::UInt8
end
function findmatchingarray(ls::LoopSet, array::Symbol)
    id = 0x01
    for as ∈ ls.refs_aliasing_syms
        vptr(as) === array && return id
        id += 0x01
    end
    0x00
end
filled_4byte_chunks(u::UInt64) = leading_zeros(u) >>> 2
num_loop_deps(os::OperationStruct) = filled_4byte_chunks(os.loopdeps)
num_reduced_deps(os::OperationStruct) = filled_4byte_chunks(os.reduced_deps)
num_parents(os::OperationStruct) = filled_4byte_chunks(os.parents)

function loopdeps_uint(ls::LoopSet, loopsyms::Vector{Symbol})
    ld = zero(UInt64) # leading_zeros(ld) >> 2 yields the number of loopdeps
    for d ∈ loopsyms
        ld <<= 4
        ld |= getloopid(ls, d)
    end
    ld
end
loopdeps_uint(ls::LoopSet, op::Operation) = shifted_loopset(ls, loopdependencies(op))
reduceddeps_uint(ls::LoopSet, op::Operation) = shifted_loopset(ls, reduceddependencies(op))
function parents_uint(ls::LoopSet, op::Operation)
    p = zero(UInt64)
    for parent ∈ parents(op)
        p <<= 8
        p |= identifier(op)
    end
    p
end
function OperationStruct(ls::LoopSet, op::Operation)
    instr = instruction(op)
    ld = loopdeps_uint(ls, op)
    rd = reduceddeps_uint(ls, op)
    p = parents_uint(ls, op)
    array = accesses_memory(op) ? findmatchingarray(ls, vptr(op.ref)) : 0x00
    OperationStruct(
        instr, ld, rd, p, array
    )
end
## turn a LoopSet into a type object which can be used to reconstruct the LoopSet.


function loop_boundaries(ls::LoopSet)
    lbd = Expr(:tuple)
    for loop ∈ ls.loops
        startexact = loop.startexact
        stopexact = loop.stopexact
        lexpr = if startexact & stopexact
            Expr(:call, Expr(:curly, lv(:StaticUnitRange), loop.starthint, loop.stophint))
        elseif startexact
            Expr(:call, Expr(:curly, lv(:StaticLowerUnitRange), loop.starthint), loop.stopsym)
        elseif stopexact
            Expr(:call, Expr(:curly, lv(:StaticUpperUnitRange), loop.stophint), loop.startsym)
        else
            Expr(:call, Expr(:call, :(:), loop.startsym, loop.stopsym))
        end
        push!(lbd, lexpr)
    end
    lbd
end

function argmeta_and_costs_description(ls::LoopSet, arraysymbolinds)
    Expr(
        :curly, :Tuple,
        length(arraysymbolinds),
        Expr(:curly, :Tuple, ls.outer_reductions...),
        Expr(:curly, :Tuple, first.(ls.preamble_symsym)...),
        Expr(:curly, :Tuple, ls.preamble_symint...),
        Expr(:curly, :Tuple, ls.preamble_symfloat...),
        Expr(:curly, :Tuple, ls.preamble_zeros...),
        Expr(:curly, :Tuple, ls.preamble_ones...)
    )
end

# Try to condense in type stable manner
function generate_call(ls::LoopSet)
    operation_descriptions = Expr(:curly, :Tuple)
    foreach(op -> push!(operation_descriptions.args, OperationStruct(ls, op)), operations(ls))
    arraysymbolinds = Symbol[]
    arrayref_descriptions = Expr(:curly, :Tuple)
    foreach(ref -> push!(arrayref_descriptions.args, ArrayRefStruct(ls, ref, arraysymbolinds)), ls.refs_aliasing_syms)
    argmeta = argmeta_and_consts_description(ls, arraysymbolinds)
    loop_bounds = loop_boundaries(ls)

    q = Expr(:call, :_avx!, operation_descriptions, arrayref_descriptions, argmeta, loop_bounds)

    foreach(ref -> push!(q.args, vptr(ref)), ls.refs_aliasing_syms)
    foreach(is -> push!(q.args, last(is)), ls.preamble_symsym)
    append!(q.args, arraysymbolinds)
    q
end



