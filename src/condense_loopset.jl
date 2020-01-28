
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
        if mref.loopedindex[n]
            index_types |= LoopIndex
            indices |= getloopid(ls, ind)
        else
            parent = get(ls.opdict, ind, nothing)
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
    # instruction::Instruction
    loopdeps::UInt64
    reduceddeps::UInt64
    childdeps::UInt64
    parents::UInt64
    node_type::OperationType
    array::UInt8
    symid::UInt8
end
isload(os::OperationStruct) = os.node_type == memload
isstore(os::OperationStruct) = os.node_type == memstore
iscompute(os::OperationStruct) = os.node_type == compute
isconstant(os::OperationStruct) = os.node_type == constant
function findmatchingarray(ls::LoopSet, array::Symbol)
    id = 0x01
    for as ∈ ls.refs_aliasing_syms
        vptr(as) === array && return id
        id += 0x01
    end
    0x00
end
filled_4byte_chunks(u::UInt64) = 16 - (leading_zeros(u) >>> 2)
filled_8byte_chunks(u::UInt64) = 8 - (leading_zeros(u) >>> 3)

num_loop_deps(os::OperationStruct) = filled_4byte_chunks(os.loopdeps)
num_reduced_deps(os::OperationStruct) = filled_4byte_chunks(os.reduceddeps)
num_child_deps(os::OperationStruct) = filled_4byte_chunks(os.childdeps)
num_parents(os::OperationStruct) = filled_4byte_chunks(os.parents)

function shifted_loopset(ls::LoopSet, loopsyms::Vector{Symbol})
    ld = zero(UInt64) # leading_zeros(ld) >> 2 yields the number of loopdeps
    for d ∈ loopsyms
        ld <<= 4
        ld |= getloopid(ls, d)::Int
    end
    ld
end
loopdeps_uint(ls::LoopSet, op::Operation) = shifted_loopset(ls, loopdependencies(op))
reduceddeps_uint(ls::LoopSet, op::Operation) = shifted_loopset(ls, reduceddependencies(op))
childdeps_uint(ls::LoopSet, op::Operation) = shifted_loopset(ls, reducedchildren(op))
function parents_uint(ls::LoopSet, op::Operation)
    p = zero(UInt64)
    for parent ∈ parents(op)
        p <<= 8
        p |= identifier(parent)
    end
    p
end
function OperationStruct!(varnames::Vector{Symbol}, ls::LoopSet, op::Operation)
    instr = instruction(op)
    ld = loopdeps_uint(ls, op)
    rd = reduceddeps_uint(ls, op)
    cd = childdeps_uint(ls, op)
    p = parents_uint(ls, op)
    array = accesses_memory(op) ? findmatchingarray(ls, vptr(op.ref)) : 0x00
    OperationStruct(
        ld, rd, cd, p, op.node_type, array, findindoradd!(varnames, name(op))
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
            Expr(:call, :(:), loop.startsym, loop.stopsym)
        end
        push!(lbd.args, lexpr)
    end
    lbd
end

function argmeta_and_consts_description(ls::LoopSet, arraysymbolinds)
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

function loopset_return_value(ls::LoopSet, ::Val{extract}) where {extract}
    if length(ls.outer_reductions) == 1
        if extract
            Expr(:call, :extract_data, Symbol(mangledvar(operations(ls)[ls.outer_reductions[1]]), 0))
        else
            Symbol(mangledvar(operations(ls)[ls.outer_reductions[1]]), 0)
        end
    elseif length(ls.outer_reductions) > 1
        ret = Expr(:tuple)
        ops = operations(ls)
        for or ∈ ls.outer_reductions
            if extract
                push!(ret.args, Expr(:call, :extract_data, Symbol(mangledvar(ops[or]), 0)))
            else
                push!(ret.args, Symbol(mangledvar(ops[or]), 0))
            end
        end
        ret
    else
        nothing
    end
end

function add_reassigned_syms!(q::Expr, ls::LoopSet)
    for op ∈ operations(ls)
        if isconstant(op)
            instr = instruction(op)
            (instr == LOOPCONSTANT || instr.mod === :numericconstant) || push!(q.args, instr.instr)
        end
    end
end
function add_external_functions!(q::Expr, ls::LoopSet)
    for op ∈ operations(ls)
        if iscompute(op)
            instr = instruction(op)
            if instr.mod !== :LoopVectorization
                push!(q.args, Expr(:(.), instr.mod, QuoteNode(instr.instr)))
            end
        end
    end
end

# Try to condense in type stable manner
function generate_call(ls::LoopSet, IUT)
    operation_descriptions = Expr(:curly, :Tuple)
    varnames = Symbol[]
    for op ∈ operations(ls)
        instr = instruction(op)
        push!(operation_descriptions.args, QuoteNode(instr.mod))
        push!(operation_descriptions.args, QuoteNode(instr.instr))
        push!(operation_descriptions.args, OperationStruct!(varnames, ls, op))
    end
    arraysymbolinds = Symbol[]
    arrayref_descriptions = Expr(:curly, :Tuple)
    foreach(ref -> push!(arrayref_descriptions.args, ArrayRefStruct(ls, ref, arraysymbolinds)), ls.refs_aliasing_syms)
    argmeta = argmeta_and_consts_description(ls, arraysymbolinds)
    loop_bounds = loop_boundaries(ls)

    q = Expr(:call, lv(:_avx_!), Expr(:call, Expr(:curly, :Val, IUT)), operation_descriptions, arrayref_descriptions, argmeta, loop_bounds)
    foreach(ref -> push!(q.args, vptr(ref)), ls.refs_aliasing_syms)
    foreach(is -> push!(q.args, last(is)), ls.preamble_symsym)
    append!(q.args, arraysymbolinds)
    add_reassigned_syms!(q, ls)
    add_external_functions!(q, ls)
    q
end

function setup_call(ls::LoopSet, inline = Int8(2), U = zero(Int8), T = zero(Int8))
    call = generate_call(ls, (inline,U,T))
    hasouterreductions = length(ls.outer_reductions) > 0
    if hasouterreductions
        retv = loopset_return_value(ls, Val(false))
        call = Expr(:(=), retv, call)
    end
    q = Expr(:block,gc_preserve(ls, call))
    outer_reducts = Expr(:local)
    for or ∈ ls.outer_reductions
        op = ls.operations[or]
        var = name(op)
        mvar = mangledvar(op)
        instr = instruction(op)
        out = Symbol(mvar, 0)
        push!(outer_reducts.args, out)
        # push!(q.args, Expr(:(=), var, Expr(:call, lv(reduction_scalar_combine(instr)), Expr(:call, lv(:SVec), out), var)))
        push!(q.args, Expr(:(=), var, Expr(:call, lv(reduction_scalar_combine(instr)), out, var)))
    end
    hasouterreductions && pushpreamble!(ls, outer_reducts)
    append!(ls.preamble.args, q.args)
    ls.preamble
end


