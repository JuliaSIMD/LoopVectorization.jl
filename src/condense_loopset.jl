
@enum IndexType::UInt8 NotAnIndex=0 LoopIndex=1 ComputedIndex=2 SymbolicIndex=3

Base.:|(u::Unsigned, it::IndexType) = u | UInt8(it)
Base.:(==)(u::Unsigned, it::IndexType) = (u % UInt8) == UInt8(it)

struct ArrayRefStruct{array,ptr}
    index_types::UInt64
    indices::UInt64
    offsets::UInt64
end
array(ar::ArrayRefStruct{a,p}) where {a,p} = a
ptr(ar::ArrayRefStruct{a,p}) where {a,p}   = p

function findindoradd!(v::Vector{T}, s::T) where {T}
    ind = findfirst(sᵢ -> sᵢ == s, v)
    ind === nothing || return ind
    push!(v, s)
    length(v)
end
function ArrayRefStruct(ls::LoopSet, mref::ArrayReferenceMeta, arraysymbolinds::Vector{Symbol})
    index_types = zero(UInt64)
    indices = zero(UInt64)
    offsets = zero(UInt64)
    indv = mref.ref.indices
    offv = mref.ref.offsets
    # we can discard that the array was considered discontiguous, as it should be recovered from type information
    start = 1 + (first(indv) === Symbol("##DISCONTIGUOUSSUBARRAY##"))
    for (n,ind) ∈ enumerate(@view(indv[start:end]))
        index_types <<= 8
        indices <<= 8
        offsets <<= 8
        offsets |= offv[n]
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
    ArrayRefStruct{mref.ref.array,mref.ptr}( index_types, indices, offsets )
end

struct OperationStruct <: AbstractLoopOperation
    # instruction::Instruction
    loopdeps::UInt64
    reduceddeps::UInt64
    childdeps::UInt64
    parents::UInt64
    node_type::OperationType
    array::UInt8
    symid::UInt8
end
optype(os) = os.node_type

function findmatchingarray(ls::LoopSet, mref::ArrayReferenceMeta)
    id = 0x01
    for r ∈ ls.refs_aliasing_syms
        r == mref && return id
        id += 0x01
    end
    0x00
end
filled_4byte_chunks(u::UInt64) = 16 - (leading_zeros(u) >>> 2)
filled_8byte_chunks(u::UInt64) = 8 - (leading_zeros(u) >>> 3)

# num_loop_deps(os::OperationStruct) = filled_4byte_chunks(os.loopdeps)
# num_reduced_deps(os::OperationStruct) = filled_4byte_chunks(os.reduceddeps)
# num_child_deps(os::OperationStruct) = filled_4byte_chunks(os.childdeps)
# num_parents(os::OperationStruct) = filled_4byte_chunks(os.parents)

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
    array = accesses_memory(op) ? findmatchingarray(ls, op.ref) : 0x00
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
            Expr(:call, :extract_data, Symbol(mangledvar(getop(ls, ls.outer_reductions[1])), 0))
        else
            Symbol(mangledvar(getop(ls, ls.outer_reductions[1])), 0)
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

@inline unwrap_array(A) = A
@inline unwrap_array(A::Union{SubArray,Transpose,Adjoint}) = parent(A)
@inline array_wrapper(A) = nothing
@inline array_wrapper(A::Transpose) = Transpose
@inline array_wrapper(A::Adjoint) = Adjoint
@inline array_wrapper(A::SubArray) = A.indices


# If you change the number of arguments here, make commensurate changes
# to the `insert!` locations in `setup_call_noinline`.
@generated function __avx__!(
    ::Val{UT}, ::Type{OPS}, ::Type{ARF}, ::Type{AM}, ::Type{LPSYM}, lb::LB,
    ::Val{AR}, ::Val{D}, ::Val{IND}, subsetvals, arraydescript, vargs::Vararg{<:Any,N}
) where {UT, OPS, ARF, AM, LPSYM, LB, N, AR, D, IND}
    num_vptrs = length(ARF.parameters)::Int
    vptrs = [gensym(:vptr) for _ ∈ 1:num_vptrs]
    call = Expr(:call, lv(:_avx_!), Val{UT}(), OPS, ARF, AM, LPSYM, :lb)
    for n ∈ 1:num_vptrs
        push!(call.args, vptrs[n])
    end
    q = Expr(:block)
    j = 0
    assigned_names = Vector{Symbol}(undef, length(AR))
    num_arrays = 0
    for i ∈ eachindex(AR)
        ari = (AR[i])::Int
        ind = (IND[i])::Union{Nothing,Int}
        LHS = ind === nothing ? gensym() : vptrs[ind]
        assigned_names[i] = LHS
        d = (D[i])::Union{Nothing,Int}
        if d === nothing # stridedpointer
            num_arrays += 1
            RHS = Expr(:call, lv(:stridedpointer), Expr(:ref, :vargs, ari), Expr(:ref, :arraydescript, ari))
        else #subsetview
            j += 1
            RHS = Expr(:call, :subsetview, assigned_names[ari], Expr(:call, Expr(:curly, :Val, d)), Expr(:ref, :subsetvals, j))
        end
        push!(q.args, Expr(:(=), LHS, RHS))
    end
    for n ∈ num_arrays+1:N
        push!(call.args, Expr(:ref, :vargs, n))
    end
    push!(q.args, call)
    Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, Symbol(@__FILE__)), q)
end

# Try to condense in type stable manner
function generate_call(ls::LoopSet, IUT, debug::Bool = false)
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
    loop_syms = Expr(:curly, :Tuple, map(QuoteNode, ls.loopsymbols)...)
    inline, U, T = IUT
    if inline | debug
        func = debug ? lv(:_avx_loopset_debug) : lv(:_avx_!)
        lbarg = debug ? Expr(:call, :typeof, loop_bounds) : loop_bounds
        q = Expr(
            :call, func, Expr(:call, Expr(:curly, :Val, (U,T))),
            operation_descriptions, arrayref_descriptions, argmeta, loop_syms, lbarg
        )
        debug && deleteat!(q.args, 2)
        foreach(ref -> push!(q.args, vptr(ref)), ls.refs_aliasing_syms)
    else
        arraydescript = Expr(:tuple)
        q = Expr(
            :call, lv(:__avx__!), Expr(:call, Expr(:curly, :Val, (U,T))),
            operation_descriptions, arrayref_descriptions, argmeta, loop_syms, loop_bounds, arraydescript
        )
        for array ∈ ls.includedactualarrays
            push!(q.args, Expr(:call, lv(:unwrap_array), array))
            push!(arraydescript.args, Expr(:call, lv(:array_wrapper), array))
        end
    end
    foreach(is -> push!(q.args, last(is)), ls.preamble_symsym)
    append!(q.args, arraysymbolinds)
    add_reassigned_syms!(q, ls)
    add_external_functions!(q, ls)
    q
end

function setup_call_noinline(ls::LoopSet, U = zero(Int8), T = zero(Int8))
    call = generate_call(ls, (false,U,T))
    hasouterreductions = length(ls.outer_reductions) > 0
    q = Expr(:block)
    vptrarrays = Expr(:tuple)
    vptrsubsetvals = Expr(:tuple)
    vptrsubsetdims = Expr(:tuple)
    vptrindices = Expr(:tuple)
    stridedpointerLHS = Symbol[]
    loopvalueLHS = Symbol[]
    for ex ∈ ls.preamble.args
        # vptrcalls = Expr(:tuple)
        if ex isa Expr && ex.head === :(=) && length(ex.args) == 2
            if ex.args[2] isa Expr && ex.args[2].head === :call
                gr = first(ex.args[2].args)
                if gr == lv(:stridedpointer)
                    array = ex.args[2].args[2]
                    arrayid = findfirst(a -> a === array, ls.includedactualarrays)
                    if arrayid isa Int
                        push!(vptrarrays.args, arrayid)
                    else
                        @assert array ∈ loopvalueLHS
                        push!(vptrarrays.args, -1)
                    end
                    push!(vptrsubsetdims.args, nothing)
                    vp = first(ex.args)::Symbol
                    push!(stridedpointerLHS, vp)
                    push!(vptrindices.args, findfirst(a -> vptr(a) == vp, ls.refs_aliasing_syms))
                elseif gr == lv(:subsetview)
                    array = ex.args[2].args[2]
                    vptrarrayid = findfirst(a -> a === array, stridedpointerLHS)#::Int
                    if vptrarrayid === nothing
                        @show array, stridedpointerLHS
                        @assert vptrarrayid isa Int
                    end
                    push!(vptrarrays.args, vptrarrayid::Int)
                    push!(vptrsubsetdims.args, ex.args[2].args[3].args[1].args[2])
                    push!(vptrsubsetvals.args, ex.args[2].args[4])
                    vp = first(ex.args)::Symbol
                    push!(stridedpointerLHS, vp)
                    push!(vptrindices.args, findfirst(a -> vptr(a) == vp, ls.refs_aliasing_syms))
                end
            end
        end
        push!(q.args, ex)
    end
    insert!(call.args, 8, Expr(:call, Expr(:curly, :Val, vptrarrays)))
    insert!(call.args, 9, Expr(:call, Expr(:curly, :Val, vptrsubsetdims)))
    insert!(call.args, 10, Expr(:call, Expr(:curly, :Val, vptrindices)))
    insert!(call.args, 11, vptrsubsetvals)
    if hasouterreductions
        outer_reducts = Expr(:local)
        for or ∈ ls.outer_reductions
            op = ls.operations[or]
            var = name(op)
            mvar = mangledvar(op)
            out = Symbol(mvar, 0)
            push!(outer_reducts.args, out)
            # push!(call.args, Symbol("##TYPEOF##", var))
        end
        push!(q.args, outer_reducts)
        retv = loopset_return_value(ls, Val(false))
        call = Expr(:(=), retv, call)
        push!(q.args, gc_preserve(ls, call))
        for or ∈ ls.outer_reductions
            op = ls.operations[or]
            var = name(op)
            mvar = mangledvar(op)
            instr = instruction(op)
            out = Symbol(mvar, 0)
            push!(q.args, Expr(:(=), var, Expr(:call, lv(reduction_scalar_combine(instr)), out, var)))
        end
    else
        push!(q.args, gc_preserve(ls, call))
    end
    q
end
function setup_call_inline(ls::LoopSet, U = zero(Int8), T = zero(Int8))
    call = generate_call(ls, (true,U,T))
    hasouterreductions = length(ls.outer_reductions) > 0
    if !hasouterreductions
        q = Expr(:block,gc_preserve(ls, call))
        append!(ls.preamble.args, q.args)
        return ls.preamble
    end
    retv = loopset_return_value(ls, Val(false))
    outer_reducts = Expr(:local)
    q = Expr(:block,gc_preserve(ls, Expr(:(=), retv, call)))
    for or ∈ ls.outer_reductions
        op = ls.operations[or]
        var = name(op)
        # push!(call.args, Symbol("##TYPEOF##", var))
        mvar = mangledvar(op)
        instr = instruction(op)
        out = Symbol(mvar, 0)
        push!(outer_reducts.args, out)
        push!(q.args, Expr(:(=), var, Expr(:call, lv(reduction_scalar_combine(instr)), out, var)))
    end
    hasouterreductions && pushpreamble!(ls, outer_reducts)
    append!(ls.preamble.args, q.args)
    ls.preamble
end
function setup_call_debug(ls::LoopSet)
    # avx_loopset(instr, ops, arf, AM, LB, vargs)
    pushpreamble!(ls, generate_call(ls, (true,zero(Int8),zero(Int8)), true))
    ls.preamble
end
function setup_call(ls::LoopSet, inline = Int8(2), U = zero(Int8), T = zero(Int8))
    # We outline/inline at the macro level by creating/not creating an anonymous function.
    # The old API instead was based on inlining or not inline the generated function, but
    # the generated function must be inlined into the initial loop preamble for performance reasons.
    # Creating an anonymous function and calling it also achieves the outlining, while still
    # inlining the generated function into the loop preamble.
    if inline == Int8(2)
        if num_loops(ls) == 1
            iszero(U) ? lower(ls) : lower(ls, U, -one(U))
        else
            setup_call_inline(ls, U, T)
        end
    else
        setup_call_noinline(ls, U, T)
    end
end

