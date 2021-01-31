
@enum IndexType::UInt8 NotAnIndex=0 LoopIndex=1 ComputedIndex=2 SymbolicIndex=3

Base.:|(u::Unsigned, it::IndexType) = u | UInt8(it)
Base.:(==)(u::Unsigned, it::IndexType) = (u % UInt8) == UInt8(it)

"""
    ArrayRefStruct

A condensed representation of an [`ArrayReference`](@ref).
It supports array-references with up to 8 indexes, where the data for each consecutive index is packed into corresponding 8-bit fields
of `index_types` (storing the enum `IndexType`), `indices` (the `id` for each index symbol), and `offsets` (currently unused).
"""
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
function ArrayRefStruct(ls::LoopSet, mref::ArrayReferenceMeta, arraysymbolinds::Vector{Symbol}, ids::Vector{Int})
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
        offsets |= (offv[n] % UInt8)
        if mref.loopedindex[n]
            index_types |= LoopIndex
            indices |= getloopid(ls, ind)
        else
            parent = get(ls.opdict, ind, nothing)
            @assert !isnothing(parent) "Index $ind not found in array."
            # if parent === nothing
            #     index_types |= SymbolicIndex
            #     indices |= findindoradd!(arraysymbolinds, ind)
            # else
            index_types |= ComputedIndex
            indices |= ids[identifier(parent)]
            # end
        end
    end
    ArrayRefStruct{mref.ref.array,mref.ptr}( index_types, indices, offsets )
end

"""
    OperationStruct

A condensed representation of an [`Operation`](@ref).
"""
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
# filled_4byte_chunks(u::UInt64) = 16 - (leading_zeros(u) >>> 2)
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
function OperationStruct!(varnames::Vector{Symbol}, ids::Vector{Int}, ls::LoopSet, op::Operation)
    instr = instruction(op)
    ld = loopdeps_uint(ls, op)
    rd = reduceddeps_uint(ls, op)
    cd = childdeps_uint(ls, op)
    p = parents_uint(ls, op)
    array = accesses_memory(op) ? findmatchingarray(ls, op.ref) : 0x00
    ids[identifier(op)] = id = findindoradd!(varnames, name(op))
    OperationStruct(
        ld, rd, cd, p, op.node_type, array, id
    )
end
## turn a LoopSet into a type object which can be used to reconstruct the LoopSet.

function loop_boundary!(q::Expr, loop::Loop)
    if loop.startexact & loop.stopexact
        push!(q.args, Expr(:call, lv(:OptionallyStaticUnitRange), staticexpr(loop.starthint), staticexpr(loop.stophint)))
    elseif loop.rangesym === Symbol("")
        lb = if startexact
            Expr(:call, lv(:OptionallyStaticUnitRange), staticexpr(loop.starthint), loop.stopsym)
        elseif stopexact
            Expr(:call, lv(:OptionallyStaticUnitRange), loop.startsym, staticexpr(loop.stophint))
        else
            Expr(:call, :(:), loop.startsym, loop.stopsym)
        end
        push!(q.args, lb)
    else
        push!(q.args, loop.rangesym)
    end
end

function loop_boundaries(ls::LoopSet)
    lbd = Expr(:tuple)
    foreach(loop -> loop_boundary!(lbd, loop), ls.loops)
    lbd
end

tuple_expr(v) = tuple_expr(identity, v)
function tuple_expr(f, v)
    t = Expr(:tuple)
    for vᵢ ∈ v
        push!(t.args, f(vᵢ))
    end
    t
end

function argmeta_and_consts_description(ls::LoopSet, arraysymbolinds)
    Expr(
        :tuple,
        length(arraysymbolinds),
        tuple_expr(ls.outer_reductions),
        tuple_expr(first, ls.preamble_symsym),
        tuple_expr(ls.preamble_symint),
        tuple_expr(ls.preamble_symfloat),
        tuple_expr(ls.preamble_zeros),
        tuple_expr(ls.preamble_funcofeltypes)
    )
end

function loopset_return_value(ls::LoopSet, ::Val{extract}) where {extract}
    @assert !iszero(length(ls.outer_reductions))
    if isone(length(ls.outer_reductions))
        op = getop(ls, ls.outer_reductions[1])
        if extract
            if (isu₁unrolled(op) | isu₂unrolled(op))
                Expr(:call, :data, Symbol(mangledvar(op), 0))
            else
                Expr(:call, :data, mangledvar(op))
            end
        else
            Symbol(mangledvar(op), 0)
        end
    else#if length(ls.outer_reductions) > 1
        ret = Expr(:tuple)
        ops = operations(ls)
        for or ∈ ls.outer_reductions
            op = ops[or]
            if extract
                push!(ret.args, Expr(:call, :data, Symbol(mangledvar(op), 0)))
            else
                push!(ret.args, Symbol(mangledvar(ops[or]), 0))
            end
        end
        ret
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
                push!(q.args, instr.instr)
            end
        end
    end
end

function check_if_empty(ls::LoopSet, q::Expr)
    lb = loop_boundaries(ls)
    Expr(:if, Expr(:call, :!, Expr(:call, :any, :isempty, lb)), q)
end

val(x) = Expr(:call, Expr(:curly, :Val, x))
# Try to condense in type stable manner
function generate_call(ls::LoopSet, inline_unroll::NTuple{3,Int8}, debug::Bool = false)
    operation_descriptions = Expr(:tuple)
    varnames = Symbol[]; ids = Vector{Int}(undef, length(operations(ls)))
    for op ∈ operations(ls)
        instr = instruction(op)
        push!(operation_descriptions.args, QuoteNode(instr.mod))
        push!(operation_descriptions.args, QuoteNode(instr.instr))
        push!(operation_descriptions.args, OperationStruct!(varnames, ids, ls, op))
    end
    arraysymbolinds = Symbol[]
    arrayref_descriptions = Expr(:tuple)
    foreach(ref -> push!(arrayref_descriptions.args, ArrayRefStruct(ls, ref, arraysymbolinds, ids)), ls.refs_aliasing_syms)
    argmeta = argmeta_and_consts_description(ls, arraysymbolinds)
    loop_bounds = loop_boundaries(ls)
    loop_syms = tuple_expr(QuoteNode, ls.loopsymbols)
    inline, u₁, u₂ = inline_unroll
    func = debug ? lv(:_avx_loopset_debug) : lv(:_avx_!)
    lbarg = debug ? Expr(:call, :typeof, loop_bounds) : loop_bounds
    unroll_param_tup = Expr(
        :tuple, inline, u₁, u₂,
        Expr(:call, lv(:unwrap), VECTORWIDTHSYMBOL),
        Expr(:call, lv(:unwrap), Expr(:call, lv(:register_size))),
        Expr(:call, lv(:unwrap),
            Expr(:call, lv(:ifelse),
                Expr(:call, lv(:unwrap), Expr(:call, lv(:has_opmask_registers))),
                Expr(:call, lv(:unwrap), Expr(:call, lv(:register_count))),
                Expr(:call, lv(:unwrap), Expr(:call, :(-), Expr(:call, lv(:register_count)), Expr(:call, lv(:One))))
            )
        ),
        Expr(:call, lv(:unwrap), Expr(:call, lv(:cache_linesize)))
    )
    q = Expr(
        :call, func, val(unroll_param_tup),
        val(operation_descriptions), val(arrayref_descriptions), val(argmeta), val(loop_syms)
    )
    # debug && deleteat!(q.args, 2)
    vargs_as_tuple = true#!debug
    vargs_as_tuple || push!(q.args, lbarg)
    extra_args = vargs_as_tuple ? Expr(:tuple) : q
    foreach(ref -> push!(extra_args.args, vptr(ref)), ls.refs_aliasing_syms)

    foreach(is -> push!(extra_args.args, last(is)), ls.preamble_symsym)
    append!(extra_args.args, arraysymbolinds)
    add_reassigned_syms!(extra_args, ls)
    add_external_functions!(extra_args, ls)
    # debug && return q
    vargs_as_tuple && push!(q.args, Expr(:tuple, lbarg, extra_args))
    vecwidthdefq = Expr(:block)
    define_eltype_vec_width!(vecwidthdefq, ls, nothing)
    Expr(:block, vecwidthdefq, q)
end


"""
    check_args(::Vararg{AbstractArray})


LoopVectorization will optimize an `@avx` loop if `check_args` on each on the indexed abstract arrays returns true.
It returns true for `AbstractArray{T}`s when `check_type(T) == true` and the array or its parent is a `StridedArray` or `AbstractRange`.

To provide support for a custom array type, ensure that `check_args` returns true, either through overloading it or subtyping `DenseArray`.
Additionally, define `pointer` and `stride` methods.
"""
@inline function check_args(A::AbstractArray{T}) where {T}
    check_type(T) && ArrayInterface.device(A) === ArrayInterface.CPUPointer()
end
@inline check_args(A::BitVector) = true
@inline check_args(A::BitArray) = iszero(size(A,1) & 7)
@inline check_args(::VectorizationBase.AbstractStridedPointer) = true
@inline check_args(_) = false
@inline check_args(A, B, C::Vararg{Any,K}) where {K} = check_args(A) && check_args(B, C...)
@inline check_args(::AbstractRange{T}) where {T} = check_type(T)
@inline check_args(::Type{T}) where {T <: VectorizationBase.NativeTypesV} = true
"""
    check_type(::Type{T}) where {T}

Returns true if the element type is supported.
"""
check_type(::Type{T}) where {T <: NativeTypes} = true
check_type(::Type{T}) where {T} = false

function check_args_call(ls::LoopSet)
    q = Expr(:call, lv(:check_args))
    append!(q.args, ls.includedactualarrays)
    for r ∈ ls.outer_reductions
        push!(q.args, Expr(:call, :typeof, name(ls.operations[r])))
    end
    q
end

make_fast(q) = Expr(:macrocall, Symbol("@fastmath"), LineNumberNode(@__LINE__,Symbol(@__FILE__)), q)
make_crashy(q) = Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__,Symbol(@__FILE__)), q)

@inline vecmemaybe(x::NativeTypes) = x
@inline vecmemaybe(x::VectorizationBase._Vec) = Vec(x)
@inline vecmemaybe(x::Tuple) = VectorizationBase.VecUnroll(x)

function setup_call_inline(ls::LoopSet, inline::Int8 = zero(Int8), U::Int8 = zero(Int8), T::Int8 = zero(Int8))
    call = generate_call(ls, (inline,U,T))
    if iszero(length(ls.outer_reductions))
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
        push!(q.args, Expr(:(=), var, Expr(:call, lv(reduction_scalar_combine(instr)), Expr(:call, lv(:vecmemaybe), out), var)))
    end
    pushpreamble!(ls, outer_reducts)
    append!(ls.preamble.args, q.args)
    ls.preamble
end
function setup_call_debug(ls::LoopSet)
    # avx_loopset(instr, ops, arf, AM, LB, vargs)
    pushpreamble!(ls, generate_call(ls, (zero(Int8),zero(Int8),zero(Int8)), true))
    Expr(:block, ls.prepreamble, ls.preamble)
end
function setup_call(ls::LoopSet, q::Expr, source::LineNumberNode, inline::Int8 = zero(Int8), check_empty::Bool = false, u₁::Int8 = zero(Int8), u₂::Int8 = zero(Int8))
    # We outline/inline at the macro level by creating/not creating an anonymous function.
    # The old API instead was based on inlining or not inline the generated function, but
    # the generated function must be inlined into the initial loop preamble for performance reasons.
    # Creating an anonymous function and calling it also achieves the outlining, while still
    # inlining the generated function into the loop preamble.
    lnns = extract_all_lnns(q)
    pushfirst!(lnns, source)
    call = setup_call_inline(ls, inline, u₁, u₂)
    call = check_empty ? check_if_empty(ls, call) : call
    isnothing(q) && return Expr(:block, ls.prepreamble, call)
    result = Expr(:block, ls.prepreamble, Expr(:if, check_args_call(ls), call, make_crashy(make_fast(q))))
    prepend_lnns!(result, lnns)
    return result
end
