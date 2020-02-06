function Loop(ls::LoopSet, l::Int, ::Type{UnitRange{Int}})
    start = gensym(:loopstart); stop = gensym(:loopstop)
    pushpreamble!(ls, Expr(:(=), start, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, @__FILE__), Expr(:(.), Expr(:ref, :lb, l), QuoteNode(:start)))))
    pushpreamble!(ls, Expr(:(=), stop, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, @__FILE__), Expr(:(.), Expr(:ref, :lb, l), QuoteNode(:stop)))))
    Loop(gensym(:n), 0, 1024, start, stop, false, false)::Loop
end
function Loop(ls::LoopSet, l::Int, ::Type{StaticUpperUnitRange{U}}) where {U}
    start = gensym(:loopstart)
    pushpreamble!(ls, Expr(:(=), start, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, @__FILE__), Expr(:(.), Expr(:ref, :lb, l), QuoteNode(:L)))))
    Loop(gensym(:n), U - 1024, U, start, Symbol(""), false, true)::Loop
end
function Loop(ls::LoopSet, l::Int, ::Type{StaticLowerUnitRange{L}}) where {L}
    stop = gensym(:loopstop)
    pushpreamble!(ls, Expr(:(=), stop, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, @__FILE__), Expr(:(.), Expr(:ref, :lb, l), QuoteNode(:U)))))
    Loop(gensym(:n), L, L + 1024, Symbol(""), stop, true, false)::Loop
end
function Loop(ls, l, ::Type{StaticUnitRange{L,U}}) where {L,U}
    Loop(gensym(:n), L, U, Symbol(""), Symbol(""), true, true)::Loop
end

function add_loops!(ls::LoopSet, LB)
    loopsyms = [gensym(:n) for _ ∈ eachindex(LB)]
    for (i,l) ∈ enumerate(LB)
        add_loop!(ls, Loop(ls, i, l)::Loop)
    end
end
function ArrayReferenceMeta(
    ls::LoopSet, ar::ArrayRefStruct, arraysymbolinds::Vector{Symbol}, opsymbols::Vector{Symbol},
    array::Symbol, vp::Symbol
)
    index_types = ar.index_types
    indices = ar.indices
    ni = filled_8byte_chunks(index_types)
    index_vec = Vector{Symbol}(undef, ni)
    loopedindex = fill(false, ni)
    while index_types != zero(UInt64)
        ind = indices % UInt8
        symind = if index_types == LoopIndex
            loopedindex[ni] = true
            ls.loopsymbols[ind]
        elseif index_types == ComputedIndex
            opsymbols[ind]
        else
            @assert index_types == SymbolicIndex
            arraysymbolinds[ind] 
        end
        index_vec[ni] = symind
        index_types >>>= 8
        indices >>>= 8
        ni -= 1
    end
    ArrayReferenceMeta(
        ArrayReference(array, index_vec),
        loopedindex, vp
    )
end

function add_mref!(ls::LoopSet, ars::ArrayRefStruct, arraysymbolinds::Vector{Symbol}, opsymbols::Vector{Symbol}, i::Int, ::Type{PackedStridedPointer{T, N}}) where {T, N}
    ar = ArrayReferenceMeta(ls, ars, arraysymbolinds, opsymbols, Symbol(""), gensym())
    pushpreamble!(ls, Expr(:(=), vptr(ar), Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, @__FILE__), Expr(:ref, :vargs, i))))
    ar
end
function add_mref!(ls::LoopSet, ars::ArrayRefStruct, arraysymbolinds::Vector{Symbol}, opsymbols::Vector{Symbol}, i::Int, ::Type{RowMajorStridedPointer{T, N}}) where {T, N}
    ar = ArrayReferenceMeta(ls, ars, arraysymbolinds, opsymbols, Symbol(""), gensym())
    reverse!(ar.loopedindex); reverse!(getindices(ar)) # reverse the listed indices here, and transpose it to make it column major
    pushpreamble!(ls, Expr(:(=), vptr(ar), Expr(:call, lv(:Transpose), Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, @__FILE__), Expr(:ref, :vargs, i)))))
    ar
end
function add_mref!(ls::LoopSet, ars::ArrayRefStruct, arraysymbolinds::Vector{Symbol}, opsymbols::Vector{Symbol}, i::Int, ::Type{StaticStridedPointer{T, X}}) where {T, X <: Tuple{1,Vararg}}
    ar = ArrayReferenceMeta(ls, ars, arraysymbolinds, opsymbols, Symbol(""), gensym())
    pushpreamble!(ls, Expr(:(=), vptr(ar), Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, @__FILE__), Expr(:ref, :vargs, i))))
    ar
end
function add_mref!(ls::LoopSet, ars::ArrayRefStruct, arraysymbolinds::Vector{Symbol}, opsymbols::Vector{Symbol}, i::Int, ::Type{StaticStridedPointer{T, X}}) where {T, X <: Tuple}
    ar = ArrayReferenceMeta(ls, ars, arraysymbolinds, opsymbols, Symbol(""), gensym())
    pushpreamble!(ls, Expr(:(=), vptr(ar), Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, @__FILE__), Expr(:ref, :vargs, i))))
    pushfirst!(getindices(ar), Symbol("##DISCONTIGUOUSSUBARRAY##"))
    ar
end
function add_mref!(ls::LoopSet, ars::ArrayRefStruct, arraysymbolinds::Vector{Symbol}, opsymbols::Vector{Symbol}, i::Int, ::Type{SparseStridedPointer{T, N}}) where {T, N}
    ar = ArrayReferenceMeta(ls, ars, arraysymbolinds, opsymbols, Symbol(""), gensym())
    pushpreamble!(ls, Expr(:(=), vptr(ar), Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, @__FILE__), Expr(:ref, :vargs, i))))
    pushfirst!(getindices(ar), Symbol("##DISCONTIGUOUSSUBARRAY##"))
    ar
end
function add_mref!(ls::LoopSet, ars::ArrayRefStruct, arraysymbolinds::Vector{Symbol}, opsymbols::Vector{Symbol}, i::Int, ::Type{StaticStridedStruct{T, X}}) where {T, X <: Tuple{1,Vararg}}
    ar = ArrayReferenceMeta(ls, ars, arraysymbolinds, opsymbols, Symbol(""), gensym())
    pushpreamble!(ls, Expr(:(=), vptr(ar), Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, @__FILE__), Expr(:ref, :vargs, i))))
    ar
end
function add_mref!(ls::LoopSet, ars::ArrayRefStruct, arraysymbolinds::Vector{Symbol}, opsymbols::Vector{Symbol}, i::Int, ::Type{StaticStridedStruct{T, X}}) where {T, X <: Tuple}
    ar = ArrayReferenceMeta(ls, ars, arraysymbolinds, opsymbols, Symbol(""), gensym())
    if last(X.parameters)::Int == 1
        reverse!(ar.loopedindex); reverse!(getindices(ar))
        pushpreamble!(ls, Expr(:(=), vptr(ar), Expr(:call, lv(:Transpose), Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, @__FILE__), Expr(:ref, :vargs, i)))))
    else
        pushpreamble!(ls, Expr(:(=), vptr(ar), Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, @__FILE__), Expr(:ref, :vargs, i))))
        pushfirst!(getindices(ar), Symbol("##DISCONTIGUOUSSUBARRAY##"))
    end
    ar
end
function add_mref!(ls::LoopSet, ars::ArrayRefStruct, arraysymbolinds::Vector{Symbol}, opsymbols::Vector{Symbol}, i::Int, ::Type{LoopValue})
    ar = ArrayReferenceMeta(ls, ars, arraysymbolinds, opsymbols, Symbol(""), gensym())
    pushpreamble!(ls, Expr(:(=), vptr(ar), LoopValue()))
    ar
end
function add_mref!(ls::LoopSet, ars::ArrayRefStruct, arraysymbolinds::Vector{Symbol}, opsymbols::Vector{Symbol}, i::Int, ::Type{<:AbstractRange{T}}) where {T}
    ar = ArrayReferenceMeta(ls, ars, arraysymbolinds, opsymbols, Symbol(""), gensym())
    pushpreamble!(ls, Expr(:(=), vptr(ar), Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, @__FILE__), Expr(:ref, :vargs, i))))
    ar
end



function create_mrefs!(ls::LoopSet, arf::Vector{ArrayRefStruct}, as::Vector{Symbol}, os::Vector{Symbol}, vargs)
    mrefs = Vector{ArrayReferenceMeta}(undef, length(arf))
    for i ∈ eachindex(arf)
        ref = add_mref!(ls, arf[i], as, os, i, vargs[i])::ArrayReferenceMeta
        mrefs[i] = ref
    end
    mrefs
end
function num_parameters(AM)
    num_param::Int = AM[1]
    num_param += length(AM[2].parameters)
    num_param + length(AM[3].parameters)
end
function process_metadata!(ls::LoopSet, AM, num_arrays::Int)::Vector{Symbol}
    num_asi = (AM[1])::Int
    arraysymbolinds = [gensym(:asi) for _ ∈ 1:num_asi]
    append!(ls.outer_reductions, AM[2].parameters)
    for (i,si) ∈ enumerate(AM[3].parameters)
        sii = si::Int
        s = gensym(:symlicm)
        push!(ls.preamble_symsym, (si, s))
        pushpreamble!(ls, Expr(:(=), s, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__,@__FILE__), Expr(:ref, :vargs, num_arrays + i))))
    end
    append!(ls.preamble_symint, AM[4].parameters)
    append!(ls.preamble_symfloat, AM[5].parameters)
    append!(ls.preamble_zeros, AM[6].parameters)
    append!(ls.preamble_ones, AM[7].parameters)
    arraysymbolinds
end
function parents_symvec(ls::LoopSet, u::Unsigned)
    i = filled_4byte_chunks(u)
    loops = Vector{Symbol}(undef, i)
    while u != zero(u)
        loops[i] = getloopsym(ls, ( u % UInt8 ) & 0x0f )
        i -= 1
        u >>= 4
    end
    loops
end
loopdependencies(ls::LoopSet, os::OperationStruct) = parents_symvec(ls, os.loopdeps)
reduceddependencies(ls::LoopSet, os::OperationStruct) = parents_symvec(ls, os.reduceddeps)
childdependencies(ls::LoopSet, os::OperationStruct) = parents_symvec(ls, os.childdeps)



function add_op!(
    ls::LoopSet, instr::Instruction, os::OperationStruct, mrefs::Vector{ArrayReferenceMeta}, opsymbol, elementbytes::Int
)
    optype = os.node_type
    # opsymbol = (isconstant(os) && instr != LOOPCONSTANT) ? instr.instr : opsymbol
    op = Operation(
        length(operations(ls)), opsymbol, elementbytes, instr,
        optype, loopdependencies(ls, os), reduceddependencies(ls, os),
        Operation[], (isload(os) | isstore(os)) ? mrefs[os.array] : NOTAREFERENCE,
        childdependencies(ls, os)
    )
    push!(ls.operations, op)
    op
end
function add_parents_to_op!(ls::LoopSet, parents::Vector{Operation}, up::Unsigned)
    ops = operations(ls)
    while up != zero(up)
        pushfirst!(parents, ops[ up % UInt8 ])
        up >>>= 8
    end
end
function add_parents_to_ops!(ls::LoopSet, ops::Vector{OperationStruct}, constoffset)
    for (i,op) ∈ enumerate(operations(ls))
        add_parents_to_op!(ls, parents(op), ops[i].parents)
        if isconstant(op)
            instr = instruction(op)
            if instr != LOOPCONSTANT && instr.mod !== :numericconstant
                constoffset += 1
                pushpreamble!(ls, Expr(:(=), instr.instr, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, @__FILE__), Expr(:ref, :vargs, constoffset))))
            end
        end
    end
    constoffset
end
function add_ops!(
    ls::LoopSet, instr::Vector{Instruction}, ops::Vector{OperationStruct}, mrefs::Vector{ArrayReferenceMeta}, opsymbols::Vector{Symbol}, constoffset::Int, elementbytes::Int
)
    for i ∈ eachindex(ops)
        os = ops[i]
        opsymbol = opsymbols[os.symid]
        add_op!(ls, instr[i], os, mrefs, opsymbol, elementbytes)
    end
    add_parents_to_ops!(ls, ops, constoffset)
end

# elbytes(::VectorizationBase.AbstractPointer{T}) where {T} = sizeof(T)::Int
typeeltype(::Type{P}) where {T,P<:VectorizationBase.AbstractPointer{T}} = T
typeeltype(::Type{LoopValue}) = Int8
typeeltype(::Type{<:AbstractRange{T}}) where {T} = T

function add_array_symbols!(ls::LoopSet, arraysymbolinds::Vector{Symbol}, offset::Int)
    for (i,as) ∈ enumerate(arraysymbolinds)
        pushpreamble!(ls, Expr(:(=), as, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, @__FILE__), Expr(:ref, :vargs, i + offset))))
    end
end
function extract_external_functions!(ls::LoopSet, offset::Int)
    for op ∈ operations(ls)
        if iscompute(op)
            instr = instruction(op)
            if instr.mod != :LoopVectorization
                offset += 1
                pushpreamble!(ls, Expr(:(=), instr.instr, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, @__FILE__), Expr(:ref, :vargs, offset))))
            end
        end
    end
    offset
end
function sizeofeltypes(v, num_arrays)::Int
    T = typeeltype(v[1])
    for i ∈ 2:num_arrays
        T = promote_type(T, typeeltype(v[i]))
    end
    sizeof(T)
end


function avx_loopset(instr, ops, arf, AM, LB, vargs)
    ls = LoopSet(:LoopVectorization)
    num_arrays = length(arf)
    elementbytes = sizeofeltypes(vargs, num_arrays)
    add_loops!(ls, LB)
    resize!(ls.loop_order, length(LB))
    arraysymbolinds = process_metadata!(ls, AM, length(arf))
    opsymbols = [gensym(:op) for _ ∈ eachindex(ops)]
    mrefs = create_mrefs!(ls, arf, arraysymbolinds, opsymbols, vargs)
    pushpreamble!(ls, Expr(:(=), ls.T, Expr(:call, :promote_type, [Expr(:call, :eltype, vptr(mref)) for mref ∈ mrefs]...)))
    num_params = num_arrays + num_parameters(AM)
    num_params = add_ops!(ls, instr, ops, mrefs, opsymbols, num_params, elementbytes)
    add_array_symbols!(ls, arraysymbolinds, num_arrays + length(ls.preamble_symsym))
    num_params = extract_external_functions!(ls, num_params)
    ls
end
function avx_body(UT, instr, ops, arf, AM, LB, vargs)
    ls = avx_loopset(instr, ops, arf, AM, LB, vargs)
    U, T = UT
    q = iszero(U) ? lower(ls) : lower(ls, U, T)
    length(ls.outer_reductions) == 0 ? push!(q.args, nothing) : push!(q.args, loopset_return_value(ls, Val(true)))
    q
end

function _avx_loopset(::Val{UT}, ::Type{OPS}, ::Type{ARF}, ::Type{AM}, lb::LB, vargs...) where {UT, OPS, ARF, AM, LB}
    OPSsv = OPS.parameters
    nops = length(OPSsv) ÷ 3
    instr = Instruction[Instruction(OPSsv[3i+1], OPSsv[3i+2]) for i ∈ 0:nops-1]
    ops = OperationStruct[ OPSsv[3i] for i ∈ 1:nops ]
    avx_loopset(
        instr, ops,
        ArrayRefStruct[ARF.parameters...],
        AM.parameters, LB.parameters, typeof.(vargs)
    )
end
@generated function _avx_!(::Val{UT}, ::Type{OPS}, ::Type{ARF}, ::Type{AM}, lb::LB, vargs...) where {UT, OPS, ARF, AM, LB}
    OPSsv = OPS.parameters
    nops = length(OPSsv) ÷ 3
    instr = Instruction[Instruction(OPSsv[3i+1], OPSsv[3i+2]) for i ∈ 0:nops-1]
    ops = OperationStruct[ OPSsv[3i] for i ∈ 1:nops ]
    avx_body(
        UT, instr, ops,
        ArrayRefStruct[ARF.parameters...],
        AM.parameters, LB.parameters, vargs
    )
end

