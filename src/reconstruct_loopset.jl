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
    for l ∈ LB
        add_loop!(ls, Loop(ls, l, LB)::Loop)
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
        ArrayReference(vp, index_vec),
        loopedindex, array
    )
end

function add_mref!(ls::LoopSet, ars::ArrayRefStruct, arraysymbolinds::Vector{Symbol}, opsymbols::Vector{Symbol}, i::Int, ::Type{PackedStridedPointer{T, N}}) where {T, N}
    ar = ArrayReferenceMeta(ls, ar, arraysymbolinds, opsymbols, Symbol(""), gensym())
    pushpreamble!(ls, Expr(:(=), vptr(ar), Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, @__FILE__), Expr(:ref, :vargs, i))))
    ar
end
function add_mref!(ls::LoopSet, ars::ArrayRefStruct, arraysymbolinds::Vector{Symbol}, opsymbols::Vector{Symbol}, i::Int, ::Type{RowMajorStridedPointer{T, N}}) where {T, N}
    ar = ArrayReferenceMeta(ls, ar, arraysymbolinds, opsymbols, Symbol(""), gensym())
    reverse!(ar.loopedindex); reverse!(getindices(ar)) # reverse the listed indices here, and transpose it to make it column major
    pushpreamble!(ls, Expr(:(=), vptr(ar), Expr(:call, lv(:Transpose), Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, @__FILE__), Expr(:ref, :vargs, i)))))
    ar
end
function add_mref!(ls::LoopSet, ars::ArrayRefStruct, arraysymbolinds::Vector{Symbol}, opsymbols::Vector{Symbol}, i::Int, ::Type{StaticStridedPointer{T, X}}) where {T, X <: Tuple{1,Vararg}}
    ar = ArrayReferenceMeta(ls, ar, arraysymbolinds, opsymbols, Symbol(""), gensym())
    pushpreamble!(ls, Expr(:(=), vptr(ar), Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, @__FILE__), Expr(:ref, :vargs, i))))
    ar
end
function add_mref!(ls::LoopSet, ars::ArrayRefStruct, arraysymbolinds::Vector{Symbol}, opsymbols::Vector{Symbol}, i::Int, ::Type{StaticStridedPointer{T, X}}) where {T, X <: Tuple}
    ar = ArrayReferenceMeta(ls, ar, arraysymbolinds, opsymbols, Symbol(""), gensym())
    pushpreamble!(ls, Expr(:(=), vptr(ar), Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, @__FILE__), Expr(:ref, :vargs, i))))
    pushfirst!(getindices(ar), Symbol("##DISCONTIGUOUSSUBARRAY##"))
    ar
end
function add_mref!(ls::LoopSet, ars::ArrayRefStruct, arraysymbolinds::Vector{Symbol}, opsymbols::Vector{Symbol}, i::Int, ::Type{SparseStridedPointer{T, N}}) where {T, N}
    ar = ArrayReferenceMeta(ls, ar, arraysymbolinds, opsymbols, Symbol(""), gensym())
    pushpreamble!(ls, Expr(:(=), vptr(ar), Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, @__FILE__), Expr(:ref, :vargs, i))))
    pushfirst!(getindices(ar), Symbol("##DISCONTIGUOUSSUBARRAY##"))
    ar
end
function add_mref!(ls::LoopSet, ars::ArrayRefStruct, arraysymbolinds::Vector{Symbol}, opsymbols::Vector{Symbol}, i::Int, ::Type{StaticStridedStruct{T, X}}) where {T, X <: Tuple{1,Vararg}}
    ar = ArrayReferenceMeta(ls, ar, arraysymbolinds, opsymbols, Symbol(""), gensym())
    pushpreamble!(ls, Expr(:(=), vptr(ar), Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, @__FILE__), Expr(:ref, :vargs, i))))
    ar
end
function add_mref!(ls::LoopSet, ars::ArrayRefStruct, arraysymbolinds::Vector{Symbol}, opsymbols::Vector{Symbol}, i::Int, ::Type{StaticStridedStruct{T, X}}) where {T, X <: Tuple}
    ar = ArrayReferenceMeta(ls, ar, arraysymbolinds, opsymbols, Symbol(""), gensym())
    pushpreamble!(ls, Expr(:(=), vptr(ar), Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, @__FILE__), Expr(:ref, :vargs, i))))
    pushfirst!(getindices(ar), Symbol("##DISCONTIGUOUSSUBARRAY##"))
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
function process_metadata!(ls::LoopSet, AM, num_arrays::Int)
    num_asi = (AM[1])::Int
    arraysymbolinds = [gensym(:asi) for _ ∈ 1:num_asi]
    append!(ls.outer_reductions, AM[2].parameters)
    for (i,si) ∈ enumerate(AM[3].parameters)
        sii = si::Int
        s = gensym(:symlicm)
        push!(ls.preamble_symsym, (si,s))
        pushpreamble!(ls, Expr(:(=), s, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__,@__FILE__), Expr(:ref, :varg, num_arrays + i))))
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
loopdependencies(ls::LoopSet, os::OperationStruct) = parents_symvec(ls, op.loopdeps)
reduceddependencies(ls::LoopSet, os::OperationStruct) = parents_symvec(ls, op.reduceddeps)



function add_op!(ls::LoopSet, os::OperationStruct, mrefs::Vector{ArrayReferenceMeta}, opsymbol::Symbol, elementbytes::Int)
    optype = os.node_type
    op = Operation(
        length(operations(ls)), opsymbol, elementbytes, os.instruction,
        optype, loopdependencies(ls, os), reduceddependencies(ls, os),
        Operation[], (isload(op) | isstore(op)) ? mrefs[os.array] : NOTAREFERENCE
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
function add_parents_to_ops!(ls::LoopSet, ops::Vector{OperationStruct})
    for i ∈ eachindex(ops)
        add_parents_to_op!(ls, parents(getop(ls, i)), ops[i].parents)
    end
end
function add_ops!(ls::LoopSet, ops::Vector{OperationStruct}, mrefs::Vector{ArrayReferenceMeta}, opsymbols::Vector{Symbol}, elementbytes::Int)
    for i ∈ eachindex(ops)
        add_op!(ls, ops[i], mrefs, opsymbols[i], elementbytes)
    end
    add_parents_to_ops!(ls, ops)
end

# elbytes(::VectorizationBase.AbstractPointer{T}) where {T} = sizeof(T)::Int
typeeltype(::Type{P}) where {T,P<:VectorizationBase.AbstractPointer{T}} = T

function avx_body(ops, arf, AM, LB, vargs)
    ls = LoopSet()
    # elementbytes = mapreduce(elbytes, min, @view(vargs[Base.OneTo(length(arf))]))::Int
    elementbytes = sizeof(mapreduce(typeeltype,promote_type,@view(vargs[Base.OneTo(length(arf))])))::Int
    add_loops!(ls, LB)
    arraysymbolinds = process_metadata!(ls, AM, length(arf))
    opsymbols = [gensym(:op) for _ ∈ eachindex(ops)]
    mrefs = create_mrefs(ls, arf, arraysymbolinds, opsymbols, vargs)
    add_ops!(ls, ops, mrefs, opsymbols, elementbytes)
end


@generated function _avx!(::Type{OPS}, ::Type{ARF}, ::Type{AM}, lb::LB, vargs...)
    avx_body(
        OperationStruct[OPS.parameters...],
        ArrayRefStruct[ARF.parameters...],
        AM.parameters, LB.parameters, vargs
    )                    
end

