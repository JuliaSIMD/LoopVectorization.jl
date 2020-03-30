function Loop(ls::LoopSet, ex::Expr, sym::Symbol, ::Type{<:AbstractUnitRange})
    ssym = String(sym)
    start = gensym(ssym*"_loopstart"); stop = gensym(ssym*"_loopstop"); loopsym = gensym(ssym * "_loop")
    pushpreamble!(ls, Expr(:(=), loopsym, ex))
    pushpreamble!(ls, Expr(:(=), start, Expr(:call, :first, loopsym)))
    pushpreamble!(ls, Expr(:(=), stop, Expr(:call, :last, loopsym)))
    Loop(sym, 1, 1024, start, stop, false, false)::Loop
end
function Loop(ls::LoopSet, ex::Expr, sym::Symbol, ::Type{StaticUpperUnitRange{U}}) where {U}
    start = gensym(String(sym)*"_loopstart")
    pushpreamble!(ls, Expr(:(=), start, Expr(:(.), ex, QuoteNode(:L))))
    Loop(sym, U - 1024, U, start, Symbol(""), false, true)::Loop
end
function Loop(ls::LoopSet, ex::Expr, sym::String, ::Type{StaticLowerUnitRange{L}}) where {L}
    stop = gensym(String(sym)*"_loopstop")
    pushpreamble!(ls, Expr(:(=), stop, Expr(:(.), ex, QuoteNode(:U))))
    Loop(sym, L, L + 1024, Symbol(""), stop, true, false)::Loop
end
# Is there any likely way to generate such a range?
# function Loop(ls::LoopSet, l::Int, ::Type{StaticLengthUnitRange{N}}) where {N}
#     start = gensym(:loopstart); stop = gensym(:loopstop)
#     pushpreamble!(ls, Expr(:(=), start, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, Symbol(@__FILE__)), Expr(:(.), Expr(:ref, :lb, l), QuoteNode(:L)))))
#     pushpreamble!(ls, Expr(:(=), stop, Expr(:call, :(+), start, N - 1)))
#     Loop(gensym(:n), 0, N, start, stop, false, false)::Loop
# end
function Loop(::LoopSet, ::Expr, sym::Symbol, ::Type{StaticUnitRange{L,U}}) where {L,U}
    Loop(sym, L, U, Symbol(""), Symbol(""), true, true)::Loop
end

function extract_loop(l)
    Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, Symbol(@__FILE__)), Expr(:ref, :lb, l))
end

function add_loops!(ls::LoopSet, LPSYM, LB)
    n = max(length(LPSYM), length(LB))
    for i = 1:n
        sym, l = LPSYM[i], LB[i]
        if l<:CartesianIndices
            add_loops!(ls, i, sym, l)
        else
            add_loop!(ls, Loop(ls, extract_loop(i), sym, l)::Loop)
            push!(ls.loopsymbol_offsets, ls.loopsymbol_offsets[end]+1)
        end
    end
end
function add_loops!(ls::LoopSet, i::Int, sym::Symbol, l::Type{CartesianIndices{N,T}}) where {N,T}
    ssym = String(sym)
    for k = N:-1:1
        axisexpr = Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, Symbol(@__FILE__)), Expr(:ref, Expr(:., Expr(:ref, :lb, i), QuoteNode(:indices)), k))
        add_loop!(ls, Loop(ls, axisexpr, Symbol(ssym*'#'*string(k)*'#'), T.parameters[k])::Loop)
    end
    push!(ls.loopsymbol_offsets, ls.loopsymbol_offsets[end]+N)
end

function ArrayReferenceMeta(
    ls::LoopSet, @nospecialize(ar::ArrayRefStruct), arraysymbolinds::Vector{Symbol},
    opsymbols::Vector{Symbol}, nopsv::Vector{Int}, expandedv::Vector{Bool}
)
    index_types = ar.index_types
    indices = ar.indices
    offsets = ar.offsets
    ni = filled_8byte_chunks(index_types)
    index_vec = Symbol[]
    offset_vec = Vector{Int8}(undef, ni)
    loopedindex = Bool[]
    while index_types != zero(UInt64)
        ind = indices % UInt8
        if index_types == LoopIndex
            for inda in ls.loopsymbol_offsets[ind]+1:ls.loopsymbol_offsets[ind+1]
                pushfirst!(index_vec, ls.loopsymbols[inda])
                pushfirst!(loopedindex, true)
            end
        elseif index_types == ComputedIndex
            opsym = opsymbols[ind]
            if expandedv[ind]
                for j ∈ 0:nopsv[ind]-1
                    pushfirst!(index_vec, expandedopname(opsym, j))
                    pushfirst!(loopedindex, false)
                end
            else
                pushfirst!(index_vec, opsym)
                pushfirst!(loopedindex, false)
            end
        else
            @assert index_types == SymbolicIndex
            pushfirst!(index_vec, arraysymbolinds[ind])
            pushfirst!(loopedindex, false)
        end
        offset_vec[ni] = offsets % Int8
        index_types >>>= 8
        indices >>>= 8
        offsets >>>= 8
        ni -= 1
    end
    ArrayReferenceMeta(
        ArrayReference(array(ar), index_vec, offset_vec),
        loopedindex, ptr(ar)
    )
end

extract_varg(i) = Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, Symbol(@__FILE__)), Expr(:ref, :vargs, i))
pushvarg!(ls::LoopSet, ar::ArrayReferenceMeta, i) = pushpreamble!(ls, Expr(:(=), vptr(ar), extract_varg(i)))
function pushvarg′!(ls::LoopSet, ar::ArrayReferenceMeta, i)
    reverse!(ar.loopedindex); reverse!(getindices(ar)) # reverse the listed indices here, and transpose it to make it column major
    pushpreamble!(ls, Expr(:(=), vptr(ar), Expr(:call, lv(:transpose), extract_varg(i))))
end
function add_mref!(
    ls::LoopSet, ar::ArrayReferenceMeta, i::Int, ::Type{S}
) where {T, N, S <: AbstractColumnMajorStridedPointer{T,N}}
    pushvarg!(ls, ar, i)
end
function add_mref!(
    ls::LoopSet, ar::ArrayReferenceMeta, i::Int, ::Type{S}
) where {T, N, S <: AbstractRowMajorStridedPointer{T, N}}
    pushvarg′!(ls, ar, i)
end
function add_mref!(
    ls::LoopSet, ar::ArrayReferenceMeta, i::Int, ::Type{OffsetStridedPointer{T,N,P}}
) where {T,N,P}
    add_mref!(ls, ar, i, P)
end

function add_mref!(
    ls::LoopSet, ar::ArrayReferenceMeta, i::Int, ::Type{S}
) where {T, X <: Tuple, S <: AbstractStaticStridedPointer{T,X}}
    if last(X.parameters)::Int == 1
        pushvarg′!(ls, ar, i)
    else
        pushvarg!(ls, ar, i)
        first(X.parameters)::Int == 1 || pushfirst!(getindices(ar), Symbol("##DISCONTIGUOUSSUBARRAY##"))
    end
end
function add_mref!(
    ls::LoopSet, ar::ArrayReferenceMeta, i::Int, ::Type{S}
) where {T, N, S <: AbstractSparseStridedPointer{T, N}}
    pushvarg!(ls, ar, i)
    pushfirst!(getindices(ar), Symbol("##DISCONTIGUOUSSUBARRAY##"))
end
function add_mref!(ls::LoopSet, ar::ArrayReferenceMeta, i::Int, ::Type{VectorizationBase.MappedStridedPointer{F,T,P}}) where {F,T,P}
    add_mref!(ls, ar, i, P)
end
function add_mref!(ls::LoopSet, ar::ArrayReferenceMeta, i::Int, ::Type{<:AbstractRange{T}}) where {T}
    pushvarg!(ls, ar, i)
end
function create_mrefs!(
    ls::LoopSet, arf::Vector{ArrayRefStruct}, as::Vector{Symbol}, os::Vector{Symbol},
    nopsv::Vector{Int}, expanded::Vector{Bool}, vargs
)
    mrefs = Vector{ArrayReferenceMeta}(undef, length(arf))
    for i ∈ eachindex(arf)
        ar = ArrayReferenceMeta(ls, arf[i], as, os, nopsv, expanded)
        add_mref!(ls, ar, i, vargs[i])
        mrefs[i] = ar
    end
    mrefs
end

function num_parameters(AM)
    num_param::Int = AM[1]
    # num_param += length(AM[2].parameters)
    num_param + length(AM[3].parameters)
end
function gen_array_syminds(AM)
    Symbol[Symbol("##arraysymbolind##"*i*'#') for i ∈ 1:(AM[1])::Int]
end
function process_metadata!(ls::LoopSet, AM, num_arrays::Int)
    opoffsets = ls.operation_offsets
    expandbyoffset!(ls.outer_reductions, AM[2].parameters, opoffsets)
    for (i,si) ∈ enumerate(AM[3].parameters)
        sii = si::Int
        s = gensym(:symlicm)
        push!(ls.preamble_symsym, (opoffsets[sii] + 1, s))
        pushpreamble!(ls, Expr(:(=), s, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__,Symbol(@__FILE__)), Expr(:ref, :vargs, num_arrays + i))))
    end
    expandbyoffset!(ls.preamble_symint, AM[4].parameters, opoffsets)
    expandbyoffset!(ls.preamble_symfloat, AM[5].parameters, opoffsets)
    expandbyoffset!(ls.preamble_zeros, AM[6].parameters, opoffsets)
    expandbyoffset!(ls.preamble_ones, AM[7].parameters, opoffsets)
    nothing
end
function expandbyoffset!(indexpand::Vector{T}, inds, offsets::Vector{Int}, expand::Bool = true) where {T <: Union{Int,Tuple{Int,<:Any}}}
    for _ind ∈ inds
        ind = T === Int ? _ind : first(_ind)
        base = offsets[ind] + 1
        for inda ∈ base:(expand ? offsets[ind+1] : base)
            T === Int ? push!(indexpand, inda) : push!(indexpand, (inda,last(_ind)))
        end
    end
    indexpand
end
expandbyoffset(inds::Vector{Int}, offsets::Vector{Int}, expand::Bool) = expandbyoffset!(Int[], inds, offsets, expand)
function loopindex(ls::LoopSet, u::Unsigned, shift::Unsigned)
    mask = (one(shift) << shift) - one(shift) # mask to zero out all but shift-bits
    idxs = Int[]
    while u != zero(u)
        pushfirst!(idxs, ( u % UInt8 ) & mask)
        u >>= shift
    end
    reverse!(idxs)
end
function loopindexoffset(ls::LoopSet, u::Unsigned, li::Bool, expand::Bool = false)
    if li
        shift = 0x04
        offsets = ls.loopsymbol_offsets
    else
        shift = 0x08
        offsets = ls.operation_offsets
    end
    idxs = loopindex(ls, u, shift)
    expandbyoffset(idxs, offsets, expand)
end
function parents_symvec(ls::LoopSet, u::Unsigned, expand, offset)
    idxs = loopindexoffset(ls, u, true, expand)   # FIXME DRY  (undesirable that this gets hard-coded in multiple places)
    return Symbol[getloopsym(ls, i + offset) for i ∈ idxs]
end
loopdependencies(ls::LoopSet, os::OperationStruct, expand = false, offset = 0) = parents_symvec(ls, os.loopdeps, expand, offset)
reduceddependencies(ls::LoopSet, os::OperationStruct, expand = false, offset = 0) = parents_symvec(ls, os.reduceddeps, expand, offset)
childdependencies(ls::LoopSet, os::OperationStruct, expand = false, offset = 0) = parents_symvec(ls, os.childdeps, expand, offset)

# parents(ls::LoopSet, u::UInt64) = loopindexoffset(ls, u, false)
parents(ls::LoopSet, u::UInt64) = loopindex(ls, u, 0x08)
parents(ls::LoopSet, os::OperationStruct) = parents(ls, os.parents)

expandedopname(opsymbol::Symbol, offset::Integer) = Symbol(String(opsymbol)*'#'*string(offset+1)*'#')
function calcnops(ls::LoopSet, os::OperationStruct)
    optyp = optype(os)
    if (optyp != loopvalue) && (optyp != compute)
        return 1
    end
    offsets = ls.loopsymbol_offsets
    idxs = loopindex(ls, os.loopdeps, 0x04)  # FIXME DRY
    iszero(length(idxs)) && return 1
    Δidxs = map(i->offsets[i+1]-offsets[i], idxs)
    nops = first(Δidxs)
    @assert all(isequal(nops), Δidxs)
    nops
end
function isexpanded(ls::LoopSet, ops::Vector{OperationStruct}, nopsv::Vector{Int}, i::Int)
    nops = nopsv[i]
    isone(nops) && return false
    os = ops[i]
    optyp = optype(os)
    if optyp == compute
        any(j -> isexpanded(ls, ops, nopsv, j), parents(ls, os))
    elseif optyp == loopvalue
        true
    else
        false
    end
end

function add_op!(
    ls::LoopSet, instr::Instruction, ops::Vector{OperationStruct}, nopsv::Vector{Int}, expandedv::Vector{Bool}, i::Int,
    mrefs::Vector{ArrayReferenceMeta}, opsymbol, elementbytes::Int
)
    os = ops[i]
    nops = nopsv[i]
    # opsymbol = (isconstant(os) && instr != LOOPCONSTANT) ? instr.instr : opsymbol
    # If it's a CartesianIndex add or subtract, we may have to add multiple operations
    expanded = expandedv[i]# isexpanded(ls, ops, nopsv, i)
    opoffsets = ls.operation_offsets
    offsets = ls.loopsymbol_offsets
    optyp = optype(os)
    if !expanded
        op = Operation(
            length(operations(ls)), opsymbol, elementbytes, instr,
            optyp, loopdependencies(ls, os, true), reduceddependencies(ls, os, true),
            Operation[], (isload(os) | isstore(os)) ? mrefs[os.array] : NOTAREFERENCE,
            childdependencies(ls, os, true)
        )
        push!(ls.operations, op)
        push!(opoffsets, opoffsets[end] + 1)
        return
    end
    # if expanded, optyp must be either loopvalue, or compute (with loopvalues in its ancestry, not cutoff by loads)
    for offset = 0:nops-1
        sym = nops == 1 ? opsymbol : expandedopname(opsymbol, offset)
        op = Operation(
            length(operations(ls)), sym, elementbytes, instr,
            optyp, loopdependencies(ls, os, false, offset), reduceddependencies(ls, os, false, offset),
            Operation[], (isload(os) | isstore(os)) ? mrefs[os.array] : NOTAREFERENCE,
            childdependencies(ls, os, false, offset)
        )
        push!(ls.operations, op)
    end
    push!(opoffsets, opoffsets[end] + nops)
    nothing
end
function add_parents_to_op!(ls::LoopSet, vparents::Vector{Operation}, up::Unsigned, k::Int, Δ::Int)
    ops = operations(ls)
    offsets = ls.operation_offsets
    if isone(Δ) # not expanded
        @assert isone(k)
        for i ∈ parents(ls, up)
            for j ∈ offsets[i]+1:offsets[i+1] # if parents are expanded, add them all
                pushfirst!(vparents, ops[j])
            end
        end        
    else#if isexpanded
        # Do we want to require that all Δidxs are equal?
        # Because `CartesianIndex((2,3)) - 1` results in a methoderorr, I think this is reasonable for now
        for i ∈ parents(ls, up)
            pushfirst!(vparents, ops[offsets[i]+k])
        end
    end
end
function add_parents_to_ops!(ls::LoopSet, ops::Vector{OperationStruct}, constoffset)
    offsets = ls.operation_offsets
    for i in 1:length(offsets)-1
        pos = offsets[i]
        Δ = offsets[i+1]-pos
        for k ∈ 1:Δ
            op = ls.operations[pos+k]
            if isconstant(op)
                instr = instruction(op)
                if instr != LOOPCONSTANT && instr.mod !== :numericconstant
                    constoffset += 1
                    pushpreamble!(ls, Expr(:(=), instr.instr, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, Symbol(@__FILE__)), Expr(:ref, :vargs, constoffset))))
                end
            elseif !isloopvalue(op)
                add_parents_to_op!(ls, parents(op), ops[i].parents, k, Δ)                
            end
        end
    end
    constoffset
end
function add_ops!(
    ls::LoopSet, instr::Vector{Instruction}, ops::Vector{OperationStruct}, mrefs::Vector{ArrayReferenceMeta},
    opsymbols::Vector{Symbol}, constoffset::Int, nopsv::Vector{Int}, expandedv::Vector{Bool}, elementbytes::Int
)
    # @show ls.loopsymbols ls.loopsymbol_offsets
    for i ∈ eachindex(ops)
        os = ops[i]
        opsymbol = opsymbols[os.symid]
        add_op!(ls, instr[i], ops, nopsv, expandedv, i, mrefs, opsymbol, elementbytes)
    end
    add_parents_to_ops!(ls, ops, constoffset)
    # for op in operations(ls)
        # @show op
    # end
end

# elbytes(::VectorizationBase.AbstractPointer{T}) where {T} = sizeof(T)::Int
typeeltype(::Type{P}) where {T,P<:VectorizationBase.AbstractPointer{T}} = T
typeeltype(::Type{<:AbstractRange{T}}) where {T} = T
# typeeltype(::Any) = Int8

function add_array_symbols!(ls::LoopSet, arraysymbolinds::Vector{Symbol}, offset::Int)
    for (i,as) ∈ enumerate(arraysymbolinds)
        pushpreamble!(ls, Expr(:(=), as, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, Symbol(@__FILE__)), Expr(:ref, :vargs, i + offset))))
    end
end
function extract_external_functions!(ls::LoopSet, offset::Int)
    for op ∈ operations(ls)
        if iscompute(op)
            instr = instruction(op)
            if instr.mod != :LoopVectorization
                offset += 1
                pushpreamble!(ls, Expr(:(=), instr.instr, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, Symbol(@__FILE__)), Expr(:ref, :vargs, offset))))
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

function avx_loopset(instr, ops, arf, AM, LPSYM, LB, vargs)
    ls = LoopSet(:LoopVectorization)
    num_arrays = length(arf)
    elementbytes = sizeofeltypes(vargs, num_arrays)
    add_loops!(ls, LPSYM, LB)
    resize!(ls.loop_order, ls.loopsymbol_offsets[end])
    arraysymbolinds = gen_array_syminds(AM)
    opsymbols = [gensym(:op) for _ ∈ eachindex(ops)]
    nopsv = calcnops.(Ref(ls), ops)
    expandedv = [isexpanded(ls, ops, nopsv, i) for i ∈ eachindex(ops)]
    mrefs = create_mrefs!(ls, arf, arraysymbolinds, opsymbols, nopsv, expandedv, vargs)
    pushpreamble!(ls, Expr(:(=), ls.T, Expr(:call, :promote_type, [Expr(:call, :eltype, vptr(mref)) for mref ∈ mrefs]...)))
    pushpreamble!(ls, Expr(:(=), ls.W, Expr(:call, lv(:pick_vector_width_val), [Expr(:call, :eltype, vptr(mref)) for mref ∈ mrefs]...)))
    num_params = num_arrays + num_parameters(AM)
    add_ops!(ls, instr, ops, mrefs, opsymbols, num_params, nopsv, expandedv, elementbytes)
    process_metadata!(ls, AM, length(arf))
    add_array_symbols!(ls, arraysymbolinds, num_arrays + length(ls.preamble_symsym))
    num_params = extract_external_functions!(ls, num_params)
    ls
end
function avx_body(ls, UT)
    U, T = UT
    q = iszero(U) ? lower_and_split_loops(ls) : lower(ls, U, T)
    length(ls.outer_reductions) == 0 ? push!(q.args, nothing) : push!(q.args, loopset_return_value(ls, Val(true)))
    # @show q
    q
end

function _avx_loopset_debug(::Type{OPS}, ::Type{ARF}, ::Type{AM}, ::Type{LPSYM}, ::Type{LB}, vargs...) where {UT, OPS, ARF, AM, LPSYM, LB}
    @show OPS ARF AM LPSYM LB vargs
    _avx_loopset(OPS.parameters, ARF.parameters, AM.parameters, LPSYM.parameters, LB.parameters, typeof.(vargs))
end
function _avx_loopset(OPSsv, ARFsv, AMsv, LPSYMsv, LBsv, vargs)
    nops = length(OPSsv) ÷ 3
    instr = Instruction[Instruction(OPSsv[3i+1], OPSsv[3i+2]) for i ∈ 0:nops-1]
    ops = OperationStruct[ OPSsv[3i] for i ∈ 1:nops ]
    avx_loopset(
        instr, ops,
        ArrayRefStruct[ARFsv...],
        AMsv, LPSYMsv, LBsv, vargs
    )
end
@generated function _avx_!(::Val{UT}, ::Type{OPS}, ::Type{ARF}, ::Type{AM}, ::Type{LPSYM}, lb::LB, vargs...) where {UT, OPS, ARF, AM, LPSYM, LB}
    ls = _avx_loopset(OPS.parameters, ARF.parameters, AM.parameters, LPSYM.parameters, LB.parameters, vargs)
    avx_body(ls, UT)
end

