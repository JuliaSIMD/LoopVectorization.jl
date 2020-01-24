
isdense(::Type{<:DenseArray}) = true

# """
# ShortVector{T} simply wraps a Vector{T}, but uses a different hash function that is faster for short vectors to support using it as the keys of a Dict.
# This hash function scales O(N) with length of the vectors, so it is slow for long vectors.
# """
# struct ShortVector{T} <: DenseVector{T}
#     data::Vector{T}
# end
# Base.@propagate_inbounds Base.getindex(x::ShortVector, I...) = x.data[I...]
# Base.@propagate_inbounds Base.setindex!(x::ShortVector, v, I...) = x.data[I...] = v
# ShortVector{T}(::UndefInitializer, N::Integer) where {T} = ShortVector{T}(Vector{T}(undef, N))
# @inbounds Base.length(x::ShortVector) = length(x.data)
# @inbounds Base.size(x::ShortVector) = size(x.data)
# @inbounds Base.strides(x::ShortVector) = strides(x.data)
# @inbounds Base.push!(x::ShortVector, v) = push!(x.data, v)
# @inbounds Base.append!(x::ShortVector, v) = append!(x.data, v)
# function Base.hash(x::ShortVector, h::UInt)
#     @inbounds for n ∈ eachindex(x)
#         h = hash(x[n], h)
#     end
#     h
# end
# function Base.isequal(a::ShortVector{T}, b::ShortVector{T}) where {T}
#     length(a) == length(b) || return false
#     @inbounds for i ∈ 1:length(a)
#         a[i] === b[i] || return false
#     end
#     true
# end
# Base.convert(::Type{Vector}, sv::ShortVector) = sv.data
# Base.convert(::Type{Vector{T}}, sv::ShortVector{T}) where {T} = sv.data


# For passing options like array types and mask
# struct LoopSetOptions
    
# end

struct Loop
    itersymbol::Symbol
    starthint::Int
    stophint::Int
    startsym::Symbol
    stopsym::Symbol
    startexact::Bool
    stopexact::Bool
end
function Loop(itersymbol::Symbol, start::Int, stop::Int)
    Loop(itersymbol, start, stop, Symbol("##UNDEFINED##"), Symbol("##UNDEFINED##"), true, true)
end
function Loop(itersymbol::Symbol, start::Int, stop::Symbol)
    Loop(itersymbol, start, 1024, Symbol("##UNDEFINED##"), stop, true, false)
end
function Loop(itersymbol::Symbol, start::Symbol, stop::Int)
    Loop(itersymbol, -1, stop, start, Symbol("##UNDEFINED##"), false, true)
end
function Loop(itersymbol::Symbol, start::Symbol, stop::Symbol)
    Loop(itersymbol, -1, 1024, start, stop, false, false)
end
Base.length(loop::Loop) = loop.stophint - loop.starthint
isstaticloop(loop::Loop) = loop.startexact & loop.stopexact
function startloop(loop::Loop, isvectorized, W, itersymbol = loop.itersymbol)
    startexact = loop.startexact
    if isvectorized
        if startexact
            Expr(:(=), itersymbol, Expr(:call, lv(:_MM), W, loop.starthint))
        else
            Expr(:(=), itersymbol, Expr(:call, lv(:_MM), W, loop.startsym))
        end
    elseif startexact
        Expr(:(=), itersymbol, loop.starthint)
    else
        Expr(:(=), itersymbol, loop.startsym)
    end
end
function vec_looprange(loop::Loop, isunrolled::Bool, W::Symbol, U::Int)
    incr = if isunrolled
        Expr(:call, lv(:valmuladd), W, U, -1)
    else
        Expr(:call, lv(:valsub), W, 1)
    end
    if loop.stopexact # split for type stability
        Expr(:call, :<, loop.itersymbol, Expr(:call, :-, loop.stophint, incr))
    else
        Expr(:call, :<, loop.itersymbol, Expr(:call, :-, loop.stopsym, incr))
    end
end                       
function looprange(loop::Loop, incr::Int, mangledname::Symbol)
    incr -= 1#one(Int32)
    if iszero(incr)
        Expr(:call, :<, mangledname, loop.stopexact ? loop.stophint : loop.stopsym)
    else
        Expr(:call, :<, mangledname, loop.stopexact ? loop.stophint - incr : Expr(:call, :-, loop.stopsym, incr))
    end
end
function terminatecondition(
    loop::Loop, W::Symbol, U::Int, T::Int, isvectorized::Bool, isunrolled::Bool, istiled::Bool,
    mangledname::Symbol = loop.itersymbol, mask::Nothing = nothing
)
    if isvectorized
        vec_looprange(loop, isunrolled, W, U) # may not be tiled
    else
        looprange(loop, isunrolled ? U : (istiled ? T : 1), mangledname)
    end
end
function terminatecondition(
    loop::Loop, W::Symbol, U::Int, T::Int, isvectorized::Bool, isunrolled::Bool, istiled::Bool,
    mangledname::Symbol, mask::Symbol
)
    looprange(loop, 1, mangledname)
end

# load/compute/store × isunroled × istiled × pre/post loop × Loop number
struct LoopOrder <: AbstractArray{Vector{Operation},5}
    oporder::Vector{Vector{Operation}}
    loopnames::Vector{Symbol}
    bestorder::Vector{Symbol}
end
function LoopOrder(N::Int)
    LoopOrder(
        [ Operation[] for _ ∈ 1:8N ],
        Vector{Symbol}(undef, N), Vector{Symbol}(undef, N)
    )
end
LoopOrder() = LoopOrder(Vector{Operation}[],Symbol[],Symbol[])
Base.empty!(lo::LoopOrder) = foreach(empty!, lo.oporder)
function Base.resize!(lo::LoopOrder, N::Int)
    Nold = length(lo.loopnames)
    resize!(lo.oporder, 8N)
    for n ∈ 8Nold+1:8N
        lo.oporder[n] = Operation[]
    end
    resize!(lo.loopnames, N)
    resize!(lo.bestorder, N)
    lo
end
Base.size(lo::LoopOrder) = (2,2,2,length(lo.loopnames))
Base.@propagate_inbounds Base.getindex(lo::LoopOrder, i::Int) = lo.oporder[i]
Base.@propagate_inbounds Base.getindex(lo::LoopOrder, i...) = lo.oporder[LinearIndices(size(lo))[i...]]

# Must make it easy to iterate
# outer_reductions is a vector of indices (within operation vectors) of the reduction operation, eg the vmuladd op in a dot product
struct LoopSet
    loopsymbols::Vector{Symbol}
    loops::Vector{Loop}
    opdict::Dict{Symbol,Operation}
    operations::Vector{Operation} # Split them to make it easier to iterate over just a subset
    outer_reductions::Vector{Int} # IDs of reduction operations that need to be reduced at end.
    loop_order::LoopOrder
    # stridesets::Dict{ShortVector{Symbol},ShortVector{Symbol}}
    preamble::Expr
    preamble_symsym::Vector{Tuple{Int,Symbol}}
    preamble_symint::Vector{Tuple{Int,Int}}
    preamble_symfloat::Vector{Tuple{Int,Float64}}
    preamble_zeros::Vector{Int}
    preamble_ones::Vector{Int}
    includedarrays::Vector{Symbol}
    syms_aliasing_refs::Vector{Symbol} # O(N) search is faster at small sizes
    refs_aliasing_syms::Vector{ArrayReferenceMeta}
    cost_vec::Matrix{Float64}
    reg_pres::Matrix{Int}
    included_vars::Vector{Bool}
    place_after_loop::Vector{Bool}
    W::Symbol
    T::Symbol
    mod::Symbol
end

instruction(ls::LoopSet, f::Symbol) = instruction(f, ls.mod)

function cost_vec_buf(ls::LoopSet)
    cv = @view(ls.cost_vec[:,2])
    @inbounds for i ∈ 1:4
        cv[i] = 0.0
    end
    cv
end
function reg_pres_buf(ls::LoopSet)
    ps = @view(ls.reg_pres[:,2])
    @inbounds for i ∈ 1:4
        ps[i] = 0
    end
    ps
end
function save_tilecost!(ls::LoopSet)
    @inbounds for i ∈ 1:4
        ls.cost_vec[i,1] = ls.cost_vec[i,2]
        ls.reg_pres[i,1] = ls.reg_pres[i,2]
    end
end


# function op_to_ref(ls::LoopSet, op::Operation)
    # s = op.variable
    # id = findfirst(ls.syms_aliasing_regs)
    # @assert id !== nothing
    # ls.refs_aliasing_syms[id]
# end
function pushpreamble!(ls::LoopSet, op::Operation, v::Symbol)
    if v !== mangledvar(op)
        push!(ls.preamble_symsym, (identifier(op),v))
    end
    nothing
end
pushpreamble!(ls::LoopSet, op::Operation, v::Integer) = push!(ls.preamble_symint, (identifier(op),convert(Int,v)))
pushpreamble!(ls::LoopSet, op::Operation, v::Real) = push!(ls.preamble_symfloat, (identifier(op),convert(Float64,v)))
pushpreamble!(ls::LoopSet, ex::Expr) = push!(ls.preamble.args, ex)
function pushpreamble!(ls::LoopSet, op::Operation, RHS::Expr)
    c = gensym(:licmconst)
    if RHS.head === :call && first(RHS.args) === :zero
        push!(ls.preamble_zeros, identifier(op))
    elseif RHS.head === :call && first(RHS.args) === :one
        push!(ls.preamble_ones, identifier(op))
    else
        pushpreamble!(ls, Expr(:(=), c, RHS))
        pushpreamble!(ls, op, c)
    end
    nothing
end

includesarray(ls::LoopSet, array::Symbol) = array ∈ ls.includedarrays

function LoopSet(mod::Symbol)# = :LoopVectorization)
    LoopSet(
        Symbol[], Loop[],
        Dict{Symbol,Operation}(),
        Operation[],
        Int[],
        LoopOrder(),
        Expr(:block),
        Tuple{Int,Symbol}[],
        Tuple{Int,Int}[],
        Tuple{Int,Float64}[],
        Int[],Int[],
        Tuple{Symbol,Int}[],
        Symbol[],
        ArrayReferenceMeta[],
        Matrix{Float64}(undef, 4, 2),
        Matrix{Int}(undef, 4, 2),
        Bool[], Bool[],
        gensym(:W), gensym(:T), mod
    )
end

num_loops(ls::LoopSet) = length(ls.loops)
function oporder(ls::LoopSet)
    N = length(ls.loop_order.loopnames)
    reshape(ls.loop_order.oporder, (2,2,2,N))
end
names(ls::LoopSet) = ls.loop_order.loopnames
function getloopid(ls::LoopSet, s::Symbol)::Int
    for (loopnum,sym) ∈ enumerate(ls.loopsymbols)
        s === sym && return loopnum
    end
end
getloop(ls::LoopSet, s::Symbol) = ls.loops[getloopid(ls, s)]
getloop(ls::LoopSet, i::Integer) = ls.loops[i]
getloopsym(ls::LoopSet, i::Integer) = ls.loopsymbols[i]
Base.length(ls::LoopSet, s::Symbol) = length(getloop(ls, s))

isstaticloop(ls::LoopSet, s::Symbol) = isstaticloop(getloop(ls,s))
looprangehint(ls::LoopSet, s::Symbol) = length(getloop(ls, s))
looprangesym(ls::LoopSet, s::Symbol) = getloop(ls, s).rangesym
function getop(ls::LoopSet, var::Symbol, elementbytes::Int = 8)
    get!(ls.opdict, var) do
        add_constant!(ls, var, elementbytes)
    end
end
function getop(ls::LoopSet, var::Symbol, deps, elementbytes::Int = 8)
    get!(ls.opdict, var) do
        add_constant!(ls, var, deps, gensym(:constant), elementbytes)
    end
end
getop(ls::LoopSet, i::Int) = ls.operations[i + 1]

@inline extract_val(::Val{N}) where {N} = N

function Operation(
    ls::LoopSet, variable, elementbytes, instruction,
    node_type, dependencies, reduced_deps, parents, ref = NOTAREFERENCE
)
    Operation(
        length(operations(ls)), variable, elementbytes, instruction,
        node_type, dependencies, reduced_deps, parents, ref
    )
end
function Operation(ls::LoopSet, var, elementbytes, instr, optype, mpref::ArrayReferenceMetaPosition)
    Operation(length(operations(ls)), var, elementbytes, instr, optype, mpref)
end

# load_operations(ls::LoopSet) = ls.loadops
# compute_operations(ls::LoopSet) = ls.computeops
# store_operations(ls::LoopSet) = ls.storeops
# function operations(ls::LoopSet)
    # Base.Iterators.flatten((
        # load_operations(ls),
        # compute_operations(ls),
        # store_operations(ls)
    # ))
# end
operations(ls::LoopSet) = ls.operations
function pushop!(ls::LoopSet, op::Operation, var::Symbol = name(op))
    for opp ∈ operations(ls)
        matches(op, opp) && return opp
    end
    push!(ls.operations, op)
    ls.opdict[var] = op
    op
end
function add_block!(ls::LoopSet, ex::Expr, elementbytes::Int = 8)
    for x ∈ ex.args
        x isa Expr || continue # be that general?
        push!(ls, x, elementbytes)
    end
end
function maybestatic!(expr::Expr)
    if expr.head === :call
        f = first(expr.args)
        if f === :length
            expr.args[1] = lv(:maybestaticlength)
        elseif f === :size && length(expr.args) == 3
            i = expr.args[3]
            if i isa Integer
                expr.args[1] = lv(:maybestaticsize)
                expr.args[3] = Expr(:call, Expr(:curly, :Val, convert(Int, i)))
            end
        end
    end
    expr
end
function add_loop_bound!(ls::LoopSet, itersym::Symbol, bound, upper::Bool = true)
    (bound isa Symbol && upper) && return bound
    bound isa Expr && maybestatic!(bound)
    N = gensym(Symbol(itersym, upper ? :_loop_upper_bound : :_loop_lower_bound))
    pushpreamble!(ls, Expr(:(=), N, upper ? bound : Expr(:call, lv(:staticm1), bound)))
    N
end

"""
This function creates a loop, while switching from 1 to 0 based indices
"""
function register_single_loop!(ls::LoopSet, looprange::Expr)
    itersym = (looprange.args[1])::Symbol
    r = (looprange.args[2])::Expr
    @assert r.head === :call
    f = first(r.args)
    loop::Loop = if f === :(:)
        lower = r.args[2]
        upper = r.args[3]
        lii::Bool = lower isa Integer
        liiv::Int = lii ? (convert(Int, lower)-1) : 0
        uii::Bool = upper isa Integer
        if lii & uii # both are integers
            Loop(itersym, liiv, convert(Int, upper))
        elseif lii # only lower bound is an integer
            if upper isa Symbol
                Loop(itersym, liiv, upper)
            elseif upper isa Expr
                Loop(itersym, liiv, add_loop_bound!(ls, itersym, upper, true))
            else
                Loop(itersym, liiv, add_loop_bound!(ls, itersym, upper, true))
            end
        elseif uii # only upper bound is an integer
            uiiv = convert(Int, upper)
            Loop(itersym, add_loop_bound!(ls, itersym, lower, false), uiiv)
        else # neither are integers
            L = add_loop_bound!(ls, itersym, lower, false)
            U = add_loop_bound!(ls, itersym, upper, true)
            Loop(itersym, L, U)
        end
    elseif f === :eachindex
        N = gensym(Symbol(:loop, itersym))
        pushpreamble!(ls, Expr(:(=), N, Expr(:call, lv(:maybestaticlength), r.args[2])))
        Loop(itersym, 0, N)
    elseif f === :OneTo || f === Expr(:(.), :Base, :OneTo)
        otN = r.args[2]
        if otN isa Integer
            Loop(itersym, 0, otN)
        else
            N = gensym(Symbol(:loop, itersym))
            pushpreamble!(ls, Expr(:(=), N, maybestatic!(otN)))
            Loop(itersym, 0, N)
        end
    else
        throw("Unrecognized loop range type: $r.")
    end
    add_loop!(ls, loop, itersym)
    nothing
end
function register_loop!(ls::LoopSet, looprange::Expr)
    if looprange.head === :block # multiple loops
        for lr ∈ looprange.args
            register_single_loop!(ls, lr)
        end
    else
        @assert looprange.head === :(=)
        register_single_loop!(ls, looprange)
    end
end
function add_loop!(ls::LoopSet, q::Expr, elementbytes::Int = 8)
    register_loop!(ls, q.args[1])
    body = q.args[2]
    if body.head === :block
        add_block!(ls, body, elementbytes)
    else
        push!(ls, q, elementbytes)
    end
end
function add_loop!(ls::LoopSet, loop::Loop, itersym::Symbol = loop.itersymbol)
    push!(ls.loopsymbols, itersym)
    push!(ls.loops, loop)
    nothing
end

function instruction(x)
    x isa Symbol ? x : last(x.args).value
end

# if it is a literal, that literal has to have been assigned to var in the preamble.

# add operation assigns X to var
function add_operation!(
    ls::LoopSet, LHS::Symbol, RHS::Expr, elementbytes::Int = 8
)
    if RHS.head === :ref
        add_load_ref!(ls, LHS, RHS, elementbytes)
    elseif RHS.head === :call
        f = first(RHS.args)
        if f === :getindex
            add_load_getindex!(ls, LHS, RHS, elementbytes)
        elseif f === :zero || f === :one
            c = gensym(f)
            op = add_constant!(ls, c, copy(ls.loopsymbols), LHS, elementbytes, :numericconstant)
            push!(f === :zero ? ls.preamble_zeros : ls.preamble_ones, identifier(op))
            op
        else
            add_compute!(ls, LHS, RHS, elementbytes)
        end
    elseif RHS.head === :if
        add_if!(ls, LHS, RHS, elementbytes)
    else
        throw("Expression not recognized:\n$x")
    end
end
add_operation!(ls::LoopSet, RHS::Expr, elementbytes::Int = 8) = add_operation!(ls, gensym(:LHS), RHS, elementbytes)
function add_operation!(
    ls::LoopSet, LHS_sym::Symbol, RHS::Expr, LHS_ref::ArrayReferenceMetaPosition, elementbytes::Int = 8
)
    if RHS.head === :ref# || (RHS.head === :call && first(RHS.args) === :getindex)
        add_load!(ls, LHS_sym, LHS_ref, elementbytes)
    elseif RHS.head === :call
        f = first(RHS.args)
        if f === :getindex
            add_load!(ls, LHS_sym, LHS_ref, elementbytes)
        elseif f === :zero || f === :one
            c = gensym(f)
            op = add_constant!(ls, c, copy(ls.loopsymbols), LHS_sym, elementbytes, :numericconstant)
            push!(f === :zero ? ls.preamble_zeros : ls.preamble_ones, identifier(op))
            op
        else
            add_compute!(ls, LHS_sym, RHS, elementbytes, LHS_ref)
        end
    elseif RHS.head === :if
        add_if!(ls, LHS, RHS, elementbytes, LHS_ref)
    else
        throw("Expression not recognized:\n$x")
    end
end

function Base.push!(ls::LoopSet, ex::Expr, elementbytes::Int = 8)
    if ex.head === :call
        finex = first(ex.args)::Symbol
        if finex === :setindex!
            add_store_setindex!(ls, ex, elementbytes)
        else
            throw("Function $finex not recognized.")
        end
    elseif ex.head === :(=)
        LHS = ex.args[1]
        RHS = ex.args[2]
        if LHS isa Symbol
            if RHS isa Expr
                add_operation!(ls, LHS, RHS, elementbytes)
            else
                add_constant!(ls, RHS, copy(ls.loopsymbols), LHS, elementbytes)
            end
        elseif LHS isa Expr
            @assert LHS.head === :ref
            if RHS isa Symbol
                add_store_ref!(ls, RHS, LHS, elementbytes)
            elseif RHS isa Expr
                # need to check if LHS appears in RHS
                # assign RHS to lrhs
                array, rawindices = ref_from_expr(LHS)
                mpref = array_reference_meta!(ls, array, rawindices, elementbytes)
                ref = mpref.mref.ref
                id = findfirst(r -> r == ref, ls.refs_aliasing_syms)
                lrhs = id === nothing ? gensym(:RHS) : ls.syms_aliasing_refs[id]
                add_operation!(ls, lrhs, RHS, mpref, elementbytes)
                add_store!( ls, lrhs, mpref, elementbytes )
            end
        else
            throw("LHS not understood:\n$LHS")
        end
    elseif ex.head === :block
        add_block!(ls, ex)
    elseif ex.head === :for
        add_loop!(ls, ex)
    elseif ex.head === :&&
        add_andblock!(ls, ex)
    elseif ex.head === :||
        add_orblock!(ls, ex)
    else
        throw("Don't know how to handle expression:\n$ex")
    end
end

function addoptoorder!(
    lo::LoopOrder, included_vars::Vector{Bool}, place_after_loop::Vector{Bool}, op::Operation, loopsym::Symbol, _n::Int, unrolled::Symbol, tiled::Symbol, loopistiled::Bool
)
    id = identifier(op)
    included_vars[id] && return nothing
    loopsym ∈ loopdependencies(op) || return nothing
    included_vars[id] = true
    for opp ∈ parents(op) # ensure parents are added first
        addoptoorder!(lo, included_vars, place_after_loop, opp, loopsym, _n, unrolled, tiled, loopistiled)
    end
    isunrolled = (unrolled ∈ loopdependencies(op)) + 1
    istiled = (loopistiled ? (tiled ∈ loopdependencies(op)) : false) + 1
    # optype = Int(op.node_type) + 1
    after_loop = place_after_loop[id] + 1
    push!(lo[isunrolled,istiled,after_loop,_n], op)
    set_upstream_family!(place_after_loop, op, false) # parents that have already been included are not moved, so no need to check included_vars to filter
    nothing
end

function fillorder!(ls::LoopSet, order::Vector{Symbol}, loopistiled::Bool)
    lo = ls.loop_order
    ro = lo.loopnames # reverse order; will have same order as lo
    nloops = length(order)
    if loopistiled
        tiled    = order[1]
        unrolled = order[2]
    else
        tiled = Symbol("##UNDEFINED##")
        unrolled = first(order)
    end
    ops = operations(ls)
    nops = length(ops)
    included_vars = resize!(ls.included_vars, nops)
    fill!(included_vars, false)
    place_after_loop = resize!(ls.place_after_loop, nops)
    fill!(ls.place_after_loop, true)
    # to go inside out, we just have to include all those not-yet included depending on the current sym
    empty!(lo)
    for _n ∈ 1:nloops
        n = 1 + nloops - _n
        ro[_n] = loopsym = order[n]
        #loopsym = order[n]
        for op ∈ ops
            addoptoorder!( lo, included_vars, place_after_loop, op, loopsym, _n, unrolled, tiled, loopistiled )
        end
    end
end

# function define_remaining_ops!(
#     ls::LoopSet, vectorized::Symbol, W, unrolled, tiled, U::Int
# )
#     ops = operations(ls)
#     for (id,incl) ∈ enumerate(ls.included_vars)
#         if !incl
#             op = ops[id]
#             length(reduceddependencies(op)) == 0 && lower!( ls.preamble, op, vectorized, W, unrolled, tiled, U, nothing, nothing )
#         end
#     end
# end

# function depends_on_assigned(op::Operation, assigned::Vector{Bool})
#     for p ∈ parents(op)
#         p === op && continue # don't fall into recursive loop when we have updates, eg a = a + b
#         assigned[identifier(op)] && return true
#         depends_on_assigned(p, assigned) && return true
#     end
#     false
# end
# ind gets increased across tiles / unroll, so we need steps.
# function replace_ind_in_offset!(offset::Vector, op::Operation, ind::Int, t)
#     t == 0 && return nothing
#     var = op.variable
#     siter = op.symbolic_metadata[ind]
#     striden = op.numerical_metadata[ind]
#     strides = Symbol(:stride_, var)
#     offset[ind] = if tstriden == -1
#         Expr(:call, :*, Expr(:call, :+, strides, t), siter)
#     else
#         Expr(:call, :*, striden + t, siter)
#     end
#     nothing
# end





