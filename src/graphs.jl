# lv(x) = Expr(:(.), :LoopVectorization, QuoteNode(x))
@static if VERSION  < v"1.3.0"
    lv(x) = Expr(:(.), :LoopVectorization, QuoteNode(x))
else
    lv(x) = GlobalRef(LoopVectorization, x)
end

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



struct Loop
    itersymbol::Symbol
    rangehint::Int
    rangesym::Symbol
    hintexact::Bool # if true, rangesym ignored and rangehint used for final lowering
end
function Loop(itersymbol::Symbol, rangehint::Int)
    Loop( itersymbol, rangehint, Symbol("##UNDEFINED##"), true )
end
function Loop(itersymbol::Symbol, rangesym::Symbol, rangehint::Int = 1_024)
    Loop( itersymbol, rangehint, rangesym, false )
end

# load/compute/store × isunroled × istiled × pre/post loop × Loop number
struct LoopOrder <: AbstractArray{Vector{Operation},5}
    oporder::Vector{Vector{Operation}}
    loopnames::Vector{Symbol}
    bestorder::Vector{Symbol}
end
function LoopOrder(N::Int)
    LoopOrder( [ Operation[] for i ∈ 1:24N ], Vector{Symbol}(undef, N), Vector{Symbol}(undef, N) )
end
LoopOrder() = LoopOrder(Vector{Operation}[],Symbol[],Symbol[])
Base.empty!(lo::LoopOrder) = foreach(empty!, lo.oporder)
function Base.resize!(lo::LoopOrder, N::Int)
    Nold = length(lo.loopnames)
    resize!(lo.oporder, 32N)
    for n ∈ 32Nold+1:32N
        lo.oporder[n] = Operation[]
    end
    resize!(lo.loopnames, N)
    resize!(lo.bestorder, N)
    lo
end
Base.size(lo::LoopOrder) = (4,2,2,2,length(lo.loopnames))
Base.@propagate_inbounds Base.getindex(lo::LoopOrder, i::Int) = lo.oporder[i]
Base.@propagate_inbounds Base.getindex(lo::LoopOrder, i...) = lo.oporder[LinearIndices(size(lo))[i...]]

# Must make it easy to iterate
# outer_reductions is a vector of indixes (within operation vectors) of the reduction operation, eg the vmuladd op in a dot product
struct LoopSet
    loops::Dict{Symbol,Loop} # sym === loops[sym].itersymbol
    opdict::Dict{Symbol,Operation}
    operations::Vector{Operation} # Split them to make it easier to iterate over just a subset
    outer_reductions::Vector{Int} # IDs of reduction operations that need to be reduced at end.
    loop_order::LoopOrder
    # stridesets::Dict{ShortVector{Symbol},ShortVector{Symbol}}
    preamble::Expr # TODO: add preamble to lowering
    includedarrays::Vector{Tuple{Symbol,Int}}
    syms_aliasing_refs::Vector{Symbol} # O(N) search is faster at small sizes
    refs_aliasing_syms::Vector{ArrayReference}
    cost_vec::Matrix{Float64}
    reg_pres::Matrix{Int}
    # sym_to_ref_aliases::Dict{Symbol,ArrayReference}
    # ref_to_sym_aliases::Dict{ArrayReference,Symbol}
end

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
pushpreamble!(ls::LoopSet, ex) = push!(ls.preamble.args, ex)

function includesarray(ls::LoopSet, array::Symbol)
    for (a,i) ∈ ls.includedarrays
        a === array && return i
    end
    -1
end
function LoopSet()
    LoopSet(
        Dict{Symbol,Loop}(),
        Dict{Symbol,Operation}(),
        Operation[],
        Int[],
        LoopOrder(),
        Expr(:block,),
        Tuple{Symbol,Int}[],
        Symbol[],
        ArrayReference[],
        Matrix{Float64}(undef, 4, 2),
        Matrix{Int}(undef, 4, 2)
    )
end
num_loops(ls::LoopSet) = length(ls.loops)
function oporder(ls::LoopSet)
    N = length(ls.loop_order.loopnames)
    reshape(ls.loop_order.oporder, (4,2,2,2,N))
end
names(ls::LoopSet) = ls.loop_order.loopnames
isstaticloop(ls::LoopSet, s::Symbol) = ls.loops[s].hintexact
looprangehint(ls::LoopSet, s::Symbol) = ls.loops[s].rangehint
looprangesym(ls::LoopSet, s::Symbol) = ls.loops[s].rangesym
# itersyms(ls::LoopSet) = keys(ls.loops)
getop(ls::LoopSet, s::Symbol) = ls.opdict[s]
getop(ls::LoopSet, i::Int) = ls.operations[i + 1]

function looprange(ls::LoopSet, s::Symbol, incr::Int = 1, mangledname::Symbol = s, loop = ls.loops[s])
    incr -= 1
    if iszero(incr)
        Expr(:call, :<, mangledname, loop.hintexact ? loop.rangehint : loop.rangesym)
    else
        Expr(:call, :<, mangledname, loop.hintexact ? loop.rangehint - incr : Expr(:call, :-, loop.rangesym, incr))
    end
end
function looprange(ls::LoopSet, s::Symbol, incr::Expr, mangledname::Symbol = s, loop = ls.loops[s])
    increxpr = Expr(:call, :-, incr, 1)
    increxpr = if loop.hintexact
        Expr(:call, :-, loop.rangehint, increxpr)
    else
        Expr(:call, :-, loop.rangesym, increxpr)
    end
    Expr(:call, :<, mangledname, increxpr)
end
function Base.length(ls::LoopSet, is::Symbol)
    ls.loops[is].rangehint
end

function Operation(
    ls::LoopSet, variable, elementbytes, instruction,
    node_type, dependencies, reduced_deps, parents, ref = NOTAREFERENCE
)
    Operation(
        length(operations(ls)), variable, elementbytes, instruction,
        node_type, dependencies, reduced_deps, parents, ref
    )
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
function register_single_loop!(ls::LoopSet, looprange::Expr)
    itersym = (looprange.args[1])::Symbol
    r = (looprange.args[2])::Expr
    @assert r.head === :call
    f = first(r.args)
    loop::Loop = if f === :(:)
        lower = r.args[2]
        upper = r.args[3]
        lii::Bool = lower isa Integer
        uii::Bool = upper isa Integer
        if lii & uii
            Loop(itersym, 1 + convert(Int,upper) - convert(Int,lower))
        else
            N = gensym(Symbol(:loop, itersym))
            ex = if lii
                lower == 1 ? upper : Expr(:call, :-, upper, lower - 1)
            elseif uii
                Expr(:call, :-, upper + 1, lower)
            else
                Expr(:call, :-, Expr(:call, :+, upper, 1), lower)
            end
            pushpreamble!(ls, Expr(:(=), N, ex))
            Loop(itersym, N)
        end
    elseif f === :eachindex
        N = gensym(Symbol(:loop, itersym))
        pushpreamble!(ls, Expr(:(=), N, Expr(:call, :length, r.args[2])))
        Loop(itersym, N)
    else
        throw("Unrecognized loop range type: $r.")
    end
    ls.loops[itersym] = loop
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
        Base.push!(ls, q, elementbytes)
    end
end
function add_vptr!(ls::LoopSet, indexed::Symbol, id::Int)
    if includesarray(ls, indexed) < 0
        push!(ls.includedarrays, (indexed, id))
        pushpreamble!(ls, Expr(:(=), Symbol("##vptr##_", indexed), Expr(:call, lv(:stridedpointer), indexed)))
    end
    nothing
end

function add_load!(
    ls::LoopSet, var::Symbol, ref::ArrayReference, elementbytes::Int = 8
)
    if ref.loaded[] == true
        op = getop(ls, var)
        @assert var === op.variable
        return op
    end
    push!(ls.syms_aliasing_refs, var)
    push!(ls.refs_aliasing_syms, ref)
    ref.loaded[] = true
    # ls.sym_to_ref_aliases[ var ] = ref
    # ls.ref_to_sym_aliases[ ref ] = var
    op = Operation(
        length(operations(ls)), var, elementbytes,
        :getindex, memload, loopdependencies(ref),
        NODEPENDENCY, NOPARENTS, ref
    )
    add_vptr!(ls, ref.array, identifier(op))
    pushop!(ls, op, var)
end
function add_load_ref!(ls::LoopSet, var::Symbol, ex::Expr, elementbytes::Int = 8)
    ref = ref_from_ref(ex)
    add_load!(ls, var, ref, elementbytes)
end
function add_load_getindex!(ls::LoopSet, var::Symbol, ex::Expr, elementbytes::Int = 8)
    ref = ref_from_getindex(ex)
    add_load!(ls, var, ref, elementbytes)
end
function instruction(x)
    x isa Symbol ? x : last(x.args).value
end

function addsetv!(s::AbstractVector{T}, v::T) where {T}
    for sᵢ ∈ s
        sᵢ === v && return nothing
    end
    push!(s, v)
    nothing
end
function mergesetv!(s1::AbstractVector{T}, s2::AbstractVector{T}) where {T}
    for s ∈ s2
        addsetv!(s1, s)
    end
    nothing
end
function mergesetdiffv!(
    s1::AbstractVector{T},
    s2::AbstractVector{T},
    s3::AbstractVector{T}
) where {T}
    for s ∈ s2
        s ∉ s3 && addsetv!(s1, s)
    end
    nothing
end
function setdiffv!(s3::AbstractVector{T}, s1::AbstractVector{T}, s2::AbstractVector{T}) where {T}
    for s ∈ s1
        (s ∈ s2) || (s ∉ s3 && push!(s3, s))
    end
end
# This version has no dependencies, and thus will not be lowered
### if it is a literal, that literal is either var"##ZERO#Float##", var"##ONE#Float##", or has to have been assigned to var in the preamble.
# if it is a literal, that literal has to have been assigned to var in the preamble.
function add_constant!(ls::LoopSet, var::Symbol, elementbytes::Int = 8)
    pushop!(ls, Operation(length(operations(ls)), var, elementbytes, Symbol("##CONSTANT##"), constant, NODEPENDENCY, Symbol[], NOPARENTS), var)
end
function add_constant!(ls::LoopSet, var, elementbytes::Int = 8)
    sym = gensym(:temp)
    pushpreamble!(ls, Expr(:(=), sym, var))
    pushop!(ls, Operation(length(operations(ls)), sym, elementbytes, Symbol("##CONSTANT##"), constant, NODEPENDENCY, Symbol[], NOPARENTS), sym)
end
# This version has loop dependencies. var gets assigned to sym when lowering.
function add_constant!(ls::LoopSet, var::Symbol, deps::Vector{Symbol}, sym::Symbol = gensym(:constant), elementbytes::Int = 8)
    # length(deps) == 0 && push!(ls.preamble.args, Expr(:(=), sym, var))
    pushop!(ls, Operation(length(operations(ls)), sym, elementbytes, var, constant, deps, NODEPENDENCY, NOPARENTS), sym)
end
function add_constant!(ls::LoopSet, var, deps::Vector{Symbol}, sym::Symbol = gensym(:constant), elementbytes::Int = 8)
    sym2 = gensym(:temp)
    pushpreamble!(ls, Expr(:(=), sym2, var))
    pushop!(ls, Operation(length(operations(ls)), sym, elementbytes, sym2, constant, deps, NODEPENDENCY, NOPARENTS), sym)
end
function pushparent!(parents::Vector{Operation}, deps::Vector{Symbol}, reduceddeps::Vector{Symbol}, parent::Operation)
    push!(parents, parent)
    mergesetdiffv!(deps, loopdependencies(parent), reduceddependencies(parent))
    if !(isload(parent) || isconstant(parent))
        mergesetv!(reduceddeps, reduceddependencies(parent))
    end
    nothing
end
function maybe_cse_load!(ls::LoopSet, expr::Expr, elementbytes::Int = 8)
    if expr.head === :ref
        offset = 0
        # array = first(expr.args)::Symbol
        # args = @view expr.args[2:end]
        # ref = ref_from_ref(expr)
    elseif expr.head === :call && first(expr.args) === :getindex
        offset = 1
        # array = (expr.args[2])::Symbol
        # args = @view expr.args[3:end]
        # ref = ref_from_getindex(expr)
    else
        return add_operation!(ls, gensym(:temporary), expr, elementbytes)
    end
    ref = ArrayReference(
        expr.args[1+offset],
        @view(expr.args[2+offset:end]),
        Ref(false)
    )::ArrayReference
    # @show ref.ref
    id = findfirst(r -> r == ref, ls.refs_aliasing_syms)
    if id === nothing
        add_load!( ls, gensym(:temporary), ref, elementbytes )
    else
        getop(ls, ls.syms_aliasing_refs[id])        
    end
    # id = includesarray(ls, array)
    # if id > 0
        # ls.operations[id]
    # else
        # add_load!( ls, gensym(:temporary), array, args, elementbytes )
    # end
end
function add_parent!(
    parents::Vector{Operation}, deps::Vector{Symbol}, reduceddeps::Vector{Symbol}, ls::LoopSet, var, elementbytes::Int = 8
)
    parent = if var isa Symbol
        get!(ls.opdict, var) do
            # might add constant
            add_constant!(ls, var, elementbytes)
        end
    elseif var isa Expr #CSE candidate
        maybe_cse_load!(ls, var, elementbytes)
    else # assumed constant
        add_constant!(ls, var, elementbytes)
    end
    pushparent!(parents, deps, reduceddeps, parent)
end
function add_reduction!(
    parents::Vector{Operation}, deps::Vector{Symbol}, reduceddeps::Vector{Symbol}, ls::LoopSet, var::Symbol, elementbytes::Int = 8
)
    get!(ls.opdict, var) do
        add_constant!(ls, var, elementbytes)
        # push!(ls.outer_reductions, identifier(p))
        # p
    end
    # pushparent!(parents, deps, reduceddeps, parent)
end
function add_reduction_update_parent!(
    parents::Vector{Operation}, deps::Vector{Symbol}, reduceddeps::Vector{Symbol}, ls::LoopSet,
    var::Symbol, instr::Symbol, elementbytes::Int = 8
)
    parent = getop(ls, var)
    setdiffv!(reduceddeps, deps, loopdependencies(parent))
    pushparent!(parents, deps, reduceddeps, parent) # deps and reduced deps will not be disjoint
    op = Operation(length(operations(ls)), var, elementbytes, instr, compute, deps, reduceddeps, parents)
    parent.instruction === Symbol("##CONSTANT##") && push!(ls.outer_reductions, identifier(op))
    pushop!(ls, op, var) # note this overwrites the entry in the operations dict, but not the vector
end
function add_compute!(ls::LoopSet, var::Symbol, ex::Expr, elementbytes::Int = 8, ref = nothing)
    @assert ex.head === :call
    instr = instruction(first(ex.args))::Symbol
    args = @view(ex.args[2:end])
    parents = Operation[]
    deps = Symbol[]
    reduceddeps = Symbol[]
    # op = Operation( length(operations(ls)), var, elementbytes, instr, compute )
    reduction = false
    for arg ∈ args
        if var === arg
            reduction = true
            add_reduction!(parents, deps, reduceddeps, ls, arg, elementbytes)
        elseif ref == arg
            reduction = true
            add_load!(ls, var, ref, elementbytes)
        else
            add_parent!(parents, deps, reduceddeps, ls, arg, elementbytes)
        end
    end
    if reduction # arg[reduction] is the reduction
        add_reduction_update_parent!(parents, deps, reduceddeps, ls, var, instr, elementbytes)
    else
        op = Operation(length(operations(ls)), var, elementbytes, instr, compute, deps, reduceddeps, parents)
        pushop!(ls, op, var)
    end
end
function add_store!(
    ls::LoopSet, var::Symbol, ref::ArrayReference, elementbytes::Int = 8
)
    parent = getop(ls, var)
    op = Operation( length(operations(ls)), ref.array, elementbytes, :setindex!, memstore, loopdependencies(ref), reduceddependencies(parent), [parent], ref )
    add_vptr!(ls, ref.array, identifier(op))
    pushop!(ls, op, var)
end
function add_store_ref!(ls::LoopSet, var::Symbol, ex::Expr, elementbytes::Int = 8)
    ref = ref_from_ref(ex)
    add_store!(ls, var, ref, elementbytes)
end
function add_store_setindex!(ls::LoopSet, ex::Expr, elementbytes::Int = 8)
    ref = ref_from_setindex(ex)
    add_store!(ls, var, ref, elementbytes)
end
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
            c = gensym(:constant)
            pushpreamble!(ls, Expr(:(=), c, RHS))
            add_constant!(ls, c, [keys(ls.loops)...], LHS, elementbytes)
        else
            add_compute!(ls, LHS, RHS, elementbytes)
        end
    else
        throw("Expression not recognized:\n$x")
    end
end
function add_operation!(
    ls::LoopSet, LHS_sym::Symbol, RHS::Expr, LHS_ref::ArrayReference, elementbytes::Int = 8
)
    if RHS.head === :ref# || (RHS.head === :call && first(RHS.args) === :getindex)
        add_load!(ls, LHS_sym, LHS_ref, elementbytes)
    elseif RHS.head === :call
        if first(RHS.args) === :getindex
            add_load!(ls, LHS_sym, LHS_ref, elementbytes)
        else
            add_compute!(ls, LHS_sym, RHS, elementbytes, LHS_ref)
        end
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
                add_constant!(ls, RHS, [keys(ls.loops)...], LHS, elementbytes)
            end
        elseif LHS isa Expr
            @assert LHS.head === :ref
            local lrhs::Symbol
            if RHS isa Symbol
                lrhs = RHS
            elseif RHS isa Expr
                # need to check of LHS appears in RHS
                # assign RHS to lrhs
                ref = ArrayReference(LHS)
                id = findfirst(r -> r == ref, ls.refs_aliasing_syms)
                lrhs = id === nothing ? gensym(:RHS) : ls.syms_aliasing_refs[id]
                # we pass ref, so it can compare references within RHS, and realize
                # they equal lrhs
                add_operation!(ls, lrhs, RHS, ref, elementbytes)
            end
            add_store_ref!(ls, lrhs, LHS, elementbytes)
        else
            throw("LHS not understood:\n$LHS")
        end
    elseif ex.head === :block
        add_block!(ls, ex)
    elseif ex.head === :for
        add_loop!(ls, ex)
    else
        throw("Don't know how to handle expression:\n$ex")
    end
end

function place_after_loop(op::Operation)
    if isload(op) || length(reduceddependencies(op)) == 0
        1
    elseif length(reduceddependencies(op)) > 1
        2
    else
        rd = first(reduceddependencies(op))
        any(d -> d === rd, loopdependencies(op)) ? 1 : 2
    end
end

function fillorder!(ls::LoopSet, order::Vector{Symbol}, loopistiled::Bool)
    lo = ls.loop_order
    ro = lo.loopnames # reverse order; will have same order as lo
    # @show 1, ro, order
    # copyto!(ro, order)
    # @show 2, ro, order
    empty!(lo)
    nloops = length(order)
    if loopistiled
        tiled    = order[1]
        unrolled = order[2]
    else
        tiled = Symbol("##UNDEFINED##")
        unrolled = first(order)
    end
    included_vars = fill(false, length(operations(ls)))
    # to go inside out, we just have to include all those not-yet included depending on the current sym
    for _n ∈ 1:nloops
        n = 1 + nloops - _n
        ro[_n] = loopsym = order[n]
        #loopsym = order[n]
        for (id,op) ∈ enumerate(operations(ls))
            included_vars[id] && continue
            loopsym ∈ loopdependencies(op) || continue
            included_vars[id] = true
            isunrolled = (unrolled ∈ loopdependencies(op)) + 1
            istiled = (loopistiled ? (tiled ∈ loopdependencies(op)) : false) + 1
            optype = Int(op.node_type) + 1
            after_loop = place_after_loop(op)
            push!(lo[optype,isunrolled,istiled,after_loop,_n], op)
        end
    end    
    # 3, ro, order
end


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





