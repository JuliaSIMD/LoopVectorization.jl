# using LightGraphs


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



@enum OperationType begin
    memload
    compute
    memstore
    # accumulator
end

# const ID = Threads.Atomic{UInt}(0)

# TODO: can some computations be cached in the operations?
"""
if ooperation_type == memstore || operation_type == memstore# || operation_type == compute_new || operation_type == compute_update
symbolic metadata contains info on direct dependencies / placement within loop.

if accesses_memory(op)
Symbol(:vptr_, op.variable)
is how we access the memory.
If numerical_metadata[i] == -1
Symbol(:stride_, op.variable, :_, op.symbolic_metadata[i])
is the stride for loop index
symbolic_metadata[i]
"""
struct Operation
    identifier::UInt
    variable::Symbol
    elementbytes::Int
    instruction::Symbol
    node_type::OperationType
    dependencies::Set{Symbol}
    reduced_deps::Set{Symbol}
    parents::Vector{Operation}
    # children::Vector{Operation}
    numerical_metadata::Vector{Int} # stride of -1 indicates dynamic
    symbolic_metadata::Vector{Symbol}
    function Operation(
        identifier,
        variable,
        elementbytes,
        instruction,
        node_type,
        variable = gensym()
    )
        new(
            identifier, variable, elementbytes, instruction, node_type,
            Set{Symbol}(), Set{Symbol}(), Operation[], Int[], Symbol[]
        )
    end
end



function isreduction(op::Operation)
    (op.node_type == memstore) && (length(op.symbolic_metadata) < length(op.dependencies))# && issubset(op.symbolic_metadata, op.dependencies)
end
isload(op::Operation) = op.node_type == memload
iscompute(op::Operation) = op.node_type == compute
isstore(op::Operation) = op.node_type == memstore
accesses_memory(op::Operation) = isload(op) | isstore(op)
elsize(op::Operation) = op.elementbytes
dependson(op::Operation, sym::Symbol) = sym ∈ op.dependencies
parents(op::Operation) = op.parents
# children(op::Operation) = op.children
loopdependencies(op::Operation) = op.dependencies
reduceddependencies(op::Operation) = op.reduced_deps
identifier(op::Operation) = op.identifier
name(op::Operation) = op.variable
instruction(op::Operation) = op.instruction

function hasintersection(s1::Set{T}, s2::Set{T}) where {T}
    for x ∈ s1
        x ∈ s2 && return true
    end
    false
end

function symposition(op::Operation, sym::Symbol)
    findfirst(s -> s === sym, op.symbolic_metadata)
end
function stride(op::Operation, sym::Symbol)
    @assert accesses_memory(op) "This operation does not access memory!"
    # access stride info?
    op.numerical_metadata[symposition(op,sym)]
end
# function
function unitstride(op::Operation, sym::Symbol)
    (first(op.symbolic_metadata) === sym) && (first(op.numerical_metadata) == 1)
end
function mem_offset(op::Operation, incr::Int = 0)::Union{Symbol,Expr}
    @assert accesses_memory(op) "Computing memory offset only makes sense for operations that access memory."
    @unpack numerical_metadata, symbolic_metadata = op
    if incr == 0 && length(numerical_metadata) == 1
        firstsym = first(symbolic_metadata)
        if first(numerical_metadata) == 1
            return firstsym
        elseif first(numerical_metadata) == -1
            return Expr(:call, :*,  Symbol(:stride_, op.variable, :_, firstsym), firstsym)
        else
            return Expr(:call, :*,  first(numerical_metadata), firstsym)
        end
    end
    ret = Expr(:call, :+, )
    for i ∈ eachindex(numerical_metadata)
        sym = symbolic_metadata[i]; num = numerical_metadata[i]
        if num == 1
            push!(ret.args, sym)
        elseif num == -1
            push!(ret.args, Expr(:call, :*, Symbol(:stride_, op.variable, :_, firstsym), sym))
        else
            push!(ret.args, Expr(:call, :*, num, sym))
        end        
    end
    incr == 0 || push!(ret.args, incr)
    ret
end


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
end
function LoopOrder(N::Int)
    LoopOrder( [ Operation[] for i ∈ 1:24N ], Vector{Symbol}(undef, N) )
end
LoopOrder() = LoopOrder(Vector{Operation}[])
Base.empty!(lo::LoopOrder) = foreach(empty!, lo.oporder)
function Base.resize!(lo::LoopOrder, N::Int)
    Nold = length(lo.loopnames)
    resize!(lo.oporder, 24N)
    for n ∈ 24Nold+1:24N
        lo.oporder[n] = Operation[]
    end
    resize!(lo.loopnames, N)
    lo
end
Base.size(lo::LoopOrder) = (3,2,2,2,length(lo.loopnames))
Base.@propagate_inbounds Base.getindex(lo::LoopOrder, i::Int) = lo.oporder[i]
Base.@propagate_inbounds Base.getindex(lo::LoopOrder, i...) = lo.oporder[LinearIndices(size(lo))[i...]]

# Must make it easy to iterate
struct LoopSet
    loops::Dict{Symbol,Loop} # sym === loops[sym].itersymbol
    opdict::Dict{Symbol,Operation}
    operations::Vector{Operation} # Split them to make it easier to iterate over just a subset
    # computeops::Vector{Operation}
    # storeops::Vector{Operation}
    outer_reductions::Set{UInt} # IDs of reduction operations that need to be reduced at end.
    loop_order::LoopOrder
    preamble::Expr # TODO: add preamble to lowering
end
function LoopSet()
    LoopSet(
        Dict{Symbol,Loop}(),
        Dict{Symbol,Operation}(),
        Operation[],
        # Operation[],
        # Operation[],
        # Set{UInt}(),
        Set{UInt}(),
        LoopOrder(),
        Expr(:block,)
    )
end
num_loops(ls::LoopSet) = length(ls.loops)
function oporder(ls::LoopSet)
    N = length(ls.loop_order.loopnames)
    reshape(ls.loop_order.oporder, (3,2,2,2,N))
end
names(ls::LoopSet) = ls.loop_order.loopnames
isstaticloop(ls::LoopSet, s::Symbol) = ls.loops[s].hintexact
looprangehint(ls::LoopSets, s::Symbol) = ls.loops[s].rangehint
looprangesym(ls::LoopSets, s::Symbol) = ls.loops[s].rangesym
itersyms(ls::LoopSet) = keys(ls.loops)
function looprange(ls::LoopSet, s::Symbol, incr::Int = 1)
    loop = ls.loops[s]
    incr -= 1
    if iszero(incr)
        Expr(:call, :<, s, loop.hintexact ? loop.rangehint : loop.rangesym)
    else
        Expr(:call, :<, s, loop.hintexact ? loop.rangehint - incr : Expr(:call, :-, loop.rangesym, incr))
    end
end
function Base.length(ls::LoopSet, is::Symbol)
    ls.loops[is].rangehint
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
function add_loop!(ls::LoopSet, looprange::Expr)
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
            N = gensym(:loop, itersym)
            ex = if lii
                Expr(:call, :-, upper, lower - 1)
            elseif uii
                Expr(:call, :-, upper + 1, lower)
            else
                Expr(:call, :-, Expr(:call, :+, upper, 1), lower)
            end
            push!(ls.preamble.args, Expr(:(=), N, ex))
            Loop(itersym, N)
        end
    elseif f === :eachindex
        N = gensym(:loop, itersym)
        push!(ls.preamble.args, Expr(:(=), N, Expr(:call, :length, r.args[2])))
        Loop(itersym, N)
    else
        throw("Unrecognized loop range type: $r.")
    end
    ls.loops[itersym] = loop
    nothing
end
function add_load!(ls::LoopSet, indexed::Symbol, indices::AbstractVector)
    Ninds = length(indices)
    
    

end
function add_load_getindex!(ls::LoopSet, ex::Expr)
    add_load!(ls, ex.args[2], @view(ex.args[3:end]))
end
function add_load_ref!(ls::LoopSet, ex::Expr)
    add_load!(ls, ex.args[1], @view(ex.args[2:end]))
end
function add_compute!(ls::LoopSet, ex::Expr)

end
function add_store!(ls::LoopSet, ex::Expr)

end
function Base.push!(ls::LoopSet, ex::Expr)
    
end

function fillorder!(ls::LoopSet, order::Vector{Symbol}, loopistiled::Bool)
    lo = ls.loop_order
    ro = lo.loopnames # reverse order; will have same order as lo
    copyto!(lo.names, order)
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
        for (id,op) ∈ enumerate(operations(ls))
            included_vars[id] && continue
            loopsym ∈ dependencies(op) || continue
            included_vars[id] = true
            isunrolled = (unrolled ∈ loopdependencies(op)) + 1
            istiled = (loopistiled ? false : (tiled ∈ loopdependencies(op))) + 1
            optype = Int(op.node_type)
            after_loop = (length(reduceddependencies(op)) > 0) + 1
            push!(lo[optype,isunrolled,istiled,after_loop,_n], op)
        end
    end    
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





