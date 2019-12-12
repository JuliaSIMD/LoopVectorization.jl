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
    memstore
    compute_new
    compute_update
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
    # dependencies::Vector{Symbol}
    dependencies::Set{Symbol}
    reduced_deps::Set{Symbol}
    # dependencies::Set{Symbol}
    parents::Vector{Operation}
    children::Vector{Operation}
    numerical_metadata::Vector{Int} # stride of -1 indicates dynamic
    symbolic_metadata::Vector{Symbol}
    # strides::Dict{Symbol,Union{Symbol,Int}}
    function Operation(
        identifier,
        elementbytes,
        instruction,
        node_type,
        variable = gensym()
    )
        new(
            identifier, variable, elementbytes, instruction, node_type,
            Set{Symbol}(), Operation[], Operation[], Int[], Symbol[]#, Dict{Symbol,Union{Symbol,Int}}()
        )
    end
end



function isreduction(op::Operation)
    (op.node_type == memstore) && (length(op.symbolic_metadata) < length(op.dependencies))# && issubset(op.symbolic_metadata, op.dependencies)
end
isload(op::Operation) = op.node_type == memload
isstore(op::Operation) = op.node_type == memstore
accesses_memory(op::Operation) = isload(op) | isstore(op)
elsize(op::Operation) = op.elementbytes
dependson(op::Operation, sym::Symbol) = sym ∈ op.dependencies
parents(op::Operation) = op.parents
children(op::Operation) = op.children
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
    oporder::Array{Vector{Operation},5}
    loopnames::Vector{Symbol}
end
function LoopOrder(N::Int)
    LoopOrder( [ Operation[] for i ∈ 1:3, j ∈ 1:2, k ∈ 1:2, l ∈ 1:2, n ∈ 1:N ], Vector{Symbol}(undef, N) )
end
Base.empty!(lo::LoopOrder) = foreach(empty!, lo.oporder)
Base.size(lo::LoopOrder) = (3,2,2,2,size(lo.oporder,5))
Base.@propagate_inbounds Base.getindex(lo::LoopOrder, i...) = lo.oporder[i...]

# Must make it easy to iterate
struct LoopSet
    loops::Dict{Symbol,Loop} # sym === loops[sym].itersymbol
    # operations::Vector{Operation}
    loadops::Vector{Operation} # Split them to make it easier to iterate over just a subset
    computeops::Vector{Operation}
    storeops::Vector{Operation}
    inner_reductions::Set{UInt} # IDs of reduction operations nested within loops and stored.
    outer_reductions::Set{UInt} # IDs of reduction operations that need to be reduced at end.
    loop_order::LoopOrder
    # strideset::Vector{} 
end
num_loops(ls::LoopSet) = length(ls.loops)
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
load_operations(ls::LoopSet) = ls.loadops
compute_operations(ls::LoopSet) = ls.computeops
store_operations(ls::LoopSet) = ls.storeops
function operations(ls::LoopSet)
    Base.Iterators.flatten((
        load_operations(ls),
        compute_operations(ls),
        store_operations(ls)
    ))
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
            optype = if isload(op)
                1
            elseif isstore(op)
                3
            else#if compute
                2
            end
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





