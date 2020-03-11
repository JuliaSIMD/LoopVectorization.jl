
struct UnrollArgs{T}
    u::Int
    unrolled::Symbol
    tiled::Symbol
    suffix::T
end
function parentind(ind::Symbol, op::Operation)
    for (id,opp) ∈ enumerate(parents(op))
        name(opp) === ind && return id
    end
    -1
end
function symbolind(ind::Symbol, op::Operation, td::UnrollArgs)
    id = parentind(ind, op)
    id == -1 && return ind
    @unpack u, unrolled, tiled, suffix = td
    parent = parents(op)[id]
    pvar = if tiled ∈ loopdependencies(parent)
        variable_name(parent, suffix)
    else
        mangledvar(parent)
    end
    unrolled ∈ loopdependencies(parent) ? Symbol(pvar, u) : pvar
end
function mem_offset(op::Operation, td::UnrollArgs)
    # @assert accesses_memory(op) "Computing memory offset only makes sense for operations that access memory."
    ret = Expr(:tuple)
    indices = getindices(op)
    loopedindex = op.ref.loopedindex
    start = (first(indices) === Symbol("##DISCONTIGUOUSSUBARRAY##")) + 1
    for (n,ind) ∈ enumerate(@view(indices[start:end]))
        if ind isa Int
            push!(ret.args, ind)
        elseif loopedindex[n]
            push!(ret.args, ind)
        else
            push!(ret.args, symbolind(ind, op, td))
        end
    end
    ret
end
function mem_offset_u(op::Operation, td::UnrollArgs)
    @assert accesses_memory(op) "Computing memory offset only makes sense for operations that access memory."
    @unpack unrolled, u = td
    incr = u
    ret = Expr(:tuple)
    indices = getindices(op)
    loopedindex = op.ref.loopedindex
    if incr == 0
        return mem_offset(op, td)
        # append_inds!(ret, indices, loopedindex)
    else
        start = (first(indices) === Symbol("##DISCONTIGUOUSSUBARRAY##")) + 1
        for (n,ind) ∈ enumerate(@view(indices[start:end]))
            if ind isa Int
                push!(ret.args, ind)
            elseif ind === unrolled
                push!(ret.args, Expr(:call, :+, ind, incr))
            elseif loopedindex[n]
                push!(ret.args, ind)
            else
                push!(ret.args, symbolind(ind, op, td))
            end
        end
    end
    ret
end
function mem_offset_u(op::Operation, td::UnrollArgs, mul::Symbol)
    @assert accesses_memory(op) "Computing memory offset only makes sense for operations that access memory."
    @unpack unrolled, u = td
    incr = u
    ret = Expr(:tuple)
    indices = getindices(op)
    loopedindex = op.ref.loopedindex
    if incr == 0
        return mem_offset(op, td)
        # append_inds!(ret, indices, loopedindex)
    else
        start = (first(indices) === Symbol("##DISCONTIGUOUSSUBARRAY##")) + 1
        for (n,ind) ∈ enumerate(@view(indices[start:end]))
            if ind isa Int
                push!(ret.args, ind)
            elseif ind === unrolled
                push!(ret.args, Expr(:call, :+, ind, Expr(:call, lv(:valmul), mul, incr)))
            elseif loopedindex[n]
                push!(ret.args, ind)
            else
                push!(ret.args, symbolind(ind, op, td))
            end
        end
    end
    ret
end

# function add_expr(q, incr)
#     if q.head === :call && q.args[2] === :+
#         qc = copy(q)
#         push!(qc.args, incr)
#         qc
#     else
#         Expr(:call, :+, q, incr)
#     end
# end
function varassignname(var::Symbol, u::Int, isunrolled::Bool)
    isunrolled ? Symbol(var, u) : var
end
# name_memoffset only gets called when vectorized
function name_memoffset(var::Symbol, op::Operation, td::UnrollArgs, W::Symbol, vecnotunrolled::Bool, parentisunrolled::Bool = true)
    @unpack u, unrolled = td
    if u < 0 # sentinel value meaning not unrolled
        name = var
        mo = mem_offset(op, td)
    else
        name = parentisunrolled ? Symbol(var, u) : var
        mo = vecnotunrolled ? mem_offset_u(op, td) : mem_offset_u(op, td, W)
    end
    name, mo
end

