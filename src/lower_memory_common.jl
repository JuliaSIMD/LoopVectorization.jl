
function parentind(ind::Symbol, op::Operation)
    for (id,opp) ∈ enumerate(parents(op))
        name(opp) === ind && return id
    end
    -1
end
function symbolind(ind::Symbol, op::Operation, td::UnrollArgs)
    id = parentind(ind, op)
    id == -1 && return ind
    @unpack u₁, u₁loopsym, u₂loopsym, suffix = td
    parent = parents(op)[id]
    pvar = if u₂loopsym ∈ loopdependencies(parent)
        variable_name(parent, suffix)
    else
        mangledvar(parent)
    end
    u₁loopsym ∈ loopdependencies(parent) ? Symbol(pvar, u₁) : pvar
end

function addoffset!(ret::Expr, ex::Union{Symbol,Expr}, offset::Integer)
    if iszero(offset)
        push!(ret.args, ex)
    else
        push!(ret.args, Expr(:call, :+, ex, convert(Int, offset)))
    end
    nothing
end

function mem_offset(op::Operation, td::UnrollArgs)
    # @assert accesses_memory(op) "Computing memory offset only makes sense for operations that access memory."
    ret = Expr(:tuple)
    indices = getindicesonly(op)
    offsets = getoffsets(op)
    loopedindex = op.ref.loopedindex
    for (n,ind) ∈ enumerate(indices)
        offset = offsets[n]
        # if ind isa Int # impossible
            # push!(ret.args, ind + offset)
        # else
        if loopedindex[n]
            addoffset!(ret, ind, offset)
        else
            addoffset!(ret, symbolind(ind, op, td), offset)
        end
    end
    ret
end
# function mem_offset_u(op::Operation, td::UnrollArgs)
#     @assert accesses_memory(op) "Computing memory offset only makes sense for operations that access memory."
#     @unpack u₁, u₁loopsym, u₂loopsym, suffix = td
#     incr₁ = u₁; incr₂ = isnothing(suffix) ? 0 : suffix
#     ret = Expr(:tuple)
#     indices = getindicesonly(op)
#     offsets = getoffsets(op)
#     loopedindex = op.ref.loopedindex
#     if iszero(incr₁) & iszero(incr₂)
#         return mem_offset(op, td)
#         # append_inds!(ret, indices, loopedindex)
#     else
#         for (n,ind) ∈ enumerate(indices)
#             offset = offsets[n]
#             # if ind isa Int
#                 # push!(ret.args, ind)
#             # else
#             if ind === u₁loopsym
#                 addoffset!(ret, ind, offset + incr₁)
#             elseif ind === u₂loopsym
#                 addoffset!(ret, ind, offset + incr₂)
#             elseif loopedindex[n]
#                 addoffset!(ret, ind, offset)
#             else
#                 addoffset!(ret, symbolind(ind, op, td), offset)
#             end
#         end
#     end
#     ret
# end

function add_vectorized_offset!(ret::Expr, ind, offset, incr)
    if isone(incr)
        if iszero(offset)
            push!(ret.args, Expr(:call, lv(:valadd), VECTORWIDTHSYMBOL, ind))
        else
            push!(ret.args, Expr(:call, :+, ind, Expr(:call, lv(:valadd), VECTORWIDTHSYMBOL, convert(Int, offset))))
        end
    elseif iszero(incr)
        if iszero(offset)
            push!(ret.args, ind)
        else
            addoffset!(ret, ind, offset)
        end
    elseif iszero(offset)
        push!(ret.args, Expr(:call, lv(:valmuladd), VECTORWIDTHSYMBOL, incr, ind))
    else
        push!(ret.args, Expr(:call, :+, ind, Expr(:call, lv(:valmuladd), VECTORWIDTHSYMBOL, incr, convert(Int, offset))))
    end
end
function mem_offset_u(op::Operation, td::UnrollArgs)
    @assert accesses_memory(op) "Computing memory offset only makes sense for operations that access memory."
    @unpack u₁, u₁loopsym, u₂loopsym, vectorized, suffix = td
    incr₁ = u₁
    incr₂ = isnothing(suffix) ? 0 : suffix
    ret = Expr(:tuple)
    indices = getindicesonly(op)
    offsets = getoffsets(op)
    loopedindex = op.ref.loopedindex
    if iszero(incr₁) & iszero(incr₂)
        return mem_offset(op, td)
        # append_inds!(ret, indices, loopedindex)
    else
        for (n,ind) ∈ enumerate(indices)
            offset = offsets[n]
            # if ind isa Int # impossible
                # push!(ret.args, ind + offset)
            # else
            if ind === u₁loopsym
                if u₁loopsym === vectorized
                    add_vectorized_offset!(ret, ind, offset, incr₁)
                else
                    addoffset!(ret, ind, incr₁ + offset)
                end
            elseif ind === u₂loopsym
                if u₂loopsym === vectorized
                    add_vectorized_offset!(ret, ind, offset, incr₂)
                else
                    addoffset!(ret, ind, incr₂ + offset)
                end
            elseif loopedindex[n]
                addoffset!(ret, ind, offset)
            else
                addoffset!(ret, symbolind(ind, op, td), offset)
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
function name_memoffset(var::Symbol, op::Operation, td::UnrollArgs, u₁unrolled::Bool = td.u₁loopsym ∈ loopdependencies(op))
    @unpack u₁, u₁loopsym, u₂loopsym, suffix = td
    if isnothing(suffix) && u₁ < 0 # sentinel value meaning not unrolled
        name = var
        mo = mem_offset(op, td)
    else
        name = u₁unrolled ? Symbol(var, u₁) : var
        mo = mem_offset_u(op, td)
    end
    name, mo
end

function condvarname_and_unroll(cond, u₁loop, u₂loop, suffix, opu₂)
    if isnothing(suffix) || !opu₂
        condvar, condu₁, condu₂ = variable_name_and_unrolled(cond, u₁loop, u₂loop, nothing)
    else
        condvar, condu₁, condu₂ = variable_name_and_unrolled(cond, u₁loop, u₂loop, suffix)
    end
    condvar, condu₁
end

