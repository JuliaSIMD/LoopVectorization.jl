function add_constant!(ls::LoopSet, var::Symbol, elementbytes::Int)
    op = Operation(length(operations(ls)), var, elementbytes, LOOPCONSTANT, constant, NODEPENDENCY, Symbol[], NOPARENTS)
    pushpreamble!(ls, op, var)
    pushop!(ls, op, var)
end
# function add_constant!(ls::LoopSet, var, elementbytes::Int = 8)
#     sym = gensym(:loopconstant)
#     pushpreamble!(ls, Expr(:(=), sym, var))
#     add_constant!(ls, sym, elementbytes)
# end
function add_constant!(ls::LoopSet, var::Number, elementbytes::Int = 8)
    op = Operation(length(operations(ls)), gensym(:loopconstnumber), elementbytes, LOOPCONSTANT, constant, NODEPENDENCY, Symbol[], NOPARENTS)
    ops = operations(ls)
    typ = var isa Integer ? HardInt : HardFloat
    if iszero(var)
        for (id,typ_) ∈ ls.preamble_zeros
            (instruction(ops[id]) === LOOPCONSTANT && typ == typ_) && return ops[id]
        end
        push!(ls.preamble_zeros, (identifier(op),typ))
    elseif isone(var)
        for (id,typ_) ∈ ls.preamble_ones
            (instruction(ops[id]) === LOOPCONSTANT && typ == typ_) && return ops[id]
        end
        push!(ls.preamble_ones, (identifier(op),typ))
    elseif var isa Integer
        for (id,ivar) ∈ ls.preamble_symint
            (instruction(ops[id]) === LOOPCONSTANT && ivar == var) && return ops[id]
        end
        push!(ls.preamble_symint, (identifier(op), var))
    else#if var isa FloatX
        for (id,fvar) ∈ ls.preamble_symfloat
            (instruction(ops[id]) === LOOPCONSTANT && fvar == var) && return ops[id]
        end
        push!(ls.preamble_symfloat, (identifier(op), var))
    end
    pushop!(ls, op)
end
function add_constant!(ls::LoopSet, mpref::ArrayReferenceMetaPosition, elementbytes::Int)
    op = Operation(length(operations(ls)), varname(mpref), elementbytes, LOOPCONSTANT, constant, NODEPENDENCY, Symbol[], NOPARENTS, mpref.mref)
    add_vptr!(ls, op)
    temp = gensym(:intermediateconstref)
    vloadcall = Expr(:call, lv(:vload), mpref.mref.ptr)
    if length(getindices(op)) > 0
        push!(vloadcall.args, mem_offset(op, UnrollArgs(0, Symbol(""), Symbol(""), Symbol(""), nothing)))
    end
    pushpreamble!(ls, Expr(:(=), temp, vloadcall))
    pushpreamble!(ls, op, temp)
    pushop!(ls, op, temp)
end
# This version has loop dependencies. var gets assigned to sym when lowering.
# value is what will get assigned within the loop.
# assignedsym will be assigned to value within the preamble
function add_constant!(
    ls::LoopSet, value::Symbol, deps::Vector{Symbol}, assignedsym::Symbol, elementbytes::Int, f::Symbol = Symbol("")
)
    retop = get(ls.opdict, value, nothing)
    # @show retop, value ls.opdict
    if retop !== nothing
        op = Operation(length(operations(ls)), assignedsym, elementbytes, :identity, compute, deps, reduceddependencies(retop), [retop])
    else
        op = Operation(length(operations(ls)), assignedsym, elementbytes, Instruction(f, value), constant, deps, NODEPENDENCY, NOPARENTS)
    end
    pushop!(ls, op, assignedsym)
end
# function add_constant!(
#     ls::LoopSet, value, deps::Vector{Symbol}, assignedsym::Symbol, elementbytes::Int, f::Symbol = Symbol("")
# )
#     intermediary = gensym(:intermediate) # hack, passing meta info here
#     pushpreamble!(ls, Expr(:(=), intermediary, value))
#     add_constant!(ls, intermediary, deps, assignedsym, f, elementbytes)
# end
function add_constant!(
    ls::LoopSet, value::Number, deps::Vector{Symbol}, assignedsym::Symbol, elementbytes::Int
)
    op = add_constant!(ls, gensym(string(value)), deps, assignedsym, elementbytes, :numericconstant)
    pushpreamble!(ls, op, value)
    op
end
