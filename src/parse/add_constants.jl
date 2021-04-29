function add_constant!(ls::LoopSet, var::Symbol, elementbytes::Int)
    op = Operation(length(operations(ls)), var, elementbytes, LOOPCONSTANT, constant, NODEPENDENCY, Symbol[], NOPARENTS)
    rop = pushop!(ls, op, var)
    rop === op && pushpreamble!(ls, op, var)
    rop
end
# function add_constant!(ls::LoopSet, var, elementbytes::Int = 8)
#     sym = gensym(:loopconstant)
#     pushpreamble!(ls, Expr(:(=), sym, var))
#     add_constant!(ls, sym, elementbytes)
# end
function add_constant!(ls::LoopSet, var::Number, elementbytes::Int = 8)
    op = Operation(length(operations(ls)), gensym!(ls, "loopconstnumber"), elementbytes, LOOPCONSTANT, constant, NODEPENDENCY, Symbol[], NOPARENTS)
    ops = operations(ls)
    typ = var isa Integer ? HardInt : HardFloat
    rop = pushop!(ls, op)
    rop !== op && return rop
    if iszero(var)
        for (id,typ_) ∈ ls.preamble_zeros
            (instruction(ops[id]) == LOOPCONSTANT && typ == typ_) && return ops[id]
        end
        push!(ls.preamble_zeros, (identifier(op),typ))
    elseif var isa Integer
        idescript = integer_description(var)
        for (id,descript) ∈ ls.preamble_symint
            if (instruction(ops[id]) == LOOPCONSTANT) && (idescript == descript)
                return ops[id]
            end
        end
        push!(ls.preamble_symint, (identifier(op), idescript))
    else#if var isa FloatX
        for (id,fvar) ∈ ls.preamble_symfloat
            (instruction(ops[id]) == LOOPCONSTANT && fvar == var) && return ops[id]
        end
        push!(ls.preamble_symfloat, (identifier(op), var))
    end
    rop
end
function add_constant!(ls::LoopSet, mpref::ArrayReferenceMetaPosition, elementbytes::Int)
    op = Operation(length(operations(ls)), varname(mpref), elementbytes, LOOPCONSTANT, constant, NODEPENDENCY, Symbol[], NOPARENTS, mpref.mref)
    add_vptr!(ls, op)
    temp = gensym!(ls, "intermediateconstref")
    vloadcall = Expr(:call, lv(:_vload), mpref.mref.ptr)
    nindices = length(getindices(op))
    # getoffsets(op) .+= 1
    if nindices > 0
        dummyloop = first(ls.loops)
        push!(vloadcall.args, mem_offset(op, UnrollArgs(dummyloop, dummyloop, dummyloop, 0, 0, 0), fill(false,nindices), true, ls))
    end
    push!(vloadcall.args, Expr(:call, lv(:False)), staticexpr(reg_size(ls)))
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
    if retop === nothing
        op = Operation(length(operations(ls)), assignedsym, elementbytes, Instruction(f, value), constant, deps, NODEPENDENCY, NOPARENTS)
    else
        op = Operation(length(operations(ls)), assignedsym, elementbytes, :identity, compute, deps, reduceddependencies(retop), [retop])
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
    op = add_constant!(ls, gensym!(ls, string(value)), deps, assignedsym, elementbytes, :numericconstant)
    pushpreamble!(ls, op, value)
    op
end
