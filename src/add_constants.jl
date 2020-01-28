function add_constant!(ls::LoopSet, var::Symbol, elementbytes::Int)
    op = Operation(length(operations(ls)), var, elementbytes, LOOPCONSTANT, constant, NODEPENDENCY, Symbol[], NOPARENTS)
    pushpreamble!(ls, op, var)
    pushop!(ls, op, var)
end
function add_constant!(ls::LoopSet, var, elementbytes::Int = 8)
    sym = gensym(:loopconstant)
    pushpreamble!(ls, Expr(:(=), sym, var))
    add_constant!(ls, sym, elementbytes)
end
function add_constant!(ls::LoopSet, var::Symbol, mpref::ArrayReferenceMetaPosition, elementbytes::Int)
    op = Operation(length(operations(ls)), var, elementbytes, LOOPCONSTANT, constant, NODEPENDENCY, Symbol[], NOPARENTS, mpref.mref)
    add_vptr!(ls, op)
    temp = gensym(:intermediateconstref)
    pushpreamble!(ls, Expr(:(=), temp, Expr(:call, lv(:load), mpref.mref.ptr, mem_offset(op, UnrollArgs(0, Symbol(""), Symbol(""), nothing)))))
    pushpreamble!(ls, op, temp)
    pushop!(ls, op, temp)
end
# This version has loop dependencies. var gets assigned to sym when lowering.
# value is what will get assigned within the loop.
# assignedsym will be assigned to value within the preamble
function add_constant!(
    ls::LoopSet, value::Symbol, deps::Vector{Symbol}, assignedsym::Symbol, elementbytes::Int, f::Symbol = Symbol("")
)
    op = Operation(length(operations(ls)), assignedsym, elementbytes, Instruction(f, value), constant, deps, NODEPENDENCY, NOPARENTS)
    pushop!(ls, op, assignedsym)
end
function add_constant!(
    ls::LoopSet, value, deps::Vector{Symbol}, assignedsym::Symbol, elementbytes::Int, f::Symbol = Symbol("")
)
    intermediary = gensym(:intermediate) # hack, passing meta info here
    pushpreamble!(ls, Expr(:(=), intermediary, value))
    add_constant!(ls, intermediary, deps, assignedsym, f, elementbytes)
end
function add_constant!(
    ls::LoopSet, value::Number, deps::Vector{Symbol}, assignedsym::Symbol, elementbytes::Int
)
    op = add_constant!(ls, gensym(Symbol(value)), deps, assignedsym, elementbytes, :numericconstant)
    if iszero(value)
        push!(ls.preamble_zeros, identifier(op))
    elseif isone(value)
        push!(ls.preamble_ones, identifier(op))
    else
        pushpreamble!(ls, op, value)
    end
    op
end
