function add_constant!(ls::LoopSet, var::Symbol, elementbytes::Int = 8)
    op = Operation(length(operations(ls)), var, elementbytes, LOOPCONSTANT, constant, NODEPENDENCY, Symbol[], NOPARENTS)
    pushpreamble!(ls, op, mangledvar(op))
    pushop!(ls, op, var)
end
function add_constant!(ls::LoopSet, var, elementbytes::Int = 8)
    sym = gensym(:temp)
    op = Operation(length(operations(ls)), sym, elementbytes, LOOPCONSTANT, constant, NODEPENDENCY, Symbol[], NOPARENTS)
    pushpreamble!(ls, Expr(:(=), mangledvar(op), var))
    pushpreamble!(ls, op, mangledvar(op))
    pushop!(ls, op, sym)
end
function add_constant!(ls::LoopSet, var::Symbol, mpref::ArrayReferenceMetaPosition, elementbytes::Int)
    op = Operation(length(operations(ls)), var, elementbytes, LOOPCONSTANT, constant, NODEPENDENCY, Symbol[], NOPARENTS, mpref.mref)
    add_vptr!(ls, op)
    pushpreamble!(ls, Expr(:(=), mangledvar(op), Expr(:call, lv(:load), mpref.mref.ptr, mem_offset(op, UnrollArgs(zero(Int32), Symbol(""), Symbol(""), nothing)))))
    pushpreamble!(ls, op, mangledvar(op))
    pushop!(ls, op, var)
end
# This version has loop dependencies. var gets assigned to sym when lowering.
function add_constant!(ls::LoopSet, var::Symbol, deps::Vector{Symbol}, sym::Symbol = gensym(:constant), f::Symbol = Symbol(""), elementbytes::Int = 8)
    # length(deps) == 0 && push!(ls.preamble.args, Expr(:(=), sym, var))
    pushop!(ls, Operation(length(operations(ls)), sym, elementbytes, Instruction(f,var), constant, deps, NODEPENDENCY, NOPARENTS), sym)
end

function add_constant!(
    ls::LoopSet, var, deps::Vector{Symbol}, sym::Symbol = gensym(:constant), f::Symbol = Symbol(""), elementbytes::Int = 8
)
    sym2 = gensym(:temp) # hack, passing meta info here
    op = Operation(length(operations(ls)), sym, elementbytes, Instruction(f, sym2), constant, deps, NODEPENDENCY, NOPARENTS)
    # @show f, sym, name(op), mangledvar(op)
    # temp = gensym(:temp2)
    # pushpreamble!(ls, Expr(:(=), temp, var))
    pushpreamble!(ls, op, var)#temp)
    pushop!(ls, op, sym)
end
