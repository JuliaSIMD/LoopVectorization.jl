
## Currently, if/else will create its own local scope
## Assignments will not register in the loop's main scope
## although stores and return values will.


function add_if!(ls::LoopSet, LHS::Symbol, RHS::Expr, elementbytes::Int = 8, mpref::Union{Nothing,ArrayReferenceMetaPosition} = nothing)
    # for now, just simple 1-liners
    @assert length(RHS.args) == 3 "if statements without an else cannot be assigned to a variable."
    condition = first(RHS.args)
    condop = add_compute!(ls, gensym(:mask), condition, elementbytes, mpref)
    iftrue = RHS.args[2]
    (iftrue isa Expr && iftrue.head !== :call) && throw("Only calls or constant expressions are currently supported in if/else blocks.")
    trueop = add_operation!(ls, Symbol(:iftrue), iftrue, elementbytes)
    iffalse = RHS.args[3]
    (iffalse isa Expr && iffalse.head !== :call) && throw("Only calls or constant expressions are currently supported in if/else blocks.")
    falseop = add_operation!(ls, Symbol(:iffalse), iffalse, elementbytes)

    add_compute!(ls, LHS, :vifelse, [condop, trueop, falseop], elementbytes)
end

function add_andblock!(ls::LoopSet, condop::Operation, LHS, rhsop::Operation, elementbytes::Int)
    if LHS isa Symbol
        altop = getop(ls, LHS)
        return add_compute!(ls, LHS, :vifelse, [condop, rhsop, altop], elementbytes)
    elseif LHS isa Expr && LHS.head === :ref
        return add_conditional_store!(ls, LHS, condop, rhsop, elementbytes)
    else
        throw("Don't know how to assign onto $LHS.")
    end        
end
function add_andblock!(ls::LoopSet, condop::Operation, LHS, RHS::Expr, elementbytes::Int)
    rhsop = add_compute!(ls, gensym(:iftruerhs), RHS, elementbytes)
    add_andblock!(ls, condop, LHS, rhsop, elementbytes)
end
function add_andblock!(ls::LoopSet, condop::Operation, LHS, RHS, elementbytes::Int)
    rhsop = getop(ls, RHS)
    add_andblock!(ls, condop, LHS, rhsop, elementbytes)
end
function add_andblock!(ls::LoopSet, condexpr::Expr, condeval::Expr, elementbytes::Int)
    condop = add_compute!(ls, gensym(:mask), condexpr, elementbytes)
    @assert condeval.head === :(=)
    @assert length(condeval.args) == 2
    LHS = condeval.args[1]
    RHS = condeval.args[2]
    add_andblock!(ls, condop, LHS, RHS, elementbytes)
end
function add_andblock!(ls::LoopSet, ex::Expr, elementbytes::Int)
    add_andblock!(ls, first(ex.args)::Expr, last(ex.args)::Expr, elementbytes)
end

function add_orblock!(ls::LoopSet, condop::Operation, LHS, rhsop::Operation, elementbytes::Int)
    if LHS isa Symbol
        altop = getop(ls, LHS)
        return add_compute!(ls, LHS, :vifelse, [condop, altop, rhsop], elementbytes)
    elseif LHS isa Expr && LHS.head === :ref
        negatedcondop = add_compute!(ls, gensym(:negated_mask), :~, [condop], elementbytes)
        return add_conditional_store!(ls, LHS, negatedcondop, rhsop, elementbytes)
    else
        throw("Don't know how to assign onto $LHS.")
    end
end
function add_orblock!(ls::LoopSet, condop::Operation, LHS, RHS::Expr, elementbytes::Int)
    rhsop = add_compute!(ls, gensym(:iffalserhs), RHS, elementbytes)
    add_orblock!(ls, condop, LHS, rhsop, elementbytes)
end
function add_orblock!(ls::LoopSet, condop::Operation, LHS, RHS, elementbytes::Int)
    rhsop = getop(ls, RHS)
    add_orblock!(ls, condop, LHS, rhsop, elementbytes)
end
function add_orblock!(ls::LoopSet, condexpr::Expr, condeval::Expr, elementbytes::Int)
    condop = add_compute!(ls, gensym(:mask), condexpr, elementbytes)
    @assert condeval.head === :(=)
    @assert length(condeval.args) == 2
    LHS = condeval.args[1]
    RHS = condeval.args[2]
    add_orblock!(ls, condop, LHS, RHS, elementbytes)
end
function add_orblock!(ls::LoopSet, ex::Expr, elementbytes::Int)
    add_orblock!(ls, first(ex.args)::Expr, last(ex.args)::Expr, elementbytes)
end

