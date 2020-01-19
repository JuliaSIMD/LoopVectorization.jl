
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

function add_andblock!(ls::LoopSet, ex::Expr)
    condop = add_compute!(ls, gensym(:mask), first(ex.args), elementbytes)
    condeval = last(ex.args)::Expr
    @assert condeval.head === :(=)
    @assert length(condeval.args) == 2
    LHS = condeval.args[1]
    RHS = condeval.args[2]
    rhsop = add_compute!(ls, gensym(:iftruerhs), RHS, elementbytes)
    if LHS isa Symbol
        altop = getop(ls, LHS)
        return add_compute!(ls, LHS, :vifelse, [condop, rhsop, altop], elementbytes)
    elseif LHS isa Expr && LHS.head === :ref
        return add_conditional_store!(ls, LHS, condop, rhsop, elementbytes)
    else
        throw("Don't know how to assign onto $LHS.")
    end        
end
function add_orblock!(ls, ex::Expr)
    condop = add_compute!(ls, gensym(:mask), first(ex.args), elementbytes)
    condeval = last(ex.args)::Expr
    @assert condeval.head === :(=)
    @assert length(condeval.args) == 2
    LHS = condeval.args[1]
    RHS = condeval.args[2]
    rhsop = add_compute!(ls, gensym(:iftruerhs), RHS, elementbytes)
    if LHS isa Symbol
        altop = getop(ls, LHS)
        return add_compute!(ls, LHS, :vifelse, [condop, altop, rhsop], elementbytes)
    elseif LHS isa Expr && LHS.head === :ref
        negatedcondop = add_compute!(ls, gensym(:negated_mask), :vnot, [condop], elementbytes)
        return add_conditional_store!(ls, LHS, negatedcondop, rhsop, elementbytes)
    else
        throw("Don't know how to assign onto $LHS.")
    end        
end

