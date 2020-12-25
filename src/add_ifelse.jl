
## Currently, if/else will create its own local scope
## Assignments will not register in the loop's main scope
## although stores and return values will.
negateop!(ls::LoopSet, condop::Operation, elementbytes::Int) = add_compute!(ls, gensym(:negated_mask), :~, [condop], elementbytes)

function add_if!(ls::LoopSet, LHS::Symbol, RHS::Expr, elementbytes::Int, position::Int, mpref::Union{Nothing,ArrayReferenceMetaPosition} = nothing)
    # for now, just simple 1-liners
    @assert length(RHS.args) == 3 "if statements without an else cannot be assigned to a variable."
    condition = first(RHS.args)
    condop = if condition isa Symbol
        getop(ls, condition, elementbytes)
    elseif isnothing(mpref)
        add_operation!(ls, gensym(:mask), condition, elementbytes, position)
    else
        add_operation!(ls, gensym(:mask), condition, mpref, elementbytes, position)
    end
    iftrue = RHS.args[2]
    if iftrue isa Expr
        trueop = add_operation!(ls, gensym(:iftrue), iftrue, elementbytes, position)
        if iftrue.head === :ref && all(ld -> ld ∈ loopdependencies(trueop), loopdependencies(condop)) && !search_tree(parents(condop), trueop)
            trueop.instruction = Instruction(:conditionalload)
            push!(parents(trueop), condop)
        end
    else
        trueop = getop(ls, iftrue, elementbytes)
    end
    iffalse = RHS.args[3]
    if iffalse isa Expr
        falseop = add_operation!(ls, gensym(:iffalse), iffalse, elementbytes, position)
        if iffalse.head === :ref && all(ld -> ld ∈ loopdependencies(falseop), loopdependencies(condop)) && !search_tree(parents(condop), falseop)
            falseop.instruction = Instruction(:conditionalload)
            push!(parents(falseop), negateop!(ls, condop, elementbytes))
        end
    else
        falseop = getop(ls, iffalse, elementbytes)
    end
    add_compute_ifelse!(ls, LHS, condop, trueop, falseop, elementbytes)
end

function add_andblock!(ls::LoopSet, condop::Operation, LHS, rhsop::Operation, elementbytes::Int, position::Int)
    if LHS isa Symbol
        altop = getop(ls, LHS, elementbytes)
        return add_compute_ifelse!(ls, LHS, condop, rhsop, altop, elementbytes)
    elseif LHS isa Expr && LHS.head === :ref
        return add_conditional_store!(ls, LHS, condop, rhsop, elementbytes)
    else
        throw("Don't know how to assign onto $LHS.")
    end
end
function add_andblock!(ls::LoopSet, condop::Operation, LHS, RHS::Expr, elementbytes::Int, position::Int)
    rhsop = add_compute!(ls, gensym(:iftruerhs), RHS, elementbytes, position)
    add_andblock!(ls, condop, LHS, rhsop, elementbytes, position)
end
function add_andblock!(ls::LoopSet, condop::Operation, LHS, RHS, elementbytes::Int, position::Int)
    rhsop = getop(ls, RHS, elementbytes)
    add_andblock!(ls, condop, LHS, rhsop, elementbytes, position)
end
function add_andblock!(ls::LoopSet, condexpr::Expr, condeval::Expr, elementbytes::Int, position::Int)
    condop = add_operation!(ls, gensym(:mask), condexpr, elementbytes, position)
    if condeval.head === :call
        @assert first(condeval.args) === :setindex!
        array, raw_indices = ref_from_setindex!(ls, condeval)
        ref = Expr(:ref, array); append!(ref.args, raw_indices)
        return add_andblock!(ls, condop, ref, condeval.args[3], elementbytes, position)
    end
    @assert condeval.head === :(=)    
    @assert length(condeval.args) == 2
    LHS = condeval.args[1]
    RHS = condeval.args[2]
    add_andblock!(ls, condop, LHS, RHS, elementbytes, position)
end
function add_andblock!(ls::LoopSet, ex::Expr, elementbytes::Int, position::Int)
    add_andblock!(ls, first(ex.args)::Expr, last(ex.args)::Expr, elementbytes, position)
end

function add_orblock!(ls::LoopSet, condop::Operation, LHS, rhsop::Operation, elementbytes::Int, position::Int)
    negatedcondop = negateop!(ls, condop, elementbytes)
    if LHS isa Symbol
        altop = getop(ls, LHS, elementbytes)
        # return add_compute!(ls, LHS, :ifelse, [condop, altop, rhsop], elementbytes)
        # Placing altop second seems to let LLVM fuse operations; but as of LLVM 9.0.1 it will not if altop is first
        # therefore, we negate the condition and switch order so that the altop is second.
        return add_compute_ifelse!(ls, LHS, negatedcondop, rhsop, altop, elementbytes)
    elseif LHS isa Expr && LHS.head === :ref
        # negatedcondop = add_compute!(ls, gensym(:negated_mask), :~, [condop], elementbytes)
        return add_conditional_store!(ls, LHS, negatedcondop, rhsop, elementbytes)
    else
        throw("Don't know how to assign onto $LHS.")
    end
end
function add_orblock!(ls::LoopSet, condop::Operation, LHS, RHS::Expr, elementbytes::Int, position::Int)
    rhsop = add_compute!(ls, gensym(:iffalserhs), RHS, elementbytes, position)
    add_orblock!(ls, condop, LHS, rhsop, elementbytes, position)
end
function add_orblock!(ls::LoopSet, condop::Operation, LHS, RHS, elementbytes::Int, position::Int)
    rhsop = getop(ls, RHS, elementbytes)
    add_orblock!(ls, condop, LHS, rhsop, elementbytes, position)
end
function add_orblock!(ls::LoopSet, condexpr::Expr, condeval::Expr, elementbytes::Int, position::Int)
    condop = add_operation!(ls, gensym(:mask), condexpr, elementbytes, position)
    if condeval.head === :call
        @assert first(condeval.args) === :setindex!
        array, raw_indices = ref_from_setindex!(ls, condeval)
        ref = Expr(:ref, array); append!(ref.args, raw_indices)
        return add_orblock!(ls, condop, ref, condeval.args[3], elementbytes, position)
    end
    @assert condeval.head === :(=)
    @assert length(condeval.args) == 2
    LHS = condeval.args[1]
    RHS = condeval.args[2]
    add_orblock!(ls, condop, LHS, RHS, elementbytes, position)
end
function add_orblock!(ls::LoopSet, ex::Expr, elementbytes::Int, position::Int)
    add_orblock!(ls, first(ex.args)::Expr, last(ex.args)::Expr, elementbytes, position)
end

