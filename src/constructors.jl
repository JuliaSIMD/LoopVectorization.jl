
### This file contains convenience functions for constructing LoopSets.

function Base.copyto!(ls::LoopSet, q::Expr)
    q.head === :for || throw("Expression must be a for loop.")
    add_loop!(ls, q)
end

function add_ci_call!(q::Expr, f, args, syms, i)
    call = Expr(:call, f)
    for arg ∈ @view(args[2:end])
        if arg isa Core.SSAValue
            push!(call.args, syms[arg.id])
        else
            push!(call.args, arg)
        end
    end
    push!(q.args, Expr(:(=), syms[i], call))
end

function substitute_broadcast(q::Expr)
    ci = first(Meta.lower(LoopVectorization, q).args).code
    nargs = length(ci)-1
    ex = Expr(:block,)
    syms = [gensym() for _ ∈ 1:nargs]
    for n ∈ 1:nargs
        ciₙ = ci[n]
        ciₙargs = ciₙ.args
        f = first(ciₙargs)
        if ciₙ.head === :(=)
            push!(ex.args, Expr(:(=), f, syms[((ciₙargs[2])::Core.SSAValue).id]))
        elseif f === GlobalRef(Base, :materialize!)
            add_ci_call!(ex, lv(:vmaterialize!), ciₙargs, syms, n)
        elseif f === GlobalRef(Base, :materialize)
            add_ci_call!(ex, lv(:vmaterialize), ciₙargs, syms, n)
        else
            add_ci_call!(ex, f, ciₙargs, syms, n)
        end
    end
    ex
end

function LoopSet(q::Expr)
    q = SIMDPirates.contract_pass(q)
    ls = LoopSet()
    copyto!(ls, q)
    resize!(ls.loop_order, num_loops(ls))
    ls
end

macro avx(q)
    q2 = if q.head === :for
        lower(LoopSet(q))
    else# assume broadcast
        substitute_broadcast(q)
    end
    esc(q2)
end

    

