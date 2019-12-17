
### This file contains convenience functions for constructing LoopSets.

function walk_body!(ls::LoopSet, body::Expr)
    

end
function Base.copyto!(ls::LoopSet, q::Expr)
    q.head === :for || throw("Expression must be a for loop.")
    add_loop!(ls, q)
end

function LoopSet(q::Expr)
    q = SIMDPirates.contract_pass(q)
    ls = LoopSet()
    copyto!(ls, q)
    resize!(ls.loop_order, num_loops(ls))
    ls
end



