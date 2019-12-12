
### This file contains convenience functions for constructing LoopSets.

function loopset_from_expr(qe::Expr)
    q = contract_pass(qe)
    postwalk(q) do ex
        
    end
end



