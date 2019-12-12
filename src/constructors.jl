
### This file contains convenience functions for constructing LoopSets.

function loopset_from_expr(q::Expr)
    q = contract_pass(q)
    
    postwalk(q) do ex
        
    end
end



