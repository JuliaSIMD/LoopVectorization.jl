
### This file contains convenience functions for constructing LoopSets.

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

macro avx(q)
    esc(lower(LoopSet(q)))
end

#=
@generated function vmaterialize(
    dest::AbstractArray{T,N}, bc::BC
) where {T,N,BC <: Base.Broadcast.Broadcasted}
    # we have an N dimensional loop.
    # need to construct the LoopSet
    
end
=#

