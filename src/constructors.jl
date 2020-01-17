
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

function LoopSet(q::Expr, types::Dict{Symbol,DataType})
    q = SIMDPirates.contract_pass(q)
    ls = LoopSet()
    copyto!(ls, q, types)
    resize!(ls.loop_order, num_loops(ls))
    ls
end


"""
    @avx

Annotate a `for` loop, or a set of nested `for` loops whose bounds are constant across iterations, to optimize the computation. For example:

    function AmulBavx!(C, A, B)
        @avx for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
            Cₘₙ = zero(eltype(C))
            for k ∈ 1:size(A,2)
                Cₘₙ += A[m,k] * B[k,n]
            end
            C[m,n] = Cₘₙ
        end
    end

The macro models the set of nested loops, and chooses a 

It may also apply to broadcasts:

```jldoctest
julia> a = rand(100);

julia> b = @avx exp.(2 .* a);

julia> c = similar(b);

julia> @avx @. c = exp(2a);

julia> b ≈ c 
true
```

"""
macro avx(q)
    q2 = if q.head === :for
        lower(LoopSet(q))
    else# assume broadcast
        substitute_broadcast(q)
    end
    esc(q2)
end
macro avx(arg, q)
    @assert q.head === :for
    @assert arg.head === :(=)
    local U::Int, T::Int
    if arg.args[1] === :unroll
        U = arg.args[2]
        T = -1
    elseif arg.args[1] === :tile
        tup = arg.args[2]
        @assert tup.head === :tuple
        U = tup.args[1]
        T = tup.args[2]
    end
    esc(lower(LoopSet(q), U, T))
end
    

